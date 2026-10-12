"""
localisation.py - where is the robot on the field?

The IMU gives the heading, so only x, y (field frame, cm) are estimated. Three layers:

1. position_from_goals(): rough position straight from the two goals (no history needed).
   Used to seed / rescue the particle filter.
2. Line matching: white line points from the camera, rotated into field axes, are compared with the field
   model's distance map (field.py). A pose is good when the seen points land on the drawn lines.
3. Particle filter: a few hundred guesses of (x, y), moved each frame, re-weighted by how well the lines and goals
   match, resampled. Handles "I can only see one straight line so I could be anywhere along it".

Later: correct_walls() takes the 8 ultrasonic distances to the wall. It is written and ready, nothing calls it yet.

Field frame: origin = centre, +y = goal we attack. "relative" = robot -> thing vector with field axes.
"""
import math
import threading

import numpy as np

import field
from utils import rotate, rotate_points

# ---- TUNE
N_PARTICLES = 300
PREDICT_NOISE = 1.5            # cm per update, plus...
PREDICT_NOISE_SPEED = 90.0     # ...cm/s * dt  (how far the robot might move between frames that we don't model)
LINE_SIGMA = 4.0               # cm: how far off a matched line point typically is
LINE_CLIP = 12.0               # cm: points further than this from any line count as this (robust to junk)
LINE_WEIGHT = 10.0             # how many "independent" measurements the line points are worth together
LINE_MIN_POINTS = 8            # fewer white points than this = don't use lines this frame
LINE_MAX_POINTS = 120          # subsample to this for speed
GOAL_SIGMA = 6.0               # cm, plus...
GOAL_SIGMA_PER_CM = 0.12       # ...this much per cm of distance (far goals are less accurate in top-down)
GOAL_X_DEADBAND = 3.0          # position_from_goals: goal near point within this of straight ahead = "in front of it"
WALL_SIGMA = 5.0               # ultrasonic
WALL_OUTLIER = 0.2             # chance an ultrasonic reading is junk (hit a robot, etc.)
WALL_MAX_RANGE = 250.0
RANDOM_FRACTION = 0.02         # particles re-spawned uniformly at each resample (recovers from being wrong)
SEED_FRACTION = 0.2            # particles re-spawned around position_from_goals() when lost
CONFIDENT_STD = 15.0           # cm spread below which the estimate is trusted
LOST_STD = 40.0
VEL_SMOOTH = 0.3               # velocity estimate smoothing (0..1, higher = faster response)
VEL_MAX = 250.0                # cm/s


def position_from_goals(attack_near, own_near):
    """Rough (x, y, x_known) from the goal near points (relative, cm), or None if no goal is seen.
    The near point of a goal is the end of the goal mouth closest to the robot, so it gives y exactly but
    x only when the robot is beside the goal (|x| > GOAL_W / 2); in front of the goal x is just 'somewhere in +-30'."""
    ys, xs = [], []
    for near, gy in ((attack_near, field.ATTACK_GOAL_Y), (own_near, field.OWN_GOAL_Y)):
        if near is None:
            continue
        ys.append(gy - near[1])
        if near[0] > GOAL_X_DEADBAND:        # goal end is to our right -> we're left of the goal
            xs.append(-field.GOAL_W / 2 - near[0])
        elif near[0] < -GOAL_X_DEADBAND:
            xs.append(field.GOAL_W / 2 - near[0])
    if not ys:
        return None
    return (sum(xs) / len(xs) if xs else 0.0), sum(ys) / len(ys), bool(xs)


class Localiser:
    def __init__(self, fmap=None, n=N_PARTICLES, seed=None):
        self.map = fmap or field.FieldMap()
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.xmax = field.WALL_W / 2 - 5.0
        self.ymax = field.WALL_L / 2 - 5.0
        self.vel = np.zeros(2)
        self._last_est = None
        self._last_t = None
        self.reset_uniform()

    # ---- particles
    def _uniform(self, k):
        return np.column_stack([self.rng.uniform(-self.xmax, self.xmax, k), self.rng.uniform(-self.ymax, self.ymax, k)])

    def reset_uniform(self):
        """Forget everything (robot was picked up / just started)."""
        self.p = self._uniform(self.n)
        self.logw = np.zeros(self.n)
        self.vel[:] = 0
        self._last_est = None

    def reset_to(self, x, y, spread=10.0):
        self.p = np.column_stack([self.rng.normal(x, spread, self.n), self.rng.normal(y, spread, self.n)])
        self._clamp()
        self.logw = np.zeros(self.n)

    def _clamp(self):
        np.clip(self.p[:, 0], -self.xmax, self.xmax, out=self.p[:, 0])
        np.clip(self.p[:, 1], -self.ymax, self.ymax, out=self.p[:, 1])

    # ---- motion
    def predict(self, dt):
        dt = min(max(dt, 0.0), 0.2)
        sigma = PREDICT_NOISE + PREDICT_NOISE_SPEED * dt
        self.p += self.vel * dt + self.rng.normal(0.0, sigma, self.p.shape)
        self._clamp()

    # ---- measurements (all "relative" = field axes, robot at the origin)
    def correct_lines(self, pts_rel):
        if len(pts_rel) < LINE_MIN_POINTS:
            return False
        if len(pts_rel) > LINE_MAX_POINTS:
            pts_rel = pts_rel[self.rng.choice(len(pts_rel), LINE_MAX_POINTS, replace=False)]
        xs = self.p[:, 0:1] + pts_rel[None, :, 0]
        ys = self.p[:, 1:2] + pts_rel[None, :, 1]
        d = self.map.lookup(xs, ys, clip=LINE_CLIP)
        cost = np.mean(d * d, axis=1)
        self.logw += -LINE_WEIGHT * cost / (2 * LINE_SIGMA ** 2)
        return True

    def correct_goal(self, near_rel, goal_y):
        gx, gy = field.goal_near_point(self.p[:, 0], self.p[:, 1], goal_y)
        ex, ey = gx - self.p[:, 0], gy - self.p[:, 1]
        sigma = GOAL_SIGMA + GOAL_SIGMA_PER_CM * math.hypot(*near_rel)
        err2 = (ex - near_rel[0]) ** 2 + (ey - near_rel[1]) ** 2
        self.logw += -err2 / (2 * sigma ** 2)

    def correct_walls(self, ranges, angles_rel, sensor_radius=0.0):
        """Ultrasonic ring (for later). ranges = cm from each sensor (None = no reading),
        angles_rel = each sensor's pointing direction in FIELD axes (robot-frame angle + compass, 0 = +x, CCW)."""
        for r, a in zip(ranges, angles_rel):
            if r is None or r <= 0 or r > WALL_MAX_RANGE:
                continue
            expected = field.wall_distance(self.p[:, 0], self.p[:, 1], a) - sensor_radius
            err = expected - r
            gauss = np.exp(-err * err / (2 * WALL_SIGMA ** 2)) / (WALL_SIGMA * math.sqrt(2 * math.pi))
            self.logw += np.log((1 - WALL_OUTLIER) * gauss + WALL_OUTLIER / WALL_MAX_RANGE)

    # ---- resampling / estimate
    def _weights(self):
        w = np.exp(self.logw - self.logw.max())
        return w / w.sum()

    def resample_if_needed(self, seed_pos=None):
        w = self._weights()
        if 1.0 / np.sum(w * w) >= self.n / 2:
            return
        # systematic resampling
        positions = (self.rng.random() + np.arange(self.n)) / self.n
        idx = np.minimum(np.searchsorted(np.cumsum(w), positions), self.n - 1)
        self.p = self.p[idx]
        self.logw = np.zeros(self.n)
        k = int(self.n * RANDOM_FRACTION)
        if k:
            self.p[self.rng.choice(self.n, k, replace=False)] = self._uniform(k)
        if seed_pos is not None:
            x, y, x_known = seed_pos
            k = int(self.n * SEED_FRACTION)
            sel = self.rng.choice(self.n, k, replace=False)
            self.p[sel, 0] = self.rng.normal(x, 8.0 if x_known else field.GOAL_W / 2, k)
            self.p[sel, 1] = self.rng.normal(y, 8.0, k)
            self._clamp()

    def estimate(self):
        """(x, y, std) - weighted mean and spread in cm."""
        w = self._weights()
        m = w @ self.p
        var = w @ np.sum((self.p - m) ** 2, axis=1)
        return float(m[0]), float(m[1]), float(math.sqrt(var))

    # ---- one full step
    def update(self, t, line_pts_rel, attack_near_rel=None, own_near_rel=None, walls=None):
        """t = capture time. walls = (ranges, angles_rel) from the ultrasonic ring, or None."""
        dt = 0.0 if self._last_t is None else t - self._last_t
        self._last_t = t
        self.predict(dt)

        self.correct_lines(line_pts_rel)
        if attack_near_rel is not None:
            self.correct_goal(attack_near_rel, field.ATTACK_GOAL_Y)
        if own_near_rel is not None:
            self.correct_goal(own_near_rel, field.OWN_GOAL_Y)
        if walls is not None:
            self.correct_walls(*walls)

        x, y, std = self.estimate()
        seed = position_from_goals(attack_near_rel, own_near_rel) if std > LOST_STD else None
        self.resample_if_needed(seed)

        if std < CONFIDENT_STD and self._last_est is not None and dt > 0:
            v = (np.array([x, y]) - self._last_est) / dt
            self.vel = (1 - VEL_SMOOTH) * self.vel + VEL_SMOOTH * v
            speed = float(np.hypot(*self.vel))
            if speed > VEL_MAX:
                self.vel *= VEL_MAX / speed
        elif std >= CONFIDENT_STD:
            self.vel *= 0.5
        self._last_est = np.array([x, y])
        return x, y, std


class Pose:
    __slots__ = ("x", "y", "std", "t")

    def __init__(self, x=0.0, y=0.0, std=1e3, t=0.0):
        self.x, self.y, self.std, self.t = x, y, std, t

    @property
    def confident(self):
        return self.std < CONFIDENT_STD


class LocalisationThread(threading.Thread):
    """Runs the localiser on every new vision frame.
    attack_colour() must return "yellow" or "blue" (the goal we shoot at)."""

    def __init__(self, vision, attack_colour):
        super().__init__()
        self.daemon = True
        self.running = True
        self.vision = vision
        self.attack_colour = attack_colour
        self.loc = Localiser()
        self.lock = threading.Lock()
        self.pose = Pose()
        self.walls = None            # set to (ranges, angles_rel) once the ultrasonic ring exists
        self._reset = False

    def relocalise(self):
        """Call when the robot is picked up / put down somewhere new."""
        self._reset = True

    def run(self):
        last_id = -1
        while self.running:
            last_id, det = self.vision.wait_new(last_id, timeout=0.5)
            if det is not None:
                self.process(det)

    def process(self, det):
        """One camera frame. The thread above and simulator.py both use exactly this."""
        if self._reset:
            self._reset = False
            self.loc.reset_uniform()
        att = self.attack_colour()
        own = "blue" if att == "yellow" else "yellow"
        c = det.compass
        g_att, g_own = det.goals[att], det.goals[own]
        x, y, std = self.loc.update(
            det.t, rotate_points(det.line_pts, c),
            None if g_att is None else rotate(g_att.near, c),
            None if g_own is None else rotate(g_own.near, c),
            self.walls)
        with self.lock:
            self.pose = Pose(x, y, std, det.t)

    def get(self):
        with self.lock:
            return self.pose

    def particles(self):
        return self.loc.p.copy()

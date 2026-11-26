"""Everything the robot knows about the game (World), worked out from the camera and the PCB.
Inputs: the camera's Detections (vision.py), the IMU heading, the LDR line sensors and IR ring (PCB),
and the teammate's messages. Output: one World object for the strategy: own position (pose), ball,
goals, other robots, line / edge state and where each came from.

Sections:
  1. Line sensors: is the robot on a white line, and which way is back in (LineFusion, limit_outward)
  2. Localisation: where the robot is on the field from the lines and goals the camera sees (Localiser, Pose)
  3. Goals seen by the camera: which way we face, own goal in front of the kicker
  4. World: ball, goals, other robots and the line state put together each loop (World.update)"""
import math
import threading
import time

import numpy as np

import field
import robot_config as cfg
from utils import PointTracker, other_goal, rotate, rotate_points, wrap_pi
from vision import Goal


# ======== 1. Line sensors

# TUNE
LDR_VECTOR_IS_ESCAPE = True
CAM_NEAR = cfg.ROBOT_RADIUS + 10.0
CAM_MIN_POINTS = 4
BOUNDARY_BAND = field.LINE_W + 0.5
IGNORE_INNER_LINES = False
LDR_REACH = cfg.LDR_RING_RADIUS + field.LINE_W
INNER_CLEARANCE = LDR_REACH + 5.0
INNER_VETO = LDR_REACH + 6.0
PENALTY_LINE_NEAR = LDR_REACH + 5.0
SLOW_START = 20.0
SLOW_STOP = 2.0
OUTWARD_STD = 15.0
W_LDR, W_CAM, W_POSE = 1.0, 0.7, 1.0
TOUCH_WINDOW = 3
SPEED_MULTI = {0: 1, 1: 0.6, 2: 0.5, 3: 0.3, 4: 0.1}
SPEED_MULTI_MANY_TOUCHES = 0.1
LATCH_OUTSIDE_TIME = 0.6   # TUNE: s the latched escape is kept while all LDRs are off and the robot is past the line
FIRST_HIT_MIN_SPEED = 0.2 * cfg.BASE_SPEED
FIRST_HIT_ALONG = 0.3
LATCH_STRONG = 0.5
LATCH_FLIP = 0.5
LATCH_SAME = 0.7
MOVE_SMOOTH = 0.05


def _unit(v):
    n = math.hypot(v[0], v[1])
    return None if n < 1e-9 else [v[0] / n, v[1] / n]


class LineState:
    def __init__(self):
        self.on_line = False
        self.escape = None
        self.ldr_hit = False
        self.ldr_count = 0
        self.cam_near = False
        self.cam_under = False
        self.source = ""
        self.speed_multi = 1.0
        self.touches = 0


class LineFusion:
    def __init__(self, threshold=cfg.LDR_LINE_THRESHOLD):
        self.threshold = threshold
        self.was_on_line = False
        self.touch_times = []
        self.latch = None
        self.flipped = False
        self.clear_since = None
        self.recent_move = [0.0, 0.0]

    def ldr_vector(self, colours, compass):
        lx = ly = 0.0
        hits = 0
        for i, value in enumerate(colours):
            if value < self.threshold:
                angle = i * (2 * math.pi / len(colours)) + cfg.LDR_START_ANGLE + compass
                lx += math.cos(angle)
                ly += math.sin(angle)
                hits += 1
        return [lx, ly], hits

    @staticmethod
    def boundary_escape(pose):
        ex = ey = 0.0
        lim_x, lim_y = out_limits(pose.x)
        if abs(pose.x) > lim_x - SLOW_STOP:
            ex = -math.copysign(1.0, pose.x)
        if abs(pose.y) > lim_y - SLOW_STOP:
            ey = -math.copysign(1.0, pose.y)
        return _unit([ex, ey])

    def update(self, colours, compass, line_pts_rel, pose, moving=None):
        state = LineState()
        confident = pose is not None and pose.confident
        precise = pose is not None and pose.std < cfg.PRECISE_STD

        ldr_vec, state.ldr_count = self.ldr_vector(colours, compass)
        state.ldr_hit = state.ldr_count > 0
        if not LDR_VECTOR_IS_ESCAPE:
            ldr_vec = [-ldr_vec[0], -ldr_vec[1]]
        ldr_escape, ldr_outside = self._latched_escape(ldr_vec, state.ldr_count, moving)
        cam_escape = self._camera_escape(state, line_pts_rel, pose, confident)
        pose_escape = self.boundary_escape(pose) if confident else None

        if precise:
            lim_x, lim_y = out_limits(pose.x)
            state.on_line = abs(pose.x) > lim_x or abs(pose.y) > lim_y
        else:
            inner_line = (IGNORE_INNER_LINES and confident and near_penalty_line(pose) and
                          field.inside_play_area(pose.x, pose.y, INNER_CLEARANCE) and not state.cam_under)
            state.on_line = (state.ldr_hit or ldr_outside) and not inner_line

        past_line = confident and not field.inside_play_area(pose.x, pose.y, 0.0)
        if state.on_line and past_line:
            ex = -math.copysign(1.0, pose.x) if abs(pose.x) > field.PLAY_W / 2 else 0.0
            ey = -math.copysign(1.0, pose.y) if abs(pose.y) > field.boundary_y(pose.x) else 0.0
            state.escape, state.source = _unit([ex, ey]) or pose_escape, "pose (past line)"
        elif state.on_line:
            ex = ey = 0.0
            sources = []
            for vec, weight, name in ((ldr_escape, W_LDR, "ldr"), (cam_escape, W_CAM, "cam"),
                                      (pose_escape, W_POSE, "pose")):
                if vec is not None:
                    ex += weight * vec[0]
                    ey += weight * vec[1]
                    sources.append(name)
            state.escape = _unit([ex, ey]) or ldr_escape or pose_escape
            state.source = "+".join(sources)
            if confident and state.escape is not None:
                state.escape = inward_escape(state.escape, pose)

        state.touches = self._count_touches(state.on_line)
        state.speed_multi = SPEED_MULTI.get(state.touches, SPEED_MULTI_MANY_TOUCHES)
        return state

    def _latched_escape(self, ldr_vec, hits, moving):
        # Once the robot centre is past the middle of the line, the lit LDRs on the ring (cfg.LDR_RING_RADIUS) are on the
        # inner side and their vector points out. The escape is kept in self.latch: it follows strong one-sided readings,
        # holds while the reading is weak (centre near the middle of the line) and stays put when the reading turns
        # round (flipped). After the LDRs clear while flipped it is kept for LATCH_OUTSIDE_TIME. A new direction that
        # points the way the robot has been driving (self.recent_move, smoothed Mover.vel) was read already past the
        # middle of that line, so it is turned round.
        if moving is not None:
            self.recent_move = [m + (v - m) * MOVE_SMOOTH for m, v in zip(self.recent_move, moving)]
        now = time.monotonic()
        if hits:
            escape = _unit(ldr_vec)
            self.clear_since = None
            if escape is None:
                return self.latch, False
            strength = math.hypot(ldr_vec[0], ldr_vec[1]) / hits
            along = escape[0] * self.latch[0] + escape[1] * self.latch[1] if self.latch else -1.0
            if self.latch is not None and strength < LATCH_STRONG:
                return self.latch, False
            if self.latch is not None and along < -LATCH_FLIP:
                self.flipped = True
            elif self.latch is None or along > LATCH_FLIP or not self.flipped:
                self.latch, self.flipped = escape, False
                if along < LATCH_SAME and self._driving_along(escape):
                    self.latch, self.flipped = [-escape[0], -escape[1]], True
            return self.latch, False
        if self.latch is None:
            return None, False
        if self.clear_since is None:
            self.clear_since = now
        if self.flipped and now - self.clear_since < LATCH_OUTSIDE_TIME:
            return self.latch, True
        self.latch, self.flipped = None, False
        return None, False

    def _driving_along(self, escape):
        m = self.recent_move
        speed = math.hypot(m[0], m[1])
        return speed > FIRST_HIT_MIN_SPEED and (escape[0] * m[0] + escape[1] * m[1]) / speed > FIRST_HIT_ALONG

    @staticmethod
    def _camera_escape(state, pts, pose, confident):
        if not len(pts):
            return None
        near = pts[pts[:, 0] ** 2 + pts[:, 1] ** 2 < CAM_NEAR ** 2]
        if confident and len(near):
            fx, fy = near[:, 0] + pose.x, near[:, 1] + pose.y
            on_boundary = ((np.abs(fx) > field.PLAY_W / 2 - BOUNDARY_BAND) |
                           (np.abs(fy) > field.PLAY_L / 2 - BOUNDARY_BAND))
            near = near[on_boundary]
        if len(near) < CAM_MIN_POINTS:
            return None
        state.cam_near = True
        state.cam_under = int((near[:, 0] ** 2 + near[:, 1] ** 2 < INNER_VETO ** 2).sum()) >= CAM_MIN_POINTS
        weights = 1.0 / np.maximum(np.hypot(near[:, 0], near[:, 1]), 1.0)
        toward = (near * weights[:, None]).sum(axis=0)
        return _unit([-toward[0], -toward[1]])

    def _count_touches(self, on_line):
        now = time.monotonic()
        if on_line and not self.was_on_line:
            self.touch_times.append(now)
        self.was_on_line = on_line
        while self.touch_times and now - self.touch_times[0] > TOUCH_WINDOW:
            self.touch_times.pop(0)
        return len(self.touch_times)


def inward_escape(escape, pose):
    ex, ey = escape
    inward = [0.0, 0.0]
    if abs(pose.x) > field.PLAY_W / 2 - INNER_CLEARANCE:
        inward[0] = -math.copysign(1.0, pose.x)
        if ex * pose.x > 0:
            ex = 0.0
    if abs(pose.y) > field.boundary_y(pose.x) - INNER_CLEARANCE:
        inward[1] = -math.copysign(1.0, pose.y)
        if ey * pose.y > 0:
            ey = 0.0
    if (ex, ey) == tuple(escape):
        return escape
    return _unit([ex + inward[0], ey + inward[1]]) or escape


def near_penalty_line(pose):
    return any(abs(field.penalty_depth(pose.x, pose.y, sgn)) < PENALTY_LINE_NEAR for sgn in (1.0, -1.0))


def out_limits(x):
    return (field.PLAY_W / 2 + cfg.MAX_OUT,
            field.boundary_y(x, cfg.ROBOT_RADIUS) + cfg.MAX_OUT)


def limit_outward(vec, pose, lead=(0.0, 0.0)):
    if pose is None or pose.std >= OUTWARD_STD:
        return vec
    unsure = max(0.0, pose.std - cfg.PRECISE_STD)
    out = list(vec)
    lim_x, lim_y = out_limits(pose.x)
    for axis, limit in ((0, lim_x), (1, lim_y)):
        position = pose.x if axis == 0 else pose.y
        if out[axis] * position <= 0:
            continue
        ahead = max(0.0, lead[axis] * math.copysign(1.0, position))
        gap = limit - unsure - abs(position) - ahead
        out[axis] *= min(max((gap - SLOW_STOP) / (SLOW_START - SLOW_STOP), 0.0), 1.0)
    return out


# ======== 2. Localisation

# ---- TUNE
N_PARTICLES = 300
PREDICT_NOISE = 1.5
PREDICT_NOISE_SPEED = 90.0
LINE_SIGMA = 4.0
LINE_CLIP = 12.0
LINE_WEIGHT = 30.0
LINE_MIN_POINTS = 8
LINE_MAX_POINTS = 120
GOAL_SIGMA = 6.0
GOAL_SIGMA_PER_CM = 0.12
GOAL_OUTLIER = 0.05
GOAL_X_DEADBAND = 3.0
WALL_SIGMA = 5.0
WALL_OUTLIER = 0.2
WALL_MAX_RANGE = 250.0
RANDOM_FRACTION = 0.02
SEED_FRACTION = 0.2
CONFIDENT_STD = 15.0
LOST_STD = 40.0
KIDNAP_DIST = 50.0
KIDNAP_FRAMES = 15
KIDNAP_SPREAD = 25.0
VEL_SMOOTH = 0.3
VEL_MAX = 250.0
HEADING_FIX = True
HEADING_MIN_POINTS = 30
HEADING_TRY = math.radians(1.5)
HEADING_SHIFT = 3.0
HEADING_GAIN = 0.15
HEADING_MAX = math.radians(25.0)
HEADING_FIT = 1.5
HEADING_SETTLE = 15


def position_from_goals(attack_near, own_near):
    ys, xs = [], []
    for near, gy in ((attack_near, field.ATTACK_GOAL_Y), (own_near, field.OWN_GOAL_Y)):
        if near is None:
            continue
        ys.append(gy - near[1])
        if near[0] > GOAL_X_DEADBAND:
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
        self.dh = 0.0
        self._settled = 0
        self._kidnap = 0
        self._last_est = None
        self._last_t = None
        self.reset_uniform()

    def _uniform(self, k):
        return np.column_stack([self.rng.uniform(-self.xmax, self.xmax, k), self.rng.uniform(-self.ymax, self.ymax, k)])

    def reset_uniform(self):
        self.p = self._uniform(self.n)
        self.logw = np.zeros(self.n)
        self.vel[:] = 0
        self._last_est = None
        self._settled = 0

    def reset_to(self, x, y, spread=10.0):
        self.p = np.column_stack([self.rng.normal(x, spread, self.n), self.rng.normal(y, spread, self.n)])
        self._clamp()
        self.logw = np.zeros(self.n)

    def _clamp(self):
        np.clip(self.p[:, 0], -self.xmax, self.xmax, out=self.p[:, 0])
        np.clip(self.p[:, 1], -self.ymax, self.ymax, out=self.p[:, 1])

    def predict(self, dt):
        dt = min(max(dt, 0.0), 0.2)
        sigma = PREDICT_NOISE + PREDICT_NOISE_SPEED * dt
        self.p += self.vel * dt + self.rng.normal(0.0, sigma, self.p.shape)
        self._clamp()

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
        gx, gy = field.goal_seen_point(self.p[:, 0], self.p[:, 1], goal_y)
        ex, ey = gx - self.p[:, 0], gy - self.p[:, 1]
        sigma = GOAL_SIGMA + GOAL_SIGMA_PER_CM * math.hypot(*near_rel)
        err2 = (ex - near_rel[0]) ** 2 + (ey - near_rel[1]) ** 2
        self.logw += np.log(np.exp(-err2 / (2 * sigma ** 2)) + GOAL_OUTLIER)

    def correct_walls(self, ranges, angles_rel, sensor_radius=0.0):
        for r, a in zip(ranges, angles_rel):
            if r is None or r <= 0 or r > WALL_MAX_RANGE:
                continue
            expected = field.wall_distance(self.p[:, 0], self.p[:, 1], a) - sensor_radius
            err = expected - r
            gauss = np.exp(-err * err / (2 * WALL_SIGMA ** 2)) / (WALL_SIGMA * math.sqrt(2 * math.pi))
            self.logw += np.log((1 - WALL_OUTLIER) * gauss + WALL_OUTLIER / WALL_MAX_RANGE)

    def _weights(self):
        w = np.exp(self.logw - self.logw.max())
        return w / w.sum()

    def resample_if_needed(self, seed_pos=None):
        w = self._weights()
        if 1.0 / np.sum(w * w) >= self.n / 2:
            return
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
        w = self._weights()
        m = w @ self.p
        var = w @ np.sum((self.p - m) ** 2, axis=1)
        return float(m[0]), float(m[1]), float(math.sqrt(var))

    _TURNS = np.arange(-2, 3)
    _SHIFTS = np.array([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)], float)

    def refine_heading(self, pts_rel, x, y):
        if len(pts_rel) < HEADING_MIN_POINTS:
            return
        if len(pts_rel) > LINE_MAX_POINTS:
            pts_rel = pts_rel[self.rng.choice(len(pts_rel), LINE_MAX_POINTS, replace=False)]
        costs = []
        for k in self._TURNS:
            c, s = math.cos(k * HEADING_TRY), math.sin(k * HEADING_TRY)
            px = pts_rel[:, 0] * c - pts_rel[:, 1] * s
            py = pts_rel[:, 0] * s + pts_rel[:, 1] * c
            sx = x + HEADING_SHIFT * self._SHIFTS[:, 0:1] + px[None, :]
            sy = y + HEADING_SHIFT * self._SHIFTS[:, 1:2] + py[None, :]
            d = self.map.lookup(sx, sy, clip=LINE_CLIP)
            costs.append(float(np.min(np.mean(d * d, axis=1))))
        costs = np.array(costs)
        if costs[len(costs) // 2] > (HEADING_FIT * LINE_SIGMA) ** 2:
            return
        a, b, _ = np.polyfit(self._TURNS, costs, 2)
        if a > 1e-9:
            best = float(np.clip(-b / (2 * a), self._TURNS[0], self._TURNS[-1]))
        else:
            best = float(self._TURNS[int(np.argmin(costs))])
        self.dh = float(np.clip(self.dh + HEADING_GAIN * best * HEADING_TRY, -HEADING_MAX, HEADING_MAX))

    def update(self, t, line_pts_rel, attack_near_rel=None, own_near_rel=None, walls=None):
        dt = 0.0 if self._last_t is None else t - self._last_t
        self._last_t = t
        self.predict(dt)
        if self.dh:
            line_pts_rel = rotate_points(line_pts_rel, self.dh)
            attack_near_rel = None if attack_near_rel is None else rotate(attack_near_rel, self.dh)
            own_near_rel = None if own_near_rel is None else rotate(own_near_rel, self.dh)
            if walls is not None:
                walls = (walls[0], [a + self.dh for a in walls[1]])

        self.correct_lines(line_pts_rel)
        if attack_near_rel is not None:
            self.correct_goal(attack_near_rel, field.ATTACK_GOAL_Y)
        if own_near_rel is not None:
            self.correct_goal(own_near_rel, field.OWN_GOAL_Y)
        if walls is not None:
            self.correct_walls(*walls)

        x, y, std = self.estimate()
        from_goals = position_from_goals(attack_near_rel, own_near_rel)
        if std < CONFIDENT_STD and from_goals is not None and abs(from_goals[1] - y) > KIDNAP_DIST:
            self._kidnap += 1
        else:
            self._kidnap = 0
        if self._kidnap >= KIDNAP_FRAMES:
            self._kidnap = 0
            self.reset_to(from_goals[0], from_goals[1], KIDNAP_SPREAD)
            self._settled = 0
            self._last_est = None
            x, y, std = self.estimate()
        seed = from_goals if std > LOST_STD else None
        self.resample_if_needed(seed)
        self._settled = self._settled + 1 if std < CONFIDENT_STD else 0
        if HEADING_FIX and self._settled > HEADING_SETTLE:
            self.refine_heading(line_pts_rel, x, y)

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
    __slots__ = ("x", "y", "std", "t", "dh")

    def __init__(self, x=0.0, y=0.0, std=1e3, t=0.0, dh=0.0):
        self.x, self.y, self.std, self.t, self.dh = x, y, std, t, dh

    @property
    def confident(self):
        return self.std < CONFIDENT_STD


class LocalisationThread(threading.Thread):

    def __init__(self, vision, attack_colour):
        super().__init__()
        self.daemon = True
        self.running = True
        self.vision = vision
        self.attack_colour = attack_colour
        self.loc = Localiser()
        self.lock = threading.Lock()
        self.pose = Pose()
        self.walls = None
        self._reset = False

    def relocalise(self):
        self._reset = True

    def run(self):
        last_id = -1
        errors = 0
        while self.running:
            try:
                last_id, det = self.vision.wait_new(last_id, timeout=0.5)
                if det is not None:
                    self.process(det)
                errors = 0
            except Exception as e:
                errors += 1
                if errors == 1:
                    print(f"[localisation] frame failed: {e!r}")
                time.sleep(0.01)

    def process(self, det):
        if self._reset:
            self._reset = False
            self.loc.reset_uniform()
            self.loc.dh = 0.0
        att = self.attack_colour()
        own = other_goal(att)
        c = det.compass
        g_att, g_own = det.goals[att], det.goals[own]
        x, y, std = self.loc.update(
            det.t, rotate_points(det.line_pts, c),
            None if g_att is None else rotate(g_att.near, c),
            None if g_own is None else rotate(g_own.near, c),
            self.walls)
        with self.lock:
            self.pose = Pose(x, y, std, det.t, self.loc.dh)

    def get(self):
        with self.lock:
            return self.pose

    def particles(self):
        return self.loc.p.copy()

# ======== 3. Goals seen by the camera

# ---- TUNE
CAM_HEADING_SNAP = math.radians(35)
MIN_GOAL_SEPARATION = 50.0
OWN_GOAL_NO_KICK = math.radians(30)
GOAL_CENTRE_PAST_BACK = 1.0      # cm behind the goal's back wall where the camera's goal centre lies
GOAL_POSE_STD = 13.0             # std given to a position worked out from the goal centres


def goal_colour_from_view(det, current):
    if det is None:
        return current
    yellow, blue = det.goals["yellow"], det.goals["blue"]
    if yellow is not None:
        return "yellow" if yellow.centre[1] > 0 else "blue"
    if blue is not None:
        return "blue" if blue.centre[1] > 0 else "yellow"
    return current


def _snap_to_axis(heading):
    if abs(heading) < CAM_HEADING_SNAP:
        return 0.0
    if abs(wrap_pi(heading - math.pi)) < CAM_HEADING_SNAP:
        return math.pi
    return heading


def heading_from_goals(det, attack, both_only=False):
    if det is None:
        return None
    ahead, behind = det.goals.get(attack), det.goals.get(other_goal(attack))
    if ahead is not None and behind is not None:
        vx, vy = ahead.centre[0] - behind.centre[0], ahead.centre[1] - behind.centre[1]
        return math.atan2(vx, vy) if vx * vx + vy * vy > MIN_GOAL_SEPARATION ** 2 else None
    if both_only or (ahead is None and behind is None):
        return None
    if ahead is not None:
        return _snap_to_axis(math.atan2(ahead.centre[0], ahead.centre[1]))
    return _snap_to_axis(wrap_pi(math.atan2(behind.centre[0], behind.centre[1]) + math.pi))


def position_from_goal_centres(attack_goal, own_goal):
    centre_y = field.ATTACK_GOAL_Y + field.GOAL_DEPTH + GOAL_CENTRE_PAST_BACK
    xs, ys, ws = [], [], []
    for goal, gy in ((attack_goal, centre_y), (own_goal, -centre_y)):
        if goal is None:
            continue
        w = 1.0 / max(math.hypot(goal.centre[0], goal.centre[1]), 20.0)
        xs.append(-goal.centre[0] * w)
        ys.append((gy - goal.centre[1]) * w)
        ws.append(w)
    if not ws:
        return None
    return sum(xs) / sum(ws), sum(ys) / sum(ws)


def own_goal_ahead(det, attack):
    if det is None:
        return False
    own = det.goals.get(other_goal(attack))
    return own is not None and own.centre[1] > 0 and abs(math.atan2(own.centre[0], own.centre[1])) < OWN_GOAL_NO_KICK


# ======== 4. World

# ---- TUNE
BALL_LOST_TIME = 0.3
CAPTURE_HOLD = 0.3
BALL_VEL_SMOOTH = 0.3
BALL_VEL_MAX = 400.0
OBSTACLE_MATCH = 20.0
OBSTACLE_MIN_HITS = 3
OBSTACLE_LOST_TIME = 0.3
OBSTACLE_MEMORY = 0.4
OBSTACLE_VISIBLE = 70.0
MATE_MATCH = cfg.ROBOT_RADIUS + 15.0
OPP_MATCH = 15.0
OPP_VEL_WINDOW = 0.3
OPP_VEL_MIN_SPAN = 0.1
OPP_TRACK_LOST = 0.3
OPP_MERGE = 15.0
IR_UNKNOWN_RANGE = 40.0       # TUNE: cm, assumed distance of a ball only the IR ring sees, with no recent camera sighting
IR_RANGE_MEMORY = 1.5
IR_RING_R = cfg.ROBOT_RADIUS + 1.0
IR_CAPTURE_ANGLE = math.radians(20)
IR_CAPTURE_R = cfg.ROBOT_RADIUS + 3.0
NEAR_HOLD_DIST = 25.0
NEAR_HOLD_TIME = 0.6
NO_IR_CAPTURE_MAX = 3.0


def front_of(o):
    if getattr(o, "centre", None) is None:
        return o.near
    d = math.hypot(o.centre[0], o.centre[1]) or 1.0
    k = max(d - cfg.ROBOT_RADIUS, 0.0) / d
    return [o.centre[0] * k, o.centre[1] * k]


def fit_velocity(track):
    n = len(track)
    mt = sum(p[0] for p in track) / n
    mx = sum(p[1] for p in track) / n
    my = sum(p[2] for p in track) / n
    var = sum((p[0] - mt) ** 2 for p in track)
    if var < 1e-9:
        return [0.0, 0.0]
    return [sum((p[0] - mt) * (p[1] - mx) for p in track) / var, sum((p[0] - mt) * (p[2] - my) for p in track) / var]


def centre_of(pose, near):
    dx, dy = near[0] - pose.x, near[1] - pose.y
    d = math.hypot(dx, dy) or 1.0
    return (near[0] + dx / d * cfg.ROBOT_RADIUS, near[1] + dy / d * cfg.ROBOT_RADIUS)


def ir_bearing(ir):
    if not ir:
        return None
    n = len(ir)
    vx = vy = 0.0
    for i, v in enumerate(ir):
        if v > 0:
            a = cfg.IR_START_ANGLE + i * 2 * math.pi / n
            vx += math.cos(a)
            vy += math.sin(a)
    if vx == 0 and vy == 0:
        return None
    return math.atan2(vy, vx)


def in_capture_zone(ball_robot):
    if ball_robot is None:
        return False
    x0, x1, y0, y1 = cfg.CAPTURE_ZONE
    return x0 <= ball_robot[0] <= x1 and y0 <= ball_robot[1] <= y1


class ObstacleTracker:

    def __init__(self):
        self.tracks = []

    def update(self, dets, now):
        for d in dets:
            best, best_d = None, OBSTACLE_MATCH
            for tr in self.tracks:
                dist = math.hypot(tr["pos"][0] - d[0][0], tr["pos"][1] - d[0][1])
                if dist < best_d and tr["seen"] != now:
                    best, best_d = tr, dist
            if best is None:
                self.tracks.append({"pos": d[0], "size": d[1], "hits": 1, "seen": now})
            else:
                best.update(pos=d[0], size=d[1], hits=best["hits"] + 1, seen=now)
        self.tracks = [tr for tr in self.tracks if now - tr["seen"] < OBSTACLE_LOST_TIME]
        return [(tr["pos"], tr["size"]) for tr in self.tracks if tr["hits"] >= OBSTACLE_MIN_HITS]

    def reset(self):
        self.tracks = []


class World:
    def __init__(self):
        self.ball_tracker = PointTracker(history=3, tolerance=30.0, lost_time=BALL_LOST_TIME)
        self.obstacle_tracker = ObstacleTracker()
        self.last_capture = 0.0
        self.frame_id = -1
        self.line_fusion = LineFusion()
        self.line = LineState()
        self.compass = 0.0
        self.reset_values()

    def reset_values(self):
        self.ball = None
        self.ball_source = "-"
        self.ball_own = None
        self.ball_own_field = None
        self.ball_robot = None
        self.ball_field = None
        self.ball_vel = [0.0, 0.0]
        self.prev_ball = None
        self.mate_pos = None
        self.ball_in_capture = False
        self.ball_hidden = None
        self.ir = None
        self.last_cam_ball = None
        self.attack_goal = None
        self.own_goal = None
        self.goal_source = ""
        self.line_pts = np.zeros((0, 2), np.float32)
        self.obstacles = []
        self.obstacles_field = []
        self.obstacle_memory = []
        self.opponents_own = []
        self.opponents = []
        self.opponent_vel = []
        self.opp_tracks = []
        self.pose = None
        self.field_visible = False

    def reset(self):
        self.ball_tracker.reset()
        self.obstacle_tracker.reset()
        self.last_capture = 0.0
        self.reset_values()

    def update(self, frame_id, det, pose, attack_colour, mate=None, ir=None, colours=None, compass=0.0, moving=None):
        """One loop of sensing. det = camera Detections, pose = Pose from localisation, mate = teammate message,
        ir = IR ring readings, colours = LDR ring readings, compass = heading (IMU + pose.dh),
        moving = the velocity the robot is driving at (Mover.vel), used to tell which side of a line it came from.
        The camera part only runs on a new frame; the line state (self.line) is updated every call."""
        self.pose = pose
        self.compass = compass
        if det is not None and frame_id != self.frame_id:
            self._update_camera(frame_id, det, pose, attack_colour, mate, ir)
        if colours is not None:
            self.line = self.line_fusion.update(colours, compass, self.line_pts, self.line_pose(pose), moving)
        return self

    def line_pose(self, pose):
        if pose is not None and pose.confident:
            return pose
        me = position_from_goal_centres(self.attack_goal, self.own_goal)
        if me is None:
            return pose
        return Pose(me[0], me[1], GOAL_POSE_STD, getattr(pose, "t", 0.0), getattr(pose, "dh", 0.0))

    def _update_camera(self, frame_id, det, pose, attack_colour, mate, ir):
        self.frame_id = frame_id
        now = time.monotonic()
        heading = det.compass + getattr(pose, "dh", 0.0)
        confident = pose is not None and pose.confident
        self.field_visible = det.field_visible

        self._update_own_ball(det, pose, heading, confident, ir, now)
        self._update_goals(det, pose, heading, attack_colour)
        self.line_pts = rotate_points(det.line_pts, heading)
        seen = [(rotate(front_of(o), heading), o.size) for o in det.obstacles]
        self.obstacles = self.obstacle_tracker.update(seen, now)

        mate_on = bool(mate) and mate.get("bot active") == 1
        mate_ball = mate.get("ball") if mate_on else None
        self.mate_pos = mate.get("pos") if mate else None
        if confident:
            self._update_field_frame(pose, mate if mate_on else None, now)
        else:
            self._forget_field_frame()
        self._choose_ball(pose, confident, mate_ball)
        self.update_ball_vel(now, None if self.ball_hidden else self.ball_own_field)

    def _update_own_ball(self, det, pose, heading, confident, ir, now):
        self.ball_robot = det.ball
        self.ball_own = self.ball_tracker.update(None if det.ball is None else rotate(det.ball, heading), now)
        self.ir = ir_bearing(ir)
        self.ball_hidden = None
        if det.ball is not None:
            relative = rotate(det.ball, heading)
            cam_ball_field = [relative[0] + pose.x, relative[1] + pose.y] if confident else None
            self.last_cam_ball = (list(det.ball), cam_ball_field, now)
        else:
            hidden = self.hidden_ball(ir is not None, pose, heading, now)
            if hidden is not None:
                self.ball_robot = hidden
                self.ball_own = rotate(hidden, heading)

        if in_capture_zone(det.ball):
            self.last_capture = now
        ir_capture = (self.ball_hidden == "ir" and math.hypot(*self.ball_robot) < IR_CAPTURE_R
                      and abs(self.ir - math.pi / 2) < IR_CAPTURE_ANGLE)
        if ir_capture:
            self.last_capture = now
        no_ir_blind_ring = ir is None and cfg.BALL_BLIND_RING > 0
        hold = NO_IR_CAPTURE_MAX if no_ir_blind_ring else CAPTURE_HOLD
        self.ball_in_capture = (in_capture_zone(det.ball) or ir_capture
                                or (det.ball is None and now - self.last_capture < hold))
        if self.ball_in_capture and det.ball is None and no_ir_blind_ring:
            self.ball_robot = [0.0, cfg.ROBOT_RADIUS + 1.0]
            self.ball_own = rotate(self.ball_robot, heading)
            self.ball_hidden = "hold"

    def _update_goals(self, det, pose, heading, attack_colour):
        sources, goals = [], []
        for colour, goal_y in ((attack_colour, field.ATTACK_GOAL_Y), (other_goal(attack_colour), field.OWN_GOAL_Y)):
            seen = det.goals[colour]
            if seen is not None:
                goals.append(Goal(rotate(seen.near, heading), rotate(seen.centre, heading)))
                sources.append("cam")
            elif pose is not None and pose.confident:
                near_x, _ = field.goal_near_point(pose.x, pose.y, goal_y)
                goals.append(Goal([float(near_x) - pose.x, goal_y - pose.y], [-pose.x, goal_y - pose.y]))
                sources.append("pose")
            else:
                goals.append(None)
                sources.append("-")
        self.attack_goal, self.own_goal = goals
        self.goal_source = "/".join(sources)

    def _drop_teammate_obstacle(self, pose):
        best = None
        for i, (p, _) in enumerate(self.obstacles):
            centre = centre_of(pose, [p[0] + pose.x, p[1] + pose.y])
            d = math.hypot(centre[0] - self.mate_pos[0], centre[1] - self.mate_pos[1])
            if d < MATE_MATCH and (best is None or d < best[0]):
                best = (d, i)
        if best is not None:
            self.obstacles = [o for i, o in enumerate(self.obstacles) if i != best[1]]

    def _update_field_frame(self, pose, active_mate, now):
        own = self.ball_own
        self.ball_own_field = None if own is None else [own[0] + pose.x, own[1] + pose.y]
        if self.mate_pos is not None:
            self._drop_teammate_obstacle(pose)
        self.obstacles_field = self.remember_obstacles(
            [([p[0] + pose.x, p[1] + pose.y], s) for p, s in self.obstacles], pose, now)
        self.opponents_own = [centre_of(pose, [p[0] + pose.x, p[1] + pose.y]) for p, _ in self.obstacles]
        self.opponents = self.merge_opponents(pose, active_mate)
        self.opponent_vel = self.opponent_velocities(now)

    def _forget_field_frame(self):
        self.ball_own_field = None
        self.obstacles_field = []
        self.obstacle_memory = []
        self.opponents_own = []
        self.opponents = []
        self.opponent_vel = []

    def _choose_ball(self, pose, confident, mate_ball):
        if self.ball_own is not None:
            self.ball, self.ball_field, self.ball_source = self.ball_own, self.ball_own_field, self.ball_hidden or "cam"
        elif confident and mate_ball is not None:
            self.ball = [mate_ball[0] - pose.x, mate_ball[1] - pose.y]
            self.ball_field, self.ball_source = list(mate_ball), "mate"
        else:
            self.ball, self.ball_field, self.ball_source = None, None, "-"

    def hidden_ball(self, has_ir, pose, c, now):
        last = self.last_cam_ball
        recent = last is not None and now - last[2] < IR_RANGE_MEMORY
        confident = pose is not None and pose.confident
        if has_ir:
            if self.ir is None:
                return None
            if last is not None and math.hypot(last[0][0], last[0][1]) < NEAR_HOLD_DIST:
                r = IR_RING_R      # last seen close and the camera lost it: it's in the blind ring
            elif recent and confident and last[1] is not None:
                dt = min(now - last[2], 0.5)
                r = math.hypot(last[1][0] + self.ball_vel[0] * dt - pose.x, last[1][1] + self.ball_vel[1] * dt - pose.y)
            elif recent:
                r = math.hypot(last[0][0], last[0][1])
            else:
                r = IR_UNKNOWN_RANGE
            r = max(r, cfg.ROBOT_RADIUS)
            self.ball_hidden = "ir"
            return [r * math.cos(self.ir), r * math.sin(self.ir)]
        if (cfg.BALL_BLIND_RING <= 0 or last is None or now - last[2] > NEAR_HOLD_TIME
                or math.hypot(last[0][0], last[0][1]) > NEAR_HOLD_DIST):
            return None
        self.ball_hidden = "hold"
        return list(last[0])

    def update_ball_vel(self, now, b):
        prev = self.prev_ball
        if b is None:
            if prev is None or now - prev[1] > 0.3:
                self.ball_vel, self.prev_ball = [0.0, 0.0], None
            return
        if prev is not None and now - prev[1] > 1e-3:
            dt = now - prev[1]
            v = [(b[0] - prev[0][0]) / dt, (b[1] - prev[0][1]) / dt]
            if math.hypot(v[0], v[1]) < BALL_VEL_MAX:
                k = BALL_VEL_SMOOTH
                self.ball_vel = [self.ball_vel[0] * (1 - k) + v[0] * k, self.ball_vel[1] * (1 - k) + v[1] * k]
        self.prev_ball = (list(b), now)

    def opponent_velocities(self, now):
        tracks = [t for t in self.opp_tracks if now - t[-1][0] < OPP_TRACK_LOST]
        free = list(range(len(tracks)))
        vels = []
        for o in self.opponents:
            k = min(free, key=lambda i: math.hypot(tracks[i][-1][1] - o[0], tracks[i][-1][2] - o[1]), default=None)
            if k is None or math.hypot(tracks[k][-1][1] - o[0], tracks[k][-1][2] - o[1]) > OPP_MATCH:
                tracks.append([(now, o[0], o[1])])
                vels.append([0.0, 0.0])
                continue
            free.remove(k)
            tr = tracks[k]
            tr.append((now, o[0], o[1]))
            while now - tr[0][0] > OPP_VEL_WINDOW:
                tr.pop(0)
            vels.append(fit_velocity(tr) if tr[-1][0] - tr[0][0] >= OPP_VEL_MIN_SPAN else [0.0, 0.0])
        self.opp_tracks = tracks
        return vels

    def merge_opponents(self, pose, mate):
        opps = [c for c in (centre_of(pose, p) for p, _ in self.obstacles_field)
                if field.inside_play_area(c[0], c[1], -cfg.ROBOT_RADIUS)]
        for o in (mate.get("opps") or []) if mate else []:
            if not field.inside_play_area(o[0], o[1], -cfg.ROBOT_RADIUS):
                continue
            if math.hypot(o[0] - pose.x, o[1] - pose.y) < OBSTACLE_VISIBLE:
                continue
            if math.hypot(o[0] - pose.x, o[1] - pose.y) < MATE_MATCH:
                continue
            if all(math.hypot(o[0] - q[0], o[1] - q[1]) > OPP_MERGE for q in opps):
                opps.append((float(o[0]), float(o[1])))
        return opps

    def remember_obstacles(self, seen, pose, now):
        mem = self.obstacle_memory
        for p, size in seen:
            match = min(mem, key=lambda m: math.hypot(m[0][0] - p[0], m[0][1] - p[1]), default=None)
            if match is not None and math.hypot(match[0][0] - p[0], match[0][1] - p[1]) < OBSTACLE_MATCH:
                match[0], match[1], match[2] = p, size, now
            else:
                mem.append([p, size, now])
        fresh = [m for m in mem if m[2] == now]
        kept = []
        for m in mem:
            if m[2] == now:
                kept.append(m)
            elif (now - m[2] < OBSTACLE_MEMORY and math.hypot(m[0][0] - pose.x, m[0][1] - pose.y) > OBSTACLE_VISIBLE
                  and all(math.hypot(m[0][0] - f[0][0], m[0][1] - f[0][1]) > 2 * OBSTACLE_MATCH for f in fresh)
                  and (self.mate_pos is None or
                       math.hypot(m[0][0] - self.mate_pos[0], m[0][1] - self.mate_pos[1]) > MATE_MATCH)):
                kept.append(m)
        self.obstacle_memory = kept
        return [(m[0], m[1]) for m in self.obstacle_memory]

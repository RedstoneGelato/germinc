"""
perception.py - one place that turns the latest camera Detections + localisation Pose into what the strategy uses.

Everything in World is "relative" (field axes, robot at the origin, cm) unless the name says _field (field frame).
"""
import math
import time

import field
import robot_config as cfg
from vision import Goal
import numpy as np

from utils import PointTracker, rotate, rotate_points

# ---- TUNE
BALL_LOST_TIME = 0.3          # seconds to keep the last ball position after losing it
CAPTURE_HOLD = 0.3            # ball vanished into the ignore box right after being in the capture zone = still ours
BALL_VEL_SMOOTH = 0.3         # 0..1: how much of each new frame's ball velocity is taken (rest = previous estimate)
BALL_VEL_MAX = 400.0          # cm/s: faster than this between two frames = a detection jump, not movement
OBSTACLE_MATCH = 20.0         # cm: same obstacle as last frame if this close
OBSTACLE_MIN_HITS = 3         # frames in a row before an obstacle is believed
OBSTACLE_LOST_TIME = 0.3
OBSTACLE_MEMORY = 0.4         # s an obstacle is remembered (field frame) after it drops out of view...
OBSTACLE_VISIBLE = 70.0       # ...unless it's closer than this (cm), where the camera would definitely still see it
MATE_MATCH = cfg.ROBOT_RADIUS + 15.0   # cm: an obstacle this close to where the teammate says it is = the teammate
OPP_MATCH = 15.0              # cm: an opponent this close to where its track was last = the same robot (velocity)
OPP_VEL_WINDOW = 0.3          # s: opponent velocity = straight-line fit through its positions over this long; 0 until
OPP_VEL_MIN_SPAN = 0.1        # it's been tracked OPP_VEL_MIN_SPAN s. A track not matched for OPP_TRACK_LOST s is
OPP_TRACK_LOST = 0.3          # dropped
OPP_MERGE = 15.0              # cm: a teammate's opponent this close to one of ours = the same robot
# the ball where the camera can't see it: the ~2 cm blind ring right round our bumper (a ball in the dribbler is never
# seen), the mirror mount bars' slices, robots in the way. With the IR ring (robot_config.IR_*) its bearing is used,
# at the range the camera last had it (moved on by its velocity) or, without one, from the IR strength
IR_NEAR = 20.0                # TUNE: cm, the IR sensors read full strength (1.0) closer than this, ~IR_NEAR / distance further
IR_CLOSE = 0.8                # strongest IR reading above this = the ball is within IR_NEAR (unseen too = in the blind ring)
IR_RANGE_MEMORY = 1.5         # s the last camera range is still used with the IR bearing
IR_RING_R = cfg.ROBOT_RADIUS + 1.0       # cm, near point of a ball IR says is close but the camera can't see (blind ring)
IR_CAPTURE_ANGLE = math.radians(20)      # ...and this near straight ahead = in the dribbler
IR_CAPTURE_R = cfg.ROBOT_RADIUS + 3.0    # (cm, as long as its estimated near point is within this)
NEAR_HOLD_DIST = 25.0         # no IR: a ball last seen this close (cm) that vanishes is in the blind ring, not gone -
NEAR_HOLD_TIME = 0.6          # kept where it was (robot frame) for this long (s)
NO_IR_CAPTURE_MAX = 3.0       # no IR: s a ball that vanished out of the capture zone still counts as in the dribbler


def front_of(o):
    """An obstacle as the point facing us, made consistent with its estimated centre (vision.Obstacle.centre):
    one robot radius in front of the centre, so `centre_of()` / strategy.obstacle_centres() get the whole robot's
    centre back."""
    if getattr(o, "centre", None) is None:
        return o.near
    d = math.hypot(o.centre[0], o.centre[1]) or 1.0
    k = max(d - cfg.ROBOT_RADIUS, 0.0) / d
    return [o.centre[0] * k, o.centre[1] * k]


def fit_velocity(track):
    """Least-squares slope (cm/s) of x and y against time through [(t, x, y), ...]."""
    n = len(track)
    mt = sum(p[0] for p in track) / n
    mx = sum(p[1] for p in track) / n
    my = sum(p[2] for p in track) / n
    var = sum((p[0] - mt) ** 2 for p in track)
    if var < 1e-9:
        return [0.0, 0.0]
    return [sum((p[0] - mt) * (p[1] - mx) for p in track) / var, sum((p[0] - mt) * (p[2] - my) for p in track) / var]


def centre_of(pose, near):
    """Obstacles are seen as their nearest point; the robot's centre is about one robot radius further away."""
    dx, dy = near[0] - pose.x, near[1] - pose.y
    d = math.hypot(dx, dy) or 1.0
    return (near[0] + dx / d * cfg.ROBOT_RADIUS, near[1] + dy / d * cfg.ROBOT_RADIUS)


def ir_bearing(ir):
    """IR ring readings (0..1 per sensor, 0 = not seeing the ball) -> (bearing, strongest reading, sensors active), or
    None if no sensor sees the ball. bearing: robot frame, radians, 0 = robot's right, anticlockwise (pi/2 = front)."""
    if not ir:
        return None
    n = len(ir)
    vx = vy = 0.0
    active = 0
    for i, v in enumerate(ir):
        if v > 0:
            a = cfg.IR_START_ANGLE + i * 2 * math.pi / n
            vx += v * math.cos(a)
            vy += v * math.sin(a)
            active += 1
    if not active or (vx == 0 and vy == 0):
        return None
    return math.atan2(vy, vx), max(ir), active


def in_capture_zone(ball_robot):
    if ball_robot is None:
        return False
    x0, x1, y0, y1 = cfg.CAPTURE_ZONE
    return x0 <= ball_robot[0] <= x1 and y0 <= ball_robot[1] <= y1


class ObstacleTracker:
    """Keeps obstacles that show up in several frames in a row (shadows / glare flicker, robots don't)."""

    def __init__(self):
        self.tracks = []      # dicts: pos (relative), size, hits, seen

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
        self.reset_values()

    def reset_values(self):
        self.ball = None              # relative, smoothed: our camera, or the teammate's if we can't see it
        self.ball_source = "-"        # "cam" / "mate" / "-"
        self.ball_own = None          # relative, our camera only
        self.ball_own_field = None    # field frame, our camera only (this is what we send to the teammate)
        self.ball_robot = None        # robot frame, raw from the latest frame
        self.ball_field = None        # field frame, ours or the teammate's
        self.ball_vel = [0.0, 0.0]    # field frame cm/s, from our own camera ([0, 0] when unknown)
        self.prev_ball = None         # (field position, time) of the last own-camera ball, for ball_vel
        self.mate_pos = None          # teammate's position (field frame) from comms, or None
        self.ball_in_capture = False
        self.ball_hidden = None       # "ir" / "hold" when the ball is where the camera can't see it (hidden_ball()), else None
        self.ir = None                # ir_bearing() of the latest IR readings, None = no sensor sees it / no IR ring
        self.last_cam_ball = None     # (robot frame point, field point or None, time) of the last camera ball
        self.attack_goal = None       # Goal(near, centre), relative
        self.own_goal = None
        self.goal_source = ""         # "cam" / "pose" per goal, for printing
        self.line_pts = np.zeros((0, 2), np.float32)   # white line points, relative
        self.obstacles = []           # [(relative pos, size cm)]
        self.obstacles_field = []     # [(field pos, size cm)]: seen now + remembered for OBSTACLE_MEMORY
        self.obstacle_memory = []     # [[field pos, size, last seen]]
        self.opponents_own = []       # field frame CENTRES of the opponents our camera sees now (sent to the teammate)
        self.opponents = []           # field frame centres: ours (incl. remembered) + the teammate's that we don't see
        self.opponent_vel = []        # field frame cm/s, one per opponents entry ([0, 0] when it's new)
        self.opp_tracks = []          # [[(t, x, y), ...]] recent positions of each opponent (opponent_velocities())
        self.pose = None
        self.field_visible = False

    def reset(self):
        self.ball_tracker.reset()
        self.obstacle_tracker.reset()
        self.last_capture = 0.0
        self.reset_values()

    def update(self, frame_id, det, pose, attack_colour, mate=None, ir=None):
        """Call every loop; only does work when there's a new camera frame.
        mate = the teammate's latest comms message (or None): its "ball" and "pos" are used.
        ir = the IR ring's readings (0..1 per sensor), None = no IR ring."""
        self.pose = pose
        if det is None or frame_id == self.frame_id:
            return
        self.frame_id = frame_id
        now = time.monotonic()
        c = det.compass + getattr(pose, "dh", 0.0)   # localisation's compass correction (0 without a pose)
        self.field_visible = det.field_visible

        # ball
        confident = pose is not None and pose.confident
        self.ball_robot = det.ball
        self.ball_own = self.ball_tracker.update(None if det.ball is None else rotate(det.ball, c), now)
        self.ir = ir_bearing(ir)
        self.ball_hidden = None
        if det.ball is not None:
            b = rotate(det.ball, c)
            cam_ball_field = [b[0] + pose.x, b[1] + pose.y] if confident else None
            self.last_cam_ball = (list(det.ball), cam_ball_field, now)
        else:
            hidden = self.hidden_ball(ir is not None, pose, c, now)
            if hidden is not None:
                self.ball_robot = hidden
                self.ball_own = rotate(hidden, c)
        if in_capture_zone(det.ball):
            self.last_capture = now
        ir_capture = (self.ball_hidden == "ir" and self.ir[1] >= IR_CLOSE and math.hypot(*self.ball_robot) < IR_CAPTURE_R
                      and abs(self.ir[0] - math.pi / 2) < IR_CAPTURE_ANGLE)
        if ir_capture:
            self.last_capture = now
        # no IR: a ball that vanished from the capture zone went into the blind ring = it's in the dribbler until the
        # camera sees it somewhere again (NO_IR_CAPTURE_MAX at most)
        hold = NO_IR_CAPTURE_MAX if ir is None and cfg.BALL_BLIND_RING > 0 else CAPTURE_HOLD
        self.ball_in_capture = in_capture_zone(det.ball) or ir_capture or \
            (det.ball is None and now - self.last_capture < hold)
        if self.ball_in_capture and det.ball is None and ir is None and cfg.BALL_BLIND_RING > 0:
            self.ball_robot = [0.0, cfg.ROBOT_RADIUS + 1.0]
            self.ball_own = rotate(self.ball_robot, c)
            self.ball_hidden = "hold"

        # goals: camera first, localisation as fallback when the goal is out of view
        own_colour = "blue" if attack_colour == "yellow" else "yellow"
        src = []
        goals = []
        for colour, gy in ((attack_colour, field.ATTACK_GOAL_Y), (own_colour, field.OWN_GOAL_Y)):
            g = det.goals[colour]
            if g is not None:
                goals.append(Goal(rotate(g.near, c), rotate(g.centre, c)))
                src.append("cam")
            elif pose is not None and pose.confident:
                nx, _ = field.goal_near_point(pose.x, pose.y, gy)
                goals.append(Goal([float(nx) - pose.x, gy - pose.y], [-pose.x, gy - pose.y]))
                src.append("pose")
            else:
                goals.append(None)
                src.append("-")
        self.attack_goal, self.own_goal = goals
        self.goal_source = "/".join(src)

        self.line_pts = rotate_points(det.line_pts, c)

        # obstacles
        self.obstacles = self.obstacle_tracker.update([(rotate(front_of(o), c), o.size) for o in det.obstacles], now)

        # field-frame copies when we know where we are (anything from the teammate needs this too)
        # a paused / taken-off teammate's ball and opponents aren't used (its "bot active" is 0)
        mate_on = bool(mate) and mate.get("bot active") == 1
        mate_ball = mate.get("ball") if mate_on else None
        self.mate_pos = mate.get("pos") if mate else None
        if confident:
            self.ball_own_field = None if self.ball_own is None else [self.ball_own[0] + pose.x, self.ball_own[1] + pose.y]
            if self.mate_pos is not None:   # the teammate shows up as an obstacle: drop it (only the one nearest
                # to where it says it is - an opponent tackling it is right next to it and must stay)
                best = None
                for i, (p, s) in enumerate(self.obstacles):
                    c = centre_of(pose, [p[0] + pose.x, p[1] + pose.y])
                    d = math.hypot(c[0] - self.mate_pos[0], c[1] - self.mate_pos[1])
                    if d < MATE_MATCH and (best is None or d < best[0]):
                        best = (d, i)
                if best is not None:
                    self.obstacles = [o for i, o in enumerate(self.obstacles) if i != best[1]]
            self.obstacles_field = self.remember_obstacles(
                [([p[0] + pose.x, p[1] + pose.y], s) for p, s in self.obstacles], pose, now)
            self.opponents_own = [centre_of(pose, [p[0] + pose.x, p[1] + pose.y]) for p, _ in self.obstacles]
            self.opponents = self.merge_opponents(pose, mate if mate_on else None)
            self.opponent_vel = self.opponent_velocities(now)
        else:
            self.ball_own_field = None
            self.obstacles_field = []
            self.obstacle_memory = []
            self.opponents_own = []
            self.opponents = []
            self.opponent_vel = []

        # ball: our own camera (or IR / where it vanished, hidden_ball) first, the teammate's sighting when we can't see it
        if self.ball_own is not None:
            self.ball, self.ball_field, self.ball_source = self.ball_own, self.ball_own_field, self.ball_hidden or "cam"
        elif confident and mate_ball is not None:
            self.ball = [mate_ball[0] - pose.x, mate_ball[1] - pose.y]
            self.ball_field, self.ball_source = list(mate_ball), "mate"
        else:
            self.ball, self.ball_field, self.ball_source = None, None, "-"
        self.update_ball_vel(now, None if self.ball_hidden else self.ball_own_field)

    def hidden_ball(self, has_ir, pose, c, now):
        """Where the ball is while the camera can't see it (robot frame, its near point like the camera gives), or None
        (then the tracker's last position is kept for BALL_LOST_TIME). Sets .ball_hidden to "ir" / "hold".
        - IR ring: along the IR bearing, at the range of the last camera ball (moved on by its velocity, IR_RANGE_MEMORY)
          kept within what the IR strength allows; strong IR + unseen = in the blind ring (IR_RING_R)
        - no IR (and cfg.BALL_BLIND_RING): a ball that vanished within NEAR_HOLD_DIST is in the blind ring - it stays where it was (robot frame)
          for NEAR_HOLD_TIME"""
        last = self.last_cam_ball
        recent = last is not None and now - last[2] < IR_RANGE_MEMORY
        confident = pose is not None and pose.confident
        if has_ir:
            if self.ir is None:
                return None
            bearing, strength, _ = self.ir
            r = None
            if recent and confident and last[1] is not None:
                dt = min(now - last[2], 0.5)
                r = math.hypot(last[1][0] + self.ball_vel[0] * dt - pose.x, last[1][1] + self.ball_vel[1] * dt - pose.y)
            elif recent:
                r = math.hypot(last[0][0], last[0][1])
            if strength >= IR_CLOSE:
                r = IR_RING_R
            else:
                r_ir = IR_NEAR / max(strength, 1e-3) - cfg.BALL_RADIUS
                r = r_ir if r is None else min(max(r, 0.6 * r_ir), 1.6 * r_ir)
            r = max(r, cfg.ROBOT_RADIUS)
            self.ball_hidden = "ir"
            return [r * math.cos(bearing), r * math.sin(bearing)]
        if (cfg.BALL_BLIND_RING <= 0 or last is None or now - last[2] > NEAR_HOLD_TIME
                or math.hypot(last[0][0], last[0][1]) > NEAR_HOLD_DIST):
            return None
        self.ball_hidden = "hold"
        return list(last[0])

    def update_ball_vel(self, now, b):
        """Smoothed ball velocity (field frame) from our own camera's ball positions (b = this frame's, or None)."""
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
        """Each opponent's velocity (field frame cm/s): it's matched to the nearest track (its recent positions), and
        the velocity is the slope of a straight-line fit through the last OPP_VEL_WINDOW s of them."""
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
        """Opponent centres: ours (incl. remembered), plus the ones the teammate sees (its "opps") that we don't -
        leaving out ourselves (the teammate sees us as an obstacle)."""
        opps = [c for c in (centre_of(pose, p) for p, _ in self.obstacles_field)
                if field.inside_play_area(c[0], c[1], -cfg.ROBOT_RADIUS)]   # a robot fully outside isn't playing
        for o in (mate.get("opps") or []) if mate else []:
            if not field.inside_play_area(o[0], o[1], -cfg.ROBOT_RADIUS):
                continue
            if math.hypot(o[0] - pose.x, o[1] - pose.y) < OBSTACLE_VISIBLE:
                continue    # somewhere our own camera sees well: if we don't see it there, it's stale / not real
            if math.hypot(o[0] - pose.x, o[1] - pose.y) < MATE_MATCH:
                continue
            if all(math.hypot(o[0] - q[0], o[1] - q[1]) > OPP_MERGE for q in opps):
                opps.append((float(o[0]), float(o[1])))
        return opps

    def remember_obstacles(self, seen, pose, now):
        """Field-frame obstacles: the ones seen now, plus ones seen in the last OBSTACLE_MEMORY seconds that are now
        out of view (far away / at the edge of the camera)."""
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
                kept.append(m)      # out of view, not a duplicate of one we see now, not our teammate
        self.obstacle_memory = kept
        return [(m[0], m[1]) for m in self.obstacle_memory]


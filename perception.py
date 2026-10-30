import math
import time

import field
import robot_config as cfg
import numpy as np

from utils import PointTracker, other_goal, rotate, rotate_points
from vision import Goal

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
IR_NEAR = 20.0                # TUNE: cm, the IR sensors read full strength (1.0) closer than this, ~IR_NEAR / distance further
IR_CLOSE = 0.8
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

    def update(self, frame_id, det, pose, attack_colour, mate=None, ir=None):
        self.pose = pose
        if det is None or frame_id == self.frame_id:
            return
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
        ir_capture = (self.ball_hidden == "ir" and self.ir[1] >= IR_CLOSE
                      and math.hypot(*self.ball_robot) < IR_CAPTURE_R
                      and abs(self.ir[0] - math.pi / 2) < IR_CAPTURE_ANGLE)
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


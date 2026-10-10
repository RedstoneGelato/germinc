"""
perception.py - one place that turns the latest camera Detections + localisation Pose into what the strategy uses.

Everything in World is "relative" (field axes, robot at the origin, cm) unless the name says _field (field frame).
"""
import math
import time

import field
import robot_config as cfg
from detection import Goal
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
OBSTACLE_MEMORY = 0.4         # s an obstacle is remembered (field frame) after it drops out of view (longer: robots
                              # that had moved on stayed behind as ghosts)...
OBSTACLE_VISIBLE = 70.0       # ...unless it's closer than this (cm), where the camera would definitely still see it
MATE_MATCH = cfg.ROBOT_RADIUS + 15.0   # cm: an obstacle this close to where the teammate says it is = the teammate
OPP_MATCH = 15.0              # cm: an opponent this close to one in the previous frame = the same robot (velocity)
OPP_VEL_SMOOTH = 0.3          # 0..1 how much of each frame's opponent velocity is taken
OPP_MERGE = 15.0              # cm: a teammate's opponent this close to one of ours = the same robot


def front_of(o):
    """An obstacle as the point facing us, made consistent with its estimated centre (detection.Obstacle.centre):
    one robot radius in front of the centre, so `centre_of()` / strategy.obstacle_centres() get the whole robot's
    centre back."""
    if getattr(o, "centre", None) is None:
        return o.near
    d = math.hypot(o.centre[0], o.centre[1]) or 1.0
    k = max(d - cfg.ROBOT_RADIUS, 0.0) / d
    return [o.centre[0] * k, o.centre[1] * k]


def centre_of(pose, near):
    """Obstacles are seen as their nearest point; the robot's centre is about one robot radius further away."""
    dx, dy = near[0] - pose.x, near[1] - pose.y
    d = math.hypot(dx, dy) or 1.0
    return (near[0] + dx / d * cfg.ROBOT_RADIUS, near[1] + dy / d * cfg.ROBOT_RADIUS)


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
        self.prev_opponents = ([], 0.0)
        self.pose = None
        self.field_visible = False

    def reset(self):
        self.ball_tracker.reset()
        self.obstacle_tracker.reset()
        self.last_capture = 0.0
        self.reset_values()

    def update(self, frame_id, det, pose, attack_colour, mate=None):
        """Call every loop; only does work when there's a new camera frame.
        mate = the teammate's latest comms message (or None): its "ball" and "pos" are used."""
        self.pose = pose
        if det is None or frame_id == self.frame_id:
            return
        self.frame_id = frame_id
        now = time.monotonic()
        c = det.compass
        self.field_visible = det.field_visible

        # ball
        self.ball_robot = det.ball
        self.ball_own = self.ball_tracker.update(None if det.ball is None else rotate(det.ball, c), now)
        if in_capture_zone(det.ball):
            self.last_capture = now
        self.ball_in_capture = in_capture_zone(det.ball) or (det.ball is None and now - self.last_capture < CAPTURE_HOLD)

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
        confident = pose is not None and pose.confident
        mate_ball = mate.get("ball") if mate else None
        self.mate_pos = mate.get("pos") if mate else None
        if confident:
            self.ball_own_field = None if self.ball_own is None else [self.ball_own[0] + pose.x, self.ball_own[1] + pose.y]
            if self.mate_pos is not None:   # the teammate shows up as an obstacle: drop it
                self.obstacles = [(p, s) for p, s in self.obstacles
                                  if math.hypot(p[0] + pose.x - self.mate_pos[0], p[1] + pose.y - self.mate_pos[1]) > MATE_MATCH]
            self.obstacles_field = self.remember_obstacles(
                [([p[0] + pose.x, p[1] + pose.y], s) for p, s in self.obstacles], pose, now)
            self.opponents_own = [centre_of(pose, [p[0] + pose.x, p[1] + pose.y]) for p, _ in self.obstacles]
            self.opponents = self.merge_opponents(pose, mate)
            self.opponent_vel = self.opponent_velocities(now)
        else:
            self.ball_own_field = None
            self.obstacles_field = []
            self.obstacle_memory = []
            self.opponents_own = []
            self.opponents = []
            self.opponent_vel = []

        # ball: our own camera first, the teammate's sighting when we can't see it
        if self.ball_own is not None:
            self.ball, self.ball_field, self.ball_source = self.ball_own, self.ball_own_field, "cam"
        elif confident and mate_ball is not None:
            self.ball = [mate_ball[0] - pose.x, mate_ball[1] - pose.y]
            self.ball_field, self.ball_source = list(mate_ball), "mate"
        else:
            self.ball, self.ball_field, self.ball_source = None, None, "-"
        self.update_ball_vel(now)

    def update_ball_vel(self, now):
        """Smoothed ball velocity (field frame) from our own camera's ball positions."""
        b, prev = self.ball_own_field, self.prev_ball
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
        """Each opponent's velocity: matched to the nearest one in the previous frame, smoothed."""
        prev, t_prev = self.prev_opponents
        prev_vel = getattr(self, "_prev_vel", [])
        vels = []
        dt = now - t_prev
        for o in self.opponents:
            v = [0.0, 0.0]
            if prev and 1e-3 < dt < 0.3:
                k = min(range(len(prev)), key=lambda i: math.hypot(prev[i][0] - o[0], prev[i][1] - o[1]))
                if math.hypot(prev[k][0] - o[0], prev[k][1] - o[1]) < OPP_MATCH:
                    raw = [(o[0] - prev[k][0]) / dt, (o[1] - prev[k][1]) / dt]
                    old = prev_vel[k] if k < len(prev_vel) else raw
                    v = [old[0] * (1 - OPP_VEL_SMOOTH) + raw[0] * OPP_VEL_SMOOTH,
                         old[1] * (1 - OPP_VEL_SMOOTH) + raw[1] * OPP_VEL_SMOOTH]
            vels.append(v)
        self.prev_opponents, self._prev_vel = (list(self.opponents), now), vels
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
        out of view (far away / at the edge of the camera). Stops planning from flip-flopping when an obstacle flickers."""
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


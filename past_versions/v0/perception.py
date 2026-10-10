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
OBSTACLE_MATCH = 20.0         # cm: same obstacle as last frame if this close
OBSTACLE_MIN_HITS = 3         # frames in a row before an obstacle is believed
OBSTACLE_LOST_TIME = 0.3
OBSTACLE_MEMORY = 1.0         # s an obstacle is remembered (field frame) after it drops out of view...
OBSTACLE_VISIBLE = 70.0       # ...unless it's closer than this (cm), where the camera would definitely still see it
MATE_MATCH = cfg.ROBOT_RADIUS + 15.0   # cm: an obstacle this close to where the teammate says it is = the teammate


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
        self.mate_pos = None          # teammate's position (field frame) from comms, or None
        self.ball_in_capture = False
        self.attack_goal = None       # Goal(near, centre), relative
        self.own_goal = None
        self.goal_source = ""         # "cam" / "pose" per goal, for printing
        self.line_pts = np.zeros((0, 2), np.float32)   # white line points, relative
        self.obstacles = []           # [(relative pos, size cm)]
        self.obstacles_field = []     # [(field pos, size cm)]: seen now + remembered for OBSTACLE_MEMORY
        self.obstacle_memory = []     # [[field pos, size, last seen]]
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
        self.obstacles = self.obstacle_tracker.update([(rotate(o.near, c), o.size) for o in det.obstacles], now)

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
        else:
            self.ball_own_field = None
            self.obstacles_field = []
            self.obstacle_memory = []

        # ball: our own camera first, the teammate's sighting when we can't see it
        if self.ball_own is not None:
            self.ball, self.ball_field, self.ball_source = self.ball_own, self.ball_own_field, "cam"
        elif confident and mate_ball is not None:
            self.ball = [mate_ball[0] - pose.x, mate_ball[1] - pose.y]
            self.ball_field, self.ball_source = list(mate_ball), "mate"
        else:
            self.ball, self.ball_field, self.ball_source = None, None, "-"

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
        self.obstacle_memory = [m for m in mem if m[2] == now or (
            now - m[2] < OBSTACLE_MEMORY and math.hypot(m[0][0] - pose.x, m[0][1] - pose.y) > OBSTACLE_VISIBLE)]
        return [(m[0], m[1]) for m in self.obstacle_memory]


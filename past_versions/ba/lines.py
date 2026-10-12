"""
lines.py - out-of-bounds avoidance from BOTH the LDR ring and the camera.

    LDR ring   sees the line right under the robot (the camera can't, the body is in the way).
               This is the only thing that can say "we're ON the line right now".
    camera     sees lines around the robot before we reach them. Gives a second escape direction.
               (Only the boundary is white; the penalty box line is black, so neither sensor reacts to it.)
    pose       when localisation is precise (spread < cfg.PRECISE_STD): the robot may go partly past the line, as
               long as part of it still touches the line (centre up to cfg.MAX_OUT past the outer edge). The pose
               decides when to escape, LDR hits on their own are allowed (touching the line is legal). Without a
               precise pose the LDRs decide: seeing the line = escape.

All vectors are "relative": field axes, robot at the origin, cm.
"""
import math
import time

import numpy as np

import field
import robot_config as cfg

# ---- TUNE
LDR_VECTOR_IS_ESCAPE = True     # the LDR sum vector was used as the escape direction at the comp. If the robot
                                # drives INTO the line on the bench test, set this to False to flip it.
CAM_NEAR = cfg.ROBOT_RADIUS + 10.0   # camera line points closer than this count as "about to touch"
CAM_MIN_POINTS = 4
BOUNDARY_BAND = field.LINE_W + 0.5   # cm: field-frame line points within this of the boundary are boundary lines
                                     # (kept tight, so only the boundary counts if inner white markings get added)
IGNORE_INNER_LINES = False      # True: confident pose + far from the boundary -> ignore LDR hits (only useful if the
                                # field has white lines inside it; ours doesn't, so it could only hide a real boundary)
INNER_CLEARANCE = 15.0          # cm robot centre must be inside the boundary for an LDR hit to count as an inner line
SLOW_START = 20.0               # cm (robot centre to the furthest-out allowed position): start limiting outward speed
SLOW_STOP = 2.0                 # cm: outward speed is zero here
W_LDR, W_CAM, W_POSE = 1.0, 0.7, 1.0   # escape direction blend
SPEED_MULTI = {0: 1, 1: 0.6, 2: 0.5, 3: 0.3, 4: 0.1}  # by number of line touches in the last 3 s


def _unit(v):
    n = math.hypot(v[0], v[1])
    return None if n < 1e-9 else [v[0] / n, v[1] / n]


class LineState:
    def __init__(self):
        self.on_line = False        # escape now
        self.escape = None          # unit vector to drive along, or None
        self.ldr_hit = False
        self.ldr_count = 0
        self.cam_near = False
        self.source = ""            # which sensors drove the escape, for printing
        self.speed_multi = 1.0
        self.touches = 0            # line touches in the last 3 s


class LineFusion:
    def __init__(self, threshold=cfg.LDR_LINE_THRESHOLD):
        self.threshold = threshold
        self.was_on_line = False
        self.touch_times = []

    def ldr_vector(self, colours, compass):
        """Sum of unit vectors of the LDRs on white (relative axes), and how many."""
        lx = ly = 0.0
        n = 0
        for i, value in enumerate(colours):
            if value < self.threshold:
                angle = i * (2 * math.pi / len(colours)) + cfg.LDR_START_ANGLE + compass
                lx += math.cos(angle)
                ly += math.sin(angle)
                n += 1
        return [lx, ly], n

    @staticmethod
    def boundary_escape(pose):
        """Direction back into the field from the pose, or None if comfortably inside."""
        ex = ey = 0.0
        lim_x, lim_y = out_limits(pose.x)
        if abs(pose.x) > lim_x - SLOW_STOP:
            ex = -math.copysign(1.0, pose.x)
        if abs(pose.y) > lim_y - SLOW_STOP:
            ey = -math.copysign(1.0, pose.y)
        return _unit([ex, ey])

    def update(self, colours, compass, line_pts_rel, pose):
        s = LineState()
        confident = pose is not None and pose.confident

        # ---- LDR
        ldr_vec, s.ldr_count = self.ldr_vector(colours, compass)
        s.ldr_hit = s.ldr_count > 0
        ldr_escape = _unit(ldr_vec if LDR_VECTOR_IS_ESCAPE else [-ldr_vec[0], -ldr_vec[1]]) if s.ldr_hit else None

        # ---- camera: line points close to the robot (boundary lines only, if we know where we are)
        cam_escape = None
        pts = line_pts_rel
        if len(pts):
            d2 = pts[:, 0] ** 2 + pts[:, 1] ** 2
            near = pts[d2 < CAM_NEAR ** 2]
            if confident and len(near):
                fx, fy = near[:, 0] + pose.x, near[:, 1] + pose.y
                boundary = ((np.abs(fx) > field.PLAY_W / 2 - BOUNDARY_BAND) |   # incl. the lines inside the goals
                            (np.abs(fy) > field.PLAY_L / 2 - BOUNDARY_BAND))
                near = near[boundary]
            if len(near) >= CAM_MIN_POINTS:
                s.cam_near = True
                w = 1.0 / np.maximum(np.hypot(near[:, 0], near[:, 1]), 1.0)
                toward = (near * w[:, None]).sum(axis=0)
                cam_escape = _unit([-toward[0], -toward[1]])

        # ---- pose
        pose_escape = None
        pose_out = False
        precise = pose is not None and pose.std < cfg.PRECISE_STD
        if confident:
            pose_escape = self.boundary_escape(pose)
        if precise:
            lim_x, lim_y = out_limits(pose.x)
            pose_out = abs(pose.x) > lim_x or abs(pose.y) > lim_y

        # ---- decide
        if precise:
            s.on_line = pose_out        # touching the line is fine, only going too far past it isn't
        else:
            inner_line = (IGNORE_INNER_LINES and confident and
                          field.inside_play_area(pose.x, pose.y, INNER_CLEARANCE) and not s.cam_near)
            s.on_line = s.ldr_hit and not inner_line

        past_line = confident and not field.inside_play_area(pose.x, pose.y, 0.0)
        if s.on_line and past_line:
            # centre already past the line: the line is on the robot's INNER side now, so the LDR and camera
            # directions ("away from the line") point further out. Only the pose knows where the field is.
            ex = -math.copysign(1.0, pose.x) if abs(pose.x) > field.PLAY_W / 2 else 0.0
            ey = -math.copysign(1.0, pose.y) if abs(pose.y) > field.boundary_y(pose.x) else 0.0
            s.escape, s.source = _unit([ex, ey]) or pose_escape, "pose (past line)"
        elif s.on_line:
            ex = ey = 0.0
            src = []
            for vec, wgt, name in ((ldr_escape, W_LDR, "ldr"), (cam_escape, W_CAM, "cam"), (pose_escape, W_POSE, "pose")):
                if vec is not None:
                    ex += wgt * vec[0]
                    ey += wgt * vec[1]
                    src.append(name)
            s.escape = _unit([ex, ey]) or ldr_escape or pose_escape
            s.source = "+".join(src)

        # repeated touches -> slow down (same as before)
        now = time.monotonic()
        if s.on_line and not self.was_on_line:
            self.touch_times.append(now)
        self.was_on_line = s.on_line
        while self.touch_times and now - self.touch_times[0] > 3:
            self.touch_times.pop(0)
        s.touches = len(self.touch_times)
        s.speed_multi = SPEED_MULTI.get(s.touches, 0.1)
        return s


def out_limits(x):
    """How far the robot centre may go in x and y (at this x) and still touch the white line (with safety)."""
    return (field.PLAY_W / 2 + cfg.MAX_OUT,
            field.boundary_y(x, cfg.ROBOT_RADIUS) + cfg.MAX_OUT)


def limit_outward(vec, pose):
    """Scale down the part of a desired movement (relative) that heads out of the field when the robot gets close
    to the furthest-out allowed position. Only with a precise pose; otherwise returns vec unchanged."""
    if pose is None or pose.std >= cfg.PRECISE_STD:
        return vec
    out = list(vec)
    lim_x, lim_y = out_limits(pose.x)
    for axis, lim in ((0, lim_x), (1, lim_y)):
        p = pose.x if axis == 0 else pose.y
        if out[axis] * p <= 0:          # moving inwards (or not at all) on this axis
            continue
        gap = lim - abs(p)
        f = min(max((gap - SLOW_STOP) / (SLOW_START - SLOW_STOP), 0.0), 1.0)
        out[axis] *= f
    return out

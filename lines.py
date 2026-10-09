"""
lines.py - out-of-bounds avoidance from BOTH the LDR ring and the camera.

    LDR ring   sees the line right under the robot (the camera can't, the body is in the way).
               This is the only thing that can say "we're ON the line right now".
    camera     sees lines around the robot before we reach them. Gives a second escape direction and,
               once the localisation is confident, tells boundary lines apart from penalty-area lines.
    pose       when localisation is confident: distance to the boundary in each direction, used to slow down
               before reaching the line and as a third escape direction.

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
BOUNDARY_BAND = 6.0             # cm: field-frame line points within this of the boundary are boundary lines
IGNORE_INNER_LINES = True       # confident pose + far from the boundary -> LDR hits are penalty lines, don't escape
INNER_CLEARANCE = 15.0          # cm robot centre must be inside the boundary for an LDR hit to count as an inner line
SLOW_START = 25.0               # cm (robot edge to boundary line): start limiting outward speed
SLOW_STOP = 3.0                 # cm: outward speed is zero here
OUT_MARGIN = 2.0                # cm past the line (robot centre) = out of bounds by pose
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
        lim_x = field.PLAY_W / 2 - cfg.ROBOT_RADIUS
        lim_y = field.boundary_y(pose.x, cfg.ROBOT_RADIUS) - cfg.ROBOT_RADIUS   # further in front of a goal (notch)
        if abs(pose.x) > lim_x:
            ex = -math.copysign(1.0, pose.x)
        if abs(pose.y) > lim_y:
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
        if confident:
            pose_escape = self.boundary_escape(pose)
            pose_out = not field.inside_play_area(pose.x, pose.y, -OUT_MARGIN)

        # ---- decide
        inner_line = (IGNORE_INNER_LINES and confident and
                      field.inside_play_area(pose.x, pose.y, INNER_CLEARANCE) and not s.cam_near)
        s.on_line = (s.ldr_hit and not inner_line) or pose_out

        if s.on_line:
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


def limit_outward(vec, pose):
    """Scale down the part of a desired movement (relative) that heads out of the field when the robot is close
    to the boundary. Only with a confident pose; otherwise returns vec unchanged."""
    if pose is None or not pose.confident:
        return vec
    out = list(vec)
    for axis, half in ((0, field.PLAY_W / 2), (1, field.boundary_y(pose.x, cfg.ROBOT_RADIUS))):
        p = pose.x if axis == 0 else pose.y
        if out[axis] * p <= 0:          # moving inwards (or not at all) on this axis
            continue
        gap = half - abs(p) - cfg.ROBOT_RADIUS
        f = min(max((gap - SLOW_STOP) / (SLOW_START - SLOW_STOP), 0.0), 1.0)
        out[axis] *= f
    return out

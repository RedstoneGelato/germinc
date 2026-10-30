import math
import time

import numpy as np

import field
import robot_config as cfg

# TUNE
LDR_VECTOR_IS_ESCAPE = True
CAM_NEAR = cfg.ROBOT_RADIUS + 10.0
CAM_MIN_POINTS = 4
BOUNDARY_BAND = field.LINE_W + 0.5
IGNORE_INNER_LINES = False
INNER_CLEARANCE = 15.0
SLOW_START = 20.0
SLOW_STOP = 2.0
OUTWARD_STD = 15.0
W_LDR, W_CAM, W_POSE = 1.0, 0.7, 1.0
TOUCH_WINDOW = 3
SPEED_MULTI = {0: 1, 1: 0.6, 2: 0.5, 3: 0.3, 4: 0.1}
SPEED_MULTI_MANY_TOUCHES = 0.1


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
        self.source = ""
        self.speed_multi = 1.0
        self.touches = 0


class LineFusion:
    def __init__(self, threshold=cfg.LDR_LINE_THRESHOLD):
        self.threshold = threshold
        self.was_on_line = False
        self.touch_times = []

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

    def update(self, colours, compass, line_pts_rel, pose):
        state = LineState()
        confident = pose is not None and pose.confident
        precise = pose is not None and pose.std < cfg.PRECISE_STD

        ldr_vec, state.ldr_count = self.ldr_vector(colours, compass)
        state.ldr_hit = state.ldr_count > 0
        ldr_escape = None
        if state.ldr_hit:
            ldr_escape = _unit(ldr_vec if LDR_VECTOR_IS_ESCAPE else [-ldr_vec[0], -ldr_vec[1]])
        cam_escape = self._camera_escape(state, line_pts_rel, pose, confident)
        pose_escape = self.boundary_escape(pose) if confident else None

        if precise:
            lim_x, lim_y = out_limits(pose.x)
            state.on_line = abs(pose.x) > lim_x or abs(pose.y) > lim_y
        else:
            inner_line = (IGNORE_INNER_LINES and confident and
                          field.inside_play_area(pose.x, pose.y, INNER_CLEARANCE) and not state.cam_near)
            state.on_line = state.ldr_hit and not inner_line

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

        state.touches = self._count_touches(state.on_line)
        state.speed_multi = SPEED_MULTI.get(state.touches, SPEED_MULTI_MANY_TOUCHES)
        return state

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

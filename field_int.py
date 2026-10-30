import math
import sys

import cv2
import numpy as np

RULES = "international"

PLAY_W = 158.0
PLAY_L = 219.0
OUTER = 12.0
LINE_W = 2.0
GOAL_W = 60.0
GOAL_DEPTH = 7.4
PENALTY_W = 80.0
PENALTY_D = 25.0
PENALTY_CORNER_R = 15.0
PENALTY_LINE_W = 2.0
PENALTY_LINE_WHITE = True

NEUTRAL_END = 45.0
NEUTRAL_X = PENALTY_W / 2
NEUTRAL_SPOTS = [(0.0, 0.0)] + [(sx * NEUTRAL_X, sy * (PLAY_L / 2 - NEUTRAL_END)) for sy in (1, -1) for sx in (-1, 1)]

WALL_W = PLAY_W + 2 * OUTER
WALL_L = PLAY_L + 2 * OUTER

ATTACK_GOAL_Y = PLAY_L / 2 - LINE_W
OWN_GOAL_Y = -ATTACK_GOAL_Y
GOAL_BACK_Y = ATTACK_GOAL_Y + GOAL_DEPTH

BALL_OUT_AT_LINE = False
LINE_IS_OUT = False
WALL_GAP = 2.0

MAP_RES = 1.0
MAP_MARGIN = 30.0


def max_out(robot_radius):
    return OUTER - robot_radius - WALL_GAP


def boundary_polyline(sgn):
    hw, hl = PLAY_W / 2 - LINE_W / 2, PLAY_L / 2 - LINE_W / 2
    pts = [(-hw, hl), (hw, hl)]
    return [(x, sgn * y) for x, y in pts]


def penalty_polyline(sgn, inset=PENALTY_LINE_W / 2, arc_steps=8):
    hw, r = PENALTY_W / 2 - inset, PENALTY_CORNER_R - inset
    cx, cd = PENALTY_W / 2 - PENALTY_CORNER_R, PENALTY_D - PENALTY_CORNER_R
    pts = [(-hw, 0.0), (-hw, cd)]
    for k in range(1, arc_steps + 1):
        a = math.pi - k * (math.pi / 2) / arc_steps
        pts.append((-cx + r * math.cos(a), cd + r * math.sin(a)))
    for k in range(0, arc_steps + 1):
        a = math.pi / 2 - k * (math.pi / 2) / arc_steps
        pts.append((cx + r * math.cos(a), cd + r * math.sin(a)))
    pts.append((hw, 0.0))
    return [(x, sgn * (ATTACK_GOAL_Y - dd)) for x, dd in pts]


def black_line_segments():
    return []


def white_line_segments():
    hw, hl = PLAY_W / 2 - LINE_W / 2, PLAY_L / 2 - LINE_W / 2
    segs = [((hw, -hl), (hw, hl)), ((-hw, hl), (-hw, -hl))]
    for sgn in (1, -1):
        pts = boundary_polyline(sgn)
        segs += list(zip(pts[:-1], pts[1:]))
    for sgn in (1, -1):
        pts = penalty_polyline(sgn)
        segs += list(zip(pts[:-1], pts[1:]))
    return segs


class FieldMap:

    def __init__(self):
        self.w = int(math.ceil((WALL_W + 2 * MAP_MARGIN) * MAP_RES))
        self.h = int(math.ceil((WALL_L + 2 * MAP_MARGIN) * MAP_RES))
        lines = np.full((self.h, self.w), 255, np.uint8)
        thick = max(1, int(round(LINE_W * MAP_RES)))
        for a, b in white_line_segments():
            cv2.line(lines, self.to_map(*a), self.to_map(*b), 0, thick)
        self.lines = 255 - lines
        self.dist = cv2.distanceTransform(lines, cv2.DIST_L2, 5).astype(np.float32) / MAP_RES

    def to_map(self, x, y):
        return (int(round((x + WALL_W / 2 + MAP_MARGIN) * MAP_RES)),
                int(round((WALL_L / 2 + MAP_MARGIN - y) * MAP_RES)))

    def lookup(self, xs, ys, clip=None):
        mx = np.rint((xs + WALL_W / 2 + MAP_MARGIN) * MAP_RES).astype(np.int32)
        my = np.rint((WALL_L / 2 + MAP_MARGIN - ys) * MAP_RES).astype(np.int32)
        inside = (mx >= 0) & (mx < self.w) & (my >= 0) & (my < self.h)
        out = np.full(xs.shape, np.float32(clip if clip is not None else 1e3))
        out[inside] = self.dist[my[inside], mx[inside]]
        if clip is not None:
            np.minimum(out, clip, out=out)
        return out


def goal_near_point(px, py, goal_y):
    return np.clip(px, -GOAL_W / 2, GOAL_W / 2), np.full_like(np.asarray(px, dtype=np.float64), goal_y)


def goal_seen_point(px, py, goal_y):
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    gw, back = GOAL_W / 2, goal_y + math.copysign(GOAL_DEPTH, goal_y)
    lo, hi = min(goal_y, back), max(goal_y, back)
    best_x, best_y = np.clip(px, -gw, gw), np.full_like(px, back)
    best_d = (best_x - px) ** 2 + (best_y - py) ** 2
    for sx in (-gw, gw):
        sy = np.clip(py, lo, hi)
        d = (sx - px) ** 2 + (sy - py) ** 2
        closer = d < best_d
        best_x, best_y, best_d = np.where(closer, sx, best_x), np.where(closer, sy, best_y), np.where(closer, d, best_d)
    return best_x, best_y


def wall_distance(px, py, angle):
    dx, dy = np.cos(angle), np.sin(angle)
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(dx > 1e-9, (WALL_W / 2 - px) / dx, np.where(dx < -1e-9, (-WALL_W / 2 - px) / dx, np.inf))
        ty = np.where(dy > 1e-9, (WALL_L / 2 - py) / dy, np.where(dy < -1e-9, (-WALL_L / 2 - py) / dy, np.inf))
    return np.maximum(np.minimum(tx, ty), 0.0)


def inside_play_area(x, y, margin=0.0):
    if abs(x) <= PLAY_W / 2 - margin and abs(y) <= PLAY_L / 2 - margin:
        return True
    return abs(x) <= GOAL_W / 2 - margin and abs(y) <= GOAL_BACK_Y - margin


def boundary_y(x, inset=0.0):
    return GOAL_BACK_Y if abs(x) <= GOAL_W / 2 - inset else PLAY_L / 2


def touches_wall(x, y, r, gap=0.0):
    return abs(x) >= WALL_W / 2 - r - gap or abs(y) >= WALL_L / 2 - r - gap


def penalty_depth(x, y, sgn):
    ax = abs(x)
    d = ATTACK_GOAL_Y - sgn * y
    cx, cd = PENALTY_W / 2 - PENALTY_CORNER_R, PENALTY_D - PENALTY_CORNER_R
    if ax > cx and d > cd:
        return PENALTY_CORNER_R - math.hypot(ax - cx, d - cd)
    return min(PENALTY_W / 2 - ax, PENALTY_D - d)


def fully_in_penalty(x, y, r, sgn):
    return penalty_depth(x, y, sgn) >= r


def partly_in_penalty(x, y, r, sgn):
    return penalty_depth(x, y, sgn) > -r


def install():
    me = sys.modules[__name__]
    old = sys.modules.get("field")
    if old is not None and old is not me:
        print("field_int.install(): field.py was already imported - some modules will use the RCJA field. "
              "Call install() first.")
    sys.modules["field"] = me
    import robot_config as cfg
    cfg.MAX_OUT = max_out(cfg.ROBOT_RADIUS)
    import lines
    lines.IGNORE_INNER_LINES = True
    lines.BOUNDARY_BAND = LINE_W + 0.5
    return me

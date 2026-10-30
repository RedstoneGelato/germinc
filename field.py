import math

import cv2
import numpy as np

PLAY_W = 132.0
PLAY_L = 193.0
OUTER = 25.0
LINE_W = 5.0
GOAL_W = 45.0
GOAL_DEPTH = 7.4
PENALTY_W = 90.0
PENALTY_D = 30.0
PENALTY_LINE_W = 5.0

NEUTRAL_SIDE = 30.0
NEUTRAL_SPOTS = [(0.0, 0.0), (-NEUTRAL_SIDE, 0.0), (NEUTRAL_SIDE, 0.0)]

WALL_W = PLAY_W + 2 * OUTER
WALL_L = PLAY_L + 2 * OUTER

ATTACK_GOAL_Y = PLAY_L / 2 - LINE_W
OWN_GOAL_Y = -ATTACK_GOAL_Y
GOAL_BACK_Y = ATTACK_GOAL_Y + GOAL_DEPTH

MAP_RES = 1.0
MAP_MARGIN = 30.0


def boundary_polyline(sgn):
    hw, hl = PLAY_W / 2 - LINE_W / 2, PLAY_L / 2 - LINE_W / 2
    pts = [(-hw, hl), (hw, hl)]
    return [(x, sgn * y) for x, y in pts]


def black_line_segments():
    hl = PLAY_L / 2 - LINE_W
    pw = PENALTY_W / 2 - PENALTY_LINE_W / 2
    segs = []
    for sgn in (1, -1):
        y0 = sgn * hl
        y1 = sgn * (ATTACK_GOAL_Y - PENALTY_D + PENALTY_LINE_W / 2)
        segs += [((-pw, y0), (-pw, y1)), ((-pw, y1), (pw, y1)), ((pw, y1), (pw, y0))]
    return segs


def white_line_segments():
    hw, hl = PLAY_W / 2 - LINE_W / 2, PLAY_L / 2 - LINE_W / 2
    segs = [((hw, -hl), (hw, hl)), ((-hw, hl), (-hw, -hl))]
    for sgn in (1, -1):
        pts = boundary_polyline(sgn)
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

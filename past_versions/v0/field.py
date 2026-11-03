"""
field.py - the field model used for localisation.

Field frame: origin = field centre, +y = towards the goal we attack, +x = right when facing it, units = cm.

Shape: the playing area is a rectangle with a notch at each goal. The white boundary line stops at the goal posts
(no line across the goal mouth) and carries on around the INSIDE of the goal (along both side walls and the back),
so the goal floor counts as part of the playing area:

        +------+ goal +------+        <- line inside the goal, against the goal walls
        |      |      |      |
   -----+      +      +      +-----   <- boundary line stops at the posts
   |                               |

!! Check every number below against this year's RCJ Soccer rules AND measure the actual competition field.
   Fields are often a few cm off, and that matters more than anything else in the localisation.
"""
import math

import cv2
import numpy as np

# ---- dimensions (cm) - VERIFY
PLAY_W = 102.0          # width of the playing area, measured between the outer edges of the white lines (x)
PLAY_L = 183.0          # length of the playing area (y, goal to goal)
OUTER = 25.0            # outer area: white line -> wall
LINE_W = 5.0            # white line thickness
GOAL_W = 45.0           # inner goal width
GOAL_DEPTH = 7.4
PENALTY_W = 90.0        # penalty area, measured like the playing area (outer line edges)
PENALTY_D = 30.0        # from the boundary line into the field
PENALTY_LINE_W = 5.0    # the penalty box is marked with a BLACK line (not white) - VERIFY its thickness
GOAL_LINE_INSET = 0.0   # gap between the goal walls and the outer edge of the white line inside the goal

# neutral spots (where the referee puts the ball) - VERIFY positions on the real field
NEUTRAL_SPOTS = [(0.0, 0.0)] + [(sx * (PLAY_W / 2 - 25.0), sy * (PLAY_L / 2 - 45.0)) for sx in (-1, 1) for sy in (-1, 1)]

WALL_W = PLAY_W + 2 * OUTER   # inside of the walls
WALL_L = PLAY_L + 2 * OUTER

ATTACK_GOAL_Y = PLAY_L / 2    # goal mouth lines
OWN_GOAL_Y = -PLAY_L / 2

MAP_RES = 1.0           # distance-map pixels per cm
MAP_MARGIN = 30.0       # map extends this far past the walls so out-of-field points still get a sensible distance


def boundary_polyline(sgn):
    """Centre line of one short edge including the goal notch, from left corner to right corner. sgn=+1 attack end."""
    hw, hl = PLAY_W / 2 - LINE_W / 2, PLAY_L / 2 - LINE_W / 2   # boundary line centres
    gx = GOAL_W / 2 - GOAL_LINE_INSET - LINE_W / 2               # line along the inside of the goal side walls
    gb = PLAY_L / 2 + GOAL_DEPTH - GOAL_LINE_INSET - LINE_W / 2  # line along the inside of the goal back wall
    pts = [(-hw, hl), (-gx, hl), (-gx, gb), (gx, gb), (gx, hl), (hw, hl)]
    return [(x, sgn * y) for x, y in pts]


def black_line_segments():
    """Centre lines of the black penalty box markings. Not used by the localisation (the camera only looks for
    white lines, and the LDRs don't react to black), but drawn by the simulator."""
    hl = PLAY_L / 2 - LINE_W                                     # inner edge of the white boundary line
    pw = PENALTY_W / 2 - PENALTY_LINE_W / 2
    segs = []
    for sgn in (1, -1):
        y0 = sgn * hl
        y1 = sgn * (PLAY_L / 2 - PENALTY_D + PENALTY_LINE_W / 2)
        segs += [((-pw, y0), (-pw, y1)), ((-pw, y1), (pw, y1)), ((pw, y1), (pw, y0))]
    return segs


def white_line_segments():
    """Centre lines of all white markings, as ((x1, y1), (x2, y2)) in field cm. Only the boundary is white
    (the penalty box is black: black_line_segments())."""
    hw, hl = PLAY_W / 2 - LINE_W / 2, PLAY_L / 2 - LINE_W / 2   # boundary line centres
    segs = [((hw, -hl), (hw, hl)), ((-hw, hl), (-hw, -hl))]     # long sides
    for sgn in (1, -1):                                          # short sides with the goal notch
        pts = boundary_polyline(sgn)
        segs += list(zip(pts[:-1], pts[1:]))
    return segs


class FieldMap:
    """Distance transform of the white lines: dist[y, x] = cm to the nearest white line."""

    def __init__(self):
        self.w = int(math.ceil((WALL_W + 2 * MAP_MARGIN) * MAP_RES))
        self.h = int(math.ceil((WALL_L + 2 * MAP_MARGIN) * MAP_RES))
        lines = np.full((self.h, self.w), 255, np.uint8)
        thick = max(1, int(round(LINE_W * MAP_RES)))
        for a, b in white_line_segments():
            cv2.line(lines, self.to_map(*a), self.to_map(*b), 0, thick)
        self.lines = 255 - lines                                    # white = line, for drawing
        self.dist = cv2.distanceTransform(lines, cv2.DIST_L2, 5).astype(np.float32) / MAP_RES

    def to_map(self, x, y):
        """Field cm -> integer map pixel (map row 0 = attacking end)."""
        return (int(round((x + WALL_W / 2 + MAP_MARGIN) * MAP_RES)),
                int(round((WALL_L / 2 + MAP_MARGIN - y) * MAP_RES)))

    def lookup(self, xs, ys, clip=None):
        """Distance to the nearest line for arrays of field points. Outside the map = the clip value."""
        mx = np.rint((xs + WALL_W / 2 + MAP_MARGIN) * MAP_RES).astype(np.int32)
        my = np.rint((WALL_L / 2 + MAP_MARGIN - ys) * MAP_RES).astype(np.int32)
        inside = (mx >= 0) & (mx < self.w) & (my >= 0) & (my < self.h)
        out = np.full(xs.shape, np.float32(clip if clip is not None else 1e3))
        out[inside] = self.dist[my[inside], mx[inside]]
        if clip is not None:
            np.minimum(out, clip, out=out)
        return out


def goal_near_point(px, py, goal_y):
    """Point of a goal mouth (x in +-GOAL_W/2 at y = goal_y) closest to (px, py). Arrays work too."""
    return np.clip(px, -GOAL_W / 2, GOAL_W / 2), np.full_like(np.asarray(px, dtype=np.float64), goal_y)


def wall_distance(px, py, angle):
    """Distance from (px, py) to the wall along a ray at field angle `angle` (radians, 0 = +x, CCW).
    For the ultrasonic ring later. Arrays work for px, py, angle."""
    dx, dy = np.cos(angle), np.sin(angle)
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(dx > 1e-9, (WALL_W / 2 - px) / dx, np.where(dx < -1e-9, (-WALL_W / 2 - px) / dx, np.inf))
        ty = np.where(dy > 1e-9, (WALL_L / 2 - py) / dy, np.where(dy < -1e-9, (-WALL_L / 2 - py) / dy, np.inf))
    return np.maximum(np.minimum(tx, ty), 0.0)


def inside_play_area(x, y, margin=0.0):
    """Inside the white boundary (including the goal notches) by at least `margin` cm. Negative margin = allow past it."""
    if abs(x) <= PLAY_W / 2 - margin and abs(y) <= PLAY_L / 2 - margin:
        return True
    return abs(x) <= GOAL_W / 2 - margin and abs(y) <= PLAY_L / 2 + GOAL_DEPTH - margin


def boundary_y(x, inset=0.0):
    """How far the playing area reaches in y at this x (further in front of a goal, because of the notch).
    inset = robot radius etc.: the notch only counts if something of that half-width fits in it."""
    return PLAY_L / 2 + (GOAL_DEPTH if abs(x) <= GOAL_W / 2 - inset else 0.0)

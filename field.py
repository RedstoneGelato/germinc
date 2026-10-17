"""
field.py - the field model used for localisation.

Field frame: origin = field centre, +y = towards the goal we attack, +x = right when facing it, units = cm.

Shape (field diagram, Oct 2026): the playing area is a rectangle. The white boundary line runs straight across the
goal mouths. Each goal's mouth is on the INNER edge of that line (the goal line, 91.5 cm from the centre) and the goal
reaches GOAL_DEPTH back from there, so its walls stand on the line and the back sticks out a little past it. The goal
floor counts as part of the playing area:

               +------+               <- goal back wall, GOAL_DEPTH behind the goal line
   ------------|------|------------   <- white boundary line (outer edge at PLAY_L / 2), straight across the goal
   |           +      +           |   <- goal line = inner edge of the white line, the goal posts are on it

!! Check every number below against this year's RCJ Soccer rules AND measure the actual competition field.
   Fields are often a few cm off, and that matters more than anything else in the localisation.
"""
import math

import cv2
import numpy as np

# ---- dimensions (cm) - VERIFY
PLAY_W = 132.0          # width of the playing area, measured between the outer edges of the white lines (x)
                        # (field diagram, Oct 2026: 1820 x 2430 outer, 250 outer area, 50 lines, centre -> goal line 915)
PLAY_L = 193.0          # length of the playing area (y, goal to goal)
OUTER = 25.0            # outer area: white line -> wall
LINE_W = 5.0            # white line thickness
GOAL_W = 45.0           # inner goal width
GOAL_DEPTH = 7.4
PENALTY_W = 90.0        # penalty area, measured like the playing area (outer line edges)
PENALTY_D = 30.0        # from the goal line (inner edge of the boundary line) to the outside of the black line
PENALTY_LINE_W = 5.0    # the penalty box is marked with a BLACK line (not white) - VERIFY its thickness

# neutral spots (where the referee puts the ball): the kickoff spot, and NEUTRAL_SIDE cm left / right of it, all on the
# halfway line (same distance to both goals) - VERIFY on the real field
NEUTRAL_SIDE = 30.0
NEUTRAL_SPOTS = [(0.0, 0.0), (-NEUTRAL_SIDE, 0.0), (NEUTRAL_SIDE, 0.0)]

WALL_W = PLAY_W + 2 * OUTER   # inside of the walls
WALL_L = PLAY_L + 2 * OUTER

ATTACK_GOAL_Y = PLAY_L / 2 - LINE_W    # goal lines (the goal mouths): the inner edge of the white boundary line
OWN_GOAL_Y = -ATTACK_GOAL_Y
GOAL_BACK_Y = ATTACK_GOAL_Y + GOAL_DEPTH  # inside of the goal back walls (attacking end; the own goal's is at -this)

MAP_RES = 1.0           # distance-map pixels per cm
MAP_MARGIN = 30.0       # map extends this far past the walls so out-of-field points still get a sensible distance


def boundary_polyline(sgn):
    """Centre line of one short edge, from left corner to right corner (straight across the goal). sgn=+1 attack end."""
    hw, hl = PLAY_W / 2 - LINE_W / 2, PLAY_L / 2 - LINE_W / 2   # boundary line centres
    pts = [(-hw, hl), (hw, hl)]
    return [(x, sgn * y) for x, y in pts]


def black_line_segments():
    """Centre lines of the black penalty box markings. Not used by the localisation (the camera only looks for
    white lines, and the LDRs don't react to black), but drawn by the simulator."""
    hl = PLAY_L / 2 - LINE_W                                     # inner edge of the white boundary line
    pw = PENALTY_W / 2 - PENALTY_LINE_W / 2
    segs = []
    for sgn in (1, -1):
        y0 = sgn * hl
        y1 = sgn * (ATTACK_GOAL_Y - PENALTY_D + PENALTY_LINE_W / 2)
        segs += [((-pw, y0), (-pw, y1)), ((-pw, y1), (pw, y1)), ((pw, y1), (pw, y0))]
    return segs


def white_line_segments():
    """Centre lines of all white markings, as ((x1, y1), (x2, y2)) in field cm. Only the boundary is white
    (the penalty box is black: black_line_segments())."""
    hw, hl = PLAY_W / 2 - LINE_W / 2, PLAY_L / 2 - LINE_W / 2   # boundary line centres
    segs = [((hw, -hl), (hw, hl)), ((-hw, hl), (-hw, -hl))]     # long sides
    for sgn in (1, -1):                                          # short sides
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


def goal_seen_point(px, py, goal_y):
    """Nearest COLOURED point of a goal from (px, py): what detection.py reports as the goal's near point. The mouth
    itself is empty, so from in front of the goal that's a post or the back wall, not a point in the mouth (which
    goal_near_point gives, up to GOAL_DEPTH / GOAL_W / 2 off). Coloured = the inside of both side walls (from the
    goal line to the back) and the back wall. Arrays work too."""
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    gw, back = GOAL_W / 2, goal_y + math.copysign(GOAL_DEPTH, goal_y)
    lo, hi = min(goal_y, back), max(goal_y, back)
    best_x, best_y = np.clip(px, -gw, gw), np.full_like(px, back)          # back wall
    best_d = (best_x - px) ** 2 + (best_y - py) ** 2
    for sx in (-gw, gw):                                                     # side walls
        sy = np.clip(py, lo, hi)
        d = (sx - px) ** 2 + (sy - py) ** 2
        closer = d < best_d
        best_x, best_y, best_d = np.where(closer, sx, best_x), np.where(closer, sy, best_y), np.where(closer, d, best_d)
    return best_x, best_y


def wall_distance(px, py, angle):
    """Distance from (px, py) to the wall along a ray at field angle `angle` (radians, 0 = +x, CCW).
    For the ultrasonic ring later. Arrays work for px, py, angle."""
    dx, dy = np.cos(angle), np.sin(angle)
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(dx > 1e-9, (WALL_W / 2 - px) / dx, np.where(dx < -1e-9, (-WALL_W / 2 - px) / dx, np.inf))
        ty = np.where(dy > 1e-9, (WALL_L / 2 - py) / dy, np.where(dy < -1e-9, (-WALL_L / 2 - py) / dy, np.inf))
    return np.maximum(np.minimum(tx, ty), 0.0)


def inside_play_area(x, y, margin=0.0):
    """Inside the white boundary (or inside a goal) by at least `margin` cm. Negative margin = allow past it."""
    if abs(x) <= PLAY_W / 2 - margin and abs(y) <= PLAY_L / 2 - margin:
        return True
    return abs(x) <= GOAL_W / 2 - margin and abs(y) <= GOAL_BACK_Y - margin


def boundary_y(x, inset=0.0):
    """How far the playing area reaches in y at this x (further in front of a goal: its back sticks out past the line).
    inset = robot radius etc.: the goal only counts if something of that half-width fits in it."""
    return GOAL_BACK_Y if abs(x) <= GOAL_W / 2 - inset else PLAY_L / 2

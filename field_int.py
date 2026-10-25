"""
field_int.py - the field model for the international RoboCupJunior Soccer rules. Same names and functions as
field.py (the RCJA field), so any module can use it in place of field.py.

To run the robot code on this field use main_int.py / main_attack_int.py (they call install() first, so every module
that does `import field` gets this one). sim_int.py does the same for the simulator.

Field frame: origin = field centre, +y = towards the goal we attack, +x = right when facing it, units = cm.
"""
import math
import sys

import cv2
import numpy as np

RULES = "international"

# ---- dimensions (cm) - VERIFY on the competition field
PLAY_W = 158.0          # width of the playing area, between the outer edges of the white lines (x)
PLAY_L = 219.0          # length of the playing area (y, goal to goal)
OUTER = 12.0            # outer area: white line -> wall
LINE_W = 2.0            # white line thickness
GOAL_W = 60.0           # inner goal width
GOAL_DEPTH = 7.4
PENALTY_W = 80.0        # penalty area, to the outer edges of its white line
PENALTY_D = 25.0        # from the goal line (inner edge of the boundary line) to the outside of its front line
PENALTY_CORNER_R = 15.0 # front corners rounded with this radius (outer edge of the line)
PENALTY_LINE_W = 2.0    # the penalty area line is WHITE here (part of the area)
PENALTY_LINE_WHITE = True

# neutral spots: the kickoff spot and four corner spots, NEUTRAL_END cm from each short edge (outer line edge), in line
# with the sides of the penalty areas - VERIFY on the real field
NEUTRAL_END = 45.0
NEUTRAL_X = PENALTY_W / 2
NEUTRAL_SPOTS = [(0.0, 0.0)] + [(sx * NEUTRAL_X, sy * (PLAY_L / 2 - NEUTRAL_END)) for sy in (1, -1) for sx in (-1, 1)]

WALL_W = PLAY_W + 2 * OUTER   # inside of the walls
WALL_L = PLAY_L + 2 * OUTER

ATTACK_GOAL_Y = PLAY_L / 2 - LINE_W    # goal lines (the goal mouths): the inner edge of the white boundary line
OWN_GOAL_Y = -ATTACK_GOAL_Y
GOAL_BACK_Y = ATTACK_GOAL_Y + GOAL_DEPTH  # inside of the goal back walls (attacking end; the own goal's is at -this)

# ---- rules (international)
BALL_OUT_AT_LINE = False   # the ball is live everywhere inside the walls
LINE_IS_OUT = False        # crossing the white line isn't out for a robot, touching a wall is
WALL_GAP = 2.0             # cm our robot's edge keeps from the walls at the furthest out it goes (touching one = out)

MAP_RES = 1.0           # distance-map pixels per cm
MAP_MARGIN = 30.0       # map extends this far past the walls so out-of-field points still get a sensible distance


def max_out(robot_radius):
    """How far a robot's centre may go past the outer edge of the white line without touching the wall (with
    WALL_GAP to spare). Negative = it has to stay that far inside the line's outer edge. (robot_config.MAX_OUT for
    this field: install() sets it.)"""
    return OUTER - robot_radius - WALL_GAP


def boundary_polyline(sgn):
    """Centre line of one short edge, from left corner to right corner (straight across the goal). sgn=+1 attack end."""
    hw, hl = PLAY_W / 2 - LINE_W / 2, PLAY_L / 2 - LINE_W / 2   # boundary line centres
    pts = [(-hw, hl), (hw, hl)]
    return [(x, sgn * y) for x, y in pts]


def penalty_polyline(sgn, inset=PENALTY_LINE_W / 2, arc_steps=8):
    """Outline of one penalty area (left side, rounded front, right side), `inset` cm inside its outer edge (default:
    the centre of its line), from the goal line at the left to the goal line at the right. sgn=+1 attack end."""
    hw, r = PENALTY_W / 2 - inset, PENALTY_CORNER_R - inset
    cx, cd = PENALTY_W / 2 - PENALTY_CORNER_R, PENALTY_D - PENALTY_CORNER_R    # left / right corner circle centres
    pts = [(-hw, 0.0), (-hw, cd)]
    for k in range(1, arc_steps + 1):                       # left corner: from the side round to the front
        a = math.pi - k * (math.pi / 2) / arc_steps
        pts.append((-cx + r * math.cos(a), cd + r * math.sin(a)))
    for k in range(0, arc_steps + 1):                       # right corner: from the front round to the side
        a = math.pi / 2 - k * (math.pi / 2) / arc_steps
        pts.append((cx + r * math.cos(a), cd + r * math.sin(a)))
    pts.append((hw, 0.0))
    return [(x, sgn * (ATTACK_GOAL_Y - dd)) for x, dd in pts]


def black_line_segments():
    """No black markings on this field (the penalty areas are white: white_line_segments())."""
    return []


def white_line_segments():
    """Centre lines of all white markings, as ((x1, y1), (x2, y2)) in field cm: the boundary AND the penalty areas."""
    hw, hl = PLAY_W / 2 - LINE_W / 2, PLAY_L / 2 - LINE_W / 2   # boundary line centres
    segs = [((hw, -hl), (hw, hl)), ((-hw, hl), (-hw, -hl))]     # long sides
    for sgn in (1, -1):                                          # short sides
        pts = boundary_polyline(sgn)
        segs += list(zip(pts[:-1], pts[1:]))
    for sgn in (1, -1):                                          # penalty areas
        pts = penalty_polyline(sgn)
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
    """Nearest COLOURED point of a goal from (px, py): what vision.py's detect() reports as the goal's near point (see
    field.py). Arrays work too."""
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


def touches_wall(x, y, r, gap=0.0):
    """A robot (centre x, y, radius r) within `gap` cm of touching an outer wall (out of bounds)."""
    return abs(x) >= WALL_W / 2 - r - gap or abs(y) >= WALL_L / 2 - r - gap


def penalty_depth(x, y, sgn):
    """How far (cm) point (x, y) is inside a penalty area (sgn=+1 the one at the attacking end, -1 ours), measured to
    its front and side lines' outer edges, the rounded corners included. Negative = outside it, by that much.
    The goal line end is left open (counted as going on behind the goal line): a robot hanging over the goal line
    into the goal isn't treated as having a part outside the area."""
    ax = abs(x)
    d = ATTACK_GOAL_Y - sgn * y                     # distance in front of that goal line
    cx, cd = PENALTY_W / 2 - PENALTY_CORNER_R, PENALTY_D - PENALTY_CORNER_R
    if ax > cx and d > cd:                          # level with a rounded corner
        return PENALTY_CORNER_R - math.hypot(ax - cx, d - cd)
    return min(PENALTY_W / 2 - ax, PENALTY_D - d)


def fully_in_penalty(x, y, r, sgn):
    """A robot (centre x, y, radius r) completely inside that penalty area (out of bounds)."""
    return penalty_depth(x, y, sgn) >= r


def partly_in_penalty(x, y, r, sgn):
    """Any part of a robot (centre x, y, radius r) in that penalty area (two of a team = multiple defense)."""
    return penalty_depth(x, y, sgn) > -r


def install():
    """Make this the field every robot module uses: `import field` anywhere afterwards gets field_int. Also sets
    robot_config.MAX_OUT for these walls and tells lines.py about the white penalty lines inside the field.
    Call it before importing anything else of the robot code (main_int.py / main_attack_int.py / sim_int.py do)."""
    me = sys.modules[__name__]
    old = sys.modules.get("field")
    if old is not None and old is not me:
        print("field_int.install(): field.py was already imported - some modules will use the RCJA field. "
              "Call install() first.")
    sys.modules["field"] = me
    import robot_config as cfg
    cfg.MAX_OUT = max_out(cfg.ROBOT_RADIUS)
    import lines
    # the penalty area lines are white too: with a confident pose well inside the boundary, an LDR hit is one of
    # those (the camera side of lines.py already only counts line points near the boundary)
    lines.IGNORE_INNER_LINES = True
    lines.BOUNDARY_BAND = LINE_W + 0.5
    return me

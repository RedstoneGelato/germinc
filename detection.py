"""
detection.py - turns the colour masks from robot_vision.py into things in cm (robot frame).

    orange -> ball
    yellow / blue -> goals
    green  -> field (convex hull = "on the field", everything outside is ignored)
    white  -> line points (for localisation + line avoidance)
    inside the field but none of the colours -> obstacles (other robots)

All positions are robot frame: origin = robot centre, +x = right, +y = forward, cm.
Floor positions use the blob point NEAREST the robot: the top-down view assumes everything is flat on the
floor, so anything tall (ball, goal walls, robots) gets stretched outwards; its nearest point is where it
touches the floor.
"""
import collections
import math

import cv2
import numpy as np

import field
import robot_config as cfg

# ---- TUNE (areas in cm^2 so they don't depend on the top-down px/cm)
BALL_MIN_AREA = 2.0
BALL_MIN_PX = 3                # ...and at least this many pixels. Counted in pixels, not by the outline's area: a far
                               # ball in the raw fisheye is only 1-2 px thick, and a 1 px thick outline has area 0
GOAL_MIN_AREA = 15.0
GREEN_MIN_AREA = 30.0          # green blobs smaller than this don't count towards the field hull
FIELD_MIN_AREA = 600.0         # less green than this in view = "can't see the field" (e.g. robot lifted)
HULL_MARGIN = 6.0              # cm the field hull is grown by before filtering lines / ball
LINE_MAX_POINTS = 250          # white pixels are subsampled to at most this many
LINE_MAX_RANGE = 120.0         # cm; far points are less accurate in the top-down view
OBSTACLE_MIN_AREA = 40.0
OBSTACLE_MIN_SIZE = 12.0       # cm across (sqrt of the blob area): a robot is never smaller, even 90 cm away; smaller
                               # blobs far away were the field corners (black outer area + blur), not robots
OBSTACLE_KERNEL = field.PENALTY_LINE_W + 2.0   # cm; opening of the "unknown colour" mask: removes anything thinner,
                                              # i.e. the black penalty box line (robots are ~20 cm across)
OBSTACLE_BODY_MARGIN = 2.0     # cm past our own body (ROBOT_RADIUS circle + ignore box) where obstacles are ignored
OBSTACLE_LINE_MARGIN = 3.0     # cm around white lines that never counts as obstacle (blurry line edges)...
OBSTACLE_LINE_MARGIN_FAR = 6.0 # ...this much further than OBSTACLE_FAR from us (the lens blurs far lines into grey,
OBSTACLE_FAR = 60.0            # which isn't white or green: the far side lines showed up as robots)
OBSTACLE_MAX_RANGE = 82.0      # cm: further than this the camera resolution is too low to tell robots from blur (the far
                               # ends of the black penalty line and the white lines blur into robot-sized blobs)
OBSTACLE_GRID_PPC = 2.0        # radial mode: px/cm of the top-down grid the obstacle mask is redrawn on (see _ObstacleGrid)
OBSTACLE_SPLIT_BIN = math.radians(3.0)   # robots touching (as we see them) make one blob: it's split by direction
OBSTACLE_SPLIT_JUMP = 6.0      # cm: ...where the blob's nearest distance jumps by more than this from one direction to
                               # the next (a robot partly behind a nearer one: the mirror sees robots side-on, so the
                               # nearer one hides a long wedge behind it, which joins onto the far one)...
OBSTACLE_SPLIT_RISE = 4.0      # ...or rises this much higher between two dips than both dips (two robots side by side)
OBSTACLE_SPLIT_MIN_W = 6.0     # cm across: a piece narrower than this isn't split off

Goal = collections.namedtuple("Goal", "near centre")      # both [x, y] robot frame cm
Obstacle = collections.namedtuple("Obstacle", "near size centre", defaults=(None,))
# near = [x, y] cm (nearest point), size = rough diameter cm, centre = [x, y] estimated centre of the whole robot


class Detections:
    """Everything seen in one frame. Robot frame cm."""

    def __init__(self, t, compass):
        self.t = t                  # capture time (time.monotonic())
        self.compass = compass      # robot heading at capture (radians)
        self.ball = None            # [x, y] or None
        self.goals = {"yellow": None, "blue": None}
        self.line_pts = np.zeros((0, 2), np.float32)
        self.obstacles = []
        self.field_visible = False
        self.hull_px = None         # convex hull of the field in pixels (for debug drawing)


def _area_px(rv, cm2):
    return cm2 * rv.px_per_cm ** 2


def _nearest_point(rv, contours):
    """Contour point closest to the robot centre, in robot cm."""
    pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float32)
    x, y = rv.to_cm(pts[:, 0], pts[:, 1])
    i = int(np.argmin(x * x + y * y))
    return [float(x[i]), float(y[i])]


def _robot_centre(rv, contour, near):
    """Centre of the whole robot behind an obstacle blob: in the middle of the blob's angular extent (as seen from
    us), one robot radius beyond its nearest point. (The nearest point alone is only the bit facing us, and not on
    the robot's centre line when it's seen at an angle or partly out of view; the far side of the blob is
    stretched outwards by the top-down view, so its depth is no use.)"""
    pts = contour.reshape(-1, 2).astype(np.float32)
    x, y = rv.to_cm(pts[:, 0], pts[:, 1])
    a_near = math.atan2(near[1], near[0])
    rel = (np.arctan2(y, x) - a_near + math.pi) % (2 * math.pi) - math.pi      # angles relative to the nearest point
    mid = a_near + (float(rel.min()) + float(rel.max())) / 2
    r = math.hypot(near[0], near[1]) + cfg.ROBOT_RADIUS
    return [r * math.cos(mid), r * math.sin(mid)]


def _size_cm(rv, contour):
    """Rough diameter (cm) of a blob: sqrt of its area measured in cm (the contour converted point by point, so it's
    right in the raw fisheye too, where a pixel is many cm far away and a fraction of one up close)."""
    pts = contour.reshape(-1, 2).astype(np.float32)
    x, y = rv.to_cm(pts[:, 0], pts[:, 1])
    area = 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    return math.sqrt(area)


def _thickness_cm(rv, contour):
    """How thick the blob really is (cm): its outline converted to cm, redrawn on a 2 px/cm grid, and the widest
    circle that fits inside it there. (Measured in cm, not pixels: a raw fisheye shows the same cm as many pixels
    close up and few far away, and squashes things along the radius.) A line, even bent into an L at a corner, is
    ~5 cm thick; a robot ~20 cm."""
    pts = contour.reshape(-1, 2).astype(np.float32)
    x, y = rv.to_cm(pts[:, 0], pts[:, 1])
    k = 2.0                                                       # grid px per cm
    gx, gy = (x - x.min()) * k + 2, (y - y.min()) * k + 2
    canvas = np.zeros((int(gy.max()) + 4, int(gx.max()) + 4), np.uint8)
    cv2.fillPoly(canvas, [np.stack([gx, gy], axis=1).round().astype(np.int32)], 255)
    depth = cv2.distanceTransform(canvas, cv2.DIST_L2, 3).max()
    return 2.0 * float(depth) / k


def _far_mask(rv, shape):
    """255 where the image is further than OBSTACLE_FAR from the robot centre (cached on rv)."""
    cached = getattr(rv, "_far_mask", None)
    if cached is None or cached.shape != shape:
        cached = np.full(shape, 255, np.uint8)
        cx, cy = (int(round(v)) for v in rv.centre)
        cv2.circle(cached, (cx, cy), int(round(rv.radius_px(OBSTACLE_FAR))), 0, -1)
        rv._far_mask = cached
    return cached


def _range_mask(rv, shape):
    """255 within LINE_MAX_RANGE of the robot centre (a few px extra; the exact cut is done in cm), cached on rv."""
    cached = getattr(rv, "_range_mask", None)
    if cached is None or cached.shape != shape:
        cached = np.zeros(shape, np.uint8)
        cx, cy = (int(round(v)) for v in rv.centre)
        cv2.circle(cached, (cx, cy), int(math.ceil(rv.radius_px(LINE_MAX_RANGE))) + 3, 255, -1)
        rv._range_mask = cached
    return cached


def _split_points(prof, nearest):
    """Indices i (into prof, the nearest distance per direction bin, NaN = no blob there) where the blob is split
    between bin i - 1 and bin i (see OBSTACLE_SPLIT_*). nearest = the blob's nearest distance (for the min width)."""
    ok = np.nonzero(~np.isnan(prof))[0]
    if len(ok) < 3:
        return []
    p = prof[ok]
    min_bins = OBSTACLE_SPLIT_MIN_W / max(nearest * OBSTACLE_SPLIT_BIN, 1e-6)
    cuts = [int(ok[i]) for i in range(1, len(p)) if abs(p[i] - p[i - 1]) > OBSTACLE_SPLIT_JUMP]
    # two dips with a rise between them: split at the highest bin between consecutive dips (only where no jump does)
    dips = [i for i in range(len(p)) if (i == 0 or p[i] <= p[i - 1]) and (i == len(p) - 1 or p[i] < p[i + 1])]
    for a, b in zip(dips, dips[1:]):
        top = a + int(np.argmax(p[a:b + 1]))
        if p[top] - max(p[a], p[b]) >= OBSTACLE_SPLIT_RISE and not any(ok[a] < c <= ok[b] for c in cuts):
            cuts.append(int(ok[top]))
    cuts = sorted(set(cuts))
    keep, last = [], int(ok[0])               # drop cuts that would leave a piece narrower than OBSTACLE_SPLIT_MIN_W
    for i, c in enumerate(cuts):
        nxt = cuts[i + 1] if i + 1 < len(cuts) else int(ok[-1]) + 1
        if c - last >= min_bins and nxt - c >= min_bins:
            keep.append(c)
            last = c
    return keep


def _split_obstacle(g, contour, cover):
    """One "unknown colour" blob -> [(near, centre, size)] for each robot in it (usually one). g = the RobotVision or
    _ObstacleGrid the contour is in, cover = (x, y) cm arrays of known things that may stand in front of a robot
    (the ball), or None: directions where that is nearer than the blob are skipped, so a ball in front of a robot
    doesn't split it. Works on the outline only: the nearest point in every direction, and the ends, are on it."""
    pts = contour.reshape(-1, 2).astype(np.float32)
    x, y = g.to_cm(pts[:, 0], pts[:, 1])
    d = np.hypot(x, y)
    j = int(np.argmin(d))
    a0 = math.atan2(y[j], x[j])
    rel = (np.arctan2(y, x) - a0 + math.pi) % (2 * math.pi) - math.pi
    lo = float(rel.min())
    b = ((rel - lo) / OBSTACLE_SPLIT_BIN).astype(np.int32)
    nb = int(b.max()) + 1
    if nb < 3:
        return [([float(x[j]), float(y[j])], _robot_centre(g, contour, [float(x[j]), float(y[j])]), _size_cm(g, contour))]
    prof = np.full(nb, np.inf)
    np.minimum.at(prof, b, d)
    if cover is not None:
        cx, cy = cover
        crel = (np.arctan2(cy, cx) - a0 + math.pi) % (2 * math.pi) - math.pi
        cb = np.floor((crel - lo) / OBSTACLE_SPLIT_BIN).astype(np.int32)
        inside = (cb >= 0) & (cb < nb)
        cprof = np.full(nb, np.inf)
        np.minimum.at(cprof, cb[inside], np.hypot(cx, cy)[inside])
        prof[cprof < prof] = np.nan
    prof[np.isinf(prof)] = np.nan
    cuts = _split_points(prof, float(d[j]))
    if not cuts:
        near = [float(x[j]), float(y[j])]
        return [(near, _robot_centre(g, contour, near), _size_cm(g, contour))]
    out = []
    edges = [0] + cuts + [nb]
    for s0, s1 in zip(edges, edges[1:]):
        sel = (b >= s0) & (b < s1)
        if not sel.any():
            continue
        k = int(np.argmin(np.where(sel, d, np.inf)))
        near = [float(x[k]), float(y[k])]
        span = float(rel[sel].max()) - float(rel[sel].min())
        mid = a0 + (float(rel[sel].min()) + float(rel[sel].max())) / 2
        r = float(d[k]) + cfg.ROBOT_RADIUS
        out.append((near, [r * math.cos(mid), r * math.sin(mid)], span * r))     # size: how wide it looks
    return out


class _ObstacleGrid:
    """Radial mode: the "unknown colour" mask redrawn on a small top-down grid (OBSTACLE_GRID_PPC px/cm, robot in
    the middle, out to OBSTACLE_MAX_RANGE + a robot), so the opening that removes thin things is the same size in cm
    everywhere. (In the raw mirror image a cm is ~4 px next to us and ~0.6 px at 80 cm, and the opening is sized for
    far away: close up it left the black penalty line in, which then joined onto a robot standing near it, and the
    robot was reported at the line.) Has the same to_cm() / px_per_cm as RobotVision, so the helpers above work on
    it; built once per RobotVision (cached on rv)."""

    def __init__(self, rv):
        k = OBSTACLE_GRID_PPC
        self.px_per_cm = k
        self.half = int(math.ceil((OBSTACLE_MAX_RANGE + 2 * cfg.ROBOT_RADIUS) * k))
        n = 2 * self.half
        u, v = np.meshgrid(np.arange(n, dtype=np.float64) + 0.5, np.arange(n, dtype=np.float64) + 0.5)
        sx, sy = rv.to_px((u - self.half) / k, (self.half - v) / k)        # grid pixel -> raw image pixel
        self.maps = cv2.convertMaps(sx.astype(np.float32), sy.astype(np.float32), cv2.CV_16SC2)

    def to_cm(self, px, py):
        return (px + 0.5 - self.half) / self.px_per_cm, (self.half - py - 0.5) / self.px_per_cm

    def remap(self, mask):
        return cv2.remap(mask, *self.maps, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _obstacle_grid(rv):
    grid = getattr(rv, "_obstacle_grid", None)
    if grid is None:
        grid = rv._obstacle_grid = _ObstacleGrid(rv)
    return grid


def _big_contours(mask, min_area, approx=cv2.CHAIN_APPROX_NONE):
    x, y, w, h = cv2.boundingRect(mask)          # only search where the mask has pixels (same contours, cheaper)
    if w == 0:
        return []
    contours, _ = cv2.findContours(mask[y:y + h, x:x + w], cv2.RETR_EXTERNAL, approx, offset=(x, y))
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def _ball(rv, orange):
    """Nearest point (robot cm) of the biggest orange blob, by pixel count, or None."""
    x, y, w, h = cv2.boundingRect(orange)
    if w == 0:
        return None
    n, lab, stats, _ = cv2.connectedComponentsWithStats(orange[y:y + h, x:x + w], connectivity=8)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < max(BALL_MIN_PX, _area_px(rv, BALL_MIN_AREA)):
        return None
    pts = cv2.findNonZero((lab == i).astype(np.uint8)).reshape(-1, 2).astype(np.float32) + (x, y)
    bx, by = rv.to_cm(pts[:, 0], pts[:, 1])
    j = int(np.argmin(bx * bx + by * by))
    return [float(bx[j]), float(by[j])]


def field_mask(rv, green):
    """Filled convex hull of all green blobs (inside the camera's view), or (None, None) if not enough green."""
    # (only areas and the hull are used: the simplified outline gives the same ones, with far fewer points)
    contours = _big_contours(green, _area_px(rv, GREEN_MIN_AREA), cv2.CHAIN_APPROX_SIMPLE)
    if not contours or sum(cv2.contourArea(c) for c in contours) < _area_px(rv, FIELD_MIN_AREA):
        return None, None
    hull = cv2.convexHull(np.vstack(contours))
    mask = np.zeros_like(green)
    cv2.fillConvexPoly(mask, hull, 255)
    return cv2.bitwise_and(mask, rv.valid), hull


def detect(rv, res, t, compass):
    """rv = RobotVision, res = rv.process(raw, align=False). Returns Detections."""
    d = Detections(t, compass)
    m = res.masks
    ppc = rv.px_per_cm

    field, d.hull_px = field_mask(rv, m["green"])
    d.field_visible = field is not None
    if field is not None:
        # hull grown by HULL_MARGIN: filling the hull and drawing its outline thick is the same as dilating it, but cheap
        field_grown = np.zeros_like(field)
        cv2.fillConvexPoly(field_grown, d.hull_px, 255)
        cv2.polylines(field_grown, [d.hull_px], True, 255, max(1, int(round(2 * HULL_MARGIN * ppc))))
        field_grown = cv2.bitwise_and(field_grown, rv.valid)   # (raw fisheye: nothing past the calibrated distance)
    else:
        field_grown = rv.valid       # can't see the field: don't filter

    # ball: biggest orange blob on (or just off) the field
    orange = cv2.bitwise_and(m["orange"], field_grown)
    d.ball = _ball(rv, orange)

    # goals: all big enough blobs of the colour together (a robot in front can split a goal in two)
    for name in ("yellow", "blue"):
        cs = _big_contours(m[name], _area_px(rv, GOAL_MIN_AREA))
        if cs:
            pts = np.vstack([c.reshape(-1, 2) for c in cs]).astype(np.float32)
            gx, gy = rv.to_cm(pts[:, 0], pts[:, 1])       # centre of the goal's extent in cm (in a raw fisheye frame
            centre = [float(gx.min() + gx.max()) / 2.0, float(gy.min() + gy.max()) / 2.0]   # a pixel box is skewed)
            d.goals[name] = Goal(_nearest_point(rv, cs), centre)

    # white line points on the field, subsampled
    # (the range disc first, so far fewer pixels get converted to cm; findNonZero keeps np.nonzero's row order)
    white = cv2.bitwise_and(cv2.bitwise_and(m["white"], field_grown), _range_mask(rv, field_grown.shape))
    nz = cv2.findNonZero(white)
    if nz is not None:
        nz = nz.reshape(-1, 2).astype(np.float32)
        lx, ly = rv.to_cm(nz[:, 0], nz[:, 1])
        keep = lx * lx + ly * ly < LINE_MAX_RANGE ** 2
        lx, ly = lx[keep], ly[keep]
        if len(lx) > LINE_MAX_POINTS:
            idx = np.linspace(0, len(lx) - 1, LINE_MAX_POINTS).astype(np.int32)
            lx, ly = lx[idx], ly[idx]
        d.line_pts = np.stack([lx, ly], axis=1).astype(np.float32)

    # obstacles: inside the field, but not any known colour
    if field is not None:
        kl = max(1, int(round(2 * OBSTACLE_LINE_MARGIN * ppc))) | 1
        white_grown = cv2.dilate(m["white"], np.ones((kl, kl), np.uint8))   # square kernel: much faster than round
        kf = max(1, int(round(2 * OBSTACLE_LINE_MARGIN_FAR * ppc))) | 1
        white_far = cv2.dilate(m["white"], np.ones((kf, kf), np.uint8))
        white_grown |= cv2.bitwise_and(white_far, _far_mask(rv, white_far.shape))
        labels = getattr(res, "labels", None)     # any colour bit set = a known colour (white is in white_grown)
        if labels is not None:
            known = cv2.bitwise_or(cv2.compare(labels, 0, cv2.CMP_NE), white_grown)
        else:
            known = m["green"] | white_grown | m["orange"] | m["yellow"] | m["blue"]
        unknown = cv2.bitwise_and(field, cv2.bitwise_not(known))
        # our own body: the ignore box no longer covers the front (so the ball in the dribbler stays visible),
        # so blank a circle the size of the robot too
        x1, y1, x2, y2 = rv.ignore_box
        pad = int(round(OBSTACLE_BODY_MARGIN * ppc))
        unknown[max(y1 - pad, 0):y2 + pad, max(x1 - pad, 0):x2 + pad] = 0
        cv2.circle(unknown, tuple(int(round(v)) for v in rv.centre),
                   int(round(rv.radius_px(cfg.ROBOT_RADIUS + OBSTACLE_BODY_MARGIN))), 0, -1)
        g = rv
        if getattr(rv, "mode", None) == "radial":      # open + find blobs on a top-down grid (see _ObstacleGrid)
            g = _obstacle_grid(rv)
            unknown = g.remap(unknown)
        cover = cv2.findNonZero(orange)                # the ball (see _split_obstacle)
        if cover is not None:
            cover = rv.to_cm(*cover.reshape(-1, 2).astype(np.float32).T)
        k = max(1, int(round(OBSTACLE_KERNEL * g.px_per_cm)))
        unknown = cv2.morphologyEx(unknown, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        found = []
        for c in _big_contours(unknown, _area_px(g, OBSTACLE_MIN_AREA)):
            if _thickness_cm(g, c) < OBSTACLE_KERNEL:
                continue        # thin in real life = a line (close up in a raw fisheye the penalty line is wide in pixels)
            for near, centre, size in _split_obstacle(g, c, cover):
                if near[0] ** 2 + near[1] ** 2 < OBSTACLE_MAX_RANGE ** 2:
                    found.append(Obstacle(near, size, centre))
        # one robot can come out as two blobs (e.g. cut in two by a white line it stands on): same centre = same robot
        for o in sorted(found, key=lambda o: o.near[0] ** 2 + o.near[1] ** 2):
            twin = next((k for k, q in enumerate(d.obstacles)
                         if math.hypot(q.centre[0] - o.centre[0], q.centre[1] - o.centre[1]) < cfg.ROBOT_RADIUS), None)
            if twin is not None:
                q = d.obstacles[twin]
                d.obstacles[twin] = q._replace(size=math.hypot(q.size, o.size))
            elif o.size >= OBSTACLE_MIN_SIZE:
                d.obstacles.append(o)
    return d

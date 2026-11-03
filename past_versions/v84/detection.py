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


def _far_mask(rv, shape):
    """255 where the image is further than OBSTACLE_FAR from the robot centre (cached on rv)."""
    cached = getattr(rv, "_far_mask", None)
    if cached is None or cached.shape != shape:
        cached = np.full(shape, 255, np.uint8)
        cx, cy = (int(round(v)) for v in rv.centre)
        cv2.circle(cached, (cx, cy), int(round(OBSTACLE_FAR * rv.px_per_cm)), 0, -1)
        rv._far_mask = cached
    return cached


def _big_contours(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def field_mask(rv, green):
    """Filled convex hull of all green blobs (inside the camera's view), or (None, None) if not enough green."""
    contours = _big_contours(green, _area_px(rv, GREEN_MIN_AREA))
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
    else:
        field_grown = rv.valid       # can't see the field: don't filter

    # ball: biggest orange blob on (or just off) the field
    orange = cv2.bitwise_and(m["orange"], field_grown)
    balls = _big_contours(orange, _area_px(rv, BALL_MIN_AREA))
    if balls:
        d.ball = _nearest_point(rv, [max(balls, key=cv2.contourArea)])

    # goals: all big enough blobs of the colour together (a robot in front can split a goal in two)
    for name in ("yellow", "blue"):
        cs = _big_contours(m[name], _area_px(rv, GOAL_MIN_AREA))
        if cs:
            x, y, w, h = cv2.boundingRect(np.vstack(cs))
            d.goals[name] = Goal(_nearest_point(rv, cs), list(rv.to_cm(x + w / 2.0, y + h / 2.0)))

    # white line points on the field, subsampled
    white = cv2.bitwise_and(m["white"], field_grown)
    ys, xs = np.nonzero(white)
    if len(xs):
        lx, ly = rv.to_cm(xs.astype(np.float32), ys.astype(np.float32))
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
        known = m["green"] | white_grown | m["orange"] | m["yellow"] | m["blue"]
        unknown = cv2.bitwise_and(field, cv2.bitwise_not(known))
        # our own body: the ignore box no longer covers the front (so the ball in the dribbler stays visible),
        # so blank a circle the size of the robot too
        x1, y1, x2, y2 = rv.ignore_box
        pad = int(round(OBSTACLE_BODY_MARGIN * ppc))
        unknown[max(y1 - pad, 0):y2 + pad, max(x1 - pad, 0):x2 + pad] = 0
        cv2.circle(unknown, tuple(int(round(v)) for v in rv.centre),
                   int(round((cfg.ROBOT_RADIUS + OBSTACLE_BODY_MARGIN) * ppc)), 0, -1)
        k = max(1, int(round(OBSTACLE_KERNEL * ppc)))
        unknown = cv2.morphologyEx(unknown, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        found = []
        for c in _big_contours(unknown, _area_px(rv, OBSTACLE_MIN_AREA)):
            size = float(np.sqrt(cv2.contourArea(c)) / ppc)
            near = _nearest_point(rv, [c])
            if near[0] ** 2 + near[1] ** 2 < OBSTACLE_MAX_RANGE ** 2:
                found.append(Obstacle(near, size, _robot_centre(rv, c, near)))
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

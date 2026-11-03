"""Camera: raw frame -> colour masks (RobotVision) -> Detections in robot-frame cm (detect) -> VisionThread."""
import base64
import collections
import json
import math
import threading
import time

import cv2
import numpy as np

import field
import robot_config as cfg

COLOURS = ("blue", "yellow", "white", "green", "orange")
BITS = {name: 1 << i for i, name in enumerate(COLOURS)}
ROTATIONS = {"90_CLOCKWISE": cv2.ROTATE_90_CLOCKWISE, "180": cv2.ROTATE_180,
             "90_COUNTERCLOCKWISE": cv2.ROTATE_90_COUNTERCLOCKWISE, "NONE": None}
QUARTER_TURNS = (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)


class Result:
    __slots__ = ("masks", "centre", "labels", "_frame", "_make_frame")

    def __init__(self, masks, centre, labels, make_frame):
        self.masks, self.centre, self.labels = masks, centre, labels
        self._frame, self._make_frame = None, make_frame

    @property
    def frame(self):
        if self._frame is None:
            self._frame = self._make_frame()
        return self._frame


def _rotate_point(point, size, rotation):
    x, y = point
    width, height = size
    if rotation == cv2.ROTATE_90_CLOCKWISE:
        return (height - 1 - y, x)
    if rotation == cv2.ROTATE_180:
        return (width - 1 - x, height - 1 - y)
    if rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return (y, width - 1 - x)
    return (x, y)


# ==================================== geometry ====================================
class IdentityWarp:
    def __init__(self, capture_size, rotation=None):
        self.rotation = rotation
        width, height = capture_size
        self.crop = (0, 0, width, height)
        self.valid = self._rotate(np.full((height, width), 255, np.uint8))

    def _rotate(self, image):
        return image if self.rotation is None else cv2.rotate(image, self.rotation)

    colour = labels = mask = _rotate


class RemapWarp:
    def __init__(self, map_x, map_y, capture_size, rotation=None, outside_label=0):
        map_x, map_y = np.asarray(map_x, np.float32), np.asarray(map_y, np.float32)
        if rotation is not None:
            map_x, map_y = cv2.rotate(map_x, rotation), cv2.rotate(map_y, rotation)
        width, height = capture_size
        inside = (map_x > -1) & (map_x < width) & (map_y > -1) & (map_y < height)
        if inside.any():
            xs, ys = map_x[inside], map_y[inside]
            x0, y0 = max(int(np.floor(xs.min())) - 1, 0), max(int(np.floor(ys.min())) - 1, 0)
            x1, y1 = min(int(np.ceil(xs.max())) + 2, width), min(int(np.ceil(ys.max())) + 2, height)
        else:
            x0, y0, x1, y1 = 0, 0, width, height
        self.crop = (x0, y0, x1 - x0, y1 - y0)
        self._is_full_frame = self.crop == (0, 0, width, height)
        cropped_x, cropped_y = map_x - x0, map_y - y0
        cropped_x[~inside], cropped_y[~inside] = -10, -10
        self._bilinear = cv2.convertMaps(cropped_x, cropped_y, cv2.CV_16SC2)
        self._nearest = cv2.convertMaps(cropped_x, cropped_y, cv2.CV_16SC2, nninterpolation=True)[0]
        self._outside_label = int(outside_label)
        self.valid = cv2.remap(np.full((y1 - y0, x1 - x0), 255, np.uint8), self._nearest, None, cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def colour(self, raw):
        if not self._is_full_frame:
            x, y, w, h = self.crop
            raw = raw[y:y + h, x:x + w]
        return cv2.remap(raw, *self._bilinear, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def labels(self, labels):
        return cv2.remap(labels, self._nearest, None, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=self._outside_label)

    def mask(self, mask):
        x, y, w, h = self.crop
        return cv2.remap(np.ascontiguousarray(mask[y:y + h, x:x + w]), self._nearest, None, cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)


WARP_MODELS = {}


def warp_model(name):
    def register(fn):
        WARP_MODELS[name] = fn
        return fn
    return register


def _topdown_maps(homography, out_size, K, D):
    width, height = out_size
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    floor = homography @ np.stack([u.ravel(), v.ravel(), np.ones(width * height)], axis=0)
    in_front = floor[2] > 1e-9
    z = np.where(in_front, floor[2], 1.0)
    rays = np.stack([floor[0] / z, floor[1] / z], axis=1).reshape(-1, 1, 2)
    pixels = cv2.fisheye.distortPoints(rays, K, D).reshape(-1, 2)
    map_x = pixels[:, 0].reshape(height, width).astype(np.float32)
    map_y = pixels[:, 1].reshape(height, width).astype(np.float32)
    bad = (~in_front).reshape(height, width) | ~np.isfinite(map_x) | ~np.isfinite(map_y)
    map_x[bad] = -1
    map_y[bad] = -1
    return map_x, map_y


@warp_model("fisheye_homography")
def _fisheye_topdown(config, capture_size):
    lens, topdown = config["lens"], config["topdown"]
    K, D = np.array(lens["K"], np.float64), np.array(lens["D"], np.float64)
    map_x, map_y = _topdown_maps(np.array(topdown["Hn"], np.float64), tuple(topdown["out_size"]), K, D)
    return map_x, map_y, topdown["px_per_cm"]


def _undistort_maps(K, D, size, balance, fov_scale):
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, size, np.eye(3), balance=balance, fov_scale=fov_scale)
    return cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, size, cv2.CV_32FC1)


# ==================================== noise opening on all colour bits at once ====================================
def _shifted(image, axis, offset):
    n = image.shape[axis]
    dst, src = (slice(0, n - offset), slice(offset, n)) if offset > 0 else (slice(-offset, n), slice(0, n + offset))
    if axis == 0:
        return (dst, slice(None)), (src, slice(None))
    return (slice(None), dst), (slice(None), src)


def _morph_bits(image, k, combine):
    anchor = k // 2
    for axis in (1, 0):
        out = image.copy()
        for offset in range(-anchor, k - anchor):
            if offset:
                dst, src = _shifted(image, axis, offset)
                combine(out[dst], image[src], out=out[dst])
        image = out
    return image


def open_bits(labels, k):
    if k <= 1:
        return labels
    eroded = _morph_bits(labels, k, np.bitwise_and)
    return _morph_bits(eroded, k, np.bitwise_or)


FALLBACK_PX_PER_CM = 1.0
RADIAL_REF_CM = 80.0
RADIAL_KEEP_BITS = ("orange",)
RADIAL_KEEP_BALL_PX = 2.0
RADIAL_BALL_DIAMETER_CM = 4.2
RADIAL_FLAT_CENTRE_CM = 10.0


def radial_table(radial):
    if radial.get("table_px"):
        pixels = np.array(radial["table_px"], np.float64)
        cm = np.array(radial["table_cm"], np.float64)
    else:
        pixels = np.linspace(0.0, float(radial["r_max_px"]), 512)
        cm = np.polyval(np.array(radial["coeffs"], np.float64), pixels)
    cm = np.maximum.accumulate(np.maximum(cm, 0.0))
    return pixels, cm


def encode_mask(mask):
    ok, buf = cv2.imencode(".png", np.where(mask > 0, 255, 0).astype(np.uint8))
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_mask(text):
    mask = cv2.imdecode(np.frombuffer(base64.b64decode(text), np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("bad ignore mask PNG")
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def camera_seen_mask(detection, size):
    width, height = size
    mirror, ignore = detection.get("mirror_crop"), detection.get("ignore_mask")
    if not mirror and not ignore:
        return None
    seen = np.full((height, width), 255, np.uint8)
    if mirror:
        seen[:] = 0
        cv2.circle(seen, (int(round(mirror["cx"])), int(round(mirror["cy"]))), int(round(mirror["r"])), 255, -1)
    if ignore:
        painted = decode_mask(ignore["png"])
        if painted.shape != (height, width):
            painted = cv2.resize(painted, (width, height), interpolation=cv2.INTER_NEAREST)
        seen[painted > 0] = 0
    return seen


class RobotVision:
    def __init__(self, path="robot_vision_config.json"):
        with open(path) as f:
            config = json.load(f)
        if config.get("version") != 1:
            raise ValueError(f"unsupported config version {config.get('version')!r}")
        self.config = config
        self.size = tuple(config["capture_size"])
        self.rotation = ROTATIONS[config["rotation"]]
        self._build_colour_tables(config["hsv"])
        detection = config["detection"]
        self.morph_kernel = int(detection["morph_kernel"])

        base_size = self._setup_geometry(config)
        self.centre = self._robot_centre(config["robot_centre"], base_size)
        self.ignore_box = self._ignore_box(detection)
        seen = camera_seen_mask(detection, self.size)
        self.seen = None if seen is None else self.warp.mask(seen)
        self.valid = self._valid_mask()
        self._far_keep = self._far_keep_mask()

    def _build_colour_tables(self, hsv):
        self._luts = np.zeros((3, 256), np.uint8)
        for name in COLOURS:
            low, high = hsv[name]["lo"], hsv[name]["hi"]
            for channel in range(3):
                self._luts[channel, int(low[channel]):int(high[channel]) + 1] |= BITS[name]
        values = np.arange(256)
        self._mask_luts = {name: np.where(values & BITS[name], 255, 0).astype(np.uint8) for name in COLOURS}

    def _setup_geometry(self, config):
        lens, topdown, radial = config.get("lens"), config.get("topdown"), config.get("radial")
        self.mode = "raw"
        self.px_per_cm = FALLBACK_PX_PER_CM
        self._radial = None
        maps = None
        if radial:
            self._radial = radial_table(radial)
            self.mode = "radial"
            self.px_per_cm = self._radial_px_per_cm()
        elif topdown and (lens or topdown.get("model")):
            model = topdown.get("model", "fisheye_homography")
            if model not in WARP_MODELS:
                raise ValueError(f"unknown top-down model {model!r} (have: {', '.join(WARP_MODELS)})")
            map_x, map_y, self.px_per_cm = WARP_MODELS[model](config, self.size)
            maps = (map_x, map_y)
            self.mode = "topdown"
        elif lens:
            K, D = np.array(lens["K"], np.float64), np.array(lens["D"], np.float64)
            view = config["undistort_view"]
            maps = _undistort_maps(K, D, self.size, view["balance"], view["fov_scale"])
            self.mode = "undistort"
        if maps is None:
            self.warp = IdentityWarp(self.size, self.rotation)
            return self.size
        outside_label = int(self.classify(np.zeros((1, 1, 3), np.uint8))[0, 0])
        self.warp = RemapWarp(maps[0], maps[1], self.size, self.rotation, outside_label)
        return (maps[0].shape[1], maps[0].shape[0])

    def _radial_px_per_cm(self):
        pixels, cm = self._radial
        ref = min(RADIAL_REF_CM, float(cm[-1]))
        ref_px = float(np.interp(ref, cm, pixels))
        along_radius = (ref_px - float(np.interp(ref - 5.0, cm, pixels))) / 5.0
        return max(0.2, min(along_radius, ref_px / max(ref, 1e-6)))

    def _robot_centre(self, configured, base_size):
        width, height = base_size
        if configured is None:
            frame_w, frame_h = (height, width) if self.rotation in QUARTER_TURNS else (width, height)
            return (frame_w / 2.0, frame_h / 2.0)
        x = min(max(configured[0], 0), width - 1)
        y = min(max(configured[1], 0), height - 1)
        return _rotate_point((x, y), (width, height), self.rotation)

    def _ignore_box(self, detection):
        ignore_cm = detection.get("ignore_cm")
        if ignore_cm and not any(ignore_cm.values()):
            ignore_cm = None
        if ignore_cm and self._radial is not None:
            xs, ys = self.to_px(np.array([-ignore_cm["left"], ignore_cm["right"]], np.float64),
                                np.array([ignore_cm["front"], -ignore_cm["back"]], np.float64))
            box = (min(xs), min(ys), max(xs), max(ys))
        elif ignore_cm:
            cx, cy = self.centre
            ppc = self.px_per_cm
            box = (cx - ignore_cm["left"] * ppc, cy - ignore_cm["front"] * ppc,
                   cx + ignore_cm["right"] * ppc, cy + ignore_cm["back"] * ppc)
        else:
            box = detection.get("ignore_box") or (0, 0, 0, 0)
        return tuple(max(int(round(v)), 0) for v in box)

    def centre_px(self):
        return tuple(int(round(v)) for v in self.centre)

    def _valid_mask(self):
        valid = self.warp.valid
        if self.seen is not None:
            valid = cv2.bitwise_and(valid, self.seen)
        if self._radial is not None:
            calibrated = np.zeros_like(valid)
            cv2.circle(calibrated, self.centre_px(), int(self._radial[0][-1]), 255, -1)
            valid = cv2.bitwise_and(valid, calibrated)
        return cv2.erode(valid, np.ones((5, 5), np.uint8))

    def _far_keep_mask(self):
        if self._radial is None or self.morph_kernel <= 1:
            return None
        pixels, cm = self._radial
        ball_px = np.gradient(pixels, cm, edge_order=1) * RADIAL_BALL_DIAMETER_CM
        thin = np.nonzero(ball_px < RADIAL_KEEP_BALL_PX * self.morph_kernel)[0]
        thin = thin[cm[thin] > RADIAL_FLAT_CENTRE_CM]
        if not len(thin):
            return None
        far = np.full(self.valid.shape, 255, np.uint8)
        cv2.circle(far, self.centre_px(), int(pixels[thin[0]]), 0, -1)
        keep_bits = 0
        for name in RADIAL_KEEP_BITS:
            keep_bits |= BITS[name]
        return cv2.bitwise_and(far, keep_bits)

    def apply_camera(self, cam):
        controls = dict(self.config["camera"]["controls"])
        if "ColourGains" in controls:
            controls["ColourGains"] = tuple(controls["ColourGains"])
        supported = getattr(cam, "camera_controls", None)
        if isinstance(supported, dict) and supported:
            controls = {k: v for k, v in controls.items() if k in supported}
        cam.set_controls(controls)

    def to_cm(self, px, py):
        cx, cy = self.centre
        dx, dy = px - cx, cy - py
        if self._radial is None:
            return dx / self.px_per_cm, dy / self.px_per_cm
        r = np.hypot(dx, dy)
        scale = np.interp(r, self._radial[0], self._radial[1]) / np.maximum(r, 1e-6)
        return dx * scale, dy * scale

    def to_px(self, x_cm, y_cm):
        cx, cy = self.centre
        if self._radial is None:
            return cx + x_cm * self.px_per_cm, cy - y_cm * self.px_per_cm
        d = np.hypot(x_cm, y_cm)
        scale = np.interp(d, self._radial[1], self._radial[0]) / np.maximum(d, 1e-6)
        return cx + x_cm * scale, cy - y_cm * scale

    def radius_px(self, cm):
        if self._radial is None:
            return cm * self.px_per_cm
        return float(np.interp(cm, self._radial[1], self._radial[0]))

    def classify(self, image):
        h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        lut = self._luts
        return cv2.bitwise_and(cv2.bitwise_and(cv2.LUT(h, lut[0]), cv2.LUT(s, lut[1])), cv2.LUT(v, lut[2]))

    def process(self, raw):
        x, y, w, h = self.warp.crop
        labels = self.warp.labels(self.classify(raw[y:y + h, x:x + w]))
        x1, y1, x2, y2 = self.ignore_box
        labels[y1:y2, x1:x2] = 0
        if self.seen is not None:
            labels = cv2.bitwise_and(labels, self.seen)
        opened = open_bits(labels, self.morph_kernel)
        if self._far_keep is not None:
            opened = cv2.bitwise_or(opened, cv2.bitwise_and(labels, self._far_keep))
        masks = {name: cv2.LUT(opened, self._mask_luts[name]) for name in COLOURS}
        return Result(masks, self.centre, opened, lambda: self.warp.colour(raw))


# ==================================== detection: masks -> things in cm ====================================
# TUNE: areas in cm^2, distances in cm
BALL_MIN_AREA = 2.0
BALL_MIN_PX = 3
GOAL_MIN_AREA = 15.0
GREEN_MIN_AREA = 30.0
FIELD_MIN_AREA = 600.0
HULL_MARGIN = 6.0
LINE_MAX_POINTS = 250
LINE_MAX_RANGE = 120.0
OBSTACLE_MIN_AREA = 40.0
OBSTACLE_MIN_SIZE = 12.0
OBSTACLE_KERNEL = field.PENALTY_LINE_W + 2.0
OBSTACLE_BODY_MARGIN = 2.0
OBSTACLE_LINE_MARGIN = 3.0
OBSTACLE_LINE_MARGIN_FAR = 6.0
OBSTACLE_FAR = 60.0
OBSTACLE_MAX_RANGE = 82.0
OBSTACLE_GRID_PPC = 2.0
OBSTACLE_SPLIT_BIN = math.radians(3.0)
OBSTACLE_SPLIT_JUMP = 6.0
OBSTACLE_SPLIT_RISE = 4.0
OBSTACLE_SPLIT_MIN_W = 6.0
THICKNESS_GRID_PPC = 2.0
WHITE_BODY_KERNEL = 12.0
WHITE_BODY_MARGIN = OBSTACLE_LINE_MARGIN_FAR
WHITE_GRID_PPC = 1.0
assert LINE_MAX_RANGE + WHITE_BODY_KERNEL >= OBSTACLE_MAX_RANGE + 2 * cfg.ROBOT_RADIUS, "white grid must cover the obstacle grid"

Goal = collections.namedtuple("Goal", "near centre")
Obstacle = collections.namedtuple("Obstacle", "near size centre", defaults=(None,))


class Detections:
    def __init__(self, t, compass):
        self.t = t
        self.compass = compass
        self.ball = None
        self.goals = {"yellow": None, "blue": None}
        self.line_pts = np.zeros((0, 2), np.float32)
        self.obstacles = []
        self.field_visible = False
        self.hull_px = None


def _area_px(view, cm2):
    return cm2 * view.px_per_cm ** 2


def _contour_cm(view, contour):
    points = contour.reshape(-1, 2).astype(np.float32)
    return view.to_cm(points[:, 0], points[:, 1])


def _wrap_angles(angles):
    return (angles + math.pi) % (2 * math.pi) - math.pi


def _nearest_point(view, contours):
    points = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float32)
    x, y = view.to_cm(points[:, 0], points[:, 1])
    i = int(np.argmin(x * x + y * y))
    return [float(x[i]), float(y[i])]


def _robot_centre(view, contour, near):
    x, y = _contour_cm(view, contour)
    near_angle = math.atan2(near[1], near[0])
    relative = _wrap_angles(np.arctan2(y, x) - near_angle)
    middle = near_angle + (float(relative.min()) + float(relative.max())) / 2
    distance = math.hypot(near[0], near[1]) + cfg.ROBOT_RADIUS
    return [distance * math.cos(middle), distance * math.sin(middle)]


def _size_cm(view, contour):
    x, y = _contour_cm(view, contour)
    area = 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    return math.sqrt(area)


def _thickness_cm(view, contour):
    x, y = _contour_cm(view, contour)
    k = THICKNESS_GRID_PPC
    gx, gy = (x - x.min()) * k + 2, (y - y.min()) * k + 2
    canvas = np.zeros((int(gy.max()) + 4, int(gx.max()) + 4), np.uint8)
    cv2.fillPoly(canvas, [np.stack([gx, gy], axis=1).round().astype(np.int32)], 255)
    depth = cv2.distanceTransform(canvas, cv2.DIST_L2, 3).max()
    return 2.0 * float(depth) / k


def _far_mask(rv, shape):
    cached = getattr(rv, "_far_mask", None)
    if cached is None or cached.shape != shape:
        cached = np.full(shape, 255, np.uint8)
        cv2.circle(cached, rv.centre_px(), int(round(rv.radius_px(OBSTACLE_FAR))), 0, -1)
        rv._far_mask = cached
    return cached


def _range_mask(rv, shape):
    cached = getattr(rv, "_range_mask", None)
    if cached is None or cached.shape != shape:
        cached = np.zeros(shape, np.uint8)
        cv2.circle(cached, rv.centre_px(), int(math.ceil(rv.radius_px(LINE_MAX_RANGE))) + 3, 255, -1)
        rv._range_mask = cached
    return cached


def _split_points(nearest_by_bin, nearest):
    seen_bins = np.nonzero(~np.isnan(nearest_by_bin))[0]
    if len(seen_bins) < 3:
        return []
    p = nearest_by_bin[seen_bins]
    min_bins = OBSTACLE_SPLIT_MIN_W / max(nearest * OBSTACLE_SPLIT_BIN, 1e-6)
    cuts = [int(seen_bins[i]) for i in range(1, len(p)) if abs(p[i] - p[i - 1]) > OBSTACLE_SPLIT_JUMP]
    dips = [i for i in range(len(p)) if (i == 0 or p[i] <= p[i - 1]) and (i == len(p) - 1 or p[i] < p[i + 1])]
    for a, b in zip(dips, dips[1:]):
        top = a + int(np.argmax(p[a:b + 1]))
        already_cut = any(seen_bins[a] < c <= seen_bins[b] for c in cuts)
        if p[top] - max(p[a], p[b]) >= OBSTACLE_SPLIT_RISE and not already_cut:
            cuts.append(int(seen_bins[top]))
    cuts = sorted(set(cuts))
    kept, last = [], int(seen_bins[0])
    for i, cut in enumerate(cuts):
        following = cuts[i + 1] if i + 1 < len(cuts) else int(seen_bins[-1]) + 1
        if cut - last >= min_bins and following - cut >= min_bins:
            kept.append(cut)
            last = cut
    return kept


def _split_obstacle(grid, contour, cover):
    x, y = _contour_cm(grid, contour)
    d = np.hypot(x, y)
    j = int(np.argmin(d))
    near = [float(x[j]), float(y[j])]
    a0 = math.atan2(y[j], x[j])
    relative = _wrap_angles(np.arctan2(y, x) - a0)
    lowest = float(relative.min())
    bins = ((relative - lowest) / OBSTACLE_SPLIT_BIN).astype(np.int32)
    n_bins = int(bins.max()) + 1
    if n_bins < 3:
        return [(near, _robot_centre(grid, contour, near), _size_cm(grid, contour))]
    nearest_by_bin = np.full(n_bins, np.inf)
    np.minimum.at(nearest_by_bin, bins, d)
    if cover is not None:
        cover_x, cover_y = cover
        cover_bins = np.floor((_wrap_angles(np.arctan2(cover_y, cover_x) - a0) - lowest)
                              / OBSTACLE_SPLIT_BIN).astype(np.int32)
        inside = (cover_bins >= 0) & (cover_bins < n_bins)
        cover_nearest = np.full(n_bins, np.inf)
        np.minimum.at(cover_nearest, cover_bins[inside], np.hypot(cover_x, cover_y)[inside])
        nearest_by_bin[cover_nearest < nearest_by_bin] = np.nan
    nearest_by_bin[np.isinf(nearest_by_bin)] = np.nan
    cuts = _split_points(nearest_by_bin, float(d[j]))
    if not cuts:
        return [(near, _robot_centre(grid, contour, near), _size_cm(grid, contour))]
    pieces = []
    edges = [0] + cuts + [n_bins]
    for start, end in zip(edges, edges[1:]):
        in_piece = (bins >= start) & (bins < end)
        if not in_piece.any():
            continue
        k = int(np.argmin(np.where(in_piece, d, np.inf)))
        lo, hi = float(relative[in_piece].min()), float(relative[in_piece].max())
        middle = a0 + (lo + hi) / 2
        distance = float(d[k]) + cfg.ROBOT_RADIUS
        pieces.append(([float(x[k]), float(y[k])],
                       [distance * math.cos(middle), distance * math.sin(middle)], (hi - lo) * distance))
    return pieces


class TopDownGrid:
    def __init__(self, rv, px_per_cm, reach_cm):
        self.px_per_cm = px_per_cm
        self.half = int(math.ceil(reach_cm * px_per_cm))
        n = 2 * self.half
        u, v = np.meshgrid(np.arange(n, dtype=np.float64) + 0.5, np.arange(n, dtype=np.float64) + 0.5)
        sx, sy = rv.to_px((u - self.half) / px_per_cm, (self.half - v) / px_per_cm)
        self.maps = cv2.convertMaps(sx.astype(np.float32), sy.astype(np.float32), cv2.CV_16SC2)

    def to_cm(self, px, py):
        return (px + 0.5 - self.half) / self.px_per_cm, (self.half - py - 0.5) / self.px_per_cm

    def to_px(self, x_cm, y_cm):
        return x_cm * self.px_per_cm + self.half - 0.5, self.half - y_cm * self.px_per_cm - 0.5

    def remap(self, mask):
        return cv2.remap(mask, *self.maps, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _cached_grid(rv, name, px_per_cm, reach_cm):
    grid = getattr(rv, name, None)
    if grid is None:
        grid = TopDownGrid(rv, px_per_cm, reach_cm)
        setattr(rv, name, grid)
    return grid


def _obstacle_view(rv):
    if rv.mode != "radial":
        return rv
    return _cached_grid(rv, "_obstacle_grid", OBSTACLE_GRID_PPC, OBSTACLE_MAX_RANGE + 2 * cfg.ROBOT_RADIUS)


def _white_view(rv):
    if rv.mode != "radial":
        return rv
    return _cached_grid(rv, "_white_grid", WHITE_GRID_PPC, LINE_MAX_RANGE + WHITE_BODY_KERNEL)


def _to_view(view, rv, mask):
    return mask if view is rv else view.remap(mask)


def _square(size_px):
    return np.ones((size_px, size_px), np.uint8)


def _odd_kernel_px(cm, px_per_cm):
    return max(1, int(round(2 * cm * px_per_cm))) | 1


def _hidden_behind(far, near):
    far_distance, near_distance = math.hypot(*far.centre), math.hypot(*near.centre)
    if far_distance <= near_distance:
        return False
    near_half_angle = math.asin(min(1.0, cfg.ROBOT_RADIUS / near_distance))
    far_half_angle = math.asin(min(1.0, far.size / 2 / far_distance))
    gap = abs(_wrap_angles(math.atan2(far.centre[1], far.centre[0]) - math.atan2(near.centre[1], near.centre[0])))
    return gap + far_half_angle < near_half_angle


def _white_robots(rv, white):
    if WHITE_BODY_KERNEL <= 0 or cv2.countNonZero(white) == 0:
        return None
    view = _white_view(rv)
    kernel_px = max(1, int(round(WHITE_BODY_KERNEL * view.px_per_cm)))
    bodies = cv2.morphologyEx(_to_view(view, rv, white), cv2.MORPH_OPEN, _square(kernel_px))
    if cv2.countNonZero(bodies) == 0:
        return None
    grown = cv2.dilate(bodies, _square(_odd_kernel_px(WHITE_BODY_MARGIN, view.px_per_cm)))
    return view, bodies, grown


def _on_mask(view, mask, x, y):
    px, py = view.to_px(x, y)
    u, v = np.round(px).astype(np.int32), np.round(py).astype(np.int32)
    height, width = mask.shape[:2]
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    hit = np.zeros(len(x), bool)
    hit[inside] = mask[v[inside], u[inside]] > 0
    return hit


def _big_contours(mask, min_area, approx=cv2.CHAIN_APPROX_NONE):
    x, y, w, h = cv2.boundingRect(mask)
    if w == 0:
        return []
    contours, _ = cv2.findContours(mask[y:y + h, x:x + w], cv2.RETR_EXTERNAL, approx, offset=(x, y))
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def _ball(rv, orange):
    x, y, w, h = cv2.boundingRect(orange)
    if w == 0:
        return None
    n, labels, stats, _ = cv2.connectedComponentsWithStats(orange[y:y + h, x:x + w], connectivity=8)
    if n < 2:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[biggest, cv2.CC_STAT_AREA] < max(BALL_MIN_PX, _area_px(rv, BALL_MIN_AREA)):
        return None
    points = cv2.findNonZero((labels == biggest).astype(np.uint8)).reshape(-1, 2).astype(np.float32) + (x, y)
    bx, by = rv.to_cm(points[:, 0], points[:, 1])
    nearest = int(np.argmin(bx * bx + by * by))
    return [float(bx[nearest]), float(by[nearest])]


def field_mask(rv, green):
    contours = _big_contours(green, _area_px(rv, GREEN_MIN_AREA), cv2.CHAIN_APPROX_SIMPLE)
    if not contours or sum(cv2.contourArea(c) for c in contours) < _area_px(rv, FIELD_MIN_AREA):
        return None, None
    hull = cv2.convexHull(np.vstack(contours))
    mask = np.zeros_like(green)
    cv2.fillConvexPoly(mask, hull, 255)
    return cv2.bitwise_and(mask, rv.valid), hull


def _grown_field(rv, field_area, hull):
    grown = np.zeros_like(field_area)
    cv2.fillConvexPoly(grown, hull, 255)
    cv2.polylines(grown, [hull], True, 255, max(1, int(round(2 * HULL_MARGIN * rv.px_per_cm))))
    return cv2.bitwise_and(grown, rv.valid)


def _goals(rv, masks):
    found = {}
    for name in ("yellow", "blue"):
        contours = _big_contours(masks[name], _area_px(rv, GOAL_MIN_AREA))
        if contours:
            points = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float32)
            gx, gy = rv.to_cm(points[:, 0], points[:, 1])
            centre = [float(gx.min() + gx.max()) / 2.0, float(gy.min() + gy.max()) / 2.0]
            found[name] = Goal(_nearest_point(rv, contours), centre)
    return found


def _line_points(rv, white, white_robots):
    pixels = cv2.findNonZero(white)
    if pixels is None:
        return None
    pixels = pixels.reshape(-1, 2).astype(np.float32)
    lx, ly = rv.to_cm(pixels[:, 0], pixels[:, 1])
    keep = lx * lx + ly * ly < LINE_MAX_RANGE ** 2
    if white_robots is not None:
        view, _, grown = white_robots
        keep &= ~_on_mask(view, grown, lx, ly)
    lx, ly = lx[keep], ly[keep]
    if len(lx) > LINE_MAX_POINTS:
        every = np.linspace(0, len(lx) - 1, LINE_MAX_POINTS).astype(np.int32)
        lx, ly = lx[every], ly[every]
    return np.stack([lx, ly], axis=1).astype(np.float32)


def _uncoloured(rv, labels, field_area):
    uncoloured = cv2.bitwise_and(field_area, cv2.compare(labels, 0, cv2.CMP_EQ))
    x1, y1, x2, y2 = rv.ignore_box
    pad = int(round(OBSTACLE_BODY_MARGIN * rv.px_per_cm))
    uncoloured[max(y1 - pad, 0):y2 + pad, max(x1 - pad, 0):x2 + pad] = 0
    cv2.circle(uncoloured, rv.centre_px(), int(round(rv.radius_px(cfg.ROBOT_RADIUS + OBSTACLE_BODY_MARGIN))), 0, -1)
    return uncoloured


def _near_white(rv, white):
    near_lines = cv2.dilate(white, _square(_odd_kernel_px(OBSTACLE_LINE_MARGIN, rv.px_per_cm)))
    far_lines = cv2.dilate(white, _square(_odd_kernel_px(OBSTACLE_LINE_MARGIN_FAR, rv.px_per_cm)))
    near_lines |= cv2.bitwise_and(far_lines, _far_mask(rv, far_lines.shape))
    return near_lines


def _white_robots_in_view(white_robots, view, shape):
    white_view, bodies, grown = white_robots
    if white_view is view:
        return bodies, grown
    scale = view.px_per_cm / white_view.px_per_cm
    offset = (int(round(2 * white_view.half * scale)) - shape[0]) // 2

    def resized(mask):
        big = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        return big[offset:offset + shape[0], offset:offset + shape[1]]
    return resized(bodies), resized(grown)


def _obstacle_blobs(view, unknown, cover):
    found = []
    for contour in _big_contours(unknown, _area_px(view, OBSTACLE_MIN_AREA)):
        if _thickness_cm(view, contour) < OBSTACLE_KERNEL:
            continue
        for near, centre, size in _split_obstacle(view, contour, cover):
            if near[0] ** 2 + near[1] ** 2 < OBSTACLE_MAX_RANGE ** 2:
                found.append(Obstacle(near, size, centre))
    return found


def _merge_obstacles(found):
    merged = []
    for o in sorted(found, key=lambda o: o.near[0] ** 2 + o.near[1] ** 2):
        twin = next((i for i, q in enumerate(merged)
                     if math.hypot(q.centre[0] - o.centre[0], q.centre[1] - o.centre[1]) < cfg.ROBOT_RADIUS), None)
        if twin is not None:
            q = merged[twin]
            merged[twin] = q._replace(size=math.hypot(q.size, o.size))
        elif any(_hidden_behind(o, q) for q in merged):
            continue
        elif o.size >= OBSTACLE_MIN_SIZE:
            merged.append(o)
    return merged


def _obstacles(rv, res, field_area, orange, white_robots):
    uncoloured = _uncoloured(rv, res.labels, field_area)
    view = _obstacle_view(rv)
    unknown = _to_view(view, rv, cv2.bitwise_and(uncoloured, cv2.bitwise_not(_near_white(rv, res.masks["white"]))))
    if white_robots is not None:
        bodies, grown = _white_robots_in_view(white_robots, view, unknown.shape)
        unknown = cv2.bitwise_or(unknown, bodies)
        unknown = cv2.bitwise_or(unknown, cv2.bitwise_and(grown, _to_view(view, rv, uncoloured)))
    cover = cv2.findNonZero(orange)
    if cover is not None:
        cover = rv.to_cm(*cover.reshape(-1, 2).astype(np.float32).T)
    kernel_px = max(1, int(round(OBSTACLE_KERNEL * view.px_per_cm)))
    unknown = cv2.morphologyEx(unknown, cv2.MORPH_OPEN, _square(kernel_px))
    return _merge_obstacles(_obstacle_blobs(view, unknown, cover))


def detect(rv, res, t, compass):
    det = Detections(t, compass)
    masks = res.masks
    field_area, det.hull_px = field_mask(rv, masks["green"])
    det.field_visible = field_area is not None
    near_field = _grown_field(rv, field_area, det.hull_px) if det.field_visible else rv.valid

    orange = cv2.bitwise_and(masks["orange"], near_field)
    det.ball = _ball(rv, orange)
    det.goals.update(_goals(rv, masks))

    white = cv2.bitwise_and(cv2.bitwise_and(masks["white"], near_field), _range_mask(rv, near_field.shape))
    white_robots = _white_robots(rv, white)
    line_pts = _line_points(rv, white, white_robots)
    if line_pts is not None:
        det.line_pts = line_pts

    if det.field_visible and cfg.ENEMY_DETECTION:
        det.obstacles = _obstacles(rv, res, field_area, orange, white_robots)
    return det


# ==================================== camera thread ====================================
def analyse_frame(rv, raw, t, compass):
    res = rv.process(raw)
    return detect(rv, res, t, compass), res


class VisionThread(threading.Thread):
    MAX_ERRORS = 10
    STARTUP_GRACE = 2.0

    def __init__(self, compass, config_path=cfg.VISION_CONFIG, keep_frames=False, first_frame_id=0):
        super().__init__()
        import picamera2      # only on the Pi, so the rest can run in the simulator
        self.daemon = True
        self.running = True
        self.compass = compass
        self.keep_frames = keep_frames

        self.rv = RobotVision(config_path)
        if self.rv.mode not in ("topdown", "radial"):
            print(f"[vision] WARNING: no distance calibration in {config_path} (mode={self.rv.mode}). "
                  f"Distances will be wrong - do the Distance calibration in test_camera.py.")
        if cfg.VISION_CV_THREADS is not None:
            cv2.setNumThreads(cfg.VISION_CV_THREADS)
        self.cap = picamera2.Picamera2()
        controls = {}
        if cfg.VISION_MAX_FPS:
            frame_us = int(1e6 / cfg.VISION_MAX_FPS)
            controls["FrameDurationLimits"] = (frame_us, frame_us)
        self.cap.configure(self.cap.create_preview_configuration(
            main={"size": self.rv.size, "format": "RGB888"}, controls=controls))
        self.rv.apply_camera(self.cap)
        self.cap.start()

        self.cond = threading.Condition()
        self.frame_id = first_frame_id
        self.last_frame_t = time.monotonic() + self.STARTUP_GRACE
        self.detections = None
        self.last_res = None
        self.fps = 0.0
        self.ready = False

    def run(self):
        frames, fps_start = 0, time.monotonic()
        errors = 0
        while self.running:
            try:
                raw = self.cap.capture_array("main")
                t = time.monotonic()
                det, res = analyse_frame(self.rv, raw, t, self.compass())
            except Exception as e:
                if not self.running:
                    break
                errors += 1
                if errors == 1:
                    print(f"[vision] frame failed: {e!r}")
                if errors >= self.MAX_ERRORS:
                    print(f"[vision] {errors} frames failed in a row - stopping the camera thread")
                    break
                time.sleep(0.05)
                continue
            errors = 0
            with self.cond:
                self.detections = det
                self.last_res = res if self.keep_frames else None
                self.frame_id += 1
                self.cond.notify_all()
            self.last_frame_t = t
            self.ready = True
            frames += 1
            if t - fps_start >= 1.0:
                self.fps, frames, fps_start = frames / (t - fps_start), 0, t

    def wait_new(self, last_id, timeout=None):
        with self.cond:
            if not self.cond.wait_for(lambda: self.frame_id != last_id, timeout):
                return last_id, None
            return self.frame_id, self.detections

    def latest(self):
        with self.cond:
            return self.detections

    def stop(self):
        self.running = False
        try:
            self.cap.stop()
        except Exception:
            pass

    def close(self):
        try:
            self.cap.close()
        except Exception:
            pass

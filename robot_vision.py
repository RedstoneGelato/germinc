"""
robot_vision.py - use the settings tuned in test_camera.py inside the robot script.

1. Tune everything in test_camera.py's web page, then press "Generate config file".
   That writes robot_vision_config.json next to test_camera.py.
2. Copy robot_vision.py and robot_vision_config.json next to your main script and:

    from robot_vision import RobotVision

    rv = RobotVision("robot_vision_config.json")
    cam = picamera2.Picamera2()
    cam.configure(cam.create_preview_configuration(main={"size": rv.size, "format": "RGB888"}))
    rv.apply_camera(cam)                     # exposure + white balance from the file
    cam.start()
    rv.set_forward(yaw_deg)                  # optional, see below

    while True:
        raw = cam.capture_array("main")
        res = rv.process(raw, yaw_deg)       # yaw_deg from your BNO08x (degrees, counter-clockwise positive)
        goalpos, own_goalpos = rv.goal_positions(res)
        # res.masks["blue" | "yellow" | "white" | "green" | "orange"] are uint8 masks,
        # res.frame is the (field-aligned) colour frame, res.centre the robot position in it.

Radial mode (test_camera.py "Distance calibration"): no undistortion at all. Every pixel's direction is its angle from
the robot centre, and its distance comes from a measured pixels -> cm curve (rv.polar() gives both).

process() gives the same masks as test_camera.py: top-down (or undistort, or raw/radial) -> rotation -> colour
masks with the ignore box -> rotate by yaw about the robot centre. It just does it in a cheaper order: the colours
are classified on the camera frame first (one HSV conversion + one lookup table per channel, all five colours at
once into one "label" image with a bit per colour), and only that single-channel label image goes through the
top-down remap (nearest neighbour). The rotation is folded into the remap, and the opening runs on the label image
(all colours together). The colour frame (res.frame) is only made if something asks for it (debug views).
Positions are in pixels of res.frame, relative to the robot centre, +y = forward (up in the frame).
The robot code (vision.py) calls process(raw, align=False) and rotates the detected points instead,
which is much cheaper than rotating six images every frame. rv.to_cm() turns pixels into cm.

The geometry step is a Warp (see below): the camera frame as it is (raw / radial), or a fixed per-pixel remap
(top-down, undistort). The omni mirror (Camera Module 3 Wide looking up into it) gives a picture that is already
radial - direction = angle round the mirror centre, distance = a curve of the radius - so radial mode should fit
it as it is. If it ever needs a remap instead, write a function that returns the maps, register it with
@warp_model("omni") and put "model": "omni" in the config's "topdown" section.

Yaw offset: with the IMU's game rotation vector, 0 deg is wherever the robot was when the IMU started,
so the offset saved from the web page is only right if you power up facing the same way. Either place
the robot facing forward at start-up, or call rv.set_forward(yaw_deg) once while it faces forward.

Only needs opencv and numpy.
"""
import json
import math

import cv2
import numpy as np

COLOURS = ("blue", "yellow", "white", "green", "orange")
BITS = {n: 1 << i for i, n in enumerate(COLOURS)}      # bit of each colour in the label image
_ROTATIONS = {"90_CLOCKWISE": cv2.ROTATE_90_CLOCKWISE, "180": cv2.ROTATE_180,
              "90_COUNTERCLOCKWISE": cv2.ROTATE_90_COUNTERCLOCKWISE, "NONE": None}


class Result:
    """process()'s output: masks {colour: uint8 0/255}, centre (robot centre in pixels), labels (uint8, one bit per
    colour, see BITS) and frame (the colour view - only made the first time something reads it)."""
    __slots__ = ("masks", "centre", "labels", "_frame", "_make_frame")

    def __init__(self, masks, centre, labels=None, frame=None, make_frame=None):
        self.masks, self.centre, self.labels = masks, centre, labels
        self._frame, self._make_frame = frame, make_frame

    @property
    def frame(self):
        if self._frame is None and self._make_frame is not None:
            self._frame, self._make_frame = self._make_frame(), None
        return self._frame


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def merge_blobs(mask, min_size):
    """Bounding box [x, y, w, h] around all blobs of at least min_size px, or [0,0,0,0]."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_min = y_min = float('inf')
    x_max = y_max = 0
    for c in contours:
        if cv2.contourArea(c) < min_size:
            continue
        x, y, w, h = cv2.boundingRect(c)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    if x_min < x_max and y_min < y_max:
        return [x_min, y_min, x_max - x_min, y_max - y_min]
    return [0, 0, 0, 0]


def _rotate_point(pt, size, code):
    x, y = pt
    w, h = size
    if code == cv2.ROTATE_90_CLOCKWISE:
        return (h - 1 - y, x)
    if code == cv2.ROTATE_180:
        return (w - 1 - x, h - 1 - y)
    if code == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return (y, w - 1 - x)
    return (x, y)


# ==================================== geometry (Warp) ====================================
class Warp:
    """Camera frame -> the robot-frame view everything else works in (top-down, undistorted, or just rotated).
    out_size  (w, h) of the view
    crop      (x, y, w, h) of the camera frame the view uses: only that part gets classified
    valid     uint8, 255 where the view has camera image
    colour(raw) -> BGR view (bilinear), labels(lab) -> label view (nearest), lab = classified crop."""
    out_size = crop = valid = None

    def colour(self, raw):
        raise NotImplementedError

    def labels(self, lab):
        raise NotImplementedError


class IdentityWarp(Warp):
    """No calibration: the view is the camera frame (rotated if the config says so)."""

    def __init__(self, capture_size, rotation=None):
        self.rotation = rotation
        w, h = capture_size
        self.crop = (0, 0, w, h)
        self.valid = self._rot(np.full((h, w), 255, np.uint8))
        self.out_size = (self.valid.shape[1], self.valid.shape[0])

    def _rot(self, img):
        return img if self.rotation is None else cv2.rotate(img, self.rotation)

    def colour(self, raw):
        return self._rot(raw)

    def labels(self, lab):
        return self._rot(lab)


class RemapWarp(Warp):
    """A fixed per-pixel lookup: view pixel (u, v) shows camera pixel (map_x[v, u], map_y[v, u]); anything outside
    the camera frame (e.g. -1) is black. map_x / map_y are float32 in full camera-frame pixels. The rotation is
    applied to the maps once, so it costs nothing per frame."""

    def __init__(self, map_x, map_y, capture_size, rotation=None, outside_label=0):
        map_x, map_y = np.asarray(map_x, np.float32), np.asarray(map_y, np.float32)
        if rotation is not None:
            map_x, map_y = cv2.rotate(map_x, rotation), cv2.rotate(map_y, rotation)
        w, h = capture_size
        inside = (map_x > -1) & (map_x < w) & (map_y > -1) & (map_y < h)     # touches the frame (bilinear)
        if inside.any():
            xs, ys = map_x[inside], map_y[inside]
            x0, y0 = max(int(np.floor(xs.min())) - 1, 0), max(int(np.floor(ys.min())) - 1, 0)
            x1, y1 = min(int(np.ceil(xs.max())) + 2, w), min(int(np.ceil(ys.max())) + 2, h)
        else:
            x0, y0, x1, y1 = 0, 0, w, h
        self.crop = (x0, y0, x1 - x0, y1 - y0)
        self._full = self.crop == (0, 0, w, h)
        # the crop holds every camera pixel the maps reach, so shifting them changes nothing; off-frame stays off
        mx, my = map_x - x0, map_y - y0
        mx[~inside], my[~inside] = -10, -10
        self._lin = cv2.convertMaps(mx, my, cv2.CV_16SC2)
        self._nn = cv2.convertMaps(mx, my, cv2.CV_16SC2, nninterpolation=True)[0]
        self._outside = int(outside_label)
        self.out_size = (map_x.shape[1], map_x.shape[0])
        self.valid = cv2.remap(np.full((y1 - y0, x1 - x0), 255, np.uint8), self._nn, None, cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def colour(self, raw):
        if not self._full:
            x, y, w, h = self.crop
            raw = raw[y:y + h, x:x + w]
        return cv2.remap(raw, *self._lin, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def labels(self, lab):
        return cv2.remap(lab, self._nn, None, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=self._outside)


# Top-down models: name -> f(config, capture_size) returning (map_x, map_y, px_per_cm), float32 maps from the
# top-down view (robot centre wherever the model puts it, +y forward = up) to camera pixels, before the config's
# rotation. Pick one with "model" in the config's "topdown" section (default: test_camera.py's fisheye one).
WARP_MODELS = {}


def warp_model(name):
    def register(fn):
        WARP_MODELS[name] = fn
        return fn
    return register


def _topdown_maps(Hn, out_size, K, D):
    """Top-down pixel -> floor homography -> camera ray -> fisheye model -> raw pixel (one remap, nothing cropped).
    Returns float32 (map_x, map_y), -1 where there is no image."""
    w, h = out_size
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    pts = np.stack([u.ravel(), v.ravel(), np.ones(w * h)], axis=0)
    q = Hn @ pts
    z = q[2]
    ok = z > 1e-9
    z_safe = np.where(ok, z, 1.0)
    norm = np.stack([q[0] / z_safe, q[1] / z_safe], axis=1).reshape(-1, 1, 2)
    px = cv2.fisheye.distortPoints(norm, K, D).reshape(-1, 2)
    mx = px[:, 0].reshape(h, w).astype(np.float32)
    my = px[:, 1].reshape(h, w).astype(np.float32)
    bad = (~ok).reshape(h, w) | ~np.isfinite(mx) | ~np.isfinite(my)
    mx[bad] = -1
    my[bad] = -1
    return mx, my


@warp_model("fisheye_homography")
def _fisheye_topdown(config, capture_size):
    """test_camera.py's top-down setup: fisheye lens calibration (K, D) + a floor homography from the checkerboard."""
    lens, td = config["lens"], config["topdown"]
    K, D = np.array(lens["K"], np.float64), np.array(lens["D"], np.float64)
    mx, my = _topdown_maps(np.array(td["Hn"], np.float64), tuple(td["out_size"]), K, D)
    return mx, my, td["px_per_cm"]


def _undistort_maps(K, D, size, balance, fov_scale):
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, size, np.eye(3), balance=balance, fov_scale=fov_scale)
    return cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, size, cv2.CV_32FC1)


def _make_align(shape, centre, angle, max_side=1000):
    """Rotate about the robot centre by `angle` deg (counter-clockwise +) onto a canvas with the robot in the middle."""
    h, w = shape[:2]
    cx, cy = centre
    R = max(math.hypot(cx - x, cy - y) for x in (0, w) for y in (0, h))
    side = min(int(math.ceil(2 * R)) + 2, max_side)
    side += side % 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    M[0, 2] += side / 2.0 - cx
    M[1, 2] += side / 2.0 - cy
    return M, (side, side), (side / 2.0, side / 2.0)


def _shifted(a, axis, o):
    """(dst, src) views so that dst[i] pairs with a[i + o] along axis, where i + o is inside the image."""
    n = a.shape[axis]
    lo, hi = (slice(0, n - o), slice(o, n)) if o > 0 else (slice(-o, n), slice(0, n + o))
    if axis == 0:
        return (lo, slice(None)), (hi, slice(None))
    return (slice(None), lo), (slice(None), hi)


def _morph_bits(a, k, op):
    """Erode (np.bitwise_and) or dilate (np.bitwise_or) every bit of a uint8 image with a k x k square, like
    cv2.erode / cv2.dilate on each colour's mask separately (pixels outside the image don't count)."""
    anchor = k // 2
    for axis in (1, 0):
        out = a.copy()
        for o in range(-anchor, k - anchor):
            if o:
                d, s = _shifted(a, axis, o)
                op(out[d], a[s], out=out[d])
        a = out
    return a


def open_bits(labels, k):
    """MORPH_OPEN with a k x k square on all colours of a label image at once."""
    if k <= 1:
        return labels
    return _morph_bits(_morph_bits(labels, k, np.bitwise_and), k, np.bitwise_or)


FALLBACK_PX_PER_CM = 1.0   # same as test_camera.py: used when there is no top-down setup
RADIAL_REF_CM = 80.0       # radial mode: the single px/cm used for blob areas / kernel sizes is the one at this
                           # distance (the far end of what matters: further away things look smaller in a fisheye,
                           # so this keeps the minimum sizes small enough for far balls / robots)
RADIAL_KEEP_BITS = ("orange",)   # radial mode: colours NOT cleaned by the noise opening far away (see below)
RADIAL_KEEP_BALL_PX = 2.0  # ...far away = where a ball is less than this many kernels across along the radius.
                           # The opening removes anything thinner than the kernel, and a fisheye squashes far things
                           # along the radius: with a 3 px kernel at 640x480 every ball past ~60 cm was wiped out


def radial_table(radial):
    """Radial calibration (test_camera.py "Distance calibration") -> (pixel radii, cm) lookup table, increasing.
    radial = {"table_px", "table_cm": the fitted pixels -> cm curve, sampled} (or "coeffs": a polynomial, older files)."""
    if radial.get("table_px"):
        r = np.array(radial["table_px"], np.float64)
        d = np.array(radial["table_cm"], np.float64)
    else:
        r = np.linspace(0.0, float(radial["r_max_px"]), 512)
        d = np.polyval(np.array(radial["coeffs"], np.float64), r)
    d = np.maximum.accumulate(np.maximum(d, 0.0))         # distance never goes down as the radius grows
    return r, d



class RobotVision:
    def __init__(self, path="robot_vision_config.json"):
        with open(path) as f:
            c = json.load(f)
        if c.get("version") != 1:
            raise ValueError(f"unsupported config version {c.get('version')!r}")
        self.config = c
        self.size = tuple(c["capture_size"])
        self.rotation = _ROTATIONS[c["rotation"]]

        self.lower = {n: np.array(c["hsv"][n]["lo"], np.uint8) for n in COLOURS}
        self.upper = {n: np.array(c["hsv"][n]["hi"], np.uint8) for n in COLOURS}
        # one lookup table per HSV channel: bit set = the value is inside that colour's range for that channel,
        # so H, S and V tables ANDed together = the same as cv2.inRange for every colour at once
        self._luts = np.zeros((3, 256), np.uint8)
        for n in COLOURS:
            for ch in range(3):
                self._luts[ch, int(self.lower[n][ch]):int(self.upper[n][ch]) + 1] |= BITS[n]

        det = c["detection"]
        self.min_blob_area = det["min_blob_area"]
        self.morph_kernel = int(det["morph_kernel"])
        self._kernel = np.ones((self.morph_kernel,) * 2, np.uint8)

        hd = c["heading"]
        self.align_to_field = bool(hd["align_to_field"])
        self.invert_yaw = bool(hd["invert"])
        self.yaw_offset = float(hd["offset_deg"])
        self.robot_centre = c["robot_centre"]               # in the image that goes into the rotation, or None

        # geometry: top-down if set up, else plain undistort, else raw
        lens, td, radial = c.get("lens"), c.get("topdown"), c.get("radial")
        self.mode = "raw"
        self.px_per_cm = FALLBACK_PX_PER_CM
        self._radial = None
        base_size = self.size                               # (w, h) of the image before the rotation
        outside = int(self.classify(np.zeros((1, 1, 3), np.uint8))[0, 0])   # what black (no image) counts as
        maps = None
        if radial:
            # raw fisheye frame, no undistortion: a pixel's DIRECTION is its angle from the robot centre, its
            # DISTANCE comes from the calibrated pixels -> cm curve
            self._radial = radial_table(radial)
            self.mode = "radial"
            # far away a fisheye squashes things towards the centre (radially) more than sideways: use the radial
            # scale there (pixels per cm along the radius at RADIAL_REF_CM) - the smallest, so filters sized with it
            # never wipe out a far robot / ball (thin things close up are caught by size checks in cm instead)
            ref = min(RADIAL_REF_CM, float(self._radial[1][-1]))
            r_ref = float(np.interp(ref, self._radial[1], self._radial[0]))
            radial_scale = (r_ref - float(np.interp(ref - 5.0, self._radial[1], self._radial[0]))) / 5.0
            self.px_per_cm = max(0.2, min(radial_scale, r_ref / max(ref, 1e-6)))
        elif td and (lens or td.get("model")):
            model = td.get("model", "fisheye_homography")
            if model not in WARP_MODELS:
                raise ValueError(f"unknown top-down model {model!r} (have: {', '.join(WARP_MODELS)})")
            mx, my, self.px_per_cm = WARP_MODELS[model](c, self.size)
            maps = (mx, my)
            self.mode = "topdown"
        elif lens:
            K, D = np.array(lens["K"], np.float64), np.array(lens["D"], np.float64)
            u = c["undistort_view"]
            maps = _undistort_maps(K, D, self.size, u["balance"], u["fov_scale"])
            self.mode = "undistort"
        if maps is None:
            self.warp = IdentityWarp(self.size, self.rotation)
        else:
            base_size = (maps[0].shape[1], maps[0].shape[0])
            self.warp = RemapWarp(maps[0], maps[1], self.size, self.rotation, outside)

        # robot centre in the robot-aligned frame (fixed, so worked out once)
        bw, bh = base_size
        if self.robot_centre is None:
            quarter = self.rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)
            fw, fh = (bh, bw) if quarter else (bw, bh)
            self.centre = (fw / 2.0, fh / 2.0)
        else:
            bx = min(max(self.robot_centre[0], 0), bw - 1)
            by = min(max(self.robot_centre[1], 0), bh - 1)
            self.centre = _rotate_point((bx, by), (bw, bh), self.rotation)

        # ignore box: cm around the robot centre (new configs), or the old fixed pixel box
        ign = det.get("ignore_cm")
        if ign and self._radial is not None:
            xs, ys = self.to_px(np.array([-ign["left"], ign["right"]], np.float64),
                                np.array([ign["front"], -ign["back"]], np.float64))
            box = (min(xs), min(ys), max(xs), max(ys))
        elif ign:
            cx, cy = self.centre
            ppc = self.px_per_cm
            box = (cx - ign["left"] * ppc, cy - ign["front"] * ppc, cx + ign["right"] * ppc, cy + ign["back"] * ppc)
        else:
            box = det["ignore_box"]                          # x1, y1, x2, y2 in the robot-aligned frame
        self.ignore_box = tuple(max(int(round(v)), 0) for v in box)

        # pixels the camera actually sees (the top-down remap leaves black where there is no image)
        valid = self.warp.valid
        if self._radial is not None:                        # only as far out as the distance calibration goes
            ring = np.zeros_like(valid)
            cv2.circle(ring, tuple(int(round(v)) for v in self.centre), int(self._radial[0][-1]), 255, -1)
            valid = cv2.bitwise_and(valid, ring)
        self.valid = cv2.erode(valid, np.ones((5, 5), np.uint8))

        # radial mode: past the distance where a ball gets thinner (along the radius) than RADIAL_KEEP_BALL_PX
        # kernels, the RADIAL_KEEP_BITS colours skip the opening, or far balls would vanish (detection.py still
        # needs a blob of at least BALL_MIN_AREA, so single noise pixels don't count)
        self._far_keep = None
        if self._radial is not None and self.morph_kernel > 1:
            r_px, d_cm = self._radial
            ball_px = np.gradient(r_px, d_cm, edge_order=1) * 4.2     # px a ~4 cm ball spans along the radius
            thin = np.nonzero(ball_px < RADIAL_KEEP_BALL_PX * self.morph_kernel)[0]
            thin = thin[d_cm[thin] > 10.0]                             # (not the flat bit at the centre)
            if len(thin):
                self.far_keep_cm = float(d_cm[thin[0]])
                far = np.full(self.valid.shape, 255, np.uint8)
                cv2.circle(far, tuple(int(round(v)) for v in self.centre), int(r_px[thin[0]]), 0, -1)
                bits = 0
                for n in RADIAL_KEEP_BITS:
                    bits |= BITS[n]
                self._far_keep = cv2.bitwise_and(far, bits)          # bits to keep un-opened, per pixel

    # ---- camera
    @property
    def camera_controls(self):
        ctrl = dict(self.config["camera"]["controls"])
        if "ColourGains" in ctrl:
            ctrl["ColourGains"] = tuple(ctrl["ColourGains"])    # (red gain, blue gain)
        return ctrl

    def apply_camera(self, cam):
        cam.set_controls(self.camera_controls)

    # ---- heading
    def effective_yaw(self, yaw_deg):
        sign = -1.0 if self.invert_yaw else 1.0
        return wrap180(sign * yaw_deg - self.yaw_offset)

    def set_forward(self, yaw_deg):
        """Call once while the robot faces 'forward' (the direction the field is aligned to)."""
        sign = -1.0 if self.invert_yaw else 1.0
        self.yaw_offset = wrap180(sign * yaw_deg)

    # ---- pixels <-> cm (robot frame: +x right, +y forward, origin at the robot centre)
    def to_cm(self, px, py):
        """Pixel(s) of the robot-aligned frame -> cm in the robot frame. Works on scalars or numpy arrays.
        Radial mode: direction = the pixel's angle from the robot centre, distance = the calibrated curve."""
        cx, cy = self.centre
        dx, dy = px - cx, cy - py
        if self._radial is None:
            return dx / self.px_per_cm, dy / self.px_per_cm
        r = np.hypot(dx, dy)
        d = np.interp(r, self._radial[0], self._radial[1])
        scale = d / np.maximum(r, 1e-6)
        return dx * scale, dy * scale

    def to_px(self, x_cm, y_cm):
        cx, cy = self.centre
        if self._radial is None:
            return cx + x_cm * self.px_per_cm, cy - y_cm * self.px_per_cm
        d = np.hypot(x_cm, y_cm)
        r = np.interp(d, self._radial[1], self._radial[0])
        scale = r / np.maximum(d, 1e-6)
        return cx + x_cm * scale, cy - y_cm * scale

    def radius_px(self, cm):
        """How many pixels out from the robot centre `cm` is (a circle of that radius in the image)."""
        if self._radial is None:
            return cm * self.px_per_cm
        return float(np.interp(cm, self._radial[1], self._radial[0]))

    def polar(self, px, py):
        """Pixel(s) -> (distance cm, direction radians: 0 = straight ahead, + = to the left) from the robot centre."""
        x, y = self.to_cm(px, py)
        return np.hypot(x, y), np.arctan2(-x, y)

    # ---- per-frame processing
    def classify(self, img):
        """BGR image -> label image: bit BITS[colour] set where the pixel is inside that colour's HSV range."""
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        lut = self._luts
        return cv2.bitwise_and(cv2.bitwise_and(cv2.LUT(h, lut[0]), cv2.LUT(s, lut[1])), cv2.LUT(v, lut[2]))

    def process(self, raw, yaw_deg=0.0, align=None):
        """raw = cam.capture_array("main"). Returns a Result (frame, masks, centre, labels).
        align=None uses the config's "align to field"; pass False to stay in the robot frame
        (cheaper: rotate the detected points instead of the images)."""
        x, y, w, h = self.warp.crop
        labels = self.warp.labels(self.classify(raw[y:y + h, x:x + w]))
        x1, y1, x2, y2 = self.ignore_box
        labels[y1:y2, x1:x2] = 0                             # the robot's own body
        if self._far_keep is not None:
            labels = cv2.bitwise_or(open_bits(labels, self.morph_kernel), cv2.bitwise_and(labels, self._far_keep))
        else:
            labels = open_bits(labels, self.morph_kernel)
        masks = {n: cv2.compare(cv2.bitwise_and(labels, BITS[n]), 0, cv2.CMP_NE) for n in COLOURS}

        centre = self.centre
        if self.align_to_field if align is None else align:
            frame = self.warp.colour(raw)
            M, size, centre = _make_align(frame.shape, centre, self.effective_yaw(yaw_deg))
            frame = cv2.warpAffine(frame, M, size, flags=cv2.INTER_LINEAR)
            masks = {k: cv2.warpAffine(v, M, size, flags=cv2.INTER_NEAREST) for k, v in masks.items()}
            labels = cv2.warpAffine(labels, M, size, flags=cv2.INTER_NEAREST)
            return Result(masks, centre, labels, frame=frame)
        return Result(masks, centre, labels, make_frame=lambda: self.warp.colour(raw))

    def goal_positions(self, res):
        """(goalpos, own_goalpos) as [dx, dy] pixels from the robot centre (+dy = forward).
        goalpos = yellow goal, own_goalpos = blue goal; [0, 200] / [0, -200] when not seen."""
        cx, cy = res.centre
        yb = merge_blobs(res.masks["yellow"], self.min_blob_area)
        bb = merge_blobs(res.masks["blue"], self.min_blob_area)
        goal = [0, 200] if yb == [0, 0, 0, 0] else [yb[0] - cx, cy - (yb[1] + yb[3] / 2)]
        own = [0, -200] if bb == [0, 0, 0, 0] else [bb[0] + bb[2] - cx, cy - (bb[1] + bb[3] / 2)]
        return goal, own

"""
vision.py - everything the camera does, in one file:

    1. RobotVision  - calibration from test_camera.py (robot_vision_config.json): raw frame -> colour masks
    2. detect()     - colour masks -> Detections in cm (ball, goals, lines, obstacles), robot frame
    3. VisionThread - camera thread: capture -> 1 -> 2 -> latest Detections (analyse_frame() is what the
                      robot and simulator.py both run on every frame)
"""
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
from utils import rotate


# ====================================================================================================
# 1. RobotVision - camera calibration + colour masks
# ====================================================================================================
# Camera calibration + colour masks - use the settings tuned in test_camera.py inside the robot script.
#
# 1. Tune everything in test_camera.py's web page, then press "Generate config file".
#    That writes robot_vision_config.json next to test_camera.py.
# 2. Put robot_vision_config.json next to vision.py (needs field.py, robot_config.py, utils.py too) and:
#
#     from vision import RobotVision
#
#     rv = RobotVision("robot_vision_config.json")
#     cam = picamera2.Picamera2()
#     cam.configure(cam.create_preview_configuration(main={"size": rv.size, "format": "RGB888"}))
#     rv.apply_camera(cam)                     # exposure + white balance from the file
#     cam.start()
#     rv.set_forward(yaw_deg)                  # optional, see below
#
#     while True:
#         raw = cam.capture_array("main")
#         res = rv.process(raw, yaw_deg)       # yaw_deg from your BNO08x (degrees, counter-clockwise positive)
#         goalpos, own_goalpos = rv.goal_positions(res)
#         # res.masks["blue" | "yellow" | "white" | "green" | "orange"] are uint8 masks,
#         # res.frame is the (field-aligned) colour frame, res.centre the robot position in it.
#
# Radial mode (test_camera.py "Distance calibration"): no undistortion at all. Every pixel's direction is its angle from
# the robot centre, and its distance comes from a measured pixels -> cm curve (rv.polar() gives both).
#
# process() gives the same masks as test_camera.py: top-down (or undistort, or raw/radial) -> rotation -> colour
# masks with the ignore box -> rotate by yaw about the robot centre. It just does it in a cheaper order: the colours
# are classified on the camera frame first (one HSV conversion + one lookup table per channel, all five colours at
# once into one "label" image with a bit per colour), and only that single-channel label image goes through the
# top-down remap (nearest neighbour). The rotation is folded into the remap, and the opening runs on the label image
# (all colours together). The colour frame (res.frame) is only made if something asks for it (debug views).
# Positions are in pixels of res.frame, relative to the robot centre, +y = forward (up in the frame).
# The robot code (VisionThread below) calls process(raw, align=False) and rotates the detected points instead,
# which is much cheaper than rotating six images every frame. rv.to_cm() turns pixels into cm.
#
# The geometry step is a Warp (see below): the camera frame as it is (raw / radial), or a fixed per-pixel remap
# (top-down, undistort). The omni mirror (Camera Module 3 Wide looking up into it) gives a picture that is already
# radial - direction = angle round the mirror centre, distance = a curve of the radius - so radial mode should fit
# it as it is. If it ever needs a remap instead, write a function that returns the maps, register it with
# @warp_model("omni") and put "model": "omni" in the config's "topdown" section.
#
# Yaw offset: with the IMU's game rotation vector, 0 deg is wherever the robot was when the IMU started,
# so the offset saved from the web page is only right if you power up facing the same way. Either place
# the robot facing forward at start-up, or call rv.set_forward(yaw_deg) once while it faces forward.

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
    colour(raw) -> BGR view (bilinear), labels(lab) -> label view (nearest), lab = classified crop,
    mask(m) -> a full camera-frame uint8 mask in the view (nearest, 0 where the view has no camera image)."""
    out_size = crop = valid = None

    def colour(self, raw):
        raise NotImplementedError

    def labels(self, lab):
        raise NotImplementedError

    def mask(self, m):
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

    def mask(self, m):
        return self._rot(m)


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

    def mask(self, m):
        x, y, w, h = self.crop
        return cv2.remap(np.ascontiguousarray(m[y:y + h, x:x + w]), self._nn, None, cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)


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
                           # (The opening removes anything thinner than the kernel, and a fisheye squashes far things
                           # along the radius.)


def radial_table(radial):
    """Radial calibration (test_camera.py "Distance calibration") -> (pixel radii, cm) lookup table, increasing.
    radial = {"table_px", "table_cm": the fitted pixels -> cm curve, sampled} (or "coeffs": a polynomial)."""
    if radial.get("table_px"):
        r = np.array(radial["table_px"], np.float64)
        d = np.array(radial["table_cm"], np.float64)
    else:
        r = np.linspace(0.0, float(radial["r_max_px"]), 512)
        d = np.polyval(np.array(radial["coeffs"], np.float64), r)
    d = np.maximum.accumulate(np.maximum(d, 0.0))         # distance never goes down as the radius grows
    return r, d



def encode_mask(mask):
    """uint8 mask (255 = set) -> base64 PNG text, how test_camera.py stores the painted ignore mask in the config."""
    ok, buf = cv2.imencode(".png", np.where(mask > 0, 255, 0).astype(np.uint8))
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_mask(text):
    """encode_mask() back to a uint8 mask (255 = set)."""
    m = cv2.imdecode(np.frombuffer(base64.b64decode(text), np.uint8), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError("bad ignore mask PNG")
    return np.where(m > 127, 255, 0).astype(np.uint8)


def camera_seen_mask(det, size):
    """What the camera may look at, in camera-frame pixels (before the rotation), from test_camera.py's config:
    inside the mirror circle ("mirror_crop": cx, cy, r) and not painted in the ignore mask ("ignore_mask": base64
    PNG, 255 = ignore - the mirror mount bars, our own body...). uint8, 255 = seen. None = no crop and no mask."""
    w, h = size
    crop, ign = det.get("mirror_crop"), det.get("ignore_mask")
    if not crop and not ign:
        return None
    seen = np.full((h, w), 255, np.uint8)
    if crop:
        seen[:] = 0
        cv2.circle(seen, (int(round(crop["cx"])), int(round(crop["cy"]))), int(round(crop["r"])), 255, -1)
    if ign:
        m = decode_mask(ign["png"])
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        seen[m > 0] = 0
    return seen


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

        # ignore box: cm around the robot centre, or a fixed pixel box
        ign = det.get("ignore_cm")
        if ign and not any(ign.values()):
            ign = None
        if ign and self._radial is not None:
            xs, ys = self.to_px(np.array([-ign["left"], ign["right"]], np.float64),
                                np.array([ign["front"], -ign["back"]], np.float64))
            box = (min(xs), min(ys), max(xs), max(ys))
        elif ign:
            cx, cy = self.centre
            ppc = self.px_per_cm
            box = (cx - ign["left"] * ppc, cy - ign["front"] * ppc, cx + ign["right"] * ppc, cy + ign["back"] * ppc)
        else:
            box = det.get("ignore_box") or (0, 0, 0, 0)     # x1, y1, x2, y2 in the robot-aligned frame
        self.ignore_box = tuple(max(int(round(v)), 0) for v in box)

        # mirror crop + painted ignore mask (test_camera.py): masked pixels never get a colour and don't count as seen
        seen = camera_seen_mask(det, self.size)
        self.seen = None if seen is None else self.warp.mask(seen)

        # pixels the camera actually sees (the top-down remap leaves black where there is no image)
        valid = self.warp.valid
        if self.seen is not None:
            valid = cv2.bitwise_and(valid, self.seen)
        if self._radial is not None:                        # only as far out as the distance calibration goes
            ring = np.zeros_like(valid)
            cv2.circle(ring, tuple(int(round(v)) for v in self.centre), int(self._radial[0][-1]), 255, -1)
            valid = cv2.bitwise_and(valid, ring)
        self.valid = cv2.erode(valid, np.ones((5, 5), np.uint8))

        # radial mode: past the distance where a ball gets thinner (along the radius) than RADIAL_KEEP_BALL_PX
        # kernels, the RADIAL_KEEP_BITS colours skip the opening, or far balls would vanish (detect() below still
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
        ctrl = self.camera_controls
        have = getattr(cam, "camera_controls", None)
        if isinstance(have, dict) and have:   # e.g. no AfMode/LensPosition (focus) on the HQ camera
            ctrl = {k: v for k, v in ctrl.items() if k in have}
        cam.set_controls(ctrl)

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
        if self.seen is not None:                            # mirror crop + painted ignore mask
            labels = cv2.bitwise_and(labels, self.seen)
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


# ====================================================================================================
# 2. Detection - colour masks -> things in cm
# ====================================================================================================
# Detection - turns the colour masks from RobotVision into things in cm (robot frame).
#
#     orange -> ball
#     yellow / blue -> goals
#     green  -> field (convex hull = "on the field", everything outside is ignored)
#     white  -> line points (for localisation + line avoidance)
#     inside the field but none of the colours -> obstacles (other robots)
#
# All positions are robot frame: origin = robot centre, +x = right, +y = forward, cm.
# Floor positions use the blob point NEAREST the robot: the top-down view assumes everything is flat on the
# floor, so anything tall (ball, goal walls, robots) gets stretched outwards; its nearest point is where it
# touches the floor.

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
OBSTACLE_MIN_SIZE = 12.0       # cm across (sqrt of the blob area): smaller blobs aren't robots
OBSTACLE_KERNEL = field.PENALTY_LINE_W + 2.0   # cm; opening of the "unknown colour" mask: removes anything thinner,
                                              # i.e. the black penalty box line (robots are ~20 cm across)
OBSTACLE_BODY_MARGIN = 2.0     # cm past our own body (ROBOT_RADIUS circle + ignore box) where obstacles are ignored
OBSTACLE_LINE_MARGIN = 3.0     # cm around white lines that never counts as obstacle (blurry line edges)...
OBSTACLE_LINE_MARGIN_FAR = 6.0 # ...this much further than OBSTACLE_FAR from us
OBSTACLE_FAR = 60.0
OBSTACLE_MAX_RANGE = 82.0      # cm: no obstacles further than this
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
    everywhere. Has the same to_cm() / px_per_cm as RobotVision, so the helpers above work on
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
        # our own body: blank a circle the size of the robot too
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


# ====================================================================================================
# 3. VisionThread - the camera thread
# ====================================================================================================
# Camera thread (VisionThread): capture -> RobotVision (calibration, masks) -> detect() -> latest Detections.
#
# Uses the calibration exported by test_camera.py (robot_vision_config.json). Stays in the robot frame
# (align=False) and lets the consumers rotate the few detected points by the compass instead of rotating images.

def analyse_frame(rv, raw, t, compass):
    """One camera frame -> Detections. The robot (VisionThread) and simulator.py both use exactly this."""
    res = rv.process(raw, align=False)
    return detect(rv, res, t, compass), res


class VisionThread(threading.Thread):
    MAX_ERRORS = 10           # frames failing in a row -> give up (the thread ends and common.Robot restarts the camera)

    def __init__(self, compass, config_path=cfg.VISION_CONFIG, keep_frames=False, first_frame_id=0):
        """compass() -> current robot heading in radians (sampled right after each capture).
        keep_frames=True also keeps the last processed frame/masks (for test_localisation.py).
        first_frame_id: carries the numbering on when the camera is restarted."""
        super().__init__()
        import picamera2                      # imported here so the rest of the code can be tested off the Pi
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
        config = self.cap.create_preview_configuration(main={"size": self.rv.size, "format": "RGB888"},
                                                       controls=controls)
        self.cap.configure(config)
        self.rv.apply_camera(self.cap)
        self.cap.start()

        self.cond = threading.Condition()
        self.frame_id = first_frame_id
        self.last_frame_t = time.monotonic() + 2.0   # last new frame (the watchdog allows 2 s extra to start up)
        self.detections = None
        self.last_res = None
        self.fps = 0.0
        self.ready = False

    def run(self):
        n, t0 = 0, time.monotonic()
        errors = 0
        while self.running:
            try:
                raw = self.cap.capture_array("main")
                t = time.monotonic()
                c = self.compass()
                det, res = analyse_frame(self.rv, raw, t, c)
            except Exception as e:      # one bad frame: skip it. Keeps failing: end the thread (gets restarted)
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
            n += 1
            if t - t0 >= 1.0:
                self.fps, n, t0 = n / (t - t0), 0, t

    def wait_new(self, last_id, timeout=None):
        """Block until a frame newer than last_id exists. Returns (frame_id, Detections or None on timeout)."""
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
        """Release the camera (after stop()) so a new VisionThread can open it."""
        try:
            self.cap.close()
        except Exception:
            pass


def relative(det, v):
    """Robot-frame vector from a Detections -> relative (field axes), using the compass at capture time."""
    return None if v is None else rotate(v, det.compass)

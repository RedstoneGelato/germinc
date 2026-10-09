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

process() does exactly what test_camera.py does: top-down (or undistort, or raw if not set up)
-> rotation -> colour masks with the ignore box -> rotate by yaw about the robot centre.
Positions are in pixels of res.frame, relative to the robot centre, +y = forward (up in the frame).
The robot code (vision.py) calls process(raw, align=False) and rotates the detected points instead,
which is much cheaper than rotating six images every frame. rv.to_cm() turns pixels into cm.

Yaw offset: with the IMU's game rotation vector, 0 deg is wherever the robot was when the IMU started,
so the offset saved from the web page is only right if you power up facing the same way. Either place
the robot facing forward at start-up, or call rv.set_forward(yaw_deg) once while it faces forward.

Only needs opencv and numpy.
"""
import collections
import json
import math

import cv2
import numpy as np

COLOURS = ("blue", "yellow", "white", "green", "orange")
_ROTATIONS = {"90_CLOCKWISE": cv2.ROTATE_90_CLOCKWISE, "180": cv2.ROTATE_180,
              "90_COUNTERCLOCKWISE": cv2.ROTATE_90_COUNTERCLOCKWISE, "NONE": None}

Result = collections.namedtuple("Result", "frame masks centre")


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


def _topdown_maps(Hn, out_size, K, D):
    """Top-down pixel -> floor homography -> camera ray -> fisheye model -> raw pixel (one remap, nothing cropped)."""
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
    return cv2.convertMaps(mx, my, cv2.CV_16SC2)


def _undistort_maps(K, D, size, balance, fov_scale):
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, size, np.eye(3), balance=balance, fov_scale=fov_scale)
    return cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, size, cv2.CV_16SC2)


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


FALLBACK_PX_PER_CM = 1.0   # same as test_camera.py: used when there is no top-down setup


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

        det = c["detection"]
        self.min_blob_area = det["min_blob_area"]
        self._kernel = np.ones((det["morph_kernel"],) * 2, np.uint8)

        hd = c["heading"]
        self.align_to_field = bool(hd["align_to_field"])
        self.invert_yaw = bool(hd["invert"])
        self.yaw_offset = float(hd["offset_deg"])
        self.robot_centre = c["robot_centre"]               # in the image that goes into the rotation, or None

        # geometry: top-down if set up, else plain undistort, else raw
        lens, td = c.get("lens"), c.get("topdown")
        self.mode = "raw"
        self._maps = None
        self.px_per_cm = FALLBACK_PX_PER_CM
        base_size = self.size                               # (w, h) of the image before the rotation
        if lens:
            K, D = np.array(lens["K"], np.float64), np.array(lens["D"], np.float64)
            if td:
                base_size = tuple(td["out_size"])
                self._maps = _topdown_maps(np.array(td["Hn"], np.float64), base_size, K, D)
                self.mode = "topdown"
                self.px_per_cm = td["px_per_cm"]
            else:
                u = c["undistort_view"]
                self._maps = _undistort_maps(K, D, self.size, u["balance"], u["fov_scale"])
                self.mode = "undistort"

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
        if ign:
            cx, cy = self.centre
            ppc = self.px_per_cm
            box = (cx - ign["left"] * ppc, cy - ign["front"] * ppc, cx + ign["right"] * ppc, cy + ign["back"] * ppc)
        else:
            box = det["ignore_box"]                          # x1, y1, x2, y2 in the robot-aligned frame
        self.ignore_box = tuple(max(int(round(v)), 0) for v in box)

        # pixels the camera actually sees (the top-down remap leaves black where there is no image)
        valid = np.full((self.size[1], self.size[0]), 255, np.uint8)
        if self._maps is not None:
            valid = cv2.remap(valid, self._maps[0], self._maps[1], cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if self.rotation is not None:
            valid = cv2.rotate(valid, self.rotation)
        self.valid = cv2.erode(valid, np.ones((5, 5), np.uint8))

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
        """Pixel(s) of the robot-aligned frame -> cm in the robot frame. Works on scalars or numpy arrays."""
        cx, cy = self.centre
        return (px - cx) / self.px_per_cm, (cy - py) / self.px_per_cm

    def to_px(self, x_cm, y_cm):
        cx, cy = self.centre
        return cx + x_cm * self.px_per_cm, cy - y_cm * self.px_per_cm

    # ---- per-frame processing
    def process(self, raw, yaw_deg=0.0, align=None):
        """raw = cam.capture_array("main"). Returns Result(frame, masks, centre).
        align=None uses the config's "align to field"; pass False to stay in the robot frame
        (cheaper: rotate the detected points instead of the images)."""
        if self._maps is None:
            base = raw
        else:
            base = cv2.remap(raw, self._maps[0], self._maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        frame = base if self.rotation is None else cv2.rotate(base, self.rotation)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        x1, y1, x2, y2 = self.ignore_box
        masks = {}
        for n in COLOURS:
            m = cv2.inRange(hsv, self.lower[n], self.upper[n])
            m[y1:y2, x1:x2] = 0                              # the robot's own body
            masks[n] = cv2.morphologyEx(m, cv2.MORPH_OPEN, self._kernel)

        centre = self.centre
        if self.align_to_field if align is None else align:
            M, size, centre = _make_align(frame.shape, centre, self.effective_yaw(yaw_deg))
            frame = cv2.warpAffine(frame, M, size, flags=cv2.INTER_LINEAR)
            masks = {k: cv2.warpAffine(v, M, size, flags=cv2.INTER_NEAREST) for k, v in masks.items()}
        return Result(frame, masks, centre)

    def goal_positions(self, res):
        """(goalpos, own_goalpos) as [dx, dy] pixels from the robot centre (+dy = forward).
        goalpos = yellow goal, own_goalpos = blue goal; [0, 200] / [0, -200] when not seen."""
        cx, cy = res.centre
        yb = merge_blobs(res.masks["yellow"], self.min_blob_area)
        bb = merge_blobs(res.masks["blue"], self.min_blob_area)
        goal = [0, 200] if yb == [0, 0, 0, 0] else [yb[0] - cx, cy - (yb[1] + yb[3] / 2)]
        own = [0, -200] if bb == [0, 0, 0, 0] else [bb[0] + bb[2] - cx, cy - (bb[1] + bb[3] / 2)]
        return goal, own

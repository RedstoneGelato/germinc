#!/usr/bin/env python3
"""
test_camera.py - headless camera test + calibration + colour-tuning tool for the soccer robot.

Run on the Pi (no monitor needed):
    python3 test_camera.py            # serves on port 8000
    python3 test_camera.py --port 8080

Then on your laptop open  http://<pi-ip>:8000
or, to get a real "localhost" URL, tunnel it:
    ssh -L 8000:localhost:8000 <user>@<pi-ip>     ->  http://localhost:8000

There is no login on this server - only run it on a network you trust.

Tabs
    Masks         annotated frame + one mask per colour (blue, yellow, white, green, orange)
    Detect        the annotated frame on its own (goal detection)
    Raw           raw camera frame
    Calibrate     live checkerboard detection for fisheye calibration
    Undistorted   fisheye correction (needs fisheye_calib.npz)
    Top-down      ground-plane bird's-eye view (needs topdown.npz)

Right-hand panel
    HSV thresholds     min/max sliders for each colour, applied live
    Camera             white balance (red/blue gain) + exposure (time/gain) sliders
    Lens calibration / Top-down setup   one-off setup, see the hints in the page

All slider values are saved to settings.json next to this file (on slider release) and
reloaded at start-up. "Python snippet" prints them as code to paste into the robot script.

Pipeline on every frame:
    raw -> fisheye undistort -> top-down warp -> ROTATION -> colour masks + goal detection
Calibration and the warp are done on the raw, un-rotated frame.

Only needs: picamera2, opencv, numpy (web server is pure standard library).
"""
import argparse
import copy
import json
import os
import re
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np
import picamera2

# =============================== CONFIG ====================================

CAPTURE_SIZE = (320, 240)
ROTATION = cv2.ROTATE_90_CLOCKWISE  # applied AFTER undistort + top-down

BALANCE = 0.0   # 0 = crop black borders after undistort, 1 = keep full field of view
                # (if you change this, redo the top-down setup)

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_FILE = os.path.join(HERE, "fisheye_calib.npz")
TOPDOWN_FILE = os.path.join(HERE, "topdown.npz")
SETTINGS_FILE = os.path.join(HERE, "settings.json")

# Ignore box (e.g. the robot's own body) + min blob area for the goals.
# In FINAL-frame pixels, i.e. top-down pixels once top-down is set up.
IGNORE_X1, IGNORE_X2 = 60, 160
IGNORE_Y1, IGNORE_Y2 = 90, 230
MIN_BLOB_AREA = 280
KERNEL = np.ones((3, 3), np.uint8)

MAX_TOPDOWN_SIDE = 1600  # sanity limit for the top-down output image

COLOURS = ["blue", "yellow", "white", "green", "orange"]
# BGR colour each mask is painted with in the Masks tab
MASK_BGR = {"blue": (255, 90, 0), "yellow": (0, 255, 255), "white": (255, 255, 255),
            "green": (0, 200, 0), "orange": (0, 140, 255)}

# HSV in OpenCV units: H 0-179, S 0-255, V 0-255.  blue/yellow are your old values;
# white/green/orange are just starting points - tune them with the sliders.
DEFAULT_HSV = {
    "blue":   {"lo": [90, 200, 100], "hi": [110, 255, 255]},
    "yellow": {"lo": [20, 100, 100], "hi": [40, 255, 255]},
    "white":  {"lo": [0, 0, 200],    "hi": [179, 50, 255]},
    "green":  {"lo": [40, 80, 50],   "hi": [85, 255, 255]},
    "orange": {"lo": [5, 150, 150],  "hi": [18, 255, 255]},
}
HSV_LIMITS = (179, 255, 255)

# Same behaviour as your original script by default: auto exposure, manual white balance.
# NOTE: libcamera's ColourGains order is (RED gain, BLUE gain). Your original code called
# the values (blue, red) = (2.4, 2.7) but passed them in that order, so red=2.4, blue=2.7.
DEFAULT_CAMERA = {"ae": True, "awb": False,
                  "exposure_us": 10000, "analogue_gain": 1.0,
                  "red_gain": 2.4, "blue_gain": 2.7}
CAMERA_LIMITS = {"exposure_us": (20, 100000), "analogue_gain": (1.0, 16.0),
                 "red_gain": (0.1, 8.0), "blue_gain": (0.1, 8.0)}

# ============================== SETTINGS ===================================


def clean_hsv(entry):
    lo, hi = list(entry["lo"]), list(entry["hi"])
    if len(lo) != 3 or len(hi) != 3:
        raise ValueError("need 3 values")
    lo = [max(0, min(int(round(float(v))), m)) for v, m in zip(lo, HSV_LIMITS)]
    hi = [max(0, min(int(round(float(v))), m)) for v, m in zip(hi, HSV_LIMITS)]
    for i in range(3):
        if lo[i] > hi[i]:
            lo[i], hi[i] = hi[i], lo[i]
    return {"lo": lo, "hi": hi}


class Config:
    """HSV thresholds + camera settings. `arrays` is swapped atomically, so detect() never sees a half update."""

    def __init__(self):
        self.lock = threading.Lock()
        self.hsv = copy.deepcopy(DEFAULT_HSV)
        self.camera = dict(DEFAULT_CAMERA)
        self.arrays = {}
        self.load()
        self._rebuild()

    def _rebuild(self):
        self.arrays = {n: (np.array(e["lo"], np.uint8), np.array(e["hi"], np.uint8))
                       for n, e in self.hsv.items()}

    def load(self):
        if not os.path.exists(SETTINGS_FILE):
            return
        try:
            with open(SETTINGS_FILE) as f:
                d = json.load(f)
            for n in COLOURS:
                if n in d.get("hsv", {}):
                    self.hsv[n] = clean_hsv(d["hsv"][n])
            for k in DEFAULT_CAMERA:
                if k in d.get("camera", {}):
                    self.camera[k] = d["camera"][k]
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(f"[warn] could not read {SETTINGS_FILE}: {e!r} - using defaults")

    def save(self):
        with self.lock:
            data = {"hsv": self.hsv, "camera": self.camera}
        tmp = SETTINGS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, SETTINGS_FILE)

    def set_hsv(self, name, lo, hi):
        entry = clean_hsv({"lo": lo, "hi": hi})
        with self.lock:
            self.hsv[name] = entry
            self._rebuild()
        return entry

    def reset_hsv(self, name):
        return self.set_hsv(name, DEFAULT_HSV[name]["lo"], DEFAULT_HSV[name]["hi"])

    def set_camera(self, partial):
        with self.lock:
            cam = dict(self.camera)
            for k in ("ae", "awb"):
                if k in partial:
                    cam[k] = bool(partial[k])
            for k, (lo, hi) in CAMERA_LIMITS.items():
                if k in partial:
                    cam[k] = max(lo, min(float(partial[k]), hi))
            cam["exposure_us"] = int(cam["exposure_us"])
            self.camera = cam
            return dict(cam)


CFG = Config()


def camera_controls(c):
    ctrl = {"AeEnable": bool(c["ae"]), "AwbEnable": bool(c["awb"])}
    if not c["ae"]:
        ctrl["ExposureTime"] = int(c["exposure_us"])
        ctrl["AnalogueGain"] = float(c["analogue_gain"])
    if not c["awb"]:
        ctrl["ColourGains"] = (float(c["red_gain"]), float(c["blue_gain"]))
    return ctrl


def python_snippet():
    with CFG.lock:
        hsv = copy.deepcopy(CFG.hsv)
        cam = dict(CFG.camera)
    lines = ["import numpy as np", ""]
    for n in COLOURS:
        lines.append(f"LOWER_{n.upper()} = np.array({hsv[n]['lo']})")
        lines.append(f"UPPER_{n.upper()} = np.array({hsv[n]['hi']})")
    lines += ["", "# camera (picamera2); ColourGains order is (red_gain, blue_gain)",
              f"cam.set_controls({camera_controls(cam)!r})"]
    return "\n".join(lines)

# ============================ GOAL DETECTION ===============================


def merge_blobs(mask, min_size):
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


def detect(frame, hsv_cfg):
    """Returns (annotated_frame, goalpos, own_goalpos, masks). masks = {colour: uint8 mask}."""
    h, w = frame.shape[:2]
    frame_cx, frame_cy = w / 2, h / 2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masks = {}
    for name in COLOURS:
        lo, hi = hsv_cfg[name]
        raw = cv2.inRange(hsv, lo, hi)
        raw[IGNORE_Y1:IGNORE_Y2, IGNORE_X1:IGNORE_X2] = 0
        masks[name] = cv2.morphologyEx(raw, cv2.MORPH_OPEN, KERNEL)

    blue_box = merge_blobs(masks["blue"], MIN_BLOB_AREA)
    yellow_box = merge_blobs(masks["yellow"], MIN_BLOB_AREA)

    annotated = frame.copy()
    cv2.rectangle(annotated, (IGNORE_X1, IGNORE_Y1), (IGNORE_X2, IGNORE_Y2), (0, 0, 255), 1)
    cv2.putText(annotated, "ignore", (IGNORE_X1, max(IGNORE_Y1 - 4, 10)),
                cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1)
    cv2.drawMarker(annotated, (int(frame_cx), int(frame_cy)), (0, 255, 0), cv2.MARKER_CROSS, 8, 1)

    if blue_box != [0, 0, 0, 0]:
        x, y, bw, bh = blue_box
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (255, 0, 0), 1)
        cv2.putText(annotated, "blue", (x, max(y - 4, 10)), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 0), 1)
    if yellow_box != [0, 0, 0, 0]:
        x, y, bw, bh = yellow_box
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (255, 0, 255), 1)
        cv2.putText(annotated, "yellow", (x, max(y - 4, 10)), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 255), 1)

    if yellow_box == [0, 0, 0, 0]:
        goalpos = [0, 200]
    else:
        goalx = yellow_box[0]
        goaly = yellow_box[1] + yellow_box[3] / 2
        goalpos = [goalx - frame_cx, frame_cy - goaly]

    if blue_box == [0, 0, 0, 0]:
        own_goalpos = [0, -200]
    else:
        own_goalx = blue_box[0] + blue_box[2]
        own_goaly = blue_box[1] + blue_box[3] / 2
        own_goalpos = [own_goalx - frame_cx, frame_cy - own_goaly]

    return annotated, goalpos, own_goalpos, masks


def tag(img, text):
    """Readable label on any background."""
    cv2.putText(img, text, (4, 14), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 3)
    cv2.putText(img, text, (4, 14), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
    return img


def build_montage(final, annotated, masks, show_bg):
    """3x2 grid: annotated frame + one painted mask per colour."""
    h, w = annotated.shape[:2]
    scale = min(1.0, 320.0 / max(w, h))
    tw, th = max(1, int(w * scale)), max(1, int(h * scale))

    def fit(img, interp):
        return img if scale == 1.0 else cv2.resize(img, (tw, th), interpolation=interp)

    def border(img):
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (90, 90, 90), 1)
        return img

    tiles = [border(tag(fit(annotated, cv2.INTER_AREA).copy(), "annotated"))]
    total = float(w * h)
    for name in COLOURS:
        m = masks[name]
        tile = (final // 4) if show_bg else np.zeros_like(final)
        tile = tile.copy()
        tile[m > 0] = MASK_BGR[name]
        cv2.rectangle(tile, (IGNORE_X1, IGNORE_Y1), (IGNORE_X2, IGNORE_Y2), (0, 0, 255), 1)
        pct = 100.0 * cv2.countNonZero(m) / total
        tiles.append(border(tag(fit(tile, cv2.INTER_NEAREST), f"{name} {pct:.1f}%")))
    return np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])])

# ============================ IMAGE PIPELINE ===============================


class Pipeline:
    """Holds the lens calibration and top-down homography. Safe to swap while running."""

    def __init__(self):
        self.maps = None      # (map1, map2)
        self.topdown = None   # (H, out_size, px_per_cm)
        self.load()

    def load(self):
        if os.path.exists(CALIB_FILE):
            d = np.load(CALIB_FILE)
            size = tuple(int(v) for v in d["size"])
            if size == CAPTURE_SIZE:
                self.set_calibration(d["K"], d["D"])
            else:
                print(f"[warn] {CALIB_FILE} was made at {size}, CAPTURE_SIZE is {CAPTURE_SIZE}: ignoring it.")
        if os.path.exists(TOPDOWN_FILE):
            d = np.load(TOPDOWN_FILE)
            self.topdown = (d["H"], tuple(int(v) for v in d["out_size"]), float(d["px_per_cm"]))

    def set_calibration(self, K, D):
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, CAPTURE_SIZE, np.eye(3), balance=BALANCE)
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, CAPTURE_SIZE, cv2.CV_16SC2)
        self.maps = (m1, m2)

    def set_topdown(self, H, out_size, px_per_cm, save=True):
        self.topdown = (H, out_size, px_per_cm)
        if save:
            np.savez(TOPDOWN_FILE, H=H, out_size=np.array(out_size), px_per_cm=px_per_cm)

    def clear_topdown(self):
        self.topdown = None
        if os.path.exists(TOPDOWN_FILE):
            os.remove(TOPDOWN_FILE)

    def undistort(self, raw):
        maps = self.maps
        if maps is None:
            return None
        return cv2.remap(raw, maps[0], maps[1], cv2.INTER_LINEAR)

    def to_topdown(self, undist):
        td = self.topdown
        if undist is None or td is None:
            return None
        H, size, _ = td
        return cv2.warpPerspective(undist, H, size, flags=cv2.INTER_LINEAR)


class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.fid = 0
        self.fps = 0.0
        self.raw = self.undist = self.topdown = self.final = self.detect_img = None
        self.masks = None
        self.goalpos = self.own_goalpos = None
        self.meta = {}   # what the camera is actually using right now
        # calibration capture state
        self.obj = []
        self.img = []
        self.board = None
        self.rms = None
        # frozen undistorted frame used for picking the 4 floor points
        self.snapshot = None


P = Pipeline()
S = Shared()
CAM = None


def capture_loop(cam):
    n, t0 = 0, time.time()
    while True:
        req = cam.capture_request()
        try:
            raw = req.make_array("main")
            md = req.get_metadata()
        finally:
            req.release()

        und = P.undistort(raw)
        td = P.to_topdown(und)
        base = td if td is not None else (und if und is not None else raw)
        final = cv2.rotate(base, ROTATION)
        annotated, goalpos, own_goalpos, masks = detect(final, CFG.arrays)

        gains = md.get("ColourGains") or (None, None)
        meta = {"exposure_us": md.get("ExposureTime"), "analogue_gain": md.get("AnalogueGain"),
                "red_gain": gains[0], "blue_gain": gains[1]}

        n += 1
        now = time.time()
        with S.lock:
            S.raw, S.undist, S.topdown = raw, und, td
            S.final, S.detect_img, S.masks = final, annotated, masks
            S.goalpos, S.own_goalpos, S.meta = goalpos, own_goalpos, meta
            S.fid += 1
            if now - t0 >= 1.0:
                S.fps = n / (now - t0)
                n, t0 = 0, now


def make_camera():
    cam = picamera2.Picamera2()
    config = cam.create_preview_configuration(main={"size": CAPTURE_SIZE, "format": "RGB888", "FrameDurationLimits": (16666, 16666)})
    cam.configure(config)
    cam.set_controls(camera_controls(CFG.camera))
    cam.start()
    return cam

# ============================== CALIBRATION ================================

SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def label(img, text):
    out = img.copy()
    cv2.putText(out, text, (6, 16), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1)
    return out


def cal_add(cols, rows):
    with S.lock:
        raw = S.raw
        board = S.board
        have = len(S.obj)
    if raw is None:
        return {"ok": False, "msg": "No frame yet."}
    if have and board != (cols, rows):
        return {"ok": False, "msg": f"Board size changed (was {board[0]}x{board[1]}). Press Reset first."}

    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if not found:
        return {"ok": False, "msg": "Checkerboard not detected - not saved."}
    corners = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), SUBPIX)

    objp = np.zeros((1, cols * rows, 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    with S.lock:
        S.obj.append(objp)
        S.img.append(corners.reshape(1, -1, 2).astype(np.float32))
        S.board = (cols, rows)
        n = len(S.obj)
    return {"ok": True, "msg": f"Saved frame {n}."}


def cal_run():
    with S.lock:
        obj, img = list(S.obj), list(S.img)
    if len(obj) < 10:
        return {"ok": False, "msg": f"Need at least 10 frames (have {len(obj)}); 15-25 is better."}

    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND
             + cv2.fisheye.CALIB_FIX_SKEW)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    dropped = 0
    while True:
        if len(obj) < 10:
            return {"ok": False, "msg": "Too many bad frames were dropped. Reset and retake with more variety."}
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        try:
            rms, K, D, _, _ = cv2.fisheye.calibrate(obj, img, CAPTURE_SIZE, K, D, None, None, flags, crit)
            break
        except cv2.error as e:
            m = re.search(r"input array (\d+)", str(e))
            if m and int(m.group(1)) < len(obj):  # OpenCV tells us which frame is ill-conditioned
                idx = int(m.group(1))
                del obj[idx]
                del img[idx]
                dropped += 1
                continue
            return {"ok": False, "msg": f"Calibration failed: {str(e).strip().splitlines()[-1]}"}

    np.savez(CALIB_FILE, K=K, D=D, size=np.array(CAPTURE_SIZE))
    P.set_calibration(K, D)
    had_topdown = P.topdown is not None
    P.clear_topdown()  # the old homography was computed on the old undistortion
    with S.lock:
        S.obj, S.img, S.rms = obj, img, float(rms)
    msg = f"Calibrated with {len(obj)} frames, RMS {rms:.3f}px (aim for < ~1). Saved fisheye_calib.npz."
    if dropped:
        msg += f" Dropped {dropped} ill-conditioned frame(s)."
    if had_topdown:
        msg += " Old top-down setup was cleared - redo it."
    return {"ok": True, "msg": msg}


def td_apply(body):
    try:
        pts = np.float32(body["points"])
        w_cm, h_cm = float(body["w_cm"]), float(body["h_cm"])
        ppc, margin_cm = float(body["px_per_cm"]), float(body["margin_cm"])
    except (KeyError, ValueError, TypeError):
        return {"ok": False, "msg": "Bad input."}
    if pts.shape != (4, 2):
        return {"ok": False, "msg": "Need exactly 4 points."}
    if min(w_cm, h_cm, ppc) <= 0 or margin_cm < 0:
        return {"ok": False, "msg": "Width, height and px/cm must be positive."}

    m, w, h = margin_cm * ppc, w_cm * ppc, h_cm * ppc
    out_size = (int(w + 2 * m), int(h + 2 * m))
    if max(out_size) > MAX_TOPDOWN_SIDE:
        return {"ok": False, "msg": f"Output would be {out_size[0]}x{out_size[1]} px - lower px/cm or margin."}
    dst = np.float32([[m, m], [m + w, m], [m + w, m + h], [m, m + h]])
    try:
        H = cv2.getPerspectiveTransform(pts, dst)
    except cv2.error:
        return {"ok": False, "msg": "Points are degenerate - pick 4 distinct corners."}
    P.set_topdown(H, out_size, ppc)
    return {"ok": True, "msg": f"Top-down saved: {out_size[0]}x{out_size[1]} px at {ppc} px/cm."}

# ============================ HSV / CAMERA API =============================


def api_hsv(body):
    name = body.get("name")
    if name not in COLOURS:
        return {"ok": False, "msg": "Unknown colour."}
    try:
        entry = CFG.set_hsv(name, body["lo"], body["hi"])
    except (KeyError, ValueError, TypeError):
        return {"ok": False, "msg": "Bad HSV values."}
    if body.get("save"):
        CFG.save()
    return {"ok": True, "hsv": entry}


def api_hsv_reset(body):
    name = body.get("name")
    if name not in COLOURS:
        return {"ok": False, "msg": "Unknown colour."}
    entry = CFG.reset_hsv(name)
    CFG.save()
    return {"ok": True, "hsv": entry, "msg": f"{name} reset to defaults."}


def api_camera(body):
    partial = {k: v for k, v in body.items() if k in DEFAULT_CAMERA}
    with CFG.lock:
        cur = dict(CFG.camera)
    with S.lock:
        meta = dict(S.meta)

    # Switching auto -> manual: start from whatever the camera is using right now,
    # so the image doesn't jump. (Handy trick: let auto settle, then switch it off.)
    if partial.get("ae") is False and cur["ae"]:
        for k in ("exposure_us", "analogue_gain"):
            if k not in partial and meta.get(k):
                partial[k] = meta[k]
    if partial.get("awb") is False and cur["awb"]:
        for k in ("red_gain", "blue_gain"):
            if k not in partial and meta.get(k):
                partial[k] = meta[k]

    try:
        new = CFG.set_camera(partial)
    except (ValueError, TypeError):
        return {"ok": False, "msg": "Bad camera values."}
    try:
        CAM.set_controls(camera_controls(new))
    except Exception as e:  # keep the server alive
        return {"ok": False, "msg": f"Camera rejected controls: {e!r}", "camera": new}
    if body.get("save"):
        CFG.save()
    return {"ok": True, "camera": new}

# ================================ WEB UI ===================================

PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Robot camera test</title>
<style>
 body{margin:0;padding:16px;background:#111;color:#eee;font:14px system-ui,sans-serif;display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
 #left{flex:1 1 560px;max-width:980px}
 #bar{display:flex;flex-wrap:wrap;align-items:center;gap:4px 12px;margin-bottom:8px}
 #wrap{position:relative;background:#000;width:100%;min-height:120px}
 #view{width:100%;display:block}
 #ov{position:absolute;left:0;top:0;pointer-events:none}
 #wrap.pick{cursor:crosshair;outline:2px solid #f80}
 #right{flex:0 0 360px;display:flex;flex-direction:column;gap:10px}
 .tabs button{padding:6px 10px;margin:0 4px 0 0;background:#222;color:#eee;border:1px solid #444;border-radius:4px;cursor:pointer}
 .tabs button.on{background:#2a6;border-color:#2a6}
 details{border:1px solid #444;border-radius:6px;padding:6px 10px}
 summary{cursor:pointer;color:#9cf;font-weight:600}
 button.a{padding:5px 9px;margin:3px 3px 3px 0;background:#335;color:#eee;border:1px solid #557;border-radius:4px;cursor:pointer}
 button.a:hover{background:#447}
 input[type=number]{width:60px;background:#222;color:#eee;border:1px solid #555;border-radius:3px;padding:3px}
 label{margin-right:8px;white-space:nowrap}
 .row{display:flex;align-items:center;gap:6px;margin:3px 0}
 .row .n{width:84px;color:#bbb;font-size:13px}
 .row input[type=range]{flex:1;min-width:0}
 .row output{width:46px;text-align:right;font:12px monospace}
 #sw button{width:28px;height:28px;border-radius:50%;border:3px solid #333;margin-right:6px;cursor:pointer}
 #sw button.on{border-color:#fff}
 #huebar{height:8px;border-radius:3px;margin:6px 0 2px;background:linear-gradient(to right,#f00,#ff0,#0f0,#0ff,#00f,#f0f,#f00);opacity:.8}
 #log{height:120px;overflow:auto;background:#000;border:1px solid #333;padding:6px;font:12px monospace;white-space:pre-wrap}
 #snip{display:none;width:100%;height:190px;background:#000;color:#9f9;border:1px solid #333;font:11px monospace;box-sizing:border-box}
 .ok{color:#6e6}.bad{color:#f66}.dim{color:#999;font-size:13px}
 #status{font:12px monospace;white-space:pre}
</style></head><body>
<div id="left">
 <div id="bar"><div class="tabs" id="tabs"></div>
  <label id="bgl"><input type="checkbox" id="bg"> show image behind masks</label></div>
 <div id="wrap"><img id="view" alt="stream"><canvas id="ov"></canvas></div>
 <div class="dim" id="hint" style="margin-top:6px"></div>
</div>
<div id="right">
 <details open><summary>HSV thresholds</summary>
  <div id="sw" style="margin:8px 0"></div>
  <div class="dim">OpenCV units: H 0-179, S 0-255, V 0-255. Applied live; saved when you release a slider. % on each mask tile = share of pixels matched.</div>
  <div id="huebar"></div>
  <div id="hsvbox"></div>
  <button class="a" onclick="resetColour()">Reset this colour</button>
  <button class="a" onclick="showSnippet()">Python snippet</button>
  <textarea id="snip" readonly onclick="this.select()"></textarea>
 </details>
 <details open><summary>Camera: white balance &amp; exposure</summary>
  <div class="dim">Tip: leave auto on until the image looks right, then untick - it freezes at the current values.</div>
  <div class="row"><label><input type="checkbox" id="ae"> Auto exposure</label>
   <label><input type="checkbox" id="awb"> Auto white balance</label></div>
  <div id="cambox"></div>
 </details>
 <details><summary>Lens calibration</summary>
  <div class="dim">Checkerboard INNER corners (= squares - 1 in each direction). Hold it at many positions/tilts, esp. frame edges. Use the Calibrate tab.</div>
  <label>cols <input id="cols" type="number" value="7" min="3"></label>
  <label>rows <input id="rows" type="number" value="6" min="3"></label><br>
  <button class="a" onclick="calAdd()">Capture (Space)</button>
  <button class="a" onclick="post('/api/cal/undo')">Undo</button>
  <button class="a" onclick="post('/api/cal/reset')">Reset</button><br>
  <button class="a" onclick="post('/api/cal/run')">Run calibration</button>
 </details>
 <details><summary>Top-down setup</summary>
  <div class="dim">Put a rectangle of known size on the floor. Freeze, then click its corners: TL, TR, BR, BL (as seen in the frozen image).</div>
  <button class="a" onclick="freeze()">Freeze &amp; pick points</button>
  <button class="a" onclick="clearPts()">Clear points</button><br>
  <label>width cm <input id="w_cm" type="number" value="60"></label>
  <label>height cm <input id="h_cm" type="number" value="40"></label><br>
  <label>px/cm <input id="ppc" type="number" value="2" step="0.5"></label>
  <label>margin cm <input id="margin" type="number" value="40"></label><br>
  <button class="a" onclick="applyTd()">Apply &amp; save</button>
  <button class="a" onclick="post('/api/td/clear')">Remove top-down</button>
 </details>
 <details open><summary>Live</summary><div id="status">...</div></details>
 <div id="log"></div>
</div>
<script>
const MODES=[['masks','Masks'],['detect','Detect'],['raw','Raw'],['calib','Calibrate'],['undistorted','Undistorted'],['topdown','Top-down']];
const HINTS={
 masks:'Annotated frame (top-left) + one mask per colour. Tune the HSV sliders on the right and watch the masks change.',
 detect:'Final frame after ROTATION with goal detection (blue / yellow) - what the robot code sees.',
 raw:'Raw camera frame (not rotated).',
 calib:'Live checkerboard detection. Coloured corners = detected; press Capture.',
 undistorted:'Fisheye-corrected frame (not rotated). Straight lines should look straight.',
 topdown:'Bird\'s-eye view (before ROTATION). Floor lines should be parallel and the rectangle true to scale.'};
const SWATCH={blue:'#38f',yellow:'#fd2',white:'#fff',green:'#3b3',orange:'#f80'};
const CH=[['H',0,179],['S',0,255],['V',0,255]];
const CAMS=[['exposure_us','Exposure µs',100,33000,100,'ae'],['analogue_gain','Gain',1,16,0.1,'ae'],
            ['red_gain','Red gain',0.5,8,0.05,'awb'],['blue_gain','Blue gain',0.5,8,0.05,'awb']];
const $=id=>document.getElementById(id);
const JH={'Content-Type':'application/json'};
let mode='masks',picking=false,points=[],cfg=null,cur='blue';
const view=$('view'),ov=$('ov'),wrap=$('wrap');

// ---------- tabs / stream ----------
MODES.forEach(([m,t])=>{const b=document.createElement('button');b.textContent=t;b.id='tab_'+m;b.onclick=()=>setMode(m);$('tabs').appendChild(b)});
function setMode(m){
  mode=m;picking=false;points=[];wrap.classList.remove('pick');
  MODES.forEach(([k])=>$('tab_'+k).classList.toggle('on',k===m));
  $('hint').textContent=HINTS[m];
  $('bgl').style.display=m==='masks'?'':'none';
  view.src='/stream?mode='+m+'&cols='+$('cols').value+'&rows='+$('rows').value+'&bg='+($('bg').checked?1:0)+'&t='+Date.now();
  draw()}
$('bg').onchange=()=>setMode('masks');

// ---------- logging / posting ----------
function log(msg,ok){const d=$('log');const s=document.createElement('div');s.className=ok===true?'ok':ok===false?'bad':'';s.textContent=msg;d.appendChild(s);d.scrollTop=d.scrollHeight}
async function post(path,body){
  const r=await fetch(path,{method:'POST',headers:JH,body:JSON.stringify(body||{})});
  const j=await r.json();if(j.msg)log(j.msg,j.ok);return j}

// ---------- HSV sliders ----------
function buildColours(){
  COLOURS_.forEach(n=>{const b=document.createElement('button');b.style.background=SWATCH[n];b.title=n;b.id='sw_'+n;b.onclick=()=>selectColour(n);$('sw').appendChild(b)})}
function buildHsv(){
  const box=$('hsvbox');box.innerHTML='';
  CH.forEach(([n,mn,mx],i)=>['lo','hi'].forEach(k=>{
    const row=document.createElement('div');row.className='row';
    row.innerHTML=`<span class="n">${n} ${k==='lo'?'min':'max'}</span><input type="range" min="${mn}" max="${mx}" id="s_${k}_${i}"><output id="o_${k}_${i}"></output>`;
    box.appendChild(row);
    const s=row.querySelector('input');
    s.oninput=()=>hsvInput(k,i,+s.value,false);
    s.onchange=()=>hsvInput(k,i,+s.value,true)}))}
function refreshHsv(){
  const c=cfg.hsv[cur];
  CH.forEach((_,i)=>['lo','hi'].forEach(k=>{$('s_'+k+'_'+i).value=c[k][i];$('o_'+k+'_'+i).textContent=c[k][i]}))}
function selectColour(n){
  cur=n;COLOURS_.forEach(c=>$('sw_'+c).classList.toggle('on',c===n));refreshHsv()}
function hsvInput(k,i,v,save){
  const c=cfg.hsv[cur];c[k][i]=v;
  if(c.lo[i]>c.hi[i]){if(k==='lo')c.hi[i]=v;else c.lo[i]=v}
  refreshHsv();sendHsv(save)}
let hsvTimer=null;
function sendHsv(save){
  const go=()=>{hsvTimer=null;const c=cfg.hsv[cur];
    fetch('/api/hsv',{method:'POST',headers:JH,body:JSON.stringify({name:cur,lo:c.lo,hi:c.hi,save})})};
  if(save){clearTimeout(hsvTimer);go()}else if(!hsvTimer){hsvTimer=setTimeout(go,80)}}
async function resetColour(){
  const j=await post('/api/hsv/reset',{name:cur});
  if(j.hsv){cfg.hsv[cur]=j.hsv;refreshHsv()}}
async function showSnippet(){
  const t=await (await fetch('/api/snippet')).text();
  const a=$('snip');a.value=t;a.style.display='block';a.select()}

// ---------- camera sliders ----------
function buildCam(){
  const box=$('cambox');
  CAMS.forEach(([k,lab,mn,mx,st])=>{
    const row=document.createElement('div');row.className='row';
    row.innerHTML=`<span class="n">${lab}</span><input type="range" id="c_${k}" min="${mn}" max="${mx}" step="${st}"><output id="co_${k}"></output>`;
    box.appendChild(row);
    const s=row.querySelector('input');
    s.oninput=()=>{cfg.camera[k]=+s.value;fmtCam(k);camSend({[k]:+s.value},false)};
    s.onchange=()=>camSend({[k]:+s.value},true)});
  $('ae').onchange=()=>camSend({ae:$('ae').checked},true);
  $('awb').onchange=()=>camSend({awb:$('awb').checked},true)}
function fmtCam(k){const v=cfg.camera[k];$('co_'+k).textContent=k==='exposure_us'?Math.round(v):(+v).toFixed(2)}
function refreshCam(){
  const c=cfg.camera;$('ae').checked=c.ae;$('awb').checked=c.awb;
  CAMS.forEach(([k,,,,,auto])=>{$('c_'+k).value=c[k];$('c_'+k).disabled=c[auto];fmtCam(k)})}
let camPending={},camTimer=null;
async function camSend(p,save){
  Object.assign(camPending,p);
  const go=async()=>{camTimer=null;const body={...camPending,save};camPending={};
    const j=await (await fetch('/api/camera',{method:'POST',headers:JH,body:JSON.stringify(body)})).json();
    if(j.msg&&!j.ok)log(j.msg,false);
    // only re-sync the sliders after a toggle, otherwise they'd fight the user's finger
    if(j.camera&&('ae' in body||'awb' in body)){cfg.camera=j.camera;refreshCam()}};
  if(save){clearTimeout(camTimer);return go()}else if(!camTimer){camTimer=setTimeout(go,100)}}

// ---------- calibration / top-down ----------
async function calAdd(){await post('/api/cal/add',{cols:+$('cols').value,rows:+$('rows').value})}
async function freeze(){
  const j=await post('/api/td/snapshot');if(!j.ok)return;
  picking=true;points=[];wrap.classList.add('pick');
  $('hint').textContent='Frozen. Click 4 corners: TL, TR, BR, BL. Then Apply & save.';
  view.src='/snapshot.jpg?t='+Date.now()}
function clearPts(){points=[];draw()}
async function applyTd(){
  if(points.length!==4){log('Pick 4 points first.',false);return}
  const j=await post('/api/td/apply',{points,w_cm:+$('w_cm').value,h_cm:+$('h_cm').value,px_per_cm:+$('ppc').value,margin_cm:+$('margin').value});
  if(j.ok)setMode('topdown')}
wrap.addEventListener('click',e=>{
  if(!picking||points.length>=4)return;
  const r=view.getBoundingClientRect();
  points.push([(e.clientX-r.left)/r.width*view.naturalWidth,(e.clientY-r.top)/r.height*view.naturalHeight]);draw()});
function draw(){
  const w=view.clientWidth,h=view.clientHeight;ov.width=w;ov.height=h;
  const c=ov.getContext('2d');c.clearRect(0,0,w,h);
  if(!picking||!view.naturalWidth)return;
  const sx=w/view.naturalWidth,sy=h/view.naturalHeight;
  c.strokeStyle='#f80';c.fillStyle='#f33';c.lineWidth=1.5;c.font='bold 14px sans-serif';
  c.beginPath();points.forEach(([x,y],i)=>i?c.lineTo(x*sx,y*sy):c.moveTo(x*sx,y*sy));
  if(points.length===4)c.closePath();c.stroke();
  points.forEach(([x,y],i)=>{c.beginPath();c.arc(x*sx,y*sy,4,0,7);c.fill();c.fillText(['TL','TR','BR','BL'][i],x*sx+7,y*sy-6)})}
view.addEventListener('load',draw);window.addEventListener('resize',draw);
document.addEventListener('keydown',e=>{
  if(e.code==='Space'&&mode==='calib'&&e.target.tagName!=='INPUT'){e.preventDefault();calAdd()}});
$('cols').onchange=$('rows').onchange=()=>{if(mode==='calib')setMode('calib')};

// ---------- live status ----------
async function poll(){
  try{
    const s=await (await fetch('/api/status')).json();
    const m=s.meta||{},f=(v,d)=>v==null?'-':(+v).toFixed(d);
    $('status').textContent=
     `fps: ${s.fps.toFixed(1)}   frame: ${s.size[0]}x${s.size[1]}\n`+
     `camera now: exp ${f(m.exposure_us,0)}us  gain ${f(m.analogue_gain,2)}\n`+
     `            R ${f(m.red_gain,2)}  B ${f(m.blue_gain,2)}\n`+
     `lens calibrated: ${s.calibrated}${s.rms!=null?'  (RMS '+s.rms.toFixed(3)+')':''}\n`+
     `calib frames:    ${s.cal_frames}\n`+
     `top-down:        ${s.topdown?s.topdown.size[0]+'x'+s.topdown.size[1]+' @ '+s.topdown.px_per_cm+' px/cm':'not set'}\n`+
     `goalpos:         ${s.goalpos.map(v=>v.toFixed(1))}\n`+
     `own goal:        ${s.own_goalpos.map(v=>v.toFixed(1))}`;
  }catch(e){}
  setTimeout(poll,500)}

let COLOURS_=[];
(async()=>{
  cfg=await (await fetch('/api/settings')).json();
  COLOURS_=cfg.colours;
  buildColours();buildHsv();buildCam();selectColour('blue');refreshCam();
  setMode('masks');poll()})();
</script></body></html>
"""

# ================================ SERVER ===================================


def get_view(mode, cols, rows, show_bg):
    with S.lock:
        fid = S.fid
        raw, und, td = S.raw, S.undist, S.topdown
        final, det, masks = S.final, S.detect_img, S.masks
        saved = len(S.obj)
    if raw is None:
        return fid, None
    if mode == "raw":
        img = raw
    elif mode == "calib":
        img = raw.copy()
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
        if found:
            cv2.drawChessboardCorners(img, (cols, rows), corners, found)
        cv2.putText(img, f"saved: {saved}  {'BOARD OK' if found else 'no board'}", (6, 16),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0) if found else (0, 0, 255), 1)
    elif mode == "undistorted":
        img = und if und is not None else label(raw, "NOT CALIBRATED YET")
    elif mode == "topdown":
        if td is not None:
            img = td
        elif und is not None:
            img = label(und, "TOP-DOWN NOT SET (showing undistorted)")
        else:
            img = label(raw, "NOT CALIBRATED YET")
    elif mode == "masks":
        img = None if masks is None else build_montage(final, det, masks, show_bg)
    else:
        img = det
    return fid, img


def jpeg(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    # ---- helpers
    def send_bytes(self, data, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, obj, code=200):
        self.send_bytes(json.dumps(obj).encode(), "application/json", code)

    # ---- GET
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                self.send_bytes(PAGE.encode(), "text/html; charset=utf-8")
            elif u.path == "/stream":
                mode = q.get("mode", ["masks"])[0]
                cols = int(q.get("cols", ["7"])[0])
                rows = int(q.get("rows", ["6"])[0])
                show_bg = q.get("bg", ["0"])[0] == "1"
                self.stream(mode, cols, rows, show_bg)
            elif u.path == "/snapshot.jpg":
                with S.lock:
                    snap = S.snapshot
                if snap is None:
                    self.send_bytes(b"no snapshot", "text/plain", 404)
                else:
                    self.send_bytes(jpeg(snap), "image/jpeg")
            elif u.path == "/api/status":
                self.send_json(self.status())
            elif u.path == "/api/settings":
                with CFG.lock:
                    self.send_json({"hsv": CFG.hsv, "camera": CFG.camera, "colours": COLOURS})
            elif u.path == "/api/snippet":
                self.send_bytes(python_snippet().encode(), "text/plain; charset=utf-8")
            else:
                self.send_bytes(b"not found", "text/plain", 404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def stream(self, mode, cols, rows, show_bg):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        last = -1
        while True:
            fid, img = get_view(mode, cols, rows, show_bg)
            if img is None or fid == last:
                time.sleep(0.01)
                continue
            last = fid
            data = jpeg(img)
            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                             + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")

    def status(self):
        td = P.topdown
        with S.lock:
            return {
                "fps": S.fps,
                "size": list(CAPTURE_SIZE),
                "calibrated": P.maps is not None,
                "rms": S.rms,
                "cal_frames": len(S.obj),
                "topdown": None if td is None else {"size": list(td[1]), "px_per_cm": td[2]},
                "goalpos": S.goalpos or [0, 0],
                "own_goalpos": S.own_goalpos or [0, 0],
                "meta": S.meta,
            }

    # ---- POST
    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            path = urlparse(self.path).path

            if path == "/api/cal/add":
                res = cal_add(int(body.get("cols", 7)), int(body.get("rows", 6)))
            elif path == "/api/cal/undo":
                with S.lock:
                    if S.obj:
                        S.obj.pop()
                        S.img.pop()
                    res = {"ok": True, "msg": f"{len(S.obj)} frames saved."}
            elif path == "/api/cal/reset":
                with S.lock:
                    S.obj, S.img, S.board = [], [], None
                res = {"ok": True, "msg": "Calibration frames cleared."}
            elif path == "/api/cal/run":
                res = cal_run()
            elif path == "/api/td/snapshot":
                with S.lock:
                    snap = S.undist if S.undist is not None else S.raw
                    S.snapshot = None if snap is None else snap.copy()
                    have = S.snapshot is not None
                if not have:
                    res = {"ok": False, "msg": "No frame yet."}
                else:
                    msg = "Snapshot frozen." if P.maps is not None else \
                        "Snapshot frozen (WARNING: lens not calibrated yet - calibrate first for accurate results)."
                    res = {"ok": True, "msg": msg}
            elif path == "/api/td/apply":
                res = td_apply(body)
            elif path == "/api/td/clear":
                P.clear_topdown()
                res = {"ok": True, "msg": "Top-down removed."}
            elif path == "/api/hsv":
                res = api_hsv(body)
            elif path == "/api/hsv/reset":
                res = api_hsv_reset(body)
            elif path == "/api/camera":
                res = api_camera(body)
            else:
                return self.send_json({"ok": False, "msg": "unknown endpoint"}, 404)
            self.send_json(res)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # keep the server alive, show the error in the UI log
            self.send_json({"ok": False, "msg": f"Server error: {e!r}"}, 500)


def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return socket.gethostname()


def main():
    global CAM
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    CAM = make_camera()
    threading.Thread(target=capture_loop, args=(CAM,), daemon=True).start()
    while S.raw is None:
        time.sleep(0.05)

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    server.daemon_threads = True
    print(f"Capturing at {CAPTURE_SIZE}.  lens calibrated: {P.maps is not None}  "
          f"top-down: {P.topdown is not None}")
    print(f"Open  http://{local_ip()}:{args.port}   (or tunnel: ssh -L {args.port}:localhost:{args.port} <user>@<pi-ip>)")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
        CAM.stop()


if __name__ == "__main__":
    main()
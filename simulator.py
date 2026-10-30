#!/usr/bin/env python3
import argparse
import math
import os
import random
import sys
import time

import cv2
import numpy as np

_SIM_T = [1000.0]
time.monotonic = lambda: _SIM_T[0]

import field  # noqa: E402  (imports after the clock patch on purpose)
import hardware_attack  # noqa: E402  (plain data: dribbler / kicker settings)
import hardware_defense  # noqa: E402
import robot_config as cfg  # noqa: E402
from common import LedCalibrator, Robot  # noqa: E402
from lines import LineFusion  # noqa: E402
from localisation import LocalisationThread  # noqa: E402
from main import GoalieBrain  # noqa: E402
from main_attack import StrikerBrain  # noqa: E402
from motion import Mover  # noqa: E402
from perception import World  # noqa: E402
import vision as _vision_mod  # noqa: E402
from utils import rotate, wrap_pi  # noqa: E402
from vision import RobotVision, analyse_frame  # noqa: E402

import common as _common, lines as _lines, localisation as _localisation, main as _main  # noqa: E402,E401
import main_attack as _main_attack, motion as _motion, perception as _perception  # noqa: E402,E401

CODE_MODULES = ["field", "robot_config", "utils", "robot_vision", "detection", "vision", "localisation", "perception",
                "lines", "motion", "comms", "common", "strategy_fallback", "strategy", "main", "main_attack"]


class Code:
    def __init__(self, name, mods):
        self.name = name
        self.GoalieBrain, self.StrikerBrain = mods["main"].GoalieBrain, mods["main_attack"].StrikerBrain
        self.LocalisationThread = mods["localisation"].LocalisationThread
        self.World, self.LineFusion = mods["perception"].World, mods["lines"].LineFusion
        self.Mover, self.LedCalibrator = mods["motion"].Mover, mods["common"].LedCalibrator


CURRENT_CODE = Code("current", {"main": _main, "main_attack": _main_attack, "localisation": _localisation,
                                "perception": _perception, "lines": _lines, "motion": _motion, "common": _common})


PAST_VERSIONS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "past_versions")


def code_folder(name):
    for folder in (name, os.path.join(PAST_VERSIONS, name)):
        if os.path.isfile(os.path.join(folder, "strategy.py")):
            return folder
    have = sorted(os.listdir(PAST_VERSIONS)) if os.path.isdir(PAST_VERSIONS) else []
    raise SystemExit(f"no robot code found for '{name}' (past versions: {', '.join(d for d in have if d != 'README.md')})")


def load_code(folder):
    import importlib
    folder = code_folder(folder)
    saved = {k: sys.modules.pop(k) for k in CODE_MODULES if k in sys.modules}
    sys.path.insert(0, os.path.abspath(folder))
    try:
        mods = {k: importlib.import_module(k) for k in CODE_MODULES
                if os.path.isfile(os.path.join(folder, k + ".py"))}
    finally:
        sys.path.pop(0)
        for k in CODE_MODULES:
            sys.modules.pop(k, None)
        sys.modules.update(saved)
    return Code(os.path.basename(os.path.normpath(folder)), mods)

DT = 0.01
CAMERA_EVERY = 3
PCB_EVERY = 4
MIRROR_D = 7.0
MIRROR_RIM_Z = 21.2
MIRROR_USE_HALF = 31.0
MIRROR_RIM_ELEV = -3.0
CAM_RES = 400
MOUNT_BAR_W = 1.0
MOUNT_BAR_R = 11.0
MOUNT_BAR_ANGLES = (90.0, -90.0)
MOUNT_BAR_HALF = math.atan((MOUNT_BAR_W / 2) / (MOUNT_BAR_R - MOUNT_BAR_W / 2))
MOUNT_MASK_MARGIN = 0.5
OWN_DECK_HEIGHT = 4.0
ROBOT_HEIGHT = 18.0
GOAL_HEIGHT = 10.0
BALL_BLIND_RING = cfg.BALL_BLIND_RING
IR_HALF_FOV = 55.0
IR_NEAR = 20.0
IR_NOISE = 0.1
SIM_IR = True
TOPDOWN_PPC = 2.0
TOPDOWN_RADIUS = 150.0

TOP_SPEED = 200.0
TOP_SPIN = 6.0
MOTOR_TAU = 0.08

BALL_RADIUS = cfg.BALL_RADIUS
BALL_FRICTION = 0.8
BALL_BOUNCE = 0.5
GOAL_WALL_T = 2.0
OPP_PRESS_SPEED = 100.0
OPP_SPEED = 60.0

KICK_SPEED = 250.0
DRIBBLE_MAX_SPIN = 6.0
DRIBBLE_MAX_SIDE = 60.0
DRIBBLE_CATCH_SPEED = 120.0
NOTCH_REST_Y = cfg.ROBOT_RADIUS + BALL_RADIUS
NOTCH_HALF_W = 4.0 + BALL_RADIUS

BALL_OUT_DELAY = 0.5
BALL_PARK = (1e4, 0.0)
NO_PROGRESS_TIME = 10.0
DAMAGED_TIME = 30.0
KICKOFF_SIDE = 20.0
PUSHED_TIME = 0.5
PUSHED_OUT_SPEED = 20.0
PARK = (1e4, 1e4)

LDR_RING_RADIUS = cfg.ROBOT_RADIUS - 3.0
LDR_PHYSICAL_OFFSET = math.pi
LDR_WHITE, LDR_GREEN, LDR_GREY = 800, 3200, 2400
GREY_CARPET = False

FIELD_PPC = 2.0
UI_PPC = 2.4
MARGIN = 10.0

GREEN, GREY, WHITE, WALL = (40, 150, 40), (120, 120, 120), (250, 250, 250), (25, 25, 25)
YELLOW, BLUE, ORANGE, ROBOT = (0, 220, 255), (220, 70, 0), (0, 120, 255), (45, 45, 45)
THEIR_ROBOT = ROBOT
THEIR_DARK_BASE = 0.0
GOAL_BGR = {"yellow": YELLOW, "blue": BLUE}

MASK_BGR = {"blue": (255, 90, 0), "yellow": (0, 255, 255), "white": (255, 255, 255),
            "green": (0, 200, 0), "orange": (0, 140, 255)}

SIM_HSV = {
    "blue": {"lo": [100, 150, 80], "hi": [130, 255, 255]},
    "yellow": {"lo": [20, 100, 100], "hi": [35, 255, 255]},
    "white": {"lo": [0, 0, 200], "hi": [179, 40, 255]},
    "green": {"lo": [40, 80, 50], "hi": [85, 255, 255]},
    "orange": {"lo": [5, 150, 150], "hi": [18, 255, 255]},
}

HERE = os.path.dirname(os.path.abspath(__file__))


def mirror_design():
    R = MIRROR_D / 2
    zr = R / math.tan(math.radians(MIRROR_USE_HALF))
    c = (zr - R * math.tan(math.radians(MIRROR_RIM_ELEV))) / 2
    a = (math.hypot(R, zr) - math.hypot(R, zr - 2 * c)) / 2
    b = math.sqrt(c * c - a * a)
    cam_z = MIRROR_RIM_Z - zr
    return dict(a=a, b=b, c=c, cam_z=cam_z, view_z=cam_z + 2 * c)


def mirror_floor_table(d, n=4000):
    a, b, c = d["a"], d["b"], d["c"]
    th = np.linspace(0.0, math.radians(MIRROR_USE_HALF), n)
    t = np.tan(th)
    A = 1 / a ** 2 - t ** 2 / b ** 2
    B = -2 * c / a ** 2
    C = c ** 2 / a ** 2 - 1
    z = (-B + np.sqrt(B * B - 4 * A * C)) / (2 * A)
    dep = np.arctan2(2 * c - z, z * t)
    with np.errstate(divide="ignore"):
        dist = d["view_z"] / np.tan(dep)
    return th, np.minimum(dist, 1e4)


MIRROR = mirror_design()
VIEW_Z = MIRROR["view_z"]
MIRROR_TH, MIRROR_DIST = mirror_floor_table(MIRROR)
MIRROR_FPX = (CAM_RES / 2) / math.tan(math.radians(MIRROR_USE_HALF))
VIEW_MAX = float(MIRROR_DIST[-1])
EGO_RADIUS = 300.0
RADIAL = hasattr(_vision_mod, "radial_table")
VIEW_RADIUS = min(VIEW_MAX, EGO_RADIUS) if RADIAL else TOPDOWN_RADIUS
CAM_SIZE = CAM_RES if RADIAL else int(2 * TOPDOWN_RADIUS * TOPDOWN_PPC)
OWN_HIDDEN_R = cfg.ROBOT_RADIUS * VIEW_Z / (VIEW_Z - OWN_DECK_HEIGHT)


def radial_calibration():
    keep = MIRROR_DIST <= VIEW_RADIUS
    return {"table_px": (MIRROR_FPX * np.tan(MIRROR_TH[keep])).tolist(), "table_cm": MIRROR_DIST[keep].tolist()}


def bar_angle_off(az):
    az = np.asarray(az, np.float64)
    off = np.full(az.shape, np.inf)
    for a in MOUNT_BAR_ANGLES:
        off = np.minimum(off, np.abs((az - math.radians(a) + math.pi) % (2 * math.pi) - math.pi))
    return off


def bar_pixels(size, half):
    j, i = np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5)
    dx, dy = j - size / 2.0, size / 2.0 - i
    return bar_angle_off(np.arctan2(-dx, dy)) < half


def sim_vision_config():
    ignore = {"left": 11.0, "right": 11.0, "back": 11.0, "front": 5.0}
    real = os.path.join(HERE, cfg.VISION_CONFIG)
    if os.path.exists(real):
        import json
        with open(real) as f:
            real_ign = json.load(f)["detection"].get("ignore_cm")
        if real_ign and any(real_ign.values()):
            ignore = real_ign
    det = {"ignore_cm": ignore, "min_blob_area": 50, "morph_kernel": 3}
    if MOUNT_BAR_ANGLES and hasattr(_vision_mod, "encode_mask"):
        bars = bar_pixels(CAM_SIZE, MOUNT_BAR_HALF + math.radians(MOUNT_MASK_MARGIN))
        det["ignore_mask"] = {"size": [CAM_SIZE, CAM_SIZE], "png": _vision_mod.encode_mask(bars.astype(np.uint8) * 255)}
    return {
        "version": 1, "capture_size": [CAM_SIZE, CAM_SIZE], "rotation": "NONE",
        "camera": {"controls": {}}, "hsv": SIM_HSV,
        "detection": det,
        "lens": None, "undistort_view": {"balance": 1, "fov_scale": 1}, "topdown": None,
        "radial": radial_calibration() if RADIAL else None,
        "robot_centre": None, "heading": {"align_to_field": False, "invert": False, "offset_deg": 0},
    }


def make_rv():
    import json
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "sim_vision_config.json")
    with open(path, "w") as f:
        json.dump(sim_vision_config(), f)
    rv = RobotVision(path)
    fov = np.zeros((CAM_SIZE, CAM_SIZE), np.uint8)
    if MOUNT_BAR_ANGLES:
        bars = bar_pixels(CAM_SIZE, MOUNT_BAR_HALF + math.radians(MOUNT_MASK_MARGIN))
        rv.valid = rv.valid.copy()
        rv.valid[cv2.dilate(bars.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = 0
    if RADIAL:
        cv2.circle(fov, (CAM_SIZE // 2, CAM_SIZE // 2), int(rv.radius_px(VIEW_RADIUS)) - 2, 255, -1)
        rv.valid = cv2.bitwise_and(rv.valid, cv2.erode(fov, np.ones((5, 5), np.uint8)))
        return rv, fov
    rv.px_per_cm = TOPDOWN_PPC
    ign, (cx, cy) = rv.config["detection"]["ignore_cm"], rv.centre
    box = (cx - ign["left"] * TOPDOWN_PPC, cy - ign["front"] * TOPDOWN_PPC,
           cx + ign["right"] * TOPDOWN_PPC, cy + ign["back"] * TOPDOWN_PPC)
    rv.ignore_box = tuple(max(int(round(v)), 0) for v in box)
    cv2.circle(fov, (CAM_SIZE // 2, CAM_SIZE // 2), int(VIEW_RADIUS * TOPDOWN_PPC) - 3, 255, -1)
    rv.valid = cv2.erode(fov, np.ones((5, 5), np.uint8))
    return rv, fov


def _ring(cx, cy, r, n=20):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)


def _lift(pts, z):
    return pts * min(VIEW_Z / max(VIEW_Z - z, 1e-3), 40.0)


def cylinder_shadow(x, y, r, height):
    base = _ring(x, y, r)
    return np.vstack([base, _lift(base, height)])


def ball_shadow(x, y):
    pts = []
    for z in np.linspace(0.0, 2 * BALL_RADIUS, 5):
        rr = math.sqrt(max(BALL_RADIUS ** 2 - (z - BALL_RADIUS) ** 2, 0.0)) + 0.05
        pts.append(_lift(_ring(x, y, rr, 12), z))
    return np.vstack(pts)


def box_shadow(corners, height):
    return np.vstack([corners, _lift(corners, height)])


def ray_hidden(sim, r, pts, z=0.0, exclude=None, size=0.0):
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    dist = np.hypot(pts[:, 0], pts[:, 1])
    hidden = dist * (VIEW_Z - OWN_DECK_HEIGHT) / max(VIEW_Z - z, 1e-3) < cfg.ROBOT_RADIUS
    hidden |= dist > VIEW_RADIUS
    if MOUNT_BAR_ANGLES:
        half_size = np.arcsin(np.minimum(size / np.maximum(dist, 1e-6), 1.0))
        bar_half = MOUNT_BAR_HALF + math.radians(MOUNT_MASK_MARGIN)
        hidden |= (bar_angle_off(np.arctan2(-pts[:, 0], pts[:, 1])) + half_size < bar_half) & (dist > MOUNT_BAR_R)
    u = pts / np.maximum(dist, 1e-6)[:, None]
    for b in sim.on_field:
        if b is r or b is exclude:
            continue
        bx, by = rotate([b.x - r.x, b.y - r.y], -r.h)
        s_c = u[:, 0] * bx + u[:, 1] * by
        perp2 = bx * bx + by * by - s_c * s_c
        half = np.sqrt(np.maximum(b.r * b.r - perp2, 0.0))
        s_in, s_out = s_c - half, np.minimum(s_c + half, dist - 1e-3)
        hit = (perp2 < b.r * b.r) & (s_in < s_out) & (s_out > 0)
        ray_z = VIEW_Z - (VIEW_Z - z) * s_out / np.maximum(dist, 1e-6)
        hidden |= hit & (ray_z <= ROBOT_HEIGHT)
    return hidden


def ball_view(sim, r):
    if sim.ball[0] > BALL_PARK[0] / 2:
        return "off"
    p = rotate([sim.ball[0] - r.x, sim.ball[1] - r.y], -r.h)
    d = math.hypot(p[0], p[1])
    if BALL_BLIND_RING > 0 and d - BALL_RADIUS < cfg.ROBOT_RADIUS + BALL_BLIND_RING:
        return "ring"
    if d > FAST_BALL_RANGE:
        return "range"
    if MOUNT_BAR_ANGLES and d > MOUNT_BAR_R:
        half_size = math.asin(min(BALL_RADIUS / d, 1.0))
        if bar_angle_off(np.array([math.atan2(-p[0], p[1])]))[0] + half_size < MOUNT_BAR_HALF + math.radians(MOUNT_MASK_MARGIN):
            return "bar"
    if ray_hidden(sim, r, [p], 2 * BALL_RADIUS)[0]:
        return "robot"
    return "clear"


class OmniCamera:
    SS = 2

    def __init__(self):
        n = CAM_RES * self.SS
        c = n / 2.0
        j, i = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5)
        dx, dy = (j - c), (c - i)
        rho = np.hypot(dx, dy)
        th = np.arctan(rho / (MIRROR_FPX * self.SS))
        dist = np.interp(th, MIRROR_TH, MIRROR_DIST)
        with np.errstate(invalid="ignore", divide="ignore"):
            gx = np.where(rho > 0, dist * dx / rho, 0)
            gy = np.where(rho > 0, dist * dy / rho, 0)
        k = FIELD_PPC
        self.half_px = int(math.ceil(EGO_RADIUS * k)) + 4
        self.ego_size = 2 * self.half_px
        ex = (self.half_px + gx * k).astype(np.float32)
        ey = (self.half_px - gy * k).astype(np.float32)
        out = th > math.radians(MIRROR_USE_HALF)
        ex[out] = -10
        ey[out] = -10
        self.lens_maps = cv2.convertMaps(ex, ey, cv2.CV_16SC2)
        self.top_maps = None
        if not RADIAL:
            m = CAM_SIZE
            u, v = np.meshgrid(np.arange(m) + 0.5, np.arange(m) + 0.5)
            x, y = (u - m / 2.0) / TOPDOWN_PPC, (m / 2.0 - v) / TOPDOWN_PPC
            rr = np.hypot(x, y)
            rho_t = MIRROR_FPX * np.tan(np.interp(rr, MIRROR_DIST, MIRROR_TH))
            cr = CAM_RES / 2.0
            with np.errstate(invalid="ignore", divide="ignore"):
                map_x = np.where(rr > 0, cr + rho_t * x / rr, cr).astype(np.float32)
                map_y = np.where(rr > 0, cr - rho_t * y / rr, cr).astype(np.float32)
            fov = np.zeros((m, m), np.uint8)
            cv2.circle(fov, (m // 2, m // 2), int(VIEW_RADIUS * TOPDOWN_PPC) - 3, 255, -1)
            map_x[fov == 0] = -10
            map_y[fov == 0] = -10
            self.top_maps = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)
        self.bar_px = bar_pixels(CAM_RES, MOUNT_BAR_HALF) if MOUNT_BAR_ANGLES else None
        self.goal_boxes = [("yellow" if y0 + y1 > 0 else "blue", np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float))
                           for x0, x1, y0, y1 in GOAL_WALLS]

    def shadows(self, robot, bodies, ball):
        c, s = math.cos(-robot.h), math.sin(-robot.h)
        R = np.array([[c, -s], [s, c]])
        to_robot = lambda p: (np.asarray(p, np.float64) - [robot.x, robot.y]) @ R.T
        items = []
        for colour, corners in self.goal_boxes:
            rc = to_robot(corners)
            items.append((float(np.hypot(rc[:, 0], rc[:, 1]).min()), GOAL_BGR[colour], box_shadow(rc, GOAL_HEIGHT)))
        for b in bodies:
            if b is robot:
                continue
            bx, by = to_robot([b.x, b.y])
            colour = ROBOT if b.team == robot.team else THEIR_ROBOT
            items.append((math.hypot(bx, by) - b.r, colour, cylinder_shadow(bx, by, b.r, ROBOT_HEIGHT)))
            if colour != ROBOT and THEIR_DARK_BASE > 0:
                items.append((math.hypot(bx, by) - b.r - 1e-6, ROBOT, cylinder_shadow(bx, by, b.r, THEIR_DARK_BASE)))
        if ball is not None:
            bx, by = to_robot(ball)
            items.append((math.hypot(bx, by) - BALL_RADIUS, ORANGE, ball_shadow(bx, by)))
        items.sort(key=lambda it: -it[0])
        items.append((0.0, ROBOT, _ring(0.0, 0.0, OWN_HIDDEN_R, 32)))
        return [(col, poly) for _, col, poly in items]

    def render(self, field_img, robot, noise_rng=None, bodies=(), ball=None):
        c, s = math.cos(robot.h), math.sin(robot.h)
        k, h = FIELD_PPC, self.half_px
        M = np.float32([[c, s, (FieldImage.W / 2 + robot.x) * k - c * h - s * h],
                        [-s, c, (FieldImage.L / 2 - robot.y) * k + s * h - c * h]])
        ego = cv2.warpAffine(field_img, M, (self.ego_size, self.ego_size), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=WALL)
        for colour, poly in self.shadows(robot, bodies, ball):
            px = np.stack([h + poly[:, 0] * k, h - poly[:, 1] * k], axis=1)
            px = np.clip(px, -4 * self.ego_size, 5 * self.ego_size)
            hull = cv2.convexHull(np.round(px * 4).astype(np.int32))
            cv2.fillConvexPoly(ego, hull, colour, cv2.LINE_8, shift=2)
        img = cv2.remap(ego, *self.lens_maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        img = cv2.resize(img, (CAM_RES, CAM_RES), interpolation=cv2.INTER_AREA)
        if self.bar_px is not None:
            img[self.bar_px & (img.max(axis=2) > 0)] = ROBOT
        if noise_rng is not None:
            if not hasattr(self, "noise_bank"):
                self.noise_bank = [noise_rng.normal(0, 8, img.shape).astype(np.int16) for _ in range(16)]
            n = self.noise_bank[int(noise_rng.integers(len(self.noise_bank)))]
            n = np.roll(n, (int(noise_rng.integers(CAM_RES)), int(noise_rng.integers(CAM_RES))), axis=(0, 1))
            img = cv2.add(img, n, dtype=cv2.CV_8U)
        return img

    def topdown(self, raw):
        if self.top_maps is None:
            return raw
        return cv2.remap(raw, *self.top_maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class FieldImage:
    W = field.WALL_W + 2 * MARGIN
    L = field.WALL_L + 2 * MARGIN

    def __init__(self, ppc):
        self.ppc = ppc
        self.size = (int(round(self.W * ppc)), int(round(self.L * ppc)))
        img = np.full((self.size[1], self.size[0], 3), WALL, np.uint8)
        cv2.rectangle(img, self.px(-field.WALL_W / 2, field.WALL_L / 2), self.px(field.WALL_W / 2, -field.WALL_L / 2),
                      GREY if GREY_CARPET else GREEN, -1)
        for x0, x1, y0, y1 in GOAL_WALLS:
            cv2.rectangle(img, self.px(x0, y1), self.px(x1, y0), YELLOW if y0 + y1 > 0 else BLUE, -1)
        self.white = np.zeros(img.shape[:2], np.uint8)
        thick = max(1, int(round(field.LINE_W * ppc)))
        for a, b in field.black_line_segments():
            cv2.line(img, self.px(*a), self.px(*b), WALL, max(1, int(round(field.PENALTY_LINE_W * ppc))))
        for a, b in field.white_line_segments():
            cv2.line(img, self.px(*a), self.px(*b), WHITE, thick)
            cv2.line(self.white, self.px(*a), self.px(*b), 255, thick)
        self.img = img

    def px(self, x, y):
        return (int(round((x + self.W / 2) * self.ppc)), int(round((self.L / 2 - y) * self.ppc)))

    def is_white(self, x, y):
        u, v = self.px(x, y)
        h, w = self.white.shape
        return 0 <= u < w and 0 <= v < h and self.white[v, u] > 0


def goal_walls():
    hw, t = field.GOAL_W / 2, GOAL_WALL_T
    y0, y1 = field.ATTACK_GOAL_Y, field.GOAL_BACK_Y
    walls = []
    for sgn in (1, -1):
        for x0, x1, ya, yb in ((-hw - t, -hw, y0, y1 + t), (hw, hw + t, y0, y1 + t), (-hw - t, hw + t, y1, y1 + t)):
            walls.append((x0, x1, min(sgn * ya, sgn * yb), max(sgn * ya, sgn * yb)))
    return walls


GOAL_WALLS = goal_walls()


def push_out_of_walls(x, y, r):
    normals = []
    for x0, x1, y0, y1 in GOAL_WALLS:
        cx, cy = min(max(x, x0), x1), min(max(y, y0), y1)
        dx, dy = x - cx, y - cy
        d = math.hypot(dx, dy)
        if d >= r:
            continue
        if d > 1e-9:
            n = (dx / d, dy / d)
            pen = r - d
        else:
            pen, n = min((x - x0 + r, (-1, 0)), (x1 - x + r, (1, 0)), (y - y0 + r, (0, -1)), (y1 - y + r, (0, 1)))
        x, y = x + n[0] * pen, y + n[1] * pen
        normals.append(n)
    return x, y, normals


class SimIMU:
    def __init__(self, body):
        self.body = body
        self.heading_offset = 0.0
        self.drift = 0.0

    @property
    def heading(self):
        return self.body.h + self.drift

    def zero(self, facing=0.0):
        self.heading_offset = self.heading - facing

    def compass(self):
        return wrap_pi(self.heading - self.heading_offset)


class SimMotors:
    def __init__(self, has_dribbler):
        self.speeds = (0, 0, 0, 0)
        self.has_dribbler = has_dribbler
        self.dribbler = False

    def set(self, speeds):
        self.speeds = tuple(speeds)

    def set_dribbler(self, on):
        self.dribbler = bool(on) and self.has_dribbler

    def stop(self):
        self.speeds = (0, 0, 0, 0)
        self.dribbler = False


class SimKicker:
    def __init__(self, cooldown):
        self.cooldown = cooldown
        self.last = -1e9
        self.pending = False

    def ready(self):
        return time.monotonic() - self.last >= self.cooldown

    def kick(self):
        if not self.ready():
            return False
        self.last = time.monotonic()
        self.pending = True
        return True

    def close(self):
        pass


class SimPCB:
    def __init__(self):
        self.colours = [LDR_GREY if GREY_CARPET else LDR_GREEN] * cfg.LDR_COUNT
        self.ir = [0.0] * getattr(cfg, "IR_COUNT", 0) if SIM_IR else []

    def snapshot(self):
        return list(self.colours)

    def ir_snapshot(self):
        return list(self.ir) if self.ir else None

    def set_brightness(self, value):
        pass


class SimComms:
    def __init__(self):
        self.my_state = {"bot active": 0}
        self.partner = None
        self.enabled = True

    def teammate(self, max_age=0.5):
        if self.enabled and self.partner is not None and self.partner.enabled:
            return dict(self.partner.my_state)
        return None


class SimGoalSwitch:

    def __init__(self, body):
        self.body = body

    @property
    def is_active(self):
        attack = "yellow" if self.body.team == "us" else "blue"
        return attack == cfg.GOAL_SWITCH_CLOSED


class SimVision:
    def __init__(self, rv):
        self.rv = rv
        self.frame_id = 0
        self.detections = None
        self.last_res = None
        self.fps = 1.0 / (DT * CAMERA_EVERY)

    def latest(self):
        return self.detections


class SimBot(Robot):

    def __init__(self, body, rv, hw, code):
        self.imu = SimIMU(body)
        self.goal_switch = SimGoalSwitch(body)
        self.motors = SimMotors(hw.DRIBBLER is not None)
        self.kicker = None if hw.KICKER_PIN is None else SimKicker(hw.KICK_COOLDOWN)
        self.pcb = SimPCB()
        self.comms = SimComms()
        self.vision = SimVision(rv)
        self.attack = "yellow"
        self.loc = code.LocalisationThread(self.vision, lambda: self.attack)
        self.world = code.World()
        self.lines = code.LineFusion()
        self.mover = code.Mover()
        self.leds = code.LedCalibrator(self.pcb)


SQ2 = math.sqrt(2)
MOTOR_ANGLES = [math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4]


def motors_to_velocity(speeds):
    mx = sum(m * math.cos(a) for m, a in zip(speeds, MOTOR_ANGLES)) / 2
    my = sum(m * math.sin(a) for m, a in zip(speeds, MOTOR_ANGLES)) / 2
    k = TOP_SPEED / (SQ2 * cfg.BASE_SPEED)
    spin = -sum(speeds) / 4 / cfg.BASE_SPEED * TOP_SPIN
    return my * k, -mx * k, spin


class Body:
    def __init__(self, x, y, h, name, colour, team="them"):
        self.team = team
        self.x, self.y, self.h = x, y, h
        self.vx = self.vy = self.w = 0.0
        self.prev_v = (0.0, 0.0)
        self.name, self.colour = name, colour
        self.r = cfg.ROBOT_RADIUS
        self.removed_until = None
        self.opp_contact_t = -1e9
        self.outs = 0
        self.was_out = False

    def place(self, x, y, h=None):
        self.x, self.y = x, y
        if h is not None:
            self.h = h
        self.vx = self.vy = self.w = 0.0
        self.prev_v = (0.0, 0.0)


class SimRobot(Body):
    def __init__(self, x, y, h, name, colour, brain_cls, hw, rv, log, code=None, team="us"):
        super().__init__(x, y, h, name, colour, team)
        self.bot = SimBot(self, rv, hw, code or CURRENT_CODE)
        self.kick_flash = 0.0
        self.brain = brain_cls(self.bot, log=lambda m: log(f"{name}: {m}"))
        self.paused = True
        self.return_at = None

    def drive(self, dt):
        vx_r, vy_r, spin = motors_to_velocity(self.bot.motors.speeds)
        tvx, tvy = rotate([vx_r, vy_r], self.h)
        a = min(dt / MOTOR_TAU, 1.0)
        self.vx += (tvx - self.vx) * a
        self.vy += (tvy - self.vy) * a
        self.w += (spin - self.w) * a


class Sim:
    def __init__(self, noise=False, seed=0, opp_code=None, fast=False):
        self.seed = seed
        self.fast = fast
        self.fimg = FieldImage(FIELD_PPC)
        self.rv, self.fov = make_rv()
        self.cam = OmniCamera()
        self.held_by = None
        self.held_since = 0.0
        self.kicks = 0
        self.events = {}
        self.last_touch = None
        self.last_touch_phase = ""
        self.last_kick = -1e9
        self.shots = []
        self.last_restart = 0.0
        self.ball_out_since = None
        self.progress = ([0.0, 0.0], 0.0)
        self.logs = []
        self.robots = [
            SimRobot(0, 0, 0, "goalie", (255, 200, 0), GoalieBrain, hardware_defense, self.rv, self.log),
            SimRobot(0, 0, 0, "striker", (255, 0, 255), StrikerBrain, hardware_attack, self.rv, self.log),
        ]
        g, s = self.robots
        g.bot.comms.partner, s.bot.comms.partner = s.bot.comms, g.bot.comms
        if opp_code is None:
            self.opponents = [Body(0, 0, math.pi, "opp", ROBOT), Body(0, 0, math.pi, "opp", ROBOT)]
        else:
            self.opponents = [
                SimRobot(0, 0, math.pi, "opp striker", ROBOT, opp_code.StrikerBrain, hardware_attack, self.rv,
                         self.log, opp_code, "them"),
                SimRobot(0, 0, math.pi, "opp goalie", ROBOT, opp_code.GoalieBrain, hardware_defense, self.rv,
                         self.log, opp_code, "them"),
            ]
            o_s, o_g = self.opponents
            o_s.bot.comms.partner, o_g.bot.comms.partner = o_g.bot.comms, o_s.bot.comms
        for i, r in enumerate(self.players):
            r.bot.loc.loc.rng = np.random.default_rng(100 + i + 10 * seed)
            if fast:
                r.bot.loc = TruthLoc(r, np.random.default_rng(200 + i + 10 * seed))
        self.fast_ignore = self.rv.config["detection"]["ignore_cm"]
        self.ball = [0.0, 0.0]
        self.ball_v = [0.0, 0.0]
        self.ball_away_until = None
        self.ball_out_at = None
        self.noise = noise
        self.opp_ai = False
        self.last_scorer = None
        self.opp_style = "push"
        self.ticks = 0
        self.score = {"us": 0, "them": 0}
        self.goal_timer = None
        self.loc_err = {r.name: [] for r in self.robots}
        self.rng = np.random.default_rng(seed)
        self.ir_rng = np.random.default_rng(300 + seed)
        self.faults = None
        self.kickoff()

    @property
    def players(self):
        return self.robots + [o for o in self.opponents if isinstance(o, SimRobot)]

    def release(self, why):
        if self.held_by is not None and self.t - self.held_since > 0.3:
            self.count(f"{self.held_by.team} possession ended: {why}")
            gy = self.held_by.y if self.held_by.team == "us" else -self.held_by.y
            where = "own third" if gy < -30 else "middle third" if gy < 30 else "attacking third"
            self.count(f"{self.held_by.team} possession ended: {why} in {where}")
            self.count(f"{self.held_by.team} possession seconds", self.t - self.held_since)
        self.held_by = None

    def count(self, what, n=1):
        self.events[what] = self.events.get(what, 0) + n

    def log(self, msg):
        self.logs.append(f"{_SIM_T[0] - 1000:6.2f} {msg}")
        self.logs = self.logs[-6:]

    @property
    def t(self):
        return _SIM_T[0] - 1000.0

    def kickoff(self):
        for b in self.robots + self.opponents:
            b.removed_until = None
            if isinstance(b, SimRobot):
                b.return_at = None
        g, s = self.robots
        R = cfg.ROBOT_RADIUS
        box_y = field.OWN_GOAL_Y + field.PENALTY_D + R - 2
        g.place(0, field.OWN_GOAL_Y + R + 8, 0)
        if self.last_scorer == "us":
            s.place(KICKOFF_SIDE, box_y, 0)
        else:
            s.place(0, -(R + 15), 0)
        opp_striker = (-KICKOFF_SIDE, -box_y) if self.last_scorer == "them" else (0, R + 15)
        for o, (x, y) in zip(self.opponents, (opp_striker, (0, field.ATTACK_GOAL_Y - R - 8))):
            o.place(x, y, math.pi)
        self.ball, self.ball_v = [0.0, 0.0], [0.0, 0.0]
        self.ball_away_until = None
        self.ball_out_at = None
        if self.seed:
            for b in self.robots + self.opponents:
                b.place(b.x + self.rng.uniform(-3, 3), b.y + self.rng.uniform(-3, 3), b.h + self.rng.uniform(-0.1, 0.1))
            self.ball = [self.rng.uniform(-3, 3), self.rng.uniform(-3, 3)]
        self.held_by = None
        self.ball_out_since = None
        self.progress = (list(self.ball), self.t)
        for r in self.players:
            r.paused = True
        self.last_restart = self.t + 0.5
        self.start_at = self.t + 0.5
        self.goal_timer = None

    @property
    def on_field(self):
        return [b for b in self.robots + self.opponents if b.removed_until is None]

    def draw_dynamic(self):
        return self.fimg.img

    def camera(self, img, robot):
        ball = None if ball_view(self, robot) in ("off", "ring") else self.ball
        raw = self.cam.render(img, robot, self.rng if self.noise else None, self.on_field, ball)
        return self.cam.topdown(raw)

    _LDR_RING = None

    def ldrs(self, robot):
        if Sim._LDR_RING is None:
            ring = []
            for i in range(cfg.LDR_COUNT):
                a = i * 2 * math.pi / cfg.LDR_COUNT + cfg.LDR_START_ANGLE + LDR_PHYSICAL_OFFSET
                ring.append([LDR_RING_RADIUS * math.cos(a), LDR_RING_RADIUS * math.sin(a)])
            Sim._LDR_RING = np.array(ring)
        c, s = math.cos(robot.h), math.sin(robot.h)
        r = Sim._LDR_RING
        x = robot.x + (r[:, 0] * c - r[:, 1] * s)
        y = robot.y + (r[:, 0] * s + r[:, 1] * c)
        f = self.fimg
        u = np.round((x + f.W / 2) * f.ppc).astype(int)
        v = np.round((f.L / 2 - y) * f.ppc).astype(int)
        h, w = f.white.shape
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        white = np.zeros(len(u), bool)
        white[ok] = f.white[v[ok], u[ok]] > 0
        out = np.where(white, LDR_WHITE, LDR_GREY if GREY_CARPET else LDR_GREEN)
        if self.noise:
            out = out + np.array([int(n) for n in self.rng.normal(0, 150, len(out))])
        return [int(o) for o in out]

    _IR_DIRS = None

    def ir(self, robot):
        n = getattr(cfg, "IR_COUNT", 0)
        if not n or not SIM_IR:
            return []
        if Sim._IR_DIRS is None:
            Sim._IR_DIRS = np.array([cfg.IR_START_ANGLE + i * 2 * math.pi / n for i in range(n)])
        if self.ball[0] > BALL_PARK[0] / 2:
            return [0.0] * n
        p = rotate([self.ball[0] - robot.x, self.ball[1] - robot.y], -robot.h)
        d = max(math.hypot(p[0], p[1]), 1.0)
        a = math.atan2(p[1], p[0])
        off = np.abs((Sim._IR_DIRS - a + math.pi) % (2 * math.pi) - math.pi) / math.radians(IR_HALF_FOV)
        s = min(1.0, IR_NEAR / d)
        val = np.where(off < 1.0, s * (1.0 - 0.6 * off ** 2), 0.0)
        if self.noise:
            val = val * (1.0 + self.ir_rng.normal(0, IR_NOISE, n))
        return [float(min(max(v, 0.0), 1.0)) for v in val]

    def ball_stats(self):
        for r in self.robots:
            if r.paused or r.removed_until is not None or r.return_at is not None:
                continue
            why = ball_view(self, r)
            seen = r.bot.vision.detections is not None and r.bot.vision.detections.ball is not None
            self.count(f"ballview {r.name} {why} {'seen' if seen else 'unseen'}")
            if why != "off":
                self.count(f"ballview {r.name} {why} code:{getattr(r.bot.world, 'ball_source', '?')}")

    def step(self):
        _SIM_T[0] += DT
        self.ticks += 1
        if self.faults is not None:
            self.faults.step()
        if self.start_at is not None and self.t >= self.start_at:
            for r in self.players:
                if r.removed_until is None and r.return_at is None:
                    r.paused = False
            self.start_at = None

        if self.ticks % CAMERA_EVERY == 0 and self.fast:
            for r in self.players:
                v = r.bot.vision
                v.detections, v.last_res = fast_detect(self, r, self.fast_ignore), None
                v.frame_id += 1
        elif self.ticks % CAMERA_EVERY == 0:
            img = self.draw_dynamic()
            for r in self.players:
                v = r.bot.vision
                det, res = analyse_frame(v.rv, self.camera(img, r), time.monotonic(), r.bot.imu.compass())
                v.detections, v.last_res = det, res
                v.frame_id += 1
                r.bot.loc.process(det)
        if self.ticks % PCB_EVERY == 0:
            for r in self.players:
                r.bot.pcb.colours = self.ldrs(r)
                r.bot.pcb.ir = self.ir(r)
        if self.noise:
            for r in self.players:
                r.bot.imu.drift += self.rng.normal(0, 0.0005)

        for r in self.players:
            if self.faults is not None:
                self.faults.tick(r)
            else:
                r.brain.tick(r.paused)
            r.drive(DT)
        if self.ticks % CAMERA_EVERY == 0:
            self.ball_stats()
        self.move_opponents()
        self.physics()
        self.bookkeeping()

    def move_opponents(self):
        for o in self.opponents:
            if isinstance(o, SimRobot) or o.removed_until is not None:
                continue
            if self.opp_ai and self.start_at is None and self.ball_away_until is None:
                bx, by = self.ball
                held = self.held_by is not None and self.held_by.team == "us"
                if self.opp_style == "press" and held:
                    tx, ty = bx, by
                else:
                    tx, ty = bx, by + o.r + BALL_RADIUS - 2
                    if o.y < by:
                        tx += math.copysign(25, o.x - bx or 1)
                        ty = by + 25
                d = math.hypot(tx - o.x, ty - o.y)
                top = OPP_PRESS_SPEED if self.opp_style == "press" else OPP_SPEED
                sp = min(top, d * 4)
                o.vx, o.vy = ((tx - o.x) / d * sp, (ty - o.y) / d * sp) if d > 1e-6 else (0, 0)
            else:
                o.vx = o.vy = 0.0

    def physics(self):
        bodies = self.on_field
        for b in bodies:
            b.x += b.vx * DT
            b.y += b.vy * DT
            b.h = wrap_pi(b.h + b.w * DT)
            b.x = min(max(b.x, -field.WALL_W / 2 + b.r), field.WALL_W / 2 - b.r)
            b.y = min(max(b.y, -field.WALL_L / 2 + b.r), field.WALL_L / 2 - b.r)
            b.x, b.y, _ = push_out_of_walls(b.x, b.y, b.r)
        for i, a in enumerate(bodies):
            for b in bodies[i + 1:]:
                dx, dy = b.x - a.x, b.y - a.y
                d = math.hypot(dx, dy)
                if 1e-6 < d < a.r + b.r:
                    if a.team != b.team:
                        a.opp_contact_t = b.opp_contact_t = self.t
                    push = (a.r + b.r - d) / 2
                    a.x -= dx / d * push
                    a.y -= dy / d * push
                    b.x += dx / d * push
                    b.y += dy / d * push

        rest_y = NOTCH_REST_Y
        notch_x = NOTCH_HALF_W
        order = list(self.players)
        self.rng.shuffle(order)
        catchers = []
        for r in order:
            if r.removed_until is not None:
                continue
            fwd = rotate([0.0, 1.0], r.h)
            rv = rotate([r.vx, r.vy], -r.h)
            side = max(abs(rv[0]), -rv[1])
            r.prev_v = (r.vx, r.vy)
            rx, ry = rotate([self.ball[0] - r.x, self.ball[1] - r.y], -r.h)
            in_notch = abs(rx) < notch_x and 0 < ry < rest_y + 2.5
            k = r.bot.kicker
            if k is not None and k.pending:
                k.pending = False
                r.kick_flash = 0.15
                if in_notch:
                    self.ball = [r.x + fwd[0] * rest_y, r.y + fwd[1] * rest_y]
                    self.ball_v = [r.vx + fwd[0] * KICK_SPEED, r.vy + fwd[1] * KICK_SPEED]
                    if self.held_by is r:
                        self.release("kicked")
                    self.kicks += 1
                    self.last_kick = self.t
                    self.count(f"{r.team} kicks")
                    keeper = next((g for g in self.players if g.team != r.team and "goalie" in g.name), None)
                    gy = field.ATTACK_GOAL_Y if r.team == "us" else field.OWN_GOAL_Y
                    self.shots.append({"t": self.t, "team": r.team, "x": r.x, "y": r.y, "dist": math.hypot(r.x, r.y - gy),
                                       "keeper": None if keeper is None else (keeper.x, keeper.y),
                                       "keeper_state": None if keeper is None else keeper.brain.strategy.botstate,
                                       "since_restart": self.t - self.last_restart, "goal": False})
                    self.log(f"{r.name} KICK")
                    continue
            if self.held_by is r:
                if not r.bot.motors.dribbler or abs(r.w) > DRIBBLE_MAX_SPIN or side > DRIBBLE_MAX_SIDE:
                    why = ("dribbler off" if not r.bot.motors.dribbler else
                           "spun too fast" if abs(r.w) > DRIBBLE_MAX_SPIN else "moved sideways / backwards")
                    self.count(f"{r.team} lost: {why}")
                    self.release(why)
                else:
                    px, py = fwd[0] * rest_y, fwd[1] * rest_y
                    self.ball = [r.x + px, r.y + py]
                    self.ball_v = [r.vx - r.w * py, r.vy + r.w * px]
            elif self.held_by is None and r.bot.motors.dribbler and in_notch:
                relv = math.hypot(self.ball_v[0] - r.vx, self.ball_v[1] - r.vy)
                if relv < DRIBBLE_CATCH_SPEED:
                    catchers.append(r)
                else:
                    self.count(f"{r.team} bounced off dribbler")
                    self.count(f"{r.team} bounce speed sum", relv)
        if catchers and self.held_by is None:
            if len({r.team for r in catchers}) > 1:
                self.count("ball squeezed between both teams' dribblers")
            else:
                r = catchers[0]
                self.held_by, self.held_since = r, self.t
                self.last_touch = r
                self.count(f"{r.team} caught")

        if self.ball_away_until is not None:
            return
        bx, by = self.ball
        vx, vy = self.ball_v
        decay = math.exp(-BALL_FRICTION * DT)
        vx, vy = vx * decay, vy * decay
        if math.hypot(vx, vy) < 1.0:
            vx = vy = 0.0
        bx, by = bx + vx * DT, by + vy * DT
        lim_x, lim_y = field.WALL_W / 2 - BALL_RADIUS, field.WALL_L / 2 - BALL_RADIUS
        if abs(bx) > lim_x:
            bx, vx = math.copysign(lim_x, bx), -vx * BALL_BOUNCE
        if abs(by) > lim_y:
            by, vy = math.copysign(lim_y, by), -vy * BALL_BOUNCE
        bx, by, normals = push_out_of_walls(bx, by, BALL_RADIUS)
        for n in normals:
            vn = vx * n[0] + vy * n[1]
            if vn < 0:
                vx -= (1 + BALL_BOUNCE) * vn * n[0]
                vy -= (1 + BALL_BOUNCE) * vn * n[1]
        contact_order = list(bodies)
        self.rng.shuffle(contact_order)
        for b in contact_order:
            if b is self.held_by:
                continue
            hit = self.ball_overlap(b, bx, by, rest_y, notch_x)
            if hit is None:
                continue
            n, pen = hit
            self.last_touch = b
            if isinstance(b, SimRobot):
                st = b.brain.strategy
                self.last_touch_phase = f"b{st.botstate} {st.phase}"
            if self.held_by is not None and b.team != self.held_by.team:
                self.count(f"{self.held_by.team} lost: knocked by opponent")
                self.release("knocked")
            bx += n[0] * pen
            by += n[1] * pen
            px, py = bx - b.x, by - b.y
            cvx, cvy = b.vx - b.w * py, b.vy + b.w * px
            rel = (vx - cvx) * n[0] + (vy - cvy) * n[1]
            if rel < 0:
                vx -= (1 + BALL_BOUNCE * 0.5) * rel * n[0]
                vy -= (1 + BALL_BOUNCE * 0.5) * rel * n[1]
        for b in bodies:
            hit = self.ball_overlap(b, bx, by, rest_y, notch_x)
            if hit is not None:
                n, pen = hit
                b.x -= n[0] * pen
                b.y -= n[1] * pen
        self.ball, self.ball_v = [bx, by], [vx, vy]

    @staticmethod
    def ball_overlap(b, bx, by, rest_y, notch_x):
        rx, ry = rotate([bx - b.x, by - b.y], -b.h)
        if isinstance(b, SimRobot) and abs(rx) < notch_x and ry > 0:
            if ry >= rest_y:
                return None
            return rotate([0.0, 1.0], b.h), rest_y - ry
        d = math.hypot(rx, ry)
        if d >= b.r + BALL_RADIUS or d < 1e-6:
            return None
        return rotate([rx / d, ry / d], b.h), b.r + BALL_RADIUS - d

    def bookkeeping(self):
        bx, by = self.ball
        in_goal = (abs(bx) < field.GOAL_W / 2 and
                   field.ATTACK_GOAL_Y + BALL_RADIUS < abs(by) < field.GOAL_BACK_Y)
        if self.goal_timer is None and in_goal:
            who = "us" if by > 0 else "them"
            self.score[who] += 1
            for sh in reversed(self.shots):
                if sh["team"] == who and self.t - sh["t"] < 1.5:
                    sh["goal"] = True
                    break
            self.count(f"goal {who} " + ("kicked" if self.t - self.last_kick < 1.5 else "pushed"))
            if self.t - self.last_kick >= 1.5:
                keeper = next((g for g in self.players if g.team != who and "goalie" in g.name), None)
                lt = self.last_touch
                self.count(f"goal {who} pushed: last touch {lt.name if lt else '-'}, keeper botstate "
                           f"{keeper.brain.strategy.botstate if keeper else '-'}")
                if lt is not None and lt.team != who:
                    self.count(f"own goal by {lt.name} in {self.last_touch_phase}")
            self.log(f"GOAL for {who}  ({self.score['us']}-{self.score['them']})")
            self.last_scorer = who
            self.goal_timer = self.t + 1.0
        if self.goal_timer is not None and self.t >= self.goal_timer:
            self.kickoff()
            return
        if self.ball_away_until is not None:
            if self.t >= self.ball_away_until and self.goal_timer is None:
                self.ball = list(self.ball_out_at)
                self.ball_away_until = None
                self.place_ball_neutral("out")
        elif self.goal_timer is None:
            out = not field.inside_play_area(bx, by, -BALL_RADIUS)
            if math.hypot(bx - self.progress[0][0], by - self.progress[0][1]) > 5:
                self.progress = ([bx, by], self.t)
            why = ("out" if out else
                   "no progress" if self.t - self.progress[1] > NO_PROGRESS_TIME else None)
            if why:
                self.count(f"ball {why}")
                lt = self.last_touch
                self.count(f"ball {why} after {lt.name if lt else 'nobody'}")
                if why == "out":
                    if self.held_by is not None:
                        self.release("ball out")
                    self.ball_out_at = [bx, by]
                    self.ball, self.ball_v = list(BALL_PARK), [0.0, 0.0]
                    self.ball_away_until = self.t + BALL_OUT_DELAY
                    self.log(f"ball out - back in {BALL_OUT_DELAY:.0f} s")
                else:
                    self.place_ball_neutral(why)
        for b in self.robots + self.opponents:
            if b.removed_until is not None:
                if self.t >= b.removed_until and self.goal_timer is None:
                    self.put_back(b)
                continue
            out = not field.inside_play_area(b.x, b.y, -b.r)
            if out and not b.was_out:
                b.outs += 1
                if self.pushed_out(b):
                    self.push_back_in(b)
                    out = False
                else:
                    self.take_off(b)
            b.was_out = out
        self.multiple_defence()
        for r in self.players:
            if r.return_at is not None and self.t >= r.return_at:
                r.return_at = None
                if self.start_at is None:
                    r.paused = False
            pose = r.bot.loc.get()
            if not r.paused and pose.confident and r.name in self.loc_err:
                self.loc_err[r.name].append(math.hypot(pose.x - r.x, pose.y - r.y))

    def take_off(self, b):
        if self.held_by is b:
            self.release("robot taken off")
        b.removed_until = self.t + DAMAGED_TIME
        b.place(*PARK)
        b.was_out = False
        if isinstance(b, SimRobot):
            b.paused = True
        self.count(f"{b.name} taken off (out of bounds)")
        self.log(f"{b.name} OUT OF BOUNDS - off for {DAMAGED_TIME:.0f} s")

    def pushed_out(self, b):
        if self.t - b.opp_contact_t > PUSHED_TIME:
            return False
        n = (math.copysign(1.0, b.x), 0.0) if abs(b.x) > field.PLAY_W / 2 else (0.0, math.copysign(1.0, b.y))
        return b.vx * n[0] + b.vy * n[1] < PUSHED_OUT_SPEED

    def push_back_in(self, b):
        R = cfg.ROBOT_RADIUS
        x = max(-(field.PLAY_W / 2 - R), min(field.PLAY_W / 2 - R, b.x))
        y = max(-(field.PLAY_L / 2 - R), min(field.PLAY_L / 2 - R, b.y))
        b.x, b.y = x, y
        self.count(f"{b.name} pushed out, back at once")
        self.log(f"{b.name} pushed out - put straight back")

    def multiple_defence(self):
        bx, by = self.ball
        R = cfg.ROBOT_RADIUS
        for team, sgn in (("us", -1.0), ("them", 1.0)):
            def in_box(x, y):
                return abs(x) < field.PENALTY_W / 2 and sgn * y > field.ATTACK_GOAL_Y - field.PENALTY_D
            if not in_box(bx, by):
                continue
            mine = [b for b in self.on_field if b.team == team and in_box(b.x, b.y)]
            if len(mine) < 2:
                continue
            far = max(mine, key=lambda b: math.hypot(b.x - bx, b.y - by))
            y = sgn * (field.ATTACK_GOAL_Y - field.PENALTY_D - R - 3)
            x = far.x
            for dx in (0.0, 25.0, -25.0, 50.0, -50.0):
                cx = max(-(field.PLAY_W / 2 - R), min(field.PLAY_W / 2 - R, far.x + dx))
                if all(math.hypot(cx - o.x, y - o.y) > 2 * R + 2 for o in self.on_field if o is not far):
                    x = cx
                    break
            if self.held_by is far:
                self.release("multiple defence")
            far.x, far.y = x, y
            far.vx = far.vy = 0.0
            self.count(f"{far.name} moved out of the penalty area (multiple defence)")
            self.log(f"{far.name}: multiple defence - moved out of the area")

    def put_back(self, b):
        R = cfg.ROBOT_RADIUS
        sgn = -1.0 if b.team == "us" else 1.0
        y = sgn * (field.PLAY_L / 2 - R - 3)
        far_side = -math.copysign(1.0, self.ball[0]) if abs(self.ball[0]) > 1 else 1.0
        x = far_side * (field.GOAL_W / 2 + R + 8)
        for cx in (x, x + far_side * 15, -x):
            if all(math.hypot(cx - o.x, y - o.y) > 2 * R + 4 for o in self.on_field):
                x = cx
                break
        b.removed_until = None
        b.place(x, y, 0.0 if b.team == "us" else math.pi)
        if isinstance(b, SimRobot):
            b.paused = True
            b.return_at = self.t + 0.5
        self.log(f"{b.name} back on the field")

    def place_ball_neutral(self, why):
        bodies = self.on_field
        free = [p for p in field.NEUTRAL_SPOTS if all(math.hypot(p[0] - b.x, p[1] - b.y) > b.r + 10 for b in bodies)]
        spots = free or field.NEUTRAL_SPOTS
        spot = min(spots, key=lambda p: math.hypot(p[0] - self.ball[0], p[1] - self.ball[1]))
        if self.held_by is not None:
            self.release("ball replaced")
        self.ball, self.ball_v = list(spot), [0.0, 0.0]
        self.last_restart = self.t
        self.ball_out_since, self.progress = None, (list(spot), self.t)
        self.log(f"ball {why} -> neutral spot ({spot[0]:.0f},{spot[1]:.0f})")

    def kidnap(self, r):
        mx, my = field.PLAY_W / 2 - r.r, field.PLAY_L / 2 - r.r
        r.place(random.uniform(-mx, mx), random.uniform(-my, my), random.uniform(-math.pi, math.pi))
        if self.held_by is r:
            self.held_by = None
        r.bot.loc.relocalise()
        self.log(f"{r.name} picked up and moved")

FAST_POSE_NOISE = 1.5
FAST_BALL_RANGE = 200.0
FAST_LINE_RANGE = 150.0
FAST_LINE_STEP = 2.0
FAST_RELOCALISE = 0.3


class TruthLoc:

    def __init__(self, body, rng):
        self.body, self.rng = body, rng
        self.lost_until = 0.0

    def relocalise(self):
        self.lost_until = time.monotonic() + FAST_RELOCALISE

    def process(self, det):
        pass

    def get(self):
        from localisation import Pose
        t = time.monotonic()
        if t < self.lost_until:
            return Pose(0.0, 0.0, 1e3, t)
        n = self.rng.normal(0, FAST_POSE_NOISE, 2)
        sgn = 1.0 if self.body.team == "us" else -1.0
        return Pose(sgn * self.body.x + n[0], sgn * self.body.y + n[1], 3.0, t)

    def particles(self):
        return np.zeros((0, 2))


def _line_points():
    pts = []
    for a, b in field.white_line_segments():
        n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / FAST_LINE_STEP))
        for k in range(n + 1):
            pts.append((a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n))
    return np.array(pts, np.float32)


FAST_LINE_PTS = _line_points()


def _goal_parts(sgn):
    gw, y0, y1 = field.GOAL_W / 2, field.ATTACK_GOAL_Y, field.GOAL_BACK_Y
    rects = [(-gw - 5, -gw, y0, y1 + 2), (gw, gw + 5, y0, y1 + 2), (-gw - 5, gw + 5, y1, y1 + 2)]
    return [(x0, x1, min(sgn * ya, sgn * yb), max(sgn * ya, sgn * yb)) for x0, x1, ya, yb in rects]


GOAL_PARTS = {"yellow": _goal_parts(1), "blue": _goal_parts(-1)}


def fast_detect(sim, r, ignore):
    from vision import Detections, Goal, Obstacle
    rng = sim.rng
    det = Detections(time.monotonic(), r.bot.imu.compass())
    det.field_visible = not GREY_CARPET

    def to_robot(px, py):
        return rotate([px - r.x, py - r.y], -r.h)

    def hidden(p, z=0.0, exclude=None, size=0.0):
        return (-ignore["left"] < p[0] < ignore["right"] and -ignore["back"] < p[1] < ignore["front"]) or \
            bool(ray_hidden(sim, r, [p], z, exclude, size)[0])

    dx, dy = sim.ball[0] - r.x, sim.ball[1] - r.y
    d = math.hypot(dx, dy)
    in_ring = BALL_BLIND_RING > 0 and d - BALL_RADIUS < cfg.ROBOT_RADIUS + BALL_BLIND_RING
    if 1e-6 < d < FAST_BALL_RANGE and not in_ring:
        near = to_robot(sim.ball[0] - dx / d * BALL_RADIUS, sim.ball[1] - dy / d * BALL_RADIUS)
        if not hidden(to_robot(*sim.ball), 2 * BALL_RADIUS, size=BALL_RADIUS):
            s = 0.3 + 0.01 * d
            det.ball = [near[0] + rng.normal(0, s), near[1] + rng.normal(0, s)]
    for colour, parts in GOAL_PARTS.items():
        best = None
        for x0, x1, y0, y1 in parts:
            cx, cy = min(max(r.x, x0), x1), min(max(r.y, y0), y1)
            dd = math.hypot(cx - r.x, cy - r.y)
            if best is None or dd < best[0]:
                best = (dd, cx, cy)
        if best[0] < VIEW_RADIUS - 5 and not hidden(to_robot(best[1], best[2]), GOAL_HEIGHT / 2):
            sgn = 1 if colour == "yellow" else -1
            det.goals[colour] = Goal(to_robot(best[1], best[2]),
                                     to_robot(0.0, sgn * (field.GOAL_BACK_Y + 1)))
    for b in (sim.on_field if cfg.ENEMY_DETECTION else ()):
        if b is r:
            continue
        dx, dy = b.x - r.x, b.y - r.y
        d = math.hypot(dx, dy)
        if b.r < d < 90.0 and not hidden(to_robot(b.x, b.y), ROBOT_HEIGHT, b, b.r):
            near = to_robot(b.x - dx / d * b.r, b.y - dy / d * b.r)
            centre = to_robot(b.x, b.y)
            n = rng.normal(0, 1.0, 2)
            det.obstacles.append(Obstacle([near[0] + n[0], near[1] + n[1]], 2 * b.r, [centre[0] + n[0], centre[1] + n[1]]))
    rel = FAST_LINE_PTS - np.float32([r.x, r.y])
    rel = rel[(rel[:, 0] ** 2 + rel[:, 1] ** 2) < FAST_LINE_RANGE ** 2]
    if len(rel) > 250:
        rel = rel[np.linspace(0, len(rel) - 1, 250).astype(int)]
    c, s = math.cos(-r.h), math.sin(-r.h)
    pts = np.stack([rel[:, 0] * c - rel[:, 1] * s, rel[:, 0] * s + rel[:, 1] * c], axis=1)
    det.line_pts = pts[~ray_hidden(sim, r, pts)].astype(np.float32)
    return det


class Viewer:
    PANEL = 330

    def __init__(self, sim):
        self.sim = sim
        self.ui = FieldImage(UI_PPC)
        self.sel = 0
        self.drag = None
        self.mouse = (0.0, 0.0)
        self.speed = 1.0
        self.running = True
        self.show_particles = True
        self.cam_mode = "detections"
        cv2.namedWindow("simulator")
        cv2.setMouseCallback("simulator", self.on_mouse)

    def to_field(self, u, v):
        return u / UI_PPC - FieldImage.W / 2, FieldImage.L / 2 - v / UI_PPC

    def pick(self, x, y):
        s = self.sim
        if math.hypot(x - s.ball[0], y - s.ball[1]) < 8:
            return "ball"
        for b in s.robots + s.opponents:
            if math.hypot(x - b.x, y - b.y) < b.r + 3:
                return b
        return None

    def on_mouse(self, event, u, v, flags, param):
        if u >= self.ui.size[0]:
            return
        x, y = self.to_field(u, v)
        self.mouse = (x, y)
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            self.drag = (self.pick(x, y), event == cv2.EVENT_RBUTTONDOWN)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self.drag = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drag and self.drag[0] is not None:
            obj, turn = self.drag
            if obj == "ball":
                self.sim.ball, self.sim.ball_v = [x, y], [0.0, 0.0]
            elif turn:
                obj.h = math.atan2(-(x - obj.x), y - obj.y)
                obj.w = 0.0
            else:
                obj.place(x, y)

    def key(self, k):
        s = self.sim
        r = s.robots[self.sel]
        if k in (ord("q"), 27):
            return False
        if k == ord(" "):
            self.running = not self.running
        elif k == ord("."):
            self.running = False
            s.step()
        elif k in (ord("1"), ord("2")):
            self.sel = k - ord("1")
        elif k == ord("p"):
            r.paused = not r.paused
            s.start_at = None
        elif k == ord("k"):
            s.kickoff()
        elif k == ord("r"):
            s.kidnap(r)
        elif k == ord("o"):
            s.opponents.append(Body(*self.mouse, math.pi, "opp", ROBOT))
        elif k == ord("x"):
            obj = self.pick(*self.mouse)
            if obj in s.opponents:
                s.opponents.remove(obj)
        elif k == ord("a"):
            s.opp_ai = not s.opp_ai
        elif k == ord("n"):
            s.noise = not s.noise
        elif k == ord("c"):
            self.cam_mode = "masks" if self.cam_mode == "detections" else "detections"
        elif k == ord("h"):
            self.show_particles = not self.show_particles
        elif k in (ord("+"), ord("=")):
            self.speed = min(self.speed * 2, 8)
        elif k in (ord("-"), ord("_")):
            self.speed = max(self.speed / 2, 0.125)
        return True

    def draw_field(self):
        s, ui = self.sim, self.ui
        img = ui.img.copy()
        P = ui.px
        sel = s.robots[self.sel]
        if self.show_particles:
            for x, y in sel.bot.loc.particles()[::2]:
                cv2.circle(img, P(x, y), 1, (170, 170, 170), -1)
        pose = sel.bot.loc.get()
        if pose.confident:
            for o in sel.bot.world.opponents:
                cv2.circle(img, P(*o), int(cfg.ROBOT_RADIUS * UI_PPC), (0, 0, 255), 1, cv2.LINE_AA)
        for o in s.opponents:
            if o.removed_until is not None:
                continue
            cv2.circle(img, P(o.x, o.y), int(o.r * UI_PPC), ROBOT, -1)
            cv2.circle(img, P(o.x, o.y), int(o.r * UI_PPC), (0, 0, 200), 2)
        for r in s.robots:
            if r.removed_until is not None:
                continue
            c = P(r.x, r.y)
            rad = int(r.r * UI_PPC)
            cv2.circle(img, c, rad, ROBOT, -1)
            cv2.circle(img, c, rad, r.colour, 2 if r is not sel else 3)
            tip = P(r.x - 1.3 * r.r * math.sin(r.h), r.y + 1.3 * r.r * math.cos(r.h))
            cv2.line(img, c, tip, r.colour, 2)
            br = r.brain
            if r.bot.motors.dribbler:
                fx, fy = -math.sin(r.h), math.cos(r.h)
                a = P(r.x + fx * r.r + fy * 5, r.y + fy * r.r - fx * 5)
                b = P(r.x + fx * r.r - fy * 5, r.y + fy * r.r + fx * 5)
                cv2.line(img, a, b, (0, 200, 255), 3)
            if r.kick_flash > 0:
                cv2.circle(img, c, rad + 8, (255, 255, 255), 2)
                r.kick_flash -= 0.03
            if getattr(br, "line", None) is not None and br.line.on_line:
                cv2.circle(img, c, rad + 4, (0, 0, 255), 2)
            if not r.paused and getattr(br, "desired_pos", None) is not None:
                dp = br.desired_pos
                cv2.arrowedLine(img, c, P(r.x + dp[0], r.y + dp[1]), r.colour, 1, tipLength=0.08)
            pose = r.bot.loc.get()
            if pose.std < 200:
                e = P(pose.x, pose.y)
                cv2.drawMarker(img, e, r.colour, cv2.MARKER_TILTED_CROSS, 10, 2)
                cv2.circle(img, e, max(2, int(pose.std * UI_PPC)), r.colour, 1)
            label = "PAUSED" if r.paused else f"b{br.strategy.botstate} {br.strategy.phase}"
            cv2.putText(img, label, (c[0] + rad + 2, c[1] - rad), cv2.FONT_HERSHEY_PLAIN, 1.0, r.colour, 1)
        cv2.circle(img, P(*s.ball), max(2, int(BALL_RADIUS * UI_PPC)), ORANGE, -1)
        cv2.putText(img, f"t={s.t:5.1f}s  x{self.speed:g}{'' if self.running else '  PAUSED'}   "
                         f"score {s.score['us']}-{s.score['them']}   "
                         f"opp {getattr(s, 'opp_name', 'simple AI')}{'' if getattr(s, 'opp_name', 'simple AI') != 'simple AI' else (' on' if s.opp_ai else ' off')}"
                         f"   noise {'on' if s.noise else 'off'}   seed {s.seed}",
                    (8, 18), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
        return img

    def draw_panel(self, h):
        s = self.sim
        r = s.robots[self.sel]
        panel = np.full((h, self.PANEL, 3), 30, np.uint8)
        v = r.bot.vision
        det, res = v.detections, v.last_res
        y = 8
        if res is not None:
            if self.cam_mode == "masks":
                cam = np.zeros_like(res.frame)
                for name, m in res.masks.items():
                    cam[m > 0] = MASK_BGR[name]
            else:
                cam = draw_detections(res.frame.copy(), det)
            cam = cv2.resize(cam, (self.PANEL - 16, self.PANEL - 16), interpolation=cv2.INTER_NEAREST)
            panel[y:y + cam.shape[0], 8:8 + cam.shape[1]] = cam
            y += cam.shape[0] + 6
        pose = r.bot.loc.get()
        w = r.bot.world
        err = math.hypot(pose.x - r.x, pose.y - r.y)
        e = s.loc_err[r.name]
        lines = [
            f"[{self.sel + 1}] {r.name}  ("
            f"{f'OFF (damaged) {r.removed_until - s.t:.0f}s' if r.removed_until is not None else 'PAUSED' if r.paused else 'running'})"
            f"  camera: {self.cam_mode}",
            f"true  ({r.x:6.1f},{r.y:6.1f}) {math.degrees(r.h):6.1f}deg",
            f"est   ({pose.x:6.1f},{pose.y:6.1f}) +-{pose.std:4.1f}  err {err:4.1f}",
            f"mean err {np.mean(e) if e else 0:4.1f}cm  outs {r.outs}",
            f"ball {None if w.ball is None else [round(c) for c in w.ball]} ({w.ball_source}) capture {w.ball_in_capture}",
            f"dribbler {'ON' if r.bot.motors.dribbler else 'off'}{' (none)' if not r.bot.motors.has_dribbler else ''}"
            f"  holding {s.held_by is r}  kicker {'none' if r.bot.kicker is None else ('ready' if r.bot.kicker.ready() else 'charging')}",
            f"goals {w.goal_source}  obstacles {len(w.obstacles)}",
            f"b{r.brain.strategy.botstate} {r.brain.strategy.phase}  line {'ON ' + r.brain.line.source if getattr(r.brain, 'line', None) and r.brain.line.on_line else 'off'}",
            f"comms {r.bot.comms.my_state.get('command', '-')}",
            "",
        ] + s.logs + ["", "space pause  . step  1/2 select  p switch", "k kickoff  r kidnap  o/x opp  a opp AI",
                      "n noise  c camera  h particles  +/- speed"]
        for line in lines:
            cv2.putText(panel, line[:52], (8, y + 12), cv2.FONT_HERSHEY_PLAIN, 0.9, (230, 230, 230), 1)
            y += 15
        return panel

    def loop(self):
        wall_last = time.perf_counter()
        sim_debt = 0.0
        while True:
            now = time.perf_counter()
            if self.running:
                sim_debt += min(now - wall_last, 0.1) * self.speed
                while sim_debt >= DT:
                    self.sim.step()
                    sim_debt -= DT
                    if time.perf_counter() - now > 0.05:
                        sim_debt = 0.0
                        break
            wall_last = now
            fieldimg = self.draw_field()
            frame = np.hstack([fieldimg, self.draw_panel(fieldimg.shape[0])])
            cv2.imshow("simulator", frame)
            k = cv2.waitKey(15) & 0xFF
            if k != 255 and not self.key(k):
                break
            if cv2.getWindowProperty("simulator", cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()


RV_FOR_DRAW = []


def draw_detections(img, det):
    rv = RV_FOR_DRAW[0]
    px = lambda p: tuple(int(round(v)) for v in rv.to_px(p[0], p[1]))
    x1, y1, x2, y2 = rv.ignore_box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cz = cfg.CAPTURE_ZONE
    cv2.rectangle(img, px((cz[0], cz[3])), px((cz[1], cz[2])), (255, 255, 0), 1)
    if det.hull_px is not None:
        cv2.polylines(img, [det.hull_px], True, (0, 255, 0), 1)
    for p in det.line_pts:
        cv2.circle(img, px(p), 1, (255, 0, 255), -1)
    for name, colour in (("yellow", (0, 255, 255)), ("blue", (255, 0, 0))):
        g = det.goals[name]
        if g is not None:
            cv2.drawMarker(img, px(g.near), colour, cv2.MARKER_CROSS, 10, 2)
    for o in det.obstacles:
        cv2.circle(img, px(o.near), 3, (0, 0, 255), -1)
        if getattr(o, "centre", None) is not None:
            cv2.circle(img, px(o.centre), int(round(cfg.ROBOT_RADIUS * rv.px_per_cm)), (0, 0, 255), 2)
    if det.ball is not None:
        cv2.circle(img, px(det.ball), 6, (0, 140, 255), 2)
    return img


def headless(sim, seconds):
    sim.opp_ai = True
    t0 = time.perf_counter()
    states = {r.name: {} for r in sim.robots}
    while sim.t < seconds:
        sim.step()
        for r in sim.robots:
            b = r.brain.strategy.botstate
            states[r.name][b] = states[r.name].get(b, 0) + 1
    wall = time.perf_counter() - t0
    print(f"simulated {seconds:.0f} s in {wall:.1f} s ({seconds / wall:.1f}x real time)")
    print(f"score us-them: {sim.score['us']}-{sim.score['them']}   kicks {sim.kicks}   camera sees {VIEW_RADIUS:.0f} cm "
          f"(omni mirror, viewpoint {VIEW_Z:.1f} cm up, {CAM_RES} px, {'radial' if RADIAL else 'top-down'})")
    for r in sim.robots:
        e = sim.loc_err[r.name]
        tot = sum(states[r.name].values())
        share = ", ".join(f"b{k} {100 * v / tot:.0f}%" for k, v in sorted(states[r.name].items()))
        print(f"{r.name:8s} outs {r.outs}  localisation error mean {np.mean(e) if e else float('nan'):.1f} cm "
              f"max {max(e) if e else float('nan'):.1f} cm  states: {share}")
    if sim.faults is not None:
        print(sim.faults.summary(), f" tick errors {sim.faults.tick_errors}")
    print("log:", *sim.logs, sep="\n  ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", type=float, metavar="SECONDS", help="no window: simulate this long and print a summary")
    ap.add_argument("--noise", action="store_true", help="start with sensor noise on")
    ap.add_argument("--fast", action="store_true", help="no camera simulation: detections + pose from the truth")
    ap.add_argument("--seed", type=int, default=0, help="game seed: with --noise this replays arena.py's game of the "
                                                        "same seed (that side round: our code as 'us')")
    ap.add_argument("--opp", metavar="VERSION", help="opponents run this version of our code (a name from "
                                                     "past_versions/, e.g. v26 or comp, or a folder) instead of the "
                                                     "simple AI; 'press' = the simple AI pressing our ball carrier")
    ap.add_argument("--faults", type=float, default=0, metavar="RATE", help="break our robots on purpose (camera "
                    "freezes, comms drops, IMU jumps, crashes, restarts - see faults.py): 1 = default rates, 2 = twice as often")
    args = ap.parse_args()
    simple = args.opp in (None, "ai", "press")
    sim = Sim(noise=args.noise, seed=args.seed, fast=args.fast, opp_code=None if simple else load_code(args.opp))
    if args.faults:
        from faults import FaultInjector
        sim.faults = FaultInjector(sim, args.faults, args.seed)
    if args.opp == "press":
        sim.opp_style, sim.opp_ai = "press", True
    sim.opp_name = args.opp or "simple AI"
    RV_FOR_DRAW.append(sim.rv)
    if args.headless:
        headless(sim, args.headless)
    else:
        Viewer(sim).loop()


if __name__ == "__main__":
    sys.exit(main())

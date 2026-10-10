#!/usr/bin/env python3
"""
simulator.py - test the robots' logic on a simulated field, on a laptop (no Pi needed).

    python simulator.py                    # window with the field, both robots, two opponents
    python simulator.py --headless 30      # no window: simulate 30 s and print a summary (quick regression test)
    python simulator.py --headless 30 --fast   # same without the camera simulation (see FAST MODE), ~10x faster

It runs main.GoalieBrain and main_attack.StrikerBrain - the SAME code that runs on the robots: camera detection,
localisation, perception, line fusion, strategy and motion. Only the hardware is replaced:

    camera   the field seen through a CAM_FOV_DEG fisheye lens CAM_HEIGHT cm up, pointing straight down, at
             CAM_RES pixels (so things far away get blurry like on the real camera), turned into the top-down
             view, then the real robot_vision.py masks -> detection.py -> localisation.py
    LDRs     32 readings: white line under the sensor = low reading, green = high
    IMU      the true heading (+ drift with noise on)
    comms    the two robots' my_state handed straight to each other
    switch   per robot, P key
    motors   the 4 motor speeds -> robot movement with an ideal omni-wheel model
    dribbler holds the ball in the capture notch while on (loses it when turning / accelerating too hard, or
             when an opponent touches it)
    kicker   fires the ball forward at KICK_SPEED, with the cooldown from hardware_*.py
    goals    solid side and back walls (GOAL_WALL_T thick): the ball only gets in through the front
    referee  ball fully out of the playing area -> nearest free neutral spot at once (3 spots on the halfway line);
             a robot fully out -> taken off as damaged for DAMAGED_TIME s or until the next kickoff
             (a robot only gets a dribbler / kicker if its hardware_attack.py / hardware_defense.py has one)

NOT simulated (so test these on the real robot): lens distortion / top-down calibration errors, tall objects
stretching in the top-down view, robots hiding things behind them, real lighting, motor wiring/sign mistakes
(the sim assumes VelocityToMotor's output moves the robot the way the code intends), real dribbler grip.

Mouse:  left-drag  move the ball / a robot / an opponent
        right-drag turn a robot / opponent to face the mouse
Keys:   space pause sim     .  step one tick       1 / 2  select goalie / striker
        p  toggle the selected robot's switch (paused <-> running)
        k  kickoff positions    r  pick up the selected robot and drop it somewhere random (relocalisation test)
        o  add opponent at mouse    x  remove opponent at mouse    a  opponents chase the ball on/off
        n  sensor noise on/off      c  camera panel: detections / masks    h  particles on/off
        + / -  sim speed            q / Esc  quit
"""
import argparse
import math
import os
import random
import sys
import time

import cv2
import numpy as np

# ---- simulated clock: every module calls time.monotonic(), so the whole robot code runs on sim time
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
from robot_vision import RobotVision  # noqa: E402
from utils import rotate, wrap_pi  # noqa: E402
from vision import analyse_frame  # noqa: E402

import common as _common, lines as _lines, localisation as _localisation, main as _main  # noqa: E402,E401
import main_attack as _main_attack, motion as _motion, perception as _perception  # noqa: E402,E401

# everything SimBot builds from the robot code, so an older version of it can play against this one
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
    """A version name from past_versions/ (e.g. "v26"), or a path to any folder with the robot code."""
    for folder in (name, os.path.join(PAST_VERSIONS, name)):
        if os.path.isfile(os.path.join(folder, "strategy.py")):
            return folder
    have = sorted(os.listdir(PAST_VERSIONS)) if os.path.isdir(PAST_VERSIONS) else []
    raise SystemExit(f"no robot code found for '{name}' (past versions: {', '.join(d for d in have if d != 'README.md')})")


def load_code(folder):
    """Import another copy of the robot code (a folder with the same .py files: one of past_versions/, or e.g. an
    old commit from `git worktree add`) next to the current one, without the two mixing. Returns a Code."""
    import importlib
    folder = code_folder(folder)
    saved = {k: sys.modules.pop(k) for k in CODE_MODULES if k in sys.modules}
    sys.path.insert(0, os.path.abspath(folder))
    try:
        mods = {k: importlib.import_module(k) for k in CODE_MODULES   # (older versions don't have every file)
                if os.path.isfile(os.path.join(folder, k + ".py"))}
    finally:
        sys.path.pop(0)
        for k in CODE_MODULES:
            sys.modules.pop(k, None)
        sys.modules.update(saved)
    return Code(os.path.basename(os.path.normpath(folder)), mods)

# ==================================== SIM SETTINGS ====================================
DT = 0.01                    # physics + control tick (the robots' main loop is 100 Hz)
CAMERA_EVERY = 3             # camera frame every 3 ticks = 33 fps (set to what test_localisation.py shows)
PCB_EVERY = 4                # LDR update every 4 ticks = 25 Hz
CAM_FOV_DEG = 160.0          # circular fisheye, full angle. 140 deg at 20 cm only sees 55 cm around the robot
CAM_HEIGHT = 20.0            # cm, lens above the floor, pointing straight down, above the robot centre
CAM_RES = 240                # px across the fisheye image circle (= the capture's short side). Try 480.
VIEW_RADIUS = CAM_HEIGHT * math.tan(math.radians(CAM_FOV_DEG / 2))   # cm of floor the lens sees (113 at 160 deg)
TOPDOWN_PPC = 2.0            # px/cm of the top-down view handed to robot_vision. Use >= 2 on the real robot too:
                             # at 1 px/cm a 4 cm ball is so small the noise filter removes it
CAM_SIZE = int(2 * VIEW_RADIUS * TOPDOWN_PPC)

TOP_SPEED = 200.0            # cm/s when the motors get BASE_SPEED - MEASURE on the real robot
TOP_SPIN = 6.0               # rad/s spinning on the spot at BASE_SPEED - MEASURE
MOTOR_TAU = 0.08             # s, how quickly the robot reaches the commanded speed

BALL_RADIUS = cfg.BALL_RADIUS   # robot_config.py
BALL_FRICTION = 0.8          # 1/s velocity decay
BALL_BOUNCE = 0.5            # wall restitution
GOAL_WALL_T = 2.0            # cm, thickness of the goal's side and back walls (solid: the ball only gets in the front)
OPP_SPEED = 60.0             # cm/s, opponents chasing the ball

KICK_SPEED = 250.0           # cm/s the kicker gives the ball - MEASURE
DRIBBLE_MAX_SPIN = 6.0       # rad/s: turning faster than this while dribbling loses the ball
DRIBBLE_MAX_ACCEL = 600.0    # cm/s^2: accelerating / braking harder than this loses the ball
DRIBBLE_CATCH_SPEED = 120.0  # cm/s: a ball arriving faster than this bounces off the dribbler instead

# referee: the ball fully outside the playing area -> off the field for BALL_OUT_DELAY s (no ball), then on the
# free neutral spot nearest to where it went out
BALL_OUT_DELAY = 2.0
BALL_PARK = (1e4, 0.0)       # where the ball is kept while it's off the field (not drawn, not seen, no physics)
NO_PROGRESS_TIME = 10.0      # referee: ball hasn't moved 5 cm in this long -> nearest free neutral spot
DAMAGED_TIME = 30.0          # referee: a robot fully outside the playing area is taken off as damaged for this long
                             # (or until the next kickoff), then put back in its own half
PUSHED_TIME = 0.5            # referee: a robot that goes out within this long of touching an opponent, while not itself
PUSHED_OUT_SPEED = 20.0      # driving outwards faster than this (cm/s), was pushed out: it comes straight back in
PARK = (1e4, 1e4)            # where a taken-off robot is kept (far off the field: not drawn, not seen, no physics)

LDR_RING_RADIUS = cfg.ROBOT_RADIUS - 3.0
# Where LDR number i physically sits = the angle lines.py computes for it + this offset.
# The comp code escaped lines correctly using the LDR vector as the escape direction, which only works if the
# sensors are physically opposite to the computed angle, hence pi. Set 0 to test the layout the comments describe.
LDR_PHYSICAL_OFFSET = math.pi
LDR_WHITE, LDR_GREEN = 800, 3200

FIELD_PPC = 2.0              # sim field image resolution (px/cm)
UI_PPC = 2.4                 # window scale (px/cm)
MARGIN = 10.0                # cm drawn around the walls

GREEN, WHITE, WALL = (40, 150, 40), (250, 250, 250), (25, 25, 25)
YELLOW, BLUE, ORANGE, ROBOT = (0, 220, 255), (220, 70, 0), (0, 120, 255), (45, 45, 45)

MASK_BGR = {"blue": (255, 90, 0), "yellow": (0, 255, 255), "white": (255, 255, 255),   # masks view
            "green": (0, 200, 0), "orange": (0, 140, 255)}

# HSV ranges that match the colours above (sim only - the real ones come from test_camera.py)
SIM_HSV = {
    "blue": {"lo": [100, 150, 80], "hi": [130, 255, 255]},
    "yellow": {"lo": [20, 100, 100], "hi": [35, 255, 255]},
    "white": {"lo": [0, 0, 200], "hi": [179, 40, 255]},
    "green": {"lo": [40, 80, 50], "hi": [85, 255, 255]},
    "orange": {"lo": [5, 150, 150], "hi": [18, 255, 255]},
}

HERE = os.path.dirname(os.path.abspath(__file__))


def sim_vision_config():
    """robot_vision config for the sim camera. Uses the ignore box from your real config if there is one."""
    ignore = {"left": 11.0, "right": 11.0, "back": 11.0, "front": 5.0}
    real = os.path.join(HERE, cfg.VISION_CONFIG)
    if os.path.exists(real):
        import json
        with open(real) as f:
            ignore = json.load(f)["detection"].get("ignore_cm") or ignore
    return {
        "version": 1, "capture_size": [CAM_SIZE, CAM_SIZE], "rotation": "NONE",
        "camera": {"controls": {}}, "hsv": SIM_HSV,
        "detection": {"ignore_cm": ignore, "min_blob_area": 50, "morph_kernel": 3},
        "lens": None, "undistort_view": {"balance": 1, "fov_scale": 1}, "topdown": None,
        "robot_centre": None, "heading": {"align_to_field": False, "invert": False, "offset_deg": 0},
    }


def make_rv():
    import json
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "sim_vision_config.json")
    with open(path, "w") as f:
        json.dump(sim_vision_config(), f)
    rv = RobotVision(path)
    # "raw" mode means 1 px/cm; the sim's top-down view is TOPDOWN_PPC, like a real top-down setup would be
    rv.px_per_cm = TOPDOWN_PPC
    ign, (cx, cy) = rv.config["detection"]["ignore_cm"], rv.centre
    box = (cx - ign["left"] * TOPDOWN_PPC, cy - ign["front"] * TOPDOWN_PPC,
           cx + ign["right"] * TOPDOWN_PPC, cy + ign["back"] * TOPDOWN_PPC)
    rv.ignore_box = tuple(max(int(round(v)), 0) for v in box)
    fov = np.zeros((CAM_SIZE, CAM_SIZE), np.uint8)
    cv2.circle(fov, (CAM_SIZE // 2, CAM_SIZE // 2), int(VIEW_RADIUS * TOPDOWN_PPC) - 3, 255, -1)
    rv.valid = cv2.erode(fov, np.ones((5, 5), np.uint8))   # like the real top-down "camera sees this" mask
    return rv, fov


class FisheyeCamera:
    """Equidistant fisheye (angle from straight down proportional to distance from the image centre).
    Rendering is two steps so nothing per-pixel is computed per frame: (1) one affine warp of the field image into
    the robot's frame (rotation + position), (2) a fixed remap from there to the fisheye pixels (precomputed).
    Then the fisheye image -> top-down view with another fixed remap, like robot_vision's calibration."""
    SS = 2   # supersampling when rendering the fisheye image (pixels average over their area, like a real sensor)

    def __init__(self):
        half = math.radians(CAM_FOV_DEG / 2)
        n = CAM_RES * self.SS
        c = n / 2.0
        j, i = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5)
        dx, dy = (j - c) / c, (c - i) / c                 # -1..1, +dy = robot front
        rho = np.hypot(dx, dy)
        theta = np.minimum(rho * half, math.radians(89.5))
        r = CAM_HEIGHT * np.tan(theta)
        with np.errstate(invalid="ignore", divide="ignore"):
            gx = np.where(rho > 0, r * dx / rho, 0)        # floor point, robot frame cm
            gy = np.where(rho > 0, r * dy / rho, 0)
        # (1) robot-frame floor image at FIELD_PPC, robot in the middle, front = up
        k = FIELD_PPC
        self.half_px = int(math.ceil(VIEW_RADIUS * k)) + 4
        self.ego_size = 2 * self.half_px
        # (2) fisheye pixel -> robot-frame image pixel; outside the lens circle -> -1 (black)
        ex = (self.half_px + gx * k).astype(np.float32)
        ey = (self.half_px - gy * k).astype(np.float32)
        ex[rho > 1.0] = -10
        ey[rho > 1.0] = -10
        self.lens_maps = cv2.convertMaps(ex, ey, cv2.CV_16SC2)
        # top-down (TOPDOWN_PPC px/cm, CAM_SIZE square, robot at the centre) -> fisheye pixel (at CAM_RES);
        # outside the "camera sees this" circle -> -1 (black), same circle as make_rv's fov mask
        m = CAM_SIZE
        u, v = np.meshgrid(np.arange(m) + 0.5, np.arange(m) + 0.5)
        x, y = (u - m / 2.0) / TOPDOWN_PPC, (m / 2.0 - v) / TOPDOWN_PPC
        rr = np.hypot(x, y)
        rho_t = np.arctan(rr / CAM_HEIGHT) / half
        cr = CAM_RES / 2.0
        with np.errstate(invalid="ignore", divide="ignore"):
            map_x = np.where(rr > 0, cr + rho_t * cr * x / rr, cr).astype(np.float32)
            map_y = np.where(rr > 0, cr - rho_t * cr * y / rr, cr).astype(np.float32)
        fov = np.zeros((m, m), np.uint8)
        cv2.circle(fov, (m // 2, m // 2), int(VIEW_RADIUS * TOPDOWN_PPC) - 3, 255, -1)
        map_x[fov == 0] = -10
        map_y[fov == 0] = -10
        self.top_maps = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)

    def render(self, field_img, robot, noise_rng=None):
        """Raw fisheye image (CAM_RES square) as the camera would see the (flat) field from this robot."""
        c, s = math.cos(robot.h), math.sin(robot.h)
        k, h = FIELD_PPC, self.half_px
        # robot-frame pixel (u, v) -> field image pixel (inverse map for warpAffine)
        M = np.float32([[c, s, (FieldImage.W / 2 + robot.x) * k - c * h - s * h],
                        [-s, c, (FieldImage.L / 2 - robot.y) * k + s * h - c * h]])
        ego = cv2.warpAffine(field_img, M, (self.ego_size, self.ego_size), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=WALL)
        img = cv2.remap(ego, *self.lens_maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        img = cv2.resize(img, (CAM_RES, CAM_RES), interpolation=cv2.INTER_AREA)
        if noise_rng is not None:      # pixel noise: a random pick from a bank of noise frames, randomly shifted
            if not hasattr(self, "noise_bank"):
                self.noise_bank = [noise_rng.normal(0, 8, img.shape).astype(np.int16) for _ in range(16)]
            n = self.noise_bank[int(noise_rng.integers(len(self.noise_bank)))]
            n = np.roll(n, (int(noise_rng.integers(CAM_RES)), int(noise_rng.integers(CAM_RES))), axis=(0, 1))
            img = cv2.add(img, n, dtype=cv2.CV_8U)
        return img

    def topdown(self, raw):
        return cv2.remap(raw, *self.top_maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ==================================== FIELD DRAWING ====================================
class FieldImage:
    W = field.WALL_W + 2 * MARGIN
    L = field.WALL_L + 2 * MARGIN

    def __init__(self, ppc):
        self.ppc = ppc
        self.size = (int(round(self.W * ppc)), int(round(self.L * ppc)))
        img = np.full((self.size[1], self.size[0], 3), WALL, np.uint8)
        cv2.rectangle(img, self.px(-field.WALL_W / 2, field.WALL_L / 2), self.px(field.WALL_W / 2, -field.WALL_L / 2), GREEN, -1)
        # goals: coloured walls, and the back wall's colour reaching outwards (that's how the stretched
        # top-down view of a 10 cm tall wall looks); the goal floor is green with the white line on it
        for sgn, col in ((1, YELLOW), (-1, BLUE)):
            gy0, gy1 = field.PLAY_L / 2, field.PLAY_L / 2 + field.GOAL_DEPTH
            gw, t = field.GOAL_W / 2, 5.0   # side walls look wide: the top-down view stretches tall walls
            cv2.rectangle(img, self.px(-gw - t, sgn * gy0), self.px(gw + t, sgn * (field.WALL_L / 2)), col, -1)
            cv2.rectangle(img, self.px(-gw, sgn * gy0), self.px(gw, sgn * gy1), GREEN, -1)
        self.white = np.zeros(img.shape[:2], np.uint8)
        thick = max(1, int(round(field.LINE_W * ppc)))
        for a, b in field.black_line_segments():     # penalty box: black, the LDRs and white mask ignore it
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
    """The goals' side and back walls as solid rectangles (x0, x1, y0, y1), both ends."""
    hw, t = field.GOAL_W / 2, GOAL_WALL_T
    y0, y1 = field.PLAY_L / 2, field.PLAY_L / 2 + field.GOAL_DEPTH
    walls = []
    for sgn in (1, -1):
        for x0, x1, ya, yb in ((-hw - t, -hw, y0, y1 + t), (hw, hw + t, y0, y1 + t), (-hw - t, hw + t, y1, y1 + t)):
            walls.append((x0, x1, min(sgn * ya, sgn * yb), max(sgn * ya, sgn * yb)))
    return walls


GOAL_WALLS = goal_walls()


def push_out_of_walls(x, y, r):
    """Move a circle (centre x, y, radius r) out of the goal walls. Returns (x, y, list of push-out normals)."""
    normals = []
    for x0, x1, y0, y1 in GOAL_WALLS:
        cx, cy = min(max(x, x0), x1), min(max(y, y0), y1)      # closest point of the wall
        dx, dy = x - cx, y - cy
        d = math.hypot(dx, dy)
        if d >= r:
            continue
        if d > 1e-9:
            n = (dx / d, dy / d)
            pen = r - d
        else:                                                   # centre inside the wall: leave by the nearest side
            pen, n = min((x - x0 + r, (-1, 0)), (x1 - x + r, (1, 0)), (y - y0 + r, (0, -1)), (y1 - y + r, (0, 1)))
        x, y = x + n[0] * pen, y + n[1] * pen
        normals.append(n)
    return x, y, normals


# ==================================== SIM HARDWARE ====================================
class SimIMU:
    def __init__(self, body):
        self.body = body
        self.heading_offset = 0.0
        self.drift = 0.0

    @property
    def heading(self):
        return self.body.h + self.drift

    def zero(self):
        self.heading_offset = self.heading

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
        self.pending = False      # physics fires the ball on the next step

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
        self.colours = [LDR_GREEN] * cfg.LDR_COUNT

    def snapshot(self):
        return list(self.colours)

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
    """Same interface as common.Robot (so the brains can't tell the difference), with sim hardware.
    code = the robot code to run (CURRENT_CODE, or an older version from load_code())."""

    def __init__(self, body, rv, hw, code):
        self.imu = SimIMU(body)
        self.motors = SimMotors(hw.DRIBBLER is not None)
        self.kicker = None if hw.KICKER_PIN is None else SimKicker(hw.KICK_COOLDOWN)
        self.pcb = SimPCB()
        self.comms = SimComms()
        self.vision = SimVision(rv)
        self.attack = "yellow"
        self.loc = code.LocalisationThread(self.vision, lambda: self.attack)   # never started: process() per frame
        self.world = code.World()
        self.lines = code.LineFusion()
        self.mover = code.Mover()
        self.leds = code.LedCalibrator(self.pcb)


# ==================================== BODIES / PHYSICS ====================================
SQ2 = math.sqrt(2)
MOTOR_ANGLES = [math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4]


def motors_to_velocity(speeds):
    """Inverse of motion.VelocityToMotor + motion.Mover's axis swap: 4 motor speeds -> robot-frame (vx, vy) cm/s
    and spin rad/s (CCW +)."""
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
        self.removed_until = None    # taken off as damaged until this sim time (None = on the field)
        self.opp_contact_t = -1e9    # last time an opponent pushed against it
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
        self.return_at = None        # just put back after being taken off: stays paused (calibrating) till then

    def drive(self, dt):
        vx_r, vy_r, spin = motors_to_velocity(self.bot.motors.speeds)
        tvx, tvy = rotate([vx_r, vy_r], self.h)
        a = min(dt / MOTOR_TAU, 1.0)
        self.vx += (tvx - self.vx) * a
        self.vy += (tvy - self.vy) * a
        self.w += (spin - self.w) * a


class Sim:
    def __init__(self, noise=False, seed=0, opp_code=None, fast=False):
        """seed: changes the sensor noise, particle filters and (seed > 0) jitters the kickoff positions a little,
        so different seeds give different games. opp_code: None = the simple opponents (key a), or a robot code
        version from load_code() = the opponents run that code (self-play), attacking the blue goal.
        fast: no camera rendering / detection / localisation - Detections and pose straight from the truth (see
        FAST MODE below), ~10x faster, for screening strategy ideas."""
        self.seed = seed
        self.fast = fast
        self.fimg = FieldImage(FIELD_PPC)
        self.rv, self.fov = make_rv()
        self.cam = FisheyeCamera()
        self.held_by = None       # robot whose dribbler has the ball
        self.held_since = 0.0
        self.kicks = 0
        self.events = {}
        self.last_touch = None    # robot that touched the ball last
        self.last_touch_phase = ""
        self.last_kick = -1e9
        self.shots = []
        self.last_restart = 0.0   # time of the last kickoff / neutral spot placement           # every kick: who, from where, did it go in (within 1.5 s)          # what happened to the ball (counts), for arena.py / tuning
        self.ball_out_since = None
        self.progress = ([0.0, 0.0], 0.0)   # (ball position, time) last time it moved 5 cm
        self.logs = []
        self.robots = [
            SimRobot(0, 0, 0, "goalie", (255, 200, 0), GoalieBrain, hardware_defense, self.rv, self.log),
            SimRobot(0, 0, 0, "striker", (255, 0, 255), StrikerBrain, hardware_attack, self.rv, self.log),
        ]
        g, s = self.robots
        g.bot.comms.partner, s.bot.comms.partner = s.bot.comms, g.bot.comms
        if opp_code is None:
            self.opponents = [Body(0, 0, math.pi, "opp", ROBOT), Body(0, 0, math.pi, "opp", ROBOT)]
        else:      # same order as the kickoff spots: striker at the centre, goalie in front of the yellow goal
            self.opponents = [
                SimRobot(0, 0, math.pi, "opp striker", ROBOT, opp_code.StrikerBrain, hardware_attack, self.rv,
                         self.log, opp_code, "them"),
                SimRobot(0, 0, math.pi, "opp goalie", ROBOT, opp_code.GoalieBrain, hardware_defense, self.rv,
                         self.log, opp_code, "them"),
            ]
            o_s, o_g = self.opponents
            o_s.bot.comms.partner, o_g.bot.comms.partner = o_g.bot.comms, o_s.bot.comms
        for i, r in enumerate(self.players):        # fixed seeds: the same settings give the same game every run
            r.bot.loc.loc.rng = np.random.default_rng(100 + i + 10 * seed)
            if fast:
                r.bot.loc = TruthLoc(r, np.random.default_rng(200 + i + 10 * seed))
        self.fast_ignore = self.rv.config["detection"]["ignore_cm"]
        self.ball = [0.0, 0.0]
        self.ball_v = [0.0, 0.0]
        self.ball_away_until = None   # ball off the field (gone out) until this time
        self.ball_out_at = None       # where it went out
        self.noise = noise
        self.opp_ai = False
        self.ticks = 0
        self.score = {"us": 0, "them": 0}
        self.goal_timer = None
        self.loc_err = {r.name: [] for r in self.robots}
        self.rng = np.random.default_rng(seed)
        self.kickoff()

    @property
    def players(self):
        """Every robot running robot code (ours, plus the opponents in self-play)."""
        return self.robots + [o for o in self.opponents if isinstance(o, SimRobot)]

    def release(self, why):
        """The ball leaves the dribbler. Possessions longer than 0.3 s are counted by how they ended."""
        if self.held_by is not None and self.t - self.held_since > 0.3:
            self.count(f"{self.held_by.team} possession ended: {why}")
            gy = self.held_by.y if self.held_by.team == "us" else -self.held_by.y     # towards the goal it attacks
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
        for b in self.robots + self.opponents:     # a new point: taken-off robots come back
            b.removed_until = None
            if isinstance(b, SimRobot):
                b.return_at = None
        g, s = self.robots
        R = cfg.ROBOT_RADIUS
        g.place(0, field.OWN_GOAL_Y + R + 8, 0)
        s.place(0, -(R + 15), 0)
        for o, (x, y) in zip(self.opponents, ((0, R + 15), (0, field.ATTACK_GOAL_Y - R - 8))):
            o.place(x, y, math.pi)
        self.ball, self.ball_v = [0.0, 0.0], [0.0, 0.0]
        self.ball_away_until = None       # a kickoff always puts the ball back on the centre spot
        self.ball_out_at = None
        if self.seed:                          # a little variety between games
            for b in self.robots + self.opponents:
                b.place(b.x + self.rng.uniform(-3, 3), b.y + self.rng.uniform(-3, 3), b.h + self.rng.uniform(-0.1, 0.1))
            self.ball = [self.rng.uniform(-3, 3), self.rng.uniform(-3, 3)]
        self.held_by = None
        self.ball_out_since = None
        self.progress = (list(self.ball), self.t)
        for r in self.players:
            r.paused = True
        self.last_restart = self.t + 0.5
        self.start_at = self.t + 0.5      # stand still (paused) half a second: calibrates heading + goal colour
        self.goal_timer = None

    # ---- sensors
    @property
    def on_field(self):
        """Every body (ours and opponents) that isn't taken off."""
        return [b for b in self.robots + self.opponents if b.removed_until is None]

    def draw_dynamic(self):
        img = self.fimg.img.copy()
        ppc = FIELD_PPC
        for b in self.on_field:
            cv2.circle(img, self.fimg.px(b.x, b.y), int(b.r * ppc), ROBOT, -1)
        cv2.circle(img, self.fimg.px(*self.ball), max(1, int(round(BALL_RADIUS * ppc))), ORANGE, -1)
        return img

    def camera(self, img, robot):
        """Fisheye image from the robot -> top-down robot-frame image (robot front = up), like
        robot_vision gets from the real camera after its calibration."""
        raw = self.cam.render(img, robot, self.rng if self.noise else None)
        return self.cam.topdown(raw)        # (outside the fov circle is already black)

    def ldrs(self, robot):
        out = []
        for i in range(cfg.LDR_COUNT):
            a = i * 2 * math.pi / cfg.LDR_COUNT + cfg.LDR_START_ANGLE + LDR_PHYSICAL_OFFSET
            px, py = rotate([LDR_RING_RADIUS * math.cos(a), LDR_RING_RADIUS * math.sin(a)], robot.h)
            v = LDR_WHITE if self.fimg.is_white(robot.x + px, robot.y + py) else LDR_GREEN
            if self.noise:
                v += int(self.rng.normal(0, 150))
            out.append(v)
        return out

    # ---- one tick
    def step(self):
        _SIM_T[0] += DT
        self.ticks += 1
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
        if self.noise:
            for r in self.players:
                r.bot.imu.drift += self.rng.normal(0, 0.0005)

        for r in self.players:
            r.brain.tick(r.paused)       # <- the real robot code
            r.drive(DT)
        self.move_opponents()
        self.physics()
        self.bookkeeping()

    def move_opponents(self):
        for o in self.opponents:
            if isinstance(o, SimRobot) or o.removed_until is not None:
                continue                 # self-play: drives itself / taken off
            if self.opp_ai and self.start_at is None:     # same start signal as our robots (they're paused till then)
                # get behind the ball (on the side away from our goal) and push it towards our goal (-y)
                tx, ty = self.ball[0], self.ball[1] + o.r + BALL_RADIUS - 2
                if o.y < self.ball[1]:
                    tx += math.copysign(25, o.x - self.ball[0] or 1)
                    ty = self.ball[1] + 25
                d = math.hypot(tx - o.x, ty - o.y)
                sp = min(OPP_SPEED, d * 4)
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
            b.x, b.y, _ = push_out_of_walls(b.x, b.y, b.r)          # robots can't drive through the goals
        for i, a in enumerate(bodies):           # robots push each other apart
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

        # dribbler / kicker
        cz = cfg.CAPTURE_ZONE
        rest_y = (cz[2] + cz[3]) / 2 + BALL_RADIUS     # ball centre when sitting in the capture notch
        notch_x = (cz[1] - cz[0]) / 2 + BALL_RADIUS
        order = list(self.players)
        self.rng.shuffle(order)            # no team gets to go first every tick
        catchers = []
        for r in order:
            if r.removed_until is not None:
                continue
            fwd = rotate([0.0, 1.0], r.h)
            accel = math.hypot(r.vx - r.prev_v[0], r.vy - r.prev_v[1]) / DT
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
                if not r.bot.motors.dribbler or abs(r.w) > DRIBBLE_MAX_SPIN or accel > DRIBBLE_MAX_ACCEL:
                    why = ("dribbler off" if not r.bot.motors.dribbler else
                           "spun too fast" if abs(r.w) > DRIBBLE_MAX_SPIN else "accel too hard")
                    self.count(f"{r.team} lost: {why}")
                    self.release(why)                    # lost it: it keeps the robot's speed and rolls away
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
                self.count("ball squeezed between both teams' dribblers")    # nobody gets it
            else:
                r = catchers[0]
                self.held_by, self.held_since = r, self.t
                self.last_touch = r
                self.count(f"{r.team} caught")

        # ball
        if self.ball_away_until is not None:
            return                                          # off the field (went out): nothing to move
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
        bx, by, normals = push_out_of_walls(bx, by, BALL_RADIUS)      # goal side / back walls: bounce off
        for n in normals:
            vn = vx * n[0] + vy * n[1]
            if vn < 0:
                vx -= (1 + BALL_BOUNCE) * vn * n[0]
                vy -= (1 + BALL_BOUNCE) * vn * n[1]
        for b in bodies:
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
                self.release("knocked")                        # an opponent knocked it out of the dribbler
            bx += n[0] * pen
            by += n[1] * pen
            # contact point velocity (incl. spin) pushes the ball
            px, py = bx - b.x, by - b.y
            cvx, cvy = b.vx - b.w * py, b.vy + b.w * px
            rel = (vx - cvx) * n[0] + (vy - cvy) * n[1]
            if rel < 0:
                vx -= (1 + BALL_BOUNCE * 0.5) * rel * n[0]
                vy -= (1 + BALL_BOUNCE * 0.5) * rel * n[1]
        # the ball is solid: a robot it still overlaps (squeezed between two robots) gets pushed back instead
        for b in bodies:
            hit = self.ball_overlap(b, bx, by, rest_y, notch_x)
            if hit is not None:
                n, pen = hit
                b.x -= n[0] * pen
                b.y -= n[1] * pen
        self.ball, self.ball_v = [bx, by], [vx, vy]

    @staticmethod
    def ball_overlap(b, bx, by, rest_y, notch_x):
        """(direction to push the ball out of body b, how far) in field axes, or None if they don't touch.
        Our robots have the capture notch at the front; opponents are plain circles."""
        rx, ry = rotate([bx - b.x, by - b.y], -b.h)       # ball in the body's frame
        if isinstance(b, SimRobot) and abs(rx) < notch_x and ry > 0:
            if ry >= rest_y:
                return None
            return rotate([0.0, 1.0], b.h), rest_y - ry       # straight out of the notch
        d = math.hypot(rx, ry)
        if d >= b.r + BALL_RADIUS or d < 1e-6:
            return None
        return rotate([rx / d, ry / d], b.h), b.r + BALL_RADIUS - d

    def bookkeeping(self):
        bx, by = self.ball
        in_goal = (abs(bx) < field.GOAL_W / 2 and          # inside a goal, fully over the goal line
                   field.PLAY_L / 2 + BALL_RADIUS < abs(by) < field.PLAY_L / 2 + field.GOAL_DEPTH)
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
            self.goal_timer = self.t + 1.0
        if self.goal_timer is not None and self.t >= self.goal_timer:
            self.kickoff()
            return          # everything below would still look at where the ball was before the kickoff
        # referee: ball out of play (fully outside the playing area) -> off for BALL_OUT_DELAY, then the nearest free
        # neutral spot; lack of progress -> nearest free neutral spot at once
        if self.ball_away_until is not None:
            if self.t >= self.ball_away_until and self.goal_timer is None:
                self.ball = list(self.ball_out_at)          # (only used to pick the nearest spot)
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
            # out: the whole robot is past the white line -> taken off as damaged (unless it was pushed out)
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
        """Robot fully out of the playing area: off as damaged for DAMAGED_TIME (or until the next kickoff)."""
        if self.held_by is b:
            self.release("robot taken off")
        b.removed_until = self.t + DAMAGED_TIME
        b.place(*PARK)
        b.was_out = False
        if isinstance(b, SimRobot):
            b.paused = True                       # switched off while it's off the field (like the real robot)
        self.count(f"{b.name} taken off (out of bounds)")
        self.log(f"{b.name} OUT OF BOUNDS - off for {DAMAGED_TIME:.0f} s")

    def pushed_out(self, b):
        """Clearly pushed out: an opponent touched it just now and it wasn't driving outwards itself."""
        if self.t - b.opp_contact_t > PUSHED_TIME:
            return False
        n = (math.copysign(1.0, b.x), 0.0) if abs(b.x) > field.PLAY_W / 2 else (0.0, math.copysign(1.0, b.y))
        return b.vx * n[0] + b.vy * n[1] < PUSHED_OUT_SPEED

    def push_back_in(self, b):
        """Pushed out by an opponent: straight back, just inside the line where it went out."""
        R = cfg.ROBOT_RADIUS
        x = max(-(field.PLAY_W / 2 - R), min(field.PLAY_W / 2 - R, b.x))
        y = max(-(field.PLAY_L / 2 - R), min(field.PLAY_L / 2 - R, b.y))
        b.x, b.y = x, y
        self.count(f"{b.name} pushed out, back at once")
        self.log(f"{b.name} pushed out - put straight back")

    def multiple_defence(self):
        """Ball in a penalty area with both of the defending team's robots in it: the one further from the ball is
        moved out of the area (straight out towards the halfway line)."""
        bx, by = self.ball
        R = cfg.ROBOT_RADIUS
        for team, sgn in (("us", -1.0), ("them", 1.0)):        # sgn: which end that team's goal is at
            def in_box(x, y):
                return abs(x) < field.PENALTY_W / 2 and sgn * y > field.PLAY_L / 2 - field.PENALTY_D
            if not in_box(bx, by):
                continue
            mine = [b for b in self.on_field if b.team == team and in_box(b.x, b.y)]
            if len(mine) < 2:
                continue
            far = max(mine, key=lambda b: math.hypot(b.x - bx, b.y - by))
            y = sgn * (field.PLAY_L / 2 - field.PENALTY_D - R - 3)
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
        """Damage time over: back in level with its own goal (parallel to it, just inside the field), on the side
        further from the ball, facing the goal it attacks. Our robots stay paused (calibrating heading / goal
        colour) for 0.5 s first."""
        R = cfg.ROBOT_RADIUS
        sgn = -1.0 if b.team == "us" else 1.0                 # own end
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

# ==================================== FAST MODE ====================================
# Sim(fast=True): no camera rendering / detection / localisation. Each camera frame's Detections are made straight
# from the true positions (+ a little noise), in the same format detection.py produces, and the pose comes from
# the truth (+ noise). Everything after that (perception, lines, strategy, motion) is still the real robot code.
# ~10x faster: use it to screen ideas, then confirm the good ones in the full simulator.
FAST_POSE_NOISE = 1.5        # cm
FAST_BALL_RANGE = 75.0       # cm the camera finds the ball out to (240 px fisheye, see GUIDE.md)
FAST_LINE_RANGE = 100.0      # cm
FAST_LINE_STEP = 2.0         # cm between line points
FAST_RELOCALISE = 0.3        # s without a confident pose after being picked up


class TruthLoc:
    """Stands in for localisation.LocalisationThread in fast mode."""

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
        sgn = 1.0 if self.body.team == "us" else -1.0   # the opponents' field frame is ours turned round (they
        return Pose(sgn * self.body.x + n[0], sgn * self.body.y + n[1], 3.0, t)   # attack the other goal)

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
    """The coloured parts of a goal as rectangles (x0, x1, y0, y1): side walls (drawn 5 cm wide) and back wall."""
    gw, y0, y1 = field.GOAL_W / 2, field.PLAY_L / 2, field.PLAY_L / 2 + field.GOAL_DEPTH
    rects = [(-gw - 5, -gw, y0, y1 + 2), (gw, gw + 5, y0, y1 + 2), (-gw - 5, gw + 5, y1, y1 + 2)]
    return [(x0, x1, min(sgn * ya, sgn * yb), max(sgn * ya, sgn * yb)) for x0, x1, ya, yb in rects]


GOAL_PARTS = {"yellow": _goal_parts(1), "blue": _goal_parts(-1)}


def fast_detect(sim, r, ignore):
    """Detections for robot r from the true positions (robot frame cm), like detection.detect() would give."""
    from detection import Detections, Goal, Obstacle
    rng = sim.rng
    det = Detections(time.monotonic(), r.bot.imu.compass())
    det.field_visible = True

    def to_robot(px, py):
        return rotate([px - r.x, py - r.y], -r.h)

    def hidden(p):           # inside the ignore box (our own body)
        return -ignore["left"] < p[0] < ignore["right"] and -ignore["back"] < p[1] < ignore["front"]

    # ball: nearest point
    dx, dy = sim.ball[0] - r.x, sim.ball[1] - r.y
    d = math.hypot(dx, dy)
    if 1e-6 < d < FAST_BALL_RANGE:
        near = to_robot(sim.ball[0] - dx / d * BALL_RADIUS, sim.ball[1] - dy / d * BALL_RADIUS)
        if not hidden(near):
            s = 0.3 + 0.01 * d
            det.ball = [near[0] + rng.normal(0, s), near[1] + rng.normal(0, s)]
    # goals: nearest coloured point + centre
    for colour, parts in GOAL_PARTS.items():
        best = None
        for x0, x1, y0, y1 in parts:
            cx, cy = min(max(r.x, x0), x1), min(max(r.y, y0), y1)
            dd = math.hypot(cx - r.x, cy - r.y)
            if best is None or dd < best[0]:
                best = (dd, cx, cy)
        if best[0] < VIEW_RADIUS - 5:
            sgn = 1 if colour == "yellow" else -1
            det.goals[colour] = Goal(to_robot(best[1], best[2]),
                                     to_robot(0.0, sgn * (field.PLAY_L / 2 + field.GOAL_DEPTH + 1)))
    # other robots: nearest point
    for b in sim.on_field:
        if b is r:
            continue
        dx, dy = b.x - r.x, b.y - r.y
        d = math.hypot(dx, dy)
        if b.r < d < 90.0:
            near = to_robot(b.x - dx / d * b.r, b.y - dy / d * b.r)
            centre = to_robot(b.x, b.y)
            n = rng.normal(0, 1.0, 2)
            det.obstacles.append(Obstacle([near[0] + n[0], near[1] + n[1]], 2 * b.r, [centre[0] + n[0], centre[1] + n[1]]))
    # white line points
    rel = FAST_LINE_PTS - np.float32([r.x, r.y])
    rel = rel[(rel[:, 0] ** 2 + rel[:, 1] ** 2) < FAST_LINE_RANGE ** 2]
    if len(rel) > 250:
        rel = rel[np.linspace(0, len(rel) - 1, 250).astype(int)]
    c, s = math.cos(-r.h), math.sin(-r.h)
    det.line_pts = np.stack([rel[:, 0] * c - rel[:, 1] * s, rel[:, 0] * s + rel[:, 1] * c], axis=1).astype(np.float32)
    return det


# ==================================== WINDOW ====================================
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
                obj.h = math.atan2(-(x - obj.x), y - obj.y)    # face the mouse (0 = +y, CCW +)
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
        if pose.confident:          # opponents as the selected robot believes them (both robots' cameras): circles
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
            if r.bot.motors.dribbler:      # dribbler bar at the front
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
            if pose.std < 200:     # estimated position: cross + spread circle
                e = P(pose.x, pose.y)
                cv2.drawMarker(img, e, r.colour, cv2.MARKER_TILTED_CROSS, 10, 2)
                cv2.circle(img, e, max(2, int(pose.std * UI_PPC)), r.colour, 1)
            label = "PAUSED" if r.paused else f"b{br.strategy.botstate} {br.strategy.phase}"
            cv2.putText(img, label, (c[0] + rad + 2, c[1] - rad), cv2.FONT_HERSHEY_PLAIN, 1.0, r.colour, 1)
        cv2.circle(img, P(*s.ball), max(2, int(BALL_RADIUS * UI_PPC)), ORANGE, -1)
        cv2.putText(img, f"t={s.t:5.1f}s  x{self.speed:g}{'' if self.running else '  PAUSED'}   "
                         f"score {s.score['us']}-{s.score['them']}   "
                         f"opp {getattr(s, 'opp_name', 'simple AI')}{'' if getattr(s, 'opp_name', 'simple AI') != 'simple AI' else (' on' if s.opp_ai else ' off')}"
                         f"   noise {'on' if s.noise else 'off'}",
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
                    if time.perf_counter() - now > 0.05:   # can't keep up: drop the debt, run slower than real time
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
    """Same overlay as test_localisation.py: ignore box red, capture zone cyan, hull green, lines magenta,
    goals crossed, obstacles red, ball orange."""
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
        if getattr(o, "centre", None) is not None:      # the whole robot it belongs to
            cv2.circle(img, px(o.centre), int(round(cfg.ROBOT_RADIUS * rv.px_per_cm)), (0, 0, 255), 2)
    if det.ball is not None:
        cv2.circle(img, px(det.ball), 6, (0, 140, 255), 2)
    return img


def headless(sim, seconds):
    """Run without a window and print what happened - quick check after changing the logic."""
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
          f"({CAM_FOV_DEG:.0f} deg at {CAM_HEIGHT:.0f} cm, {CAM_RES} px)")
    for r in sim.robots:
        e = sim.loc_err[r.name]
        tot = sum(states[r.name].values())
        share = ", ".join(f"b{k} {100 * v / tot:.0f}%" for k, v in sorted(states[r.name].items()))
        print(f"{r.name:8s} outs {r.outs}  localisation error mean {np.mean(e) if e else float('nan'):.1f} cm "
              f"max {max(e) if e else float('nan'):.1f} cm  states: {share}")
    print("log:", *sim.logs, sep="\n  ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", type=float, metavar="SECONDS", help="no window: simulate this long and print a summary")
    ap.add_argument("--noise", action="store_true", help="start with sensor noise on")
    ap.add_argument("--fast", action="store_true", help="no camera simulation: detections + pose from the truth")
    ap.add_argument("--seed", type=int, default=0, help="game seed: with --noise this replays arena.py's game of the "
                                                        "same seed (that side round: our code as 'us')")
    ap.add_argument("--opp", metavar="VERSION", help="opponents run this version of our code (a name from "
                                                     "past_versions/, e.g. v26, or a folder) instead of the simple AI")
    args = ap.parse_args()
    sim = Sim(noise=args.noise, seed=args.seed, fast=args.fast, opp_code=load_code(args.opp) if args.opp else None)
    sim.opp_name = args.opp or "simple AI"
    RV_FOR_DRAW.append(sim.rv)
    if args.headless:
        headless(sim, args.headless)
    else:
        Viewer(sim).loop()


if __name__ == "__main__":
    sys.exit(main())

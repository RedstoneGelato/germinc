#!/usr/bin/env python3
"""
simulator.py - test the robots' logic on a simulated field, on a laptop (no Pi needed).

    python simulator.py                    # window with the field, both robots, two opponents
    python simulator.py --headless 30      # no window: simulate 30 s and print a summary (quick regression test)

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

BALL_OUT_TIME = 2.0          # referee: ball outside the white line this long -> nearest free neutral spot
NO_PROGRESS_TIME = 10.0      # referee: ball hasn't moved 5 cm in this long -> nearest free neutral spot

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
    Precomputes: fisheye pixel -> floor point (robot frame cm), and top-down pixel -> fisheye pixel."""
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
            self.gx = np.where(rho > 0, r * dx / rho, 0).astype(np.float32)   # floor point, robot frame cm
            self.gy = np.where(rho > 0, r * dy / rho, 0).astype(np.float32)
        self.lens = rho <= 1.0
        # top-down (TOPDOWN_PPC px/cm, CAM_SIZE square, robot at the centre) -> fisheye pixel (at CAM_RES)
        m = CAM_SIZE
        u, v = np.meshgrid(np.arange(m) + 0.5, np.arange(m) + 0.5)
        x, y = (u - m / 2.0) / TOPDOWN_PPC, (m / 2.0 - v) / TOPDOWN_PPC
        rr = np.hypot(x, y)
        rho_t = np.arctan(rr / CAM_HEIGHT) / half
        cr = CAM_RES / 2.0
        with np.errstate(invalid="ignore", divide="ignore"):
            self.map_x = np.where(rr > 0, cr + rho_t * cr * x / rr, cr).astype(np.float32)
            self.map_y = np.where(rr > 0, cr - rho_t * cr * y / rr, cr).astype(np.float32)

    def render(self, field_img, robot, noise_rng=None):
        """Raw fisheye image (CAM_RES square) as the camera would see the (flat) field from this robot."""
        c, s = math.cos(robot.h), math.sin(robot.h)
        X = robot.x + self.gx * c - self.gy * s
        Y = robot.y + self.gx * s + self.gy * c
        k = FIELD_PPC
        fu = ((X + FieldImage.W / 2) * k).astype(np.float32)
        fv = ((FieldImage.L / 2 - Y) * k).astype(np.float32)
        img = cv2.remap(field_img, fu, fv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=WALL)
        img[~self.lens] = 0
        img = cv2.resize(img, (CAM_RES, CAM_RES), interpolation=cv2.INTER_AREA)
        if noise_rng is not None:
            img = cv2.add(img, noise_rng.normal(0, 8, img.shape).astype(np.int16), dtype=cv2.CV_8U)
        return img

    def topdown(self, raw):
        return cv2.remap(raw, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


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
    """Same interface as common.Robot (so the brains can't tell the difference), with sim hardware."""

    def __init__(self, body, rv, hw):
        self.imu = SimIMU(body)
        self.motors = SimMotors(hw.DRIBBLER is not None)
        self.kicker = None if hw.KICKER_PIN is None else SimKicker(hw.KICK_COOLDOWN)
        self.pcb = SimPCB()
        self.comms = SimComms()
        self.vision = SimVision(rv)
        self.attack = "yellow"
        self.loc = LocalisationThread(self.vision, lambda: self.attack)   # never started: process() is called per frame
        self.world = World()
        self.lines = LineFusion()
        self.mover = Mover()
        self.leds = LedCalibrator(self.pcb)


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
    def __init__(self, x, y, h, name, colour):
        self.x, self.y, self.h = x, y, h
        self.vx = self.vy = self.w = 0.0
        self.prev_v = (0.0, 0.0)
        self.name, self.colour = name, colour
        self.r = cfg.ROBOT_RADIUS

    def place(self, x, y, h=None):
        self.x, self.y = x, y
        if h is not None:
            self.h = h
        self.vx = self.vy = self.w = 0.0
        self.prev_v = (0.0, 0.0)


class SimRobot(Body):
    def __init__(self, x, y, h, name, colour, brain_cls, hw, rv, log):
        super().__init__(x, y, h, name, colour)
        self.bot = SimBot(self, rv, hw)
        self.kick_flash = 0.0
        self.brain = brain_cls(self.bot, log=lambda m: log(f"{name}: {m}"))
        self.paused = True
        self.outs = 0
        self.was_out = False

    def drive(self, dt):
        vx_r, vy_r, spin = motors_to_velocity(self.bot.motors.speeds)
        tvx, tvy = rotate([vx_r, vy_r], self.h)
        a = min(dt / MOTOR_TAU, 1.0)
        self.vx += (tvx - self.vx) * a
        self.vy += (tvy - self.vy) * a
        self.w += (spin - self.w) * a


class Sim:
    def __init__(self, noise=False):
        self.fimg = FieldImage(FIELD_PPC)
        self.rv, self.fov = make_rv()
        self.cam = FisheyeCamera()
        self.held_by = None       # robot whose dribbler has the ball
        self.kicks = 0
        self.ball_out_since = None
        self.progress = ([0.0, 0.0], 0.0)   # (ball position, time) last time it moved 5 cm
        self.logs = []
        self.robots = [
            SimRobot(0, 0, 0, "goalie", (255, 200, 0), GoalieBrain, hardware_defense, self.rv, self.log),
            SimRobot(0, 0, 0, "striker", (255, 0, 255), StrikerBrain, hardware_attack, self.rv, self.log),
        ]
        g, s = self.robots
        g.bot.comms.partner, s.bot.comms.partner = s.bot.comms, g.bot.comms
        for i, r in enumerate(self.robots):          # fixed seeds: the same settings give the same game every run
            r.bot.loc.loc.rng = np.random.default_rng(100 + i)
        self.opponents = [Body(0, 0, math.pi, "opp", ROBOT), Body(0, 0, math.pi, "opp", ROBOT)]
        self.ball = [0.0, 0.0]
        self.ball_v = [0.0, 0.0]
        self.noise = noise
        self.opp_ai = False
        self.ticks = 0
        self.score = {"us": 0, "them": 0}
        self.goal_timer = None
        self.loc_err = {r.name: [] for r in self.robots}
        self.rng = np.random.default_rng(0)
        self.kickoff()

    def log(self, msg):
        self.logs.append(f"{_SIM_T[0] - 1000:6.2f} {msg}")
        self.logs = self.logs[-6:]

    @property
    def t(self):
        return _SIM_T[0] - 1000.0

    def kickoff(self):
        g, s = self.robots
        R = cfg.ROBOT_RADIUS
        g.place(0, field.OWN_GOAL_Y + R + 8, 0)
        s.place(0, -(R + 15), 0)
        for o, (x, y) in zip(self.opponents, ((0, R + 15), (0, field.ATTACK_GOAL_Y - R - 8))):
            o.place(x, y, math.pi)
        self.ball, self.ball_v = [0.0, 0.0], [0.0, 0.0]
        self.held_by = None
        self.ball_out_since = None
        self.progress = ([0.0, 0.0], self.t)
        for r in self.robots:
            r.paused = True
        self.start_at = self.t + 0.5      # stand still (paused) half a second: calibrates heading + goal colour
        self.goal_timer = None

    # ---- sensors
    def draw_dynamic(self):
        img = self.fimg.img.copy()
        ppc = FIELD_PPC
        for b in self.robots + self.opponents:
            cv2.circle(img, self.fimg.px(b.x, b.y), int(b.r * ppc), ROBOT, -1)
        cv2.circle(img, self.fimg.px(*self.ball), max(1, int(round(BALL_RADIUS * ppc))), ORANGE, -1)
        return img

    def camera(self, img, robot):
        """Fisheye image from the robot -> top-down robot-frame image (robot front = up), like
        robot_vision gets from the real camera after its calibration."""
        raw = self.cam.render(img, robot, self.rng if self.noise else None)
        cam = self.cam.topdown(raw)
        cam[self.fov == 0] = 0
        return cam

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
            for r in self.robots:
                r.paused = False
            self.start_at = None

        if self.ticks % CAMERA_EVERY == 0:
            img = self.draw_dynamic()
            for r in self.robots:
                v = r.bot.vision
                det, res = analyse_frame(v.rv, self.camera(img, r), time.monotonic(), r.bot.imu.compass())
                v.detections, v.last_res = det, res
                v.frame_id += 1
                r.bot.loc.process(det)
        if self.ticks % PCB_EVERY == 0:
            for r in self.robots:
                r.bot.pcb.colours = self.ldrs(r)
        if self.noise:
            for r in self.robots:
                r.bot.imu.drift += self.rng.normal(0, 0.0005)

        for r in self.robots:
            r.brain.tick(r.paused)       # <- the real robot code
            r.drive(DT)
        self.move_opponents()
        self.physics()
        self.bookkeeping()

    def move_opponents(self):
        for o in self.opponents:
            if self.opp_ai:
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
        bodies = self.robots + self.opponents
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
                    push = (a.r + b.r - d) / 2
                    a.x -= dx / d * push
                    a.y -= dy / d * push
                    b.x += dx / d * push
                    b.y += dy / d * push

        # dribbler / kicker
        cz = cfg.CAPTURE_ZONE
        rest_y = (cz[2] + cz[3]) / 2 + BALL_RADIUS     # ball centre when sitting in the capture notch
        notch_x = (cz[1] - cz[0]) / 2 + BALL_RADIUS
        for r in self.robots:
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
                        self.held_by = None
                    self.kicks += 1
                    self.log(f"{r.name} KICK")
                    continue
            if self.held_by is r:
                if not r.bot.motors.dribbler or abs(r.w) > DRIBBLE_MAX_SPIN or accel > DRIBBLE_MAX_ACCEL:
                    self.held_by = None                     # lost it: it keeps the robot's speed and rolls away
                else:
                    px, py = fwd[0] * rest_y, fwd[1] * rest_y
                    self.ball = [r.x + px, r.y + py]
                    self.ball_v = [r.vx - r.w * py, r.vy + r.w * px]
            elif self.held_by is None and r.bot.motors.dribbler and in_notch:
                relv = math.hypot(self.ball_v[0] - r.vx, self.ball_v[1] - r.vy)
                if relv < DRIBBLE_CATCH_SPEED:
                    self.held_by = r

        # ball
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
            if self.held_by is not None and b not in self.robots:
                self.held_by = None                         # an opponent knocked it out of the dribbler
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
            self.log(f"GOAL for {who}  ({self.score['us']}-{self.score['them']})")
            self.goal_timer = self.t + 1.0
        if self.goal_timer is not None and self.t >= self.goal_timer:
            self.kickoff()
        # referee: ball out of play / lack of progress -> nearest free neutral spot
        if self.goal_timer is None:
            out = not field.inside_play_area(bx, by, -BALL_RADIUS)
            self.ball_out_since = (self.ball_out_since or self.t) if out else None
            if math.hypot(bx - self.progress[0][0], by - self.progress[0][1]) > 5:
                self.progress = ([bx, by], self.t)
            why = ("out" if self.ball_out_since is not None and self.t - self.ball_out_since > BALL_OUT_TIME else
                   "no progress" if self.t - self.progress[1] > NO_PROGRESS_TIME else None)
            if why:
                self.place_ball_neutral(why)
        for r in self.robots:
            # out: the whole robot is past the white line
            out = not field.inside_play_area(r.x, r.y, -r.r)
            if out and not r.was_out:
                r.outs += 1
                self.log(f"{r.name} OUT OF BOUNDS")
            r.was_out = out
            pose = r.bot.loc.get()
            if not r.paused and pose.confident:
                self.loc_err[r.name].append(math.hypot(pose.x - r.x, pose.y - r.y))

    def place_ball_neutral(self, why):
        bodies = self.robots + self.opponents
        free = [p for p in field.NEUTRAL_SPOTS if all(math.hypot(p[0] - b.x, p[1] - b.y) > b.r + 10 for b in bodies)]
        spots = free or field.NEUTRAL_SPOTS
        spot = min(spots, key=lambda p: math.hypot(p[0] - self.ball[0], p[1] - self.ball[1]))
        self.ball, self.ball_v, self.held_by = list(spot), [0.0, 0.0], None
        self.ball_out_since, self.progress = None, (list(spot), self.t)
        self.log(f"ball {why} -> neutral spot ({spot[0]:.0f},{spot[1]:.0f})")

    def kidnap(self, r):
        mx, my = field.PLAY_W / 2 - r.r, field.PLAY_L / 2 - r.r
        r.place(random.uniform(-mx, mx), random.uniform(-my, my), random.uniform(-math.pi, math.pi))
        if self.held_by is r:
            self.held_by = None
        r.bot.loc.relocalise()
        self.log(f"{r.name} picked up and moved")


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
        for o in s.opponents:
            cv2.circle(img, P(o.x, o.y), int(o.r * UI_PPC), ROBOT, -1)
            cv2.circle(img, P(o.x, o.y), int(o.r * UI_PPC), (0, 0, 200), 2)
        for r in s.robots:
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
                         f"score {s.score['us']}-{s.score['them']}   opp AI {'on' if s.opp_ai else 'off'}"
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
            f"[{self.sel + 1}] {r.name}  ({'PAUSED' if r.paused else 'running'})  camera: {self.cam_mode}",
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
        cv2.circle(img, px(o.near), 5, (0, 0, 255), 2)
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
    args = ap.parse_args()
    sim = Sim(noise=args.noise)
    RV_FOR_DRAW.append(sim.rv)
    if args.headless:
        headless(sim, args.headless)
    else:
        Viewer(sim).loop()


if __name__ == "__main__":
    sys.exit(main())

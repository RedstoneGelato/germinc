"""Drive the robot by hand from a web page, to tune speeds, turning, the dribbler and the kicker.

    python3 test_drive.py attack                  striker. Then open http://localhost:8000 on the Pi (VNC)
    python3 test_drive.py defense                 goalie
    python3 test_drive.py attack --host 0.0.0.0   also reachable from a laptop: http://<address of the Pi>:8000
    python3 test_drive.py --fake                  no robot needed, to try the page on a PC
    python3 test_drive.py attack --int            international field and strategy (strategy_int.py)

Stop the normal robot program first (it uses the same motors). Put the robot on a stand for the first try.
The robot stops by itself when the page stops sending commands for STOP_AFTER seconds (page closed, Wi-Fi lost).
With --host 0.0.0.0 anyone on the same network can drive the robot, so only use it on a network you trust.
The camera view is the picture the robot's own vision works on. It needs robot_vision_config.json (test_camera.py).
Nothing is saved: copy the numbers you like into robot_config.py.

Autonomous mode (on a real robot with camera and IMU): the robot's own Brain and strategy run all the time, and the
camera view shows what they see and want. Switching "Autonomous" on lets their wheel, dribbler and kick commands reach
the robot, scaled by the autonomous speed limit. It switches off again on STOP, Space, closing the page or Wi-Fi loss.
Point the robot at the attack goal and press "Zero heading" first, like putting it down at kick-off.
"""
import argparse
import json
import math
import sys
import threading
import time
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if "--int" in sys.argv:      # must come first: the imports below then get the international field and strategy
    import field_int
    field_int.install_international()

import cv2  # noqa: E402

import robot_config as cfg  # noqa: E402
from common import LedCalibrator, Robot, TickGuard, keep_rate  # noqa: E402
from motion import Mover, wheel_speeds  # noqa: E402
from world import World  # noqa: E402
from utils import other_goal, rotate, wrap_pi  # noqa: E402

STOP_AFTER = 0.3          # seconds without a command from the page before the motors are stopped
MAX_SPEED = 1.5           # top of the speed slider, as a multiple of BASE_SPEED (same as MOTOR_MAX)
MAX_KICK_PULSE = 0.1      # seconds. Longer pulses can overheat the solenoid, so the page cannot ask for more
MAX_DRIBBLER = 546133333  # the motor drivers' own speed limit (MOTOR_SPEED_LIMIT in hardware.py)

motors = None             # set in main()
kicker = None
imu = None
camera = None             # vision.VisionThread, or None when the camera did not start
drive = None              # DriveBot when autonomous mode is possible (real robot, camera and IMU started)
guard = None              # TickGuard around the role's Brain, with drive
last_command = 0.0        # time.monotonic() of the newest command from the page
FIELD_SCALE = 2.0         # field picture pixels per cm
field_base = None         # field picture without the robot, made once
problems = []             # parts that did not start, shown at the top of the page


class FakeMotors:
    """Stands in for hardware.MotorThread with --fake."""
    has_dribbler = True
    dribbler_on_speed = 200000000
    drivers = ()
    speeds = (0, 0, 0, 0)
    dribbler_speed = 0

    def set(self, speeds):
        self.speeds = tuple(speeds)

    def set_dribbler(self, on):
        self.dribbler_speed = self.dribbler_on_speed if on else 0

    def stop(self):
        self.speeds = (0, 0, 0, 0)
        self.dribbler_speed = 0


class FakeKicker:
    """Stands in for hardware.Kicker with --fake."""
    pulse = 0.05

    def ready(self):
        return True

    def kick(self):
        print(f"KICK for {self.pulse} s")
        return True

    def close(self):
        pass


class FakeCamera:
    """Stands in for vision.VisionThread with --fake: one grey picture with an orange square."""

    def __init__(self):
        import numpy as np
        picture = np.full((240, 240, 3), 70, np.uint8)
        picture[100:140, 100:140] = (0, 140, 255)
        self.cond = threading.Lock()
        self.last_res = SimpleNamespace(frame=picture)
        self.fps = 0.0

    def stop(self):
        pass

    def close(self):
        pass


def camera_jpeg():
    """The newest camera picture as JPEG bytes, with what the robot sees and wants drawn on, or None."""
    if camera is None:
        return None
    with camera.cond:
        result, det = camera.last_res, getattr(camera, "detections", None)
    if result is None:
        return None
    img = result.frame.copy()
    if drive is not None and det is not None:
        draw_view(img, camera.rv, det)
    ok, data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return data.tobytes() if ok else None


class DriveBot(Robot):
    """The robot program's Robot, built from the parts test_drive already started (like SimBot in the simulator).
    The Brain ticks all the time; its commands only reach the motors while auto is on."""
    auto = False
    turn_on = False           # set by the page, switched on in the main loop between two ticks
    speed_limit = 0.5         # autonomous wheel speeds are multiplied by this
    intent_kick = False

    def __init__(self, hw, motors, kicker, imu, vision):
        from hardware import PCBThread
        from world import LocalisationThread
        self.hw, self.motors, self.kicker, self.imu, self.vision = hw, motors, kicker, imu, vision
        self.lock = threading.Lock()
        self.pcb = PCBThread()
        self.pcb.start()
        self.comms = make_comms(hw)
        self.attack = "yellow"
        self.loc = LocalisationThread(vision, lambda: self.attack)
        self.loc.start()
        self.world, self.mover = World(), Mover()
        self.leds = LedCalibrator(self.pcb)

    def actuate(self, speeds, dribbler_on, kick):
        self.intent_kick = kick
        with self.lock:
            if self.auto:
                super().actuate([int(s * self.speed_limit) for s in speeds], dribbler_on, kick)

    def set_auto(self, on):
        if on:
            self.turn_on = not self.auto
            return
        with self.lock:
            was_on = self.auto
            self.auto = self.turn_on = False
        if was_on:
            self.motors.stop()

    def step(self, guard):
        """One control loop: switch auto on if asked, then one Brain tick."""
        if self.turn_on:
            guard.brain.strategy.reset()
            self.mover.reset()
            with self.lock:
                self.auto, self.turn_on = self.turn_on, False
        if not self.auto:
            self.leds.step(self.pcb.snapshot())     # what the robot does while paused, so line sensing is tuned
        guard.tick(False)

    def stop_threads(self):
        for thread in (self.loc, self.pcb):
            thread.running = False
        self.comms.stop()


def make_comms(hw):
    try:
        from comms import TeammateLinkThread
        comms = TeammateLinkThread(hw.ROBOT_ID)
        comms.start()
        return comms
    except OSError as e:
        print(f"No teammate link ({e!r}): the strategy plays as if alone")
        return SimpleNamespace(my_state={}, teammate=lambda: None, stop=lambda: None)


def record_intent(brain):
    """Keeps the strategy's newest (desired_pos, desired_heading, dribbler_on) as brain.intent, for the overlay."""
    update = brain.strategy.update
    brain.intent = None

    def recorded(*args):
        brain.intent = update(*args)
        return brain.intent
    brain.strategy.update = recorded


GOAL_BGR = {"yellow": (0, 255, 255), "blue": (255, 140, 0)}
BALL_BGR, TARGET_BGR, HEADING_BGR, ESCAPE_BGR = (0, 140, 255), (255, 255, 0), (120, 255, 120), (255, 80, 200)


def label(img, text, at, colour):
    x, y = at
    cv2.putText(img, text, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)


def draw_view(img, rv, det):
    """Draws on the camera picture: this frame's detections, what the robot remembers or works out from its
    position (thin, marked), where the strategy wants to go and face, and a text box of what it is doing."""
    w, pose, brain = drive.world, drive.loc.get(), guard.brain
    heading = det.compass + pose.dh
    reach = float(rv._radial[1][-1]) if getattr(rv, "_radial", None) is not None else None
    height, width = img.shape[:2]

    def px(p, field_aligned=False):
        """Pixel of a point in cm (robot frame, or field-aligned and robot-centred). far = pulled in to the edge."""
        x, y = rotate(p, -heading) if field_aligned else (float(p[0]), float(p[1]))
        d, far = math.hypot(x, y), False
        if reach and d > reach:
            x, y, far = x * reach / d, y * reach / d, True
        u, v = (float(a) for a in rv.to_px(x, y))
        if not (0 <= u < width and 0 <= v < height):
            u, v, far = min(max(u, 0), width - 1), min(max(v, 0), height - 1), True
        return (int(round(u)), int(round(v))), far

    centre = px((0, 0))[0]
    if det.hull_px is not None:
        cv2.polylines(img, [det.hull_px], True, (0, 160, 0), 1)
    for p in det.line_pts:
        cv2.circle(img, px(p)[0], 1, (255, 0, 255), -1)
    cz = cfg.CAPTURE_ZONE
    cv2.rectangle(img, px((cz[0], cz[3]))[0], px((cz[1], cz[2]))[0], (255, 255, 0), 1)

    sources = (w.goal_source or "-/-").split("/")
    for goal, source, colour, name in zip((w.attack_goal, w.own_goal), sources,
                                          (drive.attack, other_goal(drive.attack)), ("attack", "own")):
        if goal is None:
            continue
        at, far = px(goal.centre, True)
        seen = source == "cam"
        cv2.drawMarker(img, at, GOAL_BGR[colour], cv2.MARKER_CROSS if seen else cv2.MARKER_DIAMOND, 14, 2 if seen else 1)
        label(img, f"{name} goal" + ("" if seen else " (from pos)") + (" >" if far else ""), at, GOAL_BGR[colour])

    for o in det.obstacles:
        o_centre = o.centre if getattr(o, "centre", None) is not None else o.near
        at, _ = px(o_centre)
        edge, _ = px((o_centre[0] + cfg.ROBOT_RADIUS, o_centre[1]))
        cv2.circle(img, at, max(3, int(math.hypot(edge[0] - at[0], edge[1] - at[1]))), (0, 0, 255), 2)
        label(img, "robot", at, (80, 80, 255))
    if pose.confident:
        for o in w.opponents:
            at, far = px((o[0] - pose.x, o[1] - pose.y), True)
            cv2.circle(img, at, 8, (0, 0, 255), 1)
            if far:
                label(img, "opp >", at, (80, 80, 255))

    if w.ball is not None:
        at, far = px(w.ball, True)
        seen = w.ball_source == "cam"
        cv2.circle(img, at, 7, BALL_BGR, 2 if seen else 1)
        label(img, "ball" + ("" if seen else f" ({w.ball_source})") + (" >" if far else ""), at, BALL_BGR)

    cv2.arrowedLine(img, centre, px((0, 15))[0], (0, 255, 0), 1, tipLength=0.3)
    intent = getattr(brain, "intent", None)
    if intent is not None:
        desired_pos, desired_heading, _ = intent
        at, far = px(desired_pos, True)
        cv2.arrowedLine(img, centre, at, TARGET_BGR, 2, tipLength=0.15)
        cv2.circle(img, at, 5, TARGET_BGR, 1)
        label(img, "target" + (" >" if far else ""), at, TARGET_BGR)
        turn = wrap_pi(desired_heading - heading)
        cv2.arrowedLine(img, centre, px((-25 * math.sin(turn), 25 * math.cos(turn)))[0], HEADING_BGR, 1,
                        tipLength=0.25)
    line = getattr(brain, "line", None)
    if line is not None and line.on_line and line.escape is not None:
        cv2.arrowedLine(img, centre, px((30 * line.escape[0], 30 * line.escape[1]), True)[0], ESCAPE_BGR, 2,
                        tipLength=0.25)

    for i, t in enumerate(status_lines()[:2]):    # outlined text, no box, so nothing under it is hidden
        label(img, t, (-2, 20 + 14 * i), (80, 80, 255) if i == 0 and drive.auto else (230, 230, 230))


def status_lines():
    """A few short lines of what the robot is doing, for the camera picture."""
    w, pose, brain = drive.world, drive.loc.get(), guard.brain
    s = brain.strategy
    line = getattr(brain, "line", None)
    mode = (f"AUTO {drive.speed_limit:.0%} speed" if drive.auto
            else "MANUAL - strategy shown, not driving")
    return [
        mode + (f"  ({brain.status})" if brain.status.startswith("camera blind") else ""),
        f"b{s.botstate} {s.phase}" + ("  KICK" if drive.intent_kick else "") + ("  HAS BALL" if w.ball_in_capture else ""),
        f"pos ({pose.x:.0f},{pose.y:.0f}) +-{pose.std:.0f} " + ("ok" if pose.confident else "LOST"),
        f"ball {w.ball_source}  goals {w.goal_source}  possession {getattr(s, 'possession', None) or '-'}",
        "line " + ("ON " + line.source if line is not None and line.on_line else "off"),
    ]


def field_jpeg():
    """Top view of the field with where the robot thinks it, the ball, the other robots and its target are."""
    global field_base
    if drive is None:
        return None
    fmap = drive.loc.loc.map
    if field_base is None:
        base = cv2.resize(fmap.lines, None, fx=FIELD_SCALE, fy=FIELD_SCALE, interpolation=cv2.INTER_NEAREST)
        field_base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        field_base[base <= 128] = (40, 110, 40)
        field_base[base > 128] = (255, 255, 255)

    def fpx(x, y):
        mx, my = fmap.to_map(x, y)
        return int(mx * FIELD_SCALE), int(my * FIELD_SCALE)

    img = field_base.copy()
    pose, w, brain = drive.loc.get(), drive.world, guard.brain
    for x, y in drive.loc.particles()[::3]:
        cv2.circle(img, fpx(x, y), 1, (150, 150, 150), -1)
    if pose.confident or pose.std < 100:
        centre = fpx(pose.x, pose.y)
        for o in w.opponents:
            cv2.circle(img, fpx(*o), int(cfg.ROBOT_RADIUS * FIELD_SCALE), (0, 0, 255), 2)
        if w.ball_field is not None:
            cv2.circle(img, fpx(*w.ball_field), 5, BALL_BGR, -1)
        intent = getattr(brain, "intent", None)
        if intent is not None:
            tx, ty = pose.x + intent[0][0], pose.y + intent[0][1]
            cv2.arrowedLine(img, centre, fpx(tx, ty), TARGET_BGR, 2, tipLength=0.15)
        cv2.circle(img, centre, int(cfg.ROBOT_RADIUS * FIELD_SCALE), (0, 255, 0) if pose.confident else (0, 200, 255), 2)
        a = drive.imu.compass() + pose.dh + math.pi / 2
        cv2.arrowedLine(img, centre, fpx(pose.x + 20 * math.cos(a), pose.y + 20 * math.sin(a)), (0, 255, 0), 2,
                        tipLength=0.3)
    ok, data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return data.tobytes() if ok else None


def number(data, key, low, high):
    """The number data[key], limited to low..high. The page is not trusted to send sensible values."""
    value = float(data.get(key, 0))
    if not math.isfinite(value):
        raise ValueError(f"{key} is not a number")
    return max(low, min(high, value))


def wheel_targets(cmd):
    """The four wheel speeds for one command from the page. Robot frame: x = right, y = forward, rot = turn left."""
    speed = number(cmd, "speed", 0.0, MAX_SPEED) * cfg.BASE_SPEED
    one_wheel = cmd.get("wheel")
    if one_wheel in (0, 1, 2, 3):       # spin a single wheel, to check the wiring order and direction
        speeds = [0, 0, 0, 0]
        speeds[one_wheel] = int(speed)
        return tuple(speeds)

    x, y = number(cmd, "x", -1.0, 1.0), number(cmd, "y", -1.0, 1.0)
    drive = wheel_speeds(x, y, speed * min(1.0, math.hypot(x, y)))
    rot = cfg.BASE_SPEED * number(cmd, "turn", 0.0, 1.0) * number(cmd, "rot", -1.0, 1.0)
    speeds = [w - rot for w in drive]   # same mixing and limit as motion.Mover.step
    biggest = max(abs(s) for s in speeds)
    if biggest > cfg.MOTOR_MAX:
        speeds = [s * cfg.MOTOR_MAX / biggest for s in speeds]
    return tuple(int(s) for s in speeds)


def apply_command(cmd):
    global last_command
    if drive is not None:
        if cmd.get("attack") in ("yellow", "blue") and cmd["attack"] != drive.attack:
            drive.attack = cmd["attack"]
            drive.loc.relocalise()
        drive.speed_limit = number(cmd, "auto_limit", 0.05, 1.0)
        drive.set_auto(bool(cmd.get("auto")))
        if drive.auto or drive.turn_on:
            last_command = time.monotonic()
            return
    speeds = wheel_targets(cmd)
    dribbler_speed = number(cmd, "dribbler_speed", -MAX_DRIBBLER, MAX_DRIBBLER)
    motors.set(speeds)
    motors.dribbler_on_speed = int(dribbler_speed)
    motors.set_dribbler(bool(cmd.get("dribbler")) and motors.has_dribbler)
    last_command = time.monotonic()


def kick(cmd):
    if kicker is None:
        return {"kicked": False, "why": "this robot has no kicker (KICKER_PIN is None)"}
    kicker.pulse = number(cmd, "pulse", 0.0, MAX_KICK_PULSE)
    if kicker.kick():
        return {"kicked": True, "why": f"pulse {kicker.pulse} s"}
    return {"kicked": False, "why": "still cooling down"}


def state():
    return {
        "wheels": motors.speeds,
        "dribbler": motors.dribbler_speed,
        "compass": None if imu is None else round(math.degrees(imu.compass()), 1),
        "kick_ready": kicker is not None and kicker.ready(),
        "camera_fps": None if camera is None else round(camera.fps, 1),
        "auto": drive is not None and (drive.auto or drive.turn_on),
        "brain": None if drive is None else brain_state(),
    }


def brain_state():
    w, pose, brain = drive.world, drive.loc.get(), guard.brain
    s = brain.strategy
    r1 = lambda v: None if v is None else [round(float(a), 1) for a in v]
    intent = getattr(brain, "intent", None)
    return {
        "attack": drive.attack,
        "botstate": s.botstate,
        "phase": s.phase,
        "possession": getattr(s, "possession", None),
        "command": getattr(s, "command", None),
        "chasing": s.chasing,
        "kick": drive.intent_kick,
        "dribbler": None if intent is None else bool(intent[2]),
        "target_cm": None if intent is None else r1(intent[0]),
        "face_deg": None if intent is None else round(math.degrees(intent[1]), 1),
        "pose": {"x": round(pose.x, 1), "y": round(pose.y, 1), "std": round(pose.std, 1),
                 "confident": pose.confident},
        "ball": r1(w.ball),
        "ball_source": w.ball_source,
        "has_ball": w.ball_in_capture,
        "goals": w.goal_source,
        "robots_seen": len(w.obstacles),
        "opponents": [r1(o) for o in w.opponents],
        "status": brain.status,
        "tick_errors": guard.errors,
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith(("/camera.jpg", "/field.jpg")):
            picture = camera_jpeg() if self.path.startswith("/camera") else field_jpeg()
            if picture is None:
                return self.reply(503, "text/plain", b"no picture yet")
            return self.reply(200, "image/jpeg", picture)
        if "__DEFAULTS__" in PAGE:          # main() is still starting the hardware
            return self.reply(200, "text/html; charset=utf-8", STARTING_PAGE.encode())
        self.reply(200, "text/html; charset=utf-8", PAGE.encode())

    def do_POST(self):
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            cmd = json.loads(body or b"{}")
            if self.path == "/cmd":
                apply_command(cmd)
                answer = state()
            elif self.path == "/kick":
                answer = kick(cmd)
            elif self.path in ("/zero", "/relocalise") and drive is not None:
                if self.path == "/zero":
                    drive.imu.zero()
                drive.loc.relocalise()
                answer = {"ok": True}
            else:
                return self.reply(404, "text/plain", b"not found")
        except (ValueError, TypeError, AttributeError) as e:
            return self.reply(400, "text/plain", f"bad command: {e}".encode())
        self.reply(200, "application/json", json.dumps(answer).encode())

    def reply(self, code, kind, body):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):       # the page sends 10 commands a second: do not print each one
        pass


def problem(text):
    """A part that did not start: printed here and listed at the top of the page."""
    print(text)
    problems.append(text)


def start_hardware(role):
    """Starts each part on its own, so a missing camera, dribbler or other part only takes that part away."""
    global motors, kicker, imu, camera, drive, guard
    if role is None:
        motors, kicker, camera = FakeMotors(), FakeKicker(), FakeCamera()
        return {"dribbler_speed": FakeMotors.dribbler_on_speed, "kick_pulse": FakeKicker.pulse}

    hw = cfg.AttackRobot if role == "attack" else cfg.DefenseRobot
    try:
        from hardware import MotorThread
        try:
            motors = MotorThread(hw.DRIVE_MOTORS, hw.DRIBBLER, hw.DRIBBLER_SPEED)
        except Exception as e:
            if hw.DRIBBLER is None:
                raise
            problem(f"No dribbler (its motor driver did not answer: {e!r}). Trying the wheels without it.")
            motors = MotorThread(hw.DRIVE_MOTORS, None, 0)
        motors.start()
    except Exception as e:
        problem(f"No wheels (motor drivers did not start: {e!r})")
        motors = FakeMotors()
        motors.has_dribbler = False
    try:
        from hardware import make_kicker
        kicker = make_kicker(hw)
    except Exception as e:
        problem(f"No kicker (it did not start: {e!r})")
    try:
        from hardware import IMUThread
        imu = IMUThread()
        imu.start()
    except Exception as e:
        problem(f"No compass (IMU did not start: {e!r})")
    try:
        from vision import VisionThread
        compass = imu.compass if imu is not None else (lambda: 0.0)
        camera = VisionThread(compass, keep_frames=True)    # keep_frames: it keeps the newest picture for us
        camera.start()
    except Exception as e:
        camera = None
        problem(f"No camera view (camera did not start: {e!r})")
    if camera is not None and imu is not None:
        try:
            drive = DriveBot(hw, motors, kicker, imu, camera)
            if role == "attack":
                from main_attack import StrikerBrain as brain_cls
            else:
                from main_defence import GoalieBrain as brain_cls
            guard = TickGuard(brain_cls(drive))
            record_intent(guard.brain)
        except Exception as e:
            problem(f"No autonomous mode (could not start it: {e!r})")
            drive = guard = None
    else:
        problem("No autonomous mode: it needs the camera and the IMU")
    return {"dribbler_speed": hw.DRIBBLER_SPEED if motors.has_dribbler else 0, "kick_pulse": hw.KICK_PULSE}


def stop_hardware():
    if motors is None:                  # Ctrl+C before the hardware finished starting
        return
    motors.stop()
    time.sleep(0.05)                    # lets the motor thread send the stop
    for driver in motors.drivers:
        driver.clear_faults()
    motors.running = False
    if drive is not None:
        drive.stop_threads()
    if imu is not None:
        imu.running = False
    if kicker is not None:
        kicker.close()
    if camera is not None:
        camera.stop()
        camera.close()


def main():
    global PAGE
    parser = argparse.ArgumentParser(description="Drive the robot by hand from a web page.")
    parser.add_argument("role", nargs="?", choices=["attack", "defense"], help="which robot this is")
    parser.add_argument("--fake", action="store_true", help="no robot: just try the page")
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = reachable from other computers")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--int", action="store_true", help="international field and strategy (strategy_int.py)")
    args = parser.parse_args()
    if args.fake == (args.role is not None):
        parser.error("give attack or defense, or --fake")

    # the page server starts first, so the page loads (with a "starting" note) even while a part is slow to start
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Open http://localhost:{args.port}   (Ctrl+C here to quit)")
    try:
        defaults = start_hardware(None if args.fake else args.role)
        defaults["camera"] = camera is not None
        defaults["auto"] = drive is not None
        defaults["problems"] = problems
        defaults["dribbler"] = motors.has_dribbler
        defaults["base_speed"] = cfg.BASE_SPEED
        defaults["turn_max"] = cfg.TURN_MAX
        defaults["max_speed"] = MAX_SPEED
        defaults["max_kick_pulse"] = MAX_KICK_PULSE
        PAGE = PAGE.replace("__DEFAULTS__", json.dumps(defaults))
        print("Hardware started" + (f", {len(problems)} part(s) missing (listed above)" if problems else ""))
        next_loop = time.monotonic()
        while True:
            next_loop = keep_rate(next_loop)
            if time.monotonic() - last_command > STOP_AFTER:
                if drive is not None:
                    drive.set_auto(False)
                motors.stop()
            if drive is not None:
                drive.step(guard)
    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        server.shutdown()
        stop_hardware()


STARTING_PAGE = """<!doctype html>
<meta charset="utf-8"><meta http-equiv="refresh" content="1">
<title>Robot manual drive</title>
<body style="font-family: sans-serif; background: #14171c; color: #e8eaed; padding: 16px">
Starting the robot's parts... this page reloads by itself. If it stays here, look at the terminal.
</body>
"""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Robot manual drive</title>
<style>
  body { font-family: sans-serif; max-width: 760px; margin: 0 auto; padding: 16px; background: #14171c; color: #e8eaed; }
  h1 { font-size: 20px; }
  fieldset { border: 1px solid #3a4150; border-radius: 8px; margin: 0 0 12px; padding: 12px; }
  legend { padding: 0 6px; color: #9fb4d0; }
  label { display: block; margin: 8px 0; }
  input[type=range] { width: 100%; }
  input[type=number] { width: 150px; font-size: 16px; background: #222833; color: inherit; border: 1px solid #3a4150;
                       border-radius: 6px; padding: 6px; }
  button { font-size: 16px; padding: 14px 0; border-radius: 8px; border: 1px solid #3a4150; background: #222833;
           color: inherit; touch-action: none; user-select: none; cursor: pointer; }
  button.on, button:active { background: #2f6fdf; }
  #pad { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
  #wheels { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
  #stop { width: 100%; background: #b3261e; font-weight: bold; margin-bottom: 12px; }
  #kick { width: 100%; }
  #camera { display: block; width: 100%; max-width: 640px; margin: 0 auto; }
  #field { display: block; width: 100%; max-width: 420px; margin: 8px auto 0; }
  #auto { width: 100%; font-weight: bold; }
  #auto.on { background: #b3261e; }
  select { font-size: 16px; background: #222833; color: inherit; border: 1px solid #3a4150; border-radius: 6px; padding: 6px; }
  #status { font-family: monospace; white-space: pre-wrap; }
  .hint { color: #9aa3b2; font-size: 14px; }
  .bad { color: #ff8a80; }
</style>
</head>
<body>
<h1>Robot manual drive</h1>
<div id="problems" class="bad" hidden></div>
<button id="stop">STOP (Space)</button>

<fieldset id="autoBox">
  <legend>Autonomous (the robot's own strategy drives)</legend>
  <button id="auto">Autonomous is OFF (T)</button>
  <label>Attack goal <select id="attack"><option>yellow</option><option>blue</option></select></label>
  <label>Autonomous speed limit: <b id="limitText"></b>
    <input id="limit" type="range" min="5" max="100" step="5" value="50"></label>
  <div id="pad2" style="display:flex; gap:8px">
    <button id="zero" style="flex:1">Zero heading (facing attack goal)</button>
    <button id="reloc" style="flex:1">Relocalise</button>
  </div>
  <p class="hint">The camera view always shows what the strategy wants. Thick = seen now; thin = remembered or worked
  out from the position; "&gt;" = further than the camera sees. Cyan arrow = where it wants to go, light green arrow =
  which way it wants to face, pink arrow = escaping a line. STOP, Space or closing the page switches it off.</p>
  <div id="brain" style="font-family: monospace; white-space: pre-wrap"></div>
</fieldset>

<fieldset id="cameraBox">
  <legend>Camera (what the robot's vision sees)</legend>
  <img id="camera" alt="camera view">
  <img id="field" alt="field view: where the robot thinks it is">
</fieldset>

<fieldset>
  <legend>Move and turn (hold a button or key)</legend>
  <div id="pad">
    <button data-key="q">Q<br>turn left</button>
    <button data-key="w">W<br>forward</button>
    <button data-key="e">E<br>turn right</button>
    <button data-key="a">A<br>left</button>
    <button data-key="s">S<br>back</button>
    <button data-key="d">D<br>right</button>
  </div>
  <label>Drive speed: <b id="speedText"></b>
    <input id="speed" type="range" min="0" step="1" value="30"></label>
  <label>Turn speed: <b id="turnText"></b>
    <input id="turn" type="range" min="0" max="100" step="1"></label>
</fieldset>

<fieldset>
  <legend>Single wheel (hold) - checks the motor order and direction</legend>
  <div id="wheels">
    <button data-key="1">Wheel 1</button>
    <button data-key="2">Wheel 2</button>
    <button data-key="3">Wheel 3</button>
    <button data-key="4">Wheel 4</button>
  </div>
  <p class="hint">Uses the drive speed above. Order = DRIVE_MOTORS in robot_config.py.</p>
</fieldset>

<fieldset>
  <legend>Dribbler</legend>
  <button id="dribbler" style="width:100%">Dribbler is OFF (R)</button>
  <label>DRIBBLER_SPEED (a minus sign spins it the other way)
    <input id="dribblerSpeed" type="number" step="10000000"></label>
</fieldset>

<fieldset>
  <legend>Kicker</legend>
  <button id="kick">KICK (K)</button>
  <label>KICK_PULSE in seconds (most allowed: <span id="maxPulse"></span>)
    <input id="pulse" type="number" step="0.005" min="0"></label>
  <div id="kickText" class="hint"></div>
</fieldset>

<fieldset>
  <legend>Robot</legend>
  <div id="status">connecting...</div>
</fieldset>

<script>
const D = __DEFAULTS__;
const $ = id => document.getElementById(id);
const held = new Set();          // keys and buttons held down right now
let dribblerOn = false;
let autoOn = false;

$("speed").max = D.max_speed * 100;
$("turn").value = D.turn_max * 100;
$("dribblerSpeed").value = D.dribbler_speed;
$("pulse").value = D.kick_pulse;
$("pulse").max = D.max_kick_pulse;
$("maxPulse").textContent = D.max_kick_pulse;

function showSliders() {
  const speed = $("speed").value / 100, turn = $("turn").value / 100;
  $("speedText").textContent = speed.toFixed(2) + " x BASE_SPEED = " + Math.round(speed * D.base_speed);
  $("turnText").textContent = "TURN_MAX " + turn.toFixed(2);
}
$("limitText").textContent = $("limit").value + "%";
$("limit").oninput = () => { $("limitText").textContent = $("limit").value + "%"; };
$("speed").oninput = $("turn").oninput = showSliders;
showSliders();

function showHeld() {
  document.querySelectorAll("button[data-key]").forEach(b => b.classList.toggle("on", held.has(b.dataset.key)));
  $("dribbler").classList.toggle("on", dribblerOn);
  if (D.dribbler) $("dribbler").textContent = "Dribbler is " + (dribblerOn ? "ON" : "OFF") + " (R)";
  $("auto").classList.toggle("on", autoOn);
  $("auto").textContent = "Autonomous is " + (autoOn ? "ON - press to stop" : "OFF") + " (T)";
}
function stopAll() { held.clear(); dribblerOn = false; autoOn = false; showHeld(); send(); }
function toggleAuto() { autoOn = !autoOn && D.auto; held.clear(); showHeld(); send(); }

document.querySelectorAll("button[data-key]").forEach(b => {
  b.onpointerdown = () => { held.add(b.dataset.key); showHeld(); };
  b.onpointerup = b.onpointerleave = b.onpointercancel = () => { held.delete(b.dataset.key); showHeld(); };
});
$("stop").onclick = stopAll;
$("dribbler").onclick = () => { dribblerOn = !dribblerOn; showHeld(); };
$("kick").onclick = kick;
$("auto").onclick = toggleAuto;
$("zero").onclick = () => post("/zero", {}).catch(() => {});
$("reloc").onclick = () => post("/relocalise", {}).catch(() => {});

document.addEventListener("keydown", e => {
  if (e.target.type === "number") return;          // typing a number is not driving
  const key = e.key.toLowerCase();
  if (key === " ") { e.preventDefault(); stopAll(); }
  else if (e.repeat) return;
  else if (key === "r") { dribblerOn = !dribblerOn; showHeld(); }
  else if (key === "k") kick();
  else if (key === "t") toggleAuto();
  else if ("wasdqe1234".includes(key)) { held.add(key); showHeld(); }
});
document.addEventListener("keyup", e => { held.delete(e.key.toLowerCase()); showHeld(); });
window.addEventListener("blur", stopAll);           // another window in front = keys can get stuck, so stop

const axis = (plus, minus) => (held.has(plus) ? 1 : 0) - (held.has(minus) ? 1 : 0);

async function post(path, data) {
  const r = await fetch(path, {method: "POST", body: JSON.stringify(data)});
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function send() {
  const wheel = ["1", "2", "3", "4"].findIndex(k => held.has(k));
  try {
    const s = await post("/cmd", {
      x: axis("d", "a"), y: axis("w", "s"), rot: axis("q", "e"),
      speed: $("speed").value / 100, turn: $("turn").value / 100,
      wheel: wheel < 0 ? null : wheel,
      dribbler: dribblerOn, dribbler_speed: Number($("dribblerSpeed").value) || 0,
      auto: autoOn, attack: $("attack").value, auto_limit: $("limit").value / 100,
    });
    if (autoOn && !s.auto) { autoOn = false; showHeld(); }
    if (s.brain) showBrain(s.brain);
    $("status").className = "";
    $("status").textContent = "wheel speeds sent: " + s.wheels.join(", ") +
      "\\ndribbler speed sent: " + s.dribbler +
      "\\ncompass: " + (s.compass === null ? "no IMU" : s.compass + " deg") +
      "\\nkicker: " + (s.kick_ready ? "ready" : "not ready") +
      "\\ncamera: " + (s.camera_fps === null ? "none" : s.camera_fps + " frames per second (vision speed)");
  } catch (err) {
    $("status").className = "bad";
    $("status").textContent = "NOT CONNECTED - the robot has stopped itself. " + err.message;
  }
}

function showBrain(b) {
  const f = v => v === null || v === undefined ? "-" : v;
  $("brain").textContent =
    "doing: b" + b.botstate + " " + b.phase + (b.kick ? "  KICK" : "") + (b.has_ball ? "  HAS BALL" : "") +
    "\\npossession: " + f(b.possession) + "   chasing: " + b.chasing + "   dribbler: " + f(b.dribbler) +
    "\\ntarget: " + f(b.target_cm && b.target_cm.join(", ")) + " cm from the robot, face " + f(b.face_deg) + " deg" +
    "\\nposition: (" + b.pose.x + ", " + b.pose.y + ") +-" + b.pose.std + " cm " + (b.pose.confident ? "ok" : "LOST") +
    "\\nball: " + f(b.ball && b.ball.join(", ")) + " (" + b.ball_source + ")   goals: " + b.goals +
    "\\nrobots seen: " + b.robots_seen + "   opponents known: " + b.opponents.length +
    (b.command !== null ? "   command: " + b.command : "") +
    (b.tick_errors ? "\\nSTRATEGY CRASHES: " + b.tick_errors + " (see the terminal)" : "");
}

async function kick() {
  try {
    const k = await post("/kick", {pulse: Number($("pulse").value) || 0});
    $("kickText").textContent = (k.kicked ? "Kicked: " : "No kick: ") + k.why;
  } catch (err) { $("kickText").textContent = "No kick: " + err.message; }
}

function keepLoading(img, path) {    // load the next picture a moment after each one arrives
  const next = () => { img.src = path + "?" + Date.now(); };
  img.onload = () => setTimeout(next, 100);
  img.onerror = () => setTimeout(next, 1000);
  next();
}
if (D.problems.length) {
  $("problems").hidden = false;
  $("problems").innerText = "Not connected (the rest still works):\\n- " + D.problems.join("\\n- ");
}
if (!D.dribbler) { $("dribbler").disabled = true; $("dribbler").textContent = "No dribbler on this robot"; }
if (D.camera) keepLoading($("camera"), "/camera.jpg"); else $("cameraBox").hidden = true;
if (D.auto) keepLoading($("field"), "/field.jpg"); else { $("field").hidden = true; $("autoBox").hidden = true; }

setInterval(send, 100);          // also the "I am still here" signal: the robot stops 0.3 s after the last one
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()

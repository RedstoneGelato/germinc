import json
import math
import os
import threading
import time
import traceback

import robot_config as cfg
from lines import LineFusion
from motion import Mover
from perception import World
from utils import other_goal, wrap_pi

# TUNE
CAMERA_BLIND = 0.3
VISION_STALL = 1.5
COMMS_STALL = 1.0
READY_TIMEOUT = 10.0
WATCHDOG_PERIOD = 1.0
STATE_FILE = "/dev/shm/germinc_state.json"   # RAM disk: survives a program restart, cleared on reboot
STATE_MAX_AGE = 30.0
CAM_HEADING_WRONG = math.radians(100)
CAM_HEADING_SNAP = math.radians(35)
CAM_VOTES_PAUSED = 20
CAM_VOTES_RUNNING = 50
MIN_GOAL_SEPARATION = 50.0
OWN_GOAL_NO_KICK = math.radians(30)
LED_STEP = 50
LED_MAX = 65535
LED_HEADROOM = 3000


class Robot:
    goal_switch = None
    zeroed_for = None
    _pause_shift = 0.0
    _free_zero = False
    _cam_votes = 0
    _cam_heading = 0.0

    def __init__(self, hw):
        from comms import TeammateLinkThread      # Pi-only imports, so the simulator can use this class
        from hardware import IMUThread, PCBThread
        from localisation import LocalisationThread
        from vision import VisionThread

        self.imu = IMUThread()
        self.imu.start()
        self.motors = hw.make_motors()
        self.motors.start()
        self.kicker = hw.make_kicker()
        self.pcb = PCBThread()
        self.pcb.start()
        self.comms = TeammateLinkThread(hw.ROBOT_ID)
        self.comms.start()
        self.vision = VisionThread(self.imu.compass)
        self.vision.start()

        self.attack = "yellow"
        self.loc = LocalisationThread(self.vision, lambda: self.attack)
        self.loc.start()

        self.world = World()
        self.lines = LineFusion()
        self.mover = Mover()
        self.leds = LedCalibrator(self.pcb)

        self.hw = hw
        self._next_check = 0.0
        self._restarting = set()
        self.restarts = {"vision": 0, "comms": 0}
        self.load_state()

    def wait_ready(self, timeout=READY_TIMEOUT):
        print("Waiting for sensors...")
        give_up_at = time.monotonic() + timeout
        while not (self.imu.ready and self.vision.ready and self.pcb.ready):
            if time.monotonic() > give_up_at:
                print(f"Sensors not all ready after {timeout:.0f} s (imu {self.imu.ready}, camera {self.vision.ready}, "
                      f"pcb {self.pcb.ready}) - running anyway")
                return
            time.sleep(0.05)

    def check_threads(self):
        now = time.monotonic()
        if now < self._next_check:
            return
        self._next_check = now + WATCHDOG_PERIOD
        try:
            vision, comms = self.vision, self.comms
            if not vision.is_alive() or now - vision.last_frame_t > VISION_STALL:
                self._restart("vision", self._new_vision)
            if not comms.is_alive() or now - comms.last_loop_t > COMMS_STALL:
                self._restart("comms", self._new_comms)
            self.save_state()
        except Exception:
            print("check_threads failed:")
            traceback.print_exc()

    def _restart(self, name, make):
        if name in self._restarting:
            return
        self._restarting.add(name)
        self.restarts[name] += 1
        print(f"[watchdog] {name} thread dead or stuck - restarting ({self.restarts[name]})")

        def restart_in_background():
            try:
                make()
                print(f"[watchdog] {name} restarted")
            except Exception:
                print(f"[watchdog] {name} restart failed (trying again in a second):")
                traceback.print_exc()
            finally:
                self._restarting.discard(name)
        threading.Thread(target=restart_in_background, daemon=True).start()

    def _new_vision(self):
        from vision import VisionThread
        old = self.vision
        old.stop()
        old.join(timeout=1.0)
        old.close()
        new = VisionThread(self.imu.compass, first_frame_id=old.frame_id + 1)
        new.start()
        self.vision = new
        self.loc.vision = new

    def _new_comms(self):
        from comms import TeammateLinkThread
        old = self.comms
        old.stop()
        old.join(timeout=1.0)
        new = TeammateLinkThread(self.hw.ROBOT_ID)
        new.my_state = old.my_state
        new.enabled = old.enabled
        new.start()
        self.comms = new

    def save_state(self):
        state = {"t": time.time(), "compass": self.imu.compass(), "attack": self.attack,
                 "brightness": self.leds.brightness}
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, STATE_FILE)

    def load_state(self):
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            age = time.time() - state["t"]
        except (OSError, ValueError, KeyError, TypeError):
            return
        if not 0 <= age < STATE_MAX_AGE:
            return
        print(f"Restarted {age:.0f} s after the last run: carrying on heading {state['compass']:.2f} rad, "
              f"attacking {state['attack']}")
        self.imu.continue_from(state["compass"])
        self.attack = state["attack"]
        self.zeroed_for = self.attack
        self.leds.brightness = state["brightness"]
        self.pcb.set_brightness(self.leds.brightness)

    def actuate(self, speeds, dribbler_on, kick):
        self.motors.set(speeds)
        self.motors.set_dribbler(dribbler_on)
        if kick and self.kicker is not None:
            self.kicker.kick()

    def attack_goal(self, det):
        if self.goal_switch is None or cfg.GOAL_FROM_CAMERA:
            return goal_colour_from_view(det, self.attack)
        return goal_colour_from_switch(self.goal_switch.is_active)

    def standby(self, colours, det):
        self.motors.stop()
        self.attack = self.attack_goal(det)
        self.leds.step(colours)
        self.zero_heading(det)

    def zero_heading(self, det=None):
        heading_before_pause = wrap_pi(self.imu.compass() + self._pause_shift)
        if self.zeroed_for != self.attack:
            self._free_zero = True
            self.zeroed_for = self.attack
        put_down_facing_goal = abs(heading_before_pause) < cfg.ZERO_MAX_ERR
        zero = self._free_zero or not cfg.REF_MODULE or put_down_facing_goal
        facing = 0.0 if zero else None
        seen = self._camera_says_turned(det, 0.0 if zero else self.imu.compass(), CAM_VOTES_PAUSED)
        if seen is not None:
            facing = seen
        if facing is not None:
            self.imu.zero(facing)
            self._pause_shift = wrap_pi(heading_before_pause - facing)

    def check_heading(self, det, dh=0.0):
        seen = self._camera_says_turned(det, wrap_pi(self.imu.compass() + dh), CAM_VOTES_RUNNING, both_only=True)
        if seen is None:
            return False
        self.imu.zero(wrap_pi(seen - dh))
        self._cam_votes = 0
        self.loc.relocalise()
        return True

    def _camera_says_turned(self, det, compass, votes_needed, both_only=False):
        seen = heading_from_goals(det, self.attack, both_only)
        if seen is not None:
            if abs(wrap_pi(seen - compass)) > CAM_HEADING_WRONG:
                self._cam_votes += 1
                self._cam_heading = seen
            else:
                self._cam_votes = 0
        return self._cam_heading if self._cam_votes >= votes_needed else None

    def on_pause(self):
        self.world.reset()
        self.mover.reset()
        self.loc.relocalise()
        self._pause_shift = 0.0
        self._free_zero = False
        self._cam_votes = 0

    def shutdown(self):
        print("Shutting down safely...")

        def attempt(what, fn):
            try:
                fn()
            except Exception as e:
                print(f"shutdown: {what} failed: {e!r}")

        attempt("stop motors", self.motors.stop)
        if self.kicker is not None:
            attempt("kicker", self.kicker.close)
        for driver in self.motors.drivers:
            attempt("clear faults", driver.clear_faults)
        time.sleep(0.05)      # lets the motor thread send the stop

        for thread in (self.loc, self.motors, self.imu, self.pcb):
            thread.running = False
        attempt("stop camera", self.vision.stop)
        attempt("stop comms", self.comms.stop)
        for thread in (self.vision, self.loc, self.motors, self.imu, self.pcb, self.comms):
            attempt("join", lambda: thread.join(timeout=1.0))
        attempt("close camera", self.vision.close)
        print("Robot stopped.")


class CameraWatch:
    def __init__(self):
        self.frame_id = None
        self.t = time.monotonic()

    def blind(self, frame_id):
        now = time.monotonic()
        if frame_id != self.frame_id:
            self.frame_id, self.t = frame_id, now
        return now - self.t > CAMERA_BLIND


class Brain:
    idle_state = {"bot active": 0}

    def __init__(self, bot, strategy, log=print):
        self.bot = bot
        self.log = log
        self.strategy = strategy
        self.robot_active = None
        self.status = ""
        self.camera = CameraWatch()

    def tick(self, paused):
        bot = self.bot
        strategy = self.strategy
        colours = bot.pcb.snapshot()
        frame_id, det = bot.vision.frame_id, bot.vision.latest()
        blind = self.camera.blind(frame_id)

        if paused:
            if self.robot_active or self.robot_active is None:
                self.log("Paused")
                self.robot_active = False
                strategy.reset()
                bot.on_pause()
            bot.comms.my_state.update(self.idle_state)
            bot.standby(colours, det)
            return
        if not self.robot_active:
            self.log(f"running, attacking {bot.attack}")
            self.robot_active = True
        if blind:
            bot.actuate((0, 0, 0, 0), getattr(self, "dribbler_on", False), False)
            bot.comms.my_state.update(self.idle_state)
            self.status = "camera blind - standing still"
            return

        pose = bot.loc.get()
        if bot.check_heading(det, pose.dh):
            self.log("heading turned round from the goals")
            pose = bot.loc.get()
        compass = wrap_pi(bot.imu.compass() + pose.dh)
        mate = bot.comms.teammate()
        world = bot.world
        ir = bot.pcb.ir_snapshot() if hasattr(bot.pcb, "ir_snapshot") else None
        world.update(frame_id, det, pose, bot.attack, mate, ir)
        line = bot.lines.update(colours, compass, world.line_pts, pose)

        desired_pos, desired_heading, dribbler_on = strategy.update(world, compass, mate, line.touches)
        speeds = bot.mover.step(desired_pos, desired_heading, compass, line, pose, **strategy.speed,
                                dribbler_on=dribbler_on)
        kick = strategy.kick and not own_goal_ahead(det, bot.attack)
        bot.actuate(speeds, dribbler_on, kick)
        if hasattr(strategy, "note_kick"):
            strategy.note_kick()

        bot.comms.my_state.update({
            "bot active": 1,
            **self.extra_state(),
            "chasing": strategy.chasing,
            "has ball": world.ball_in_capture,
            "pos": [round(pose.x), round(pose.y)] if pose.confident else None,
            "ball": [round(v) for v in world.ball_own_field] if world.ball_own_field else None,
            "opps": [[round(v) for v in o] for o in world.opponents_own],
        })

        self.desired_pos, self.line, self.dribbler_on = desired_pos, line, dribbler_on
        line_text = "ON " + line.source if line.on_line else "off"
        self.status = (f"botstate={strategy.botstate} {strategy.phase} ball:{strategy.possession or '-'} line={line_text} "
                       f"pos=({pose.x:.0f},{pose.y:.0f})+-{pose.std:.0f} ball={world.ball}({world.ball_source}) "
                       f"goals={world.goal_source}")

    def extra_state(self):
        return {}


class TickGuard:
    RESET_AFTER = 20

    def __init__(self, brain, log=print):
        self.brain = brain
        self.log = log
        self.errors = 0
        self.in_row = 0

    def tick(self, paused):
        try:
            self.brain.tick(paused)
        except Exception as e:
            self._crashed(e)
            return False
        if self.in_row:
            self.log(f"tick() working again after {self.in_row} bad loop(s)")
            self.in_row = 0
        return True

    def _crashed(self, e):
        self.errors += 1
        self.in_row += 1
        bot = self.brain.bot
        try:
            bot.motors.stop()
        except Exception:
            pass
        if self.in_row == 1:
            self.log(f"tick() crashed ({self.errors} so far), carrying on: {e!r}\n{traceback.format_exc()}")
        if self.in_row == self.RESET_AFTER:
            self.log(f"tick() crashed {self.in_row} loops in a row - resetting strategy, world and mover")
            for reset in (self.brain.strategy.reset, bot.world.reset, bot.mover.reset):
                try:
                    reset()
                except Exception:
                    pass


def _pause_input():
    from gpiozero import DigitalInputDevice
    if cfg.REF_MODULE:      # comms module OUT1: 3.3 V = play, pulled down so unplugged = stopped
        pin = DigitalInputDevice(cfg.SWITCH_PIN, pull_up=False)
        return lambda: not pin.is_active
    pin = DigitalInputDevice(cfg.SWITCH_PIN, pull_up=True)
    return lambda: pin.is_active


def run_robot(hw, brain_cls):
    from gpiozero import DigitalInputDevice

    is_paused = _pause_input()
    bot = Robot(hw)
    bot.goal_switch = DigitalInputDevice(cfg.GOAL_SWITCH_PIN, pull_up=True)
    if not cfg.GOAL_FROM_CAMERA:
        bot.attack = bot.attack_goal(None)
        print(f"Goal switch: attacking {bot.attack}")
    guard = TickGuard(brain_cls(bot))
    bot.wait_ready()

    print("Waiting for signal")
    failed = False
    try:
        next_loop = time.monotonic()
        while True:
            paused = is_paused()
            guard.tick(paused)
            bot.check_threads()
            if paused:
                time.sleep(0.02)
                next_loop = time.monotonic()
                continue
            print(guard.brain.status)
            next_loop = keep_rate(next_loop)
    except KeyboardInterrupt:
        print("User stopped.")
    except Exception as e:
        print(f"Unexpected error: {e!r}")
        traceback.print_exc()
        failed = True
    finally:
        bot.shutdown()
    if failed:
        raise SystemExit(1)


class LedCalibrator:
    def __init__(self, pcb):
        self.pcb = pcb
        self.brightness = cfg.LED_BRIGHTNESS_START
        pcb.set_brightness(self.brightness)

    def step(self, colours):
        brightest = max(colours)
        if brightest - LED_HEADROOM > cfg.LDR_LINE_THRESHOLD:
            self.brightness += LED_STEP
        elif brightest < cfg.LDR_LINE_THRESHOLD:
            self.brightness -= LED_STEP
        self.brightness = max(min(self.brightness, LED_MAX), 0)
        self.pcb.set_brightness(self.brightness)


def goal_colour_from_switch(closed):
    return cfg.GOAL_SWITCH_CLOSED if closed else other_goal(cfg.GOAL_SWITCH_CLOSED)


def goal_colour_from_view(det, current):
    if det is None:
        return current
    yellow, blue = det.goals["yellow"], det.goals["blue"]
    if yellow is not None:
        return "yellow" if yellow.centre[1] > 0 else "blue"
    if blue is not None:
        return "blue" if blue.centre[1] > 0 else "yellow"
    return current


def _snap_to_axis(heading):
    if abs(heading) < CAM_HEADING_SNAP:
        return 0.0
    if abs(wrap_pi(heading - math.pi)) < CAM_HEADING_SNAP:
        return math.pi
    return heading


def heading_from_goals(det, attack, both_only=False):
    if det is None:
        return None
    ahead, behind = det.goals.get(attack), det.goals.get(other_goal(attack))
    if ahead is not None and behind is not None:
        vx, vy = ahead.centre[0] - behind.centre[0], ahead.centre[1] - behind.centre[1]
        return math.atan2(vx, vy) if vx * vx + vy * vy > MIN_GOAL_SEPARATION ** 2 else None
    if both_only or (ahead is None and behind is None):
        return None
    if ahead is not None:
        return _snap_to_axis(math.atan2(ahead.centre[0], ahead.centre[1]))
    return _snap_to_axis(wrap_pi(math.atan2(behind.centre[0], behind.centre[1]) + math.pi))


def own_goal_ahead(det, attack):
    if det is None:
        return False
    own = det.goals.get(other_goal(attack))
    return own is not None and own.centre[1] > 0 and abs(math.atan2(own.centre[0], own.centre[1])) < OWN_GOAL_NO_KICK


def keep_rate(next_loop):
    next_loop += cfg.CONTROL_PERIOD
    sleep_time = next_loop - time.monotonic()
    if sleep_time > 0:
        time.sleep(sleep_time)
        return next_loop
    return time.monotonic()

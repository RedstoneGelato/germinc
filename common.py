"""
common.py - start-up / standby / shutdown pieces both main.py (goalie) and main_attack.py (striker) use,
and the 100 Hz loop they both run (run_robot), which keeps going through crashes:
    TickGuard        an exception in tick() stops the motors for that loop and the next loop carries on
    check_threads()  restarts the camera or comms thread if it died or stopped producing
    save_state()     heading / attack goal / LED brightness go to STATE_FILE every second, so when systemd restarts
                     the whole program mid-game (deploy/germinc.service, Restart=always) the robot carries on facing
                     the right way without being paused to recalibrate
"""
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
from utils import wrap_pi

# ---- TUNE
CAMERA_BLIND = 0.3        # s without a new camera frame -> the robot stands still until frames come again
VISION_STALL = 1.5        # s without a new camera frame -> restart the camera
COMMS_STALL = 1.0         # s without the comms thread going round its loop -> restart it
READY_TIMEOUT = 10.0      # s to wait for the sensors at start-up before running anyway
STATE_FILE = "/dev/shm/germinc_state.json"   # RAM disk: survives a program restart, cleared when the Pi reboots
STATE_MAX_AGE = 30.0      # s: older saved state is from an earlier run, ignore it
CAM_HEADING_WRONG = math.radians(100)  # the goals put our heading this far from the compass = the compass is wrong
CAM_HEADING_SNAP = math.radians(35)    # one goal seen: a heading this close to 0 / 180 deg is taken as exactly that
CAM_VOTES_PAUSED = 20     # loops in a row the camera must disagree while paused before the heading is turned round
CAM_VOTES_RUNNING = 50    # same while playing (both goals in view only)
OWN_GOAL_NO_KICK = math.radians(30)    # never kick while our own goal is within this either side of straight ahead


class Robot:
    """Every thread + helper object one robot needs. hw = hardware_attack or hardware_defense."""
    goal_switch = None # attack-goal switch (gpiozero input), set by run_robot. None (simulator) = goal from the camera
    zeroed_for = None  # attack goal the heading was last zeroed for (None = never zeroed)
    _pause_shift = 0.0 # how far zero_heading has turned compass 0 during this pause
    _free_zero = False # this pause re-zeroes whatever way the robot faces (start-up / goal changed)
    _cam_votes = 0     # loops in a row the goals have said the compass is the wrong way round
    _cam_heading = 0.0 # the heading they said

    def __init__(self, hw):
        # hardware imports live here so simulator.py can use this class without the Pi libraries
        from comms import TeammateLinkThread
        from hardware import IMUThread, PCBThread
        from localisation import LocalisationThread
        from vision import VisionThread

        self.imu = IMUThread()
        self.imu.start()
        self.motors = hw.make_motors()
        self.motors.start()
        self.kicker = hw.make_kicker() # None if this robot has no kicker
        self.pcb = PCBThread()
        self.pcb.start()
        self.comms = TeammateLinkThread(hw.ROBOT_ID)
        self.comms.start()
        self.vision = VisionThread(self.imu.compass)
        self.vision.start()

        self.attack = "yellow" # goal we shoot at, set while paused (goal switch, or the camera)
        self.loc = LocalisationThread(self.vision, lambda: self.attack)
        self.loc.start() #start threads

        self.world = World()
        self.lines = LineFusion()
        self.mover = Mover()
        self.leds = LedCalibrator(self.pcb)

        self.hw = hw
        self._next_check = 0.0
        self._restarting = set()   # names of threads being restarted right now
        self.restarts = {"vision": 0, "comms": 0}
        self.load_state()

    def wait_ready(self, timeout=READY_TIMEOUT):
        print("Waiting for sensors...")
        t_end = time.monotonic() + timeout
        while not (self.imu.ready and self.vision.ready and self.pcb.ready):
            if time.monotonic() > t_end:
                print(f"Sensors not all ready after {timeout:.0f} s (imu {self.imu.ready}, camera {self.vision.ready}, "
                      f"pcb {self.pcb.ready}) - running anyway")
                return
            time.sleep(0.05)

    # ---- keeping going through crashes
    def check_threads(self):
        """Call every loop (does its work once a second): restart the camera / comms thread if it died or stopped
        producing, and save the state for a restart. Never raises."""
        now = time.monotonic()
        if now < self._next_check:
            return
        self._next_check = now + 1.0
        try:
            v, c = self.vision, self.comms
            if not v.is_alive() or now - v.last_frame_t > VISION_STALL:
                self._restart("vision", self._new_vision)
            if not c.is_alive() or now - c.last_loop_t > COMMS_STALL:
                self._restart("comms", self._new_comms)
            self.save_state()
        except Exception:
            print("check_threads failed:")
            traceback.print_exc()

    def _restart(self, name, make):
        """Restart a thread in the background (opening the camera takes a while, and a hung one can block
        stop()): the main loop carries on meanwhile, with stale camera frames / no teammate."""
        if name in self._restarting:
            return
        self._restarting.add(name)
        self.restarts[name] += 1
        print(f"[watchdog] {name} thread dead or stuck - restarting ({self.restarts[name]})")

        def go():
            try:
                make()
                print(f"[watchdog] {name} restarted")
            except Exception:
                print(f"[watchdog] {name} restart failed (trying again in a second):")
                traceback.print_exc()
            finally:
                self._restarting.discard(name)
        threading.Thread(target=go, daemon=True).start()

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
        """What a restarted program needs to carry on mid-game (see load_state)."""
        state = {"t": time.time(), "compass": self.imu.compass(), "attack": self.attack,
                 "brightness": self.leds.brightness}
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, STATE_FILE)

    def load_state(self):
        """Program restarted mid-game (crash -> systemd): the IMU has reset, so carry on from the heading saved a
        moment before (the robot stood still while the program restarted), with the same attack goal and LED
        brightness. Pausing the robot recalibrates everything anyway."""
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
        self.zeroed_for = self.attack # the saved heading is already zeroed for this goal
        self.leds.brightness = state["brightness"]
        self.pcb.set_brightness(self.leds.brightness)

    def actuate(self, speeds, dribbler_on, kick):
        """Send one loop's decisions to the hardware: 4 wheel speeds, dribbler on/off, kick (ignored without a
        kicker or while it recharges)."""
        self.motors.set(speeds)
        self.motors.set_dribbler(dribbler_on)
        if kick and self.kicker is not None:
            self.kicker.kick()

    def attack_goal(self, det):
        """The goal we shoot at: from the goal switch, or from the camera if there's no switch or
        cfg.GOAL_FROM_CAMERA is on."""
        if self.goal_switch is None or cfg.GOAL_FROM_CAMERA:
            return goal_colour_from_view(det, self.attack)
        return goal_colour_from_switch(self.goal_switch.is_active)

    def standby(self, colours, det):
        """One loop while paused: motors off, goal colour (switch or camera), calibrate LEDs and heading."""
        self.motors.stop()
        self.attack = self.attack_goal(det) #goal switch (or camera)
        self.leds.step(colours) #calibrate pcb leds
        self.zero_heading(det) #calibrate imu heading, if the robot was put down facing the goal we attack

    def zero_heading(self, det=None):
        """Facing forward = facing the goal we attack. With our own on/off switch (REF_MODULE off) every pause zeroes
        whatever way the robot faces. With the comms module, only when the robot was really put down that way. Zeroes:
            - whatever way it faces, on the first pause after start-up and when the attack goal changes (half-time:
              put down facing the other goal), for that whole pause
            - otherwise only while it faces within ZERO_MAX_ERR of forward (put down facing the goal again).
              Measured from the heading before this pause.
        And whatever the IMU says, the goals the camera sees win when they put us facing the other way."""
        before = wrap_pi(self.imu.compass() + self._pause_shift) # heading as it was before this pause's zeroing
        if self.zeroed_for != self.attack:
            self._free_zero = True
            self.zeroed_for = self.attack
        zero = self._free_zero or not cfg.REF_MODULE or abs(before) < cfg.ZERO_MAX_ERR
        facing = 0.0 if zero else None
        seen = self._camera_says_turned(det, 0.0 if zero else self.imu.compass(), CAM_VOTES_PAUSED)
        if seen is not None:
            facing = seen
        if facing is not None:
            self.imu.zero(facing)
            self._pause_shift = wrap_pi(before - facing)

    def check_heading(self, det, dh=0.0):
        """While playing: both goals in view and they say we face the other way to the compass (+ localisation's
        correction dh, which can't turn more than 25 deg) for CAM_VOTES_RUNNING loops -> turn the compass round to
        what the goals say and relocalise. Returns True when it did."""
        seen = self._camera_says_turned(det, wrap_pi(self.imu.compass() + dh), CAM_VOTES_RUNNING, both_only=True)
        if seen is None:
            return False
        self.imu.zero(wrap_pi(seen - dh))
        self._cam_votes = 0
        self.loc.relocalise()
        return True

    def _camera_says_turned(self, det, compass, votes, both_only=False):
        """The heading the goals give, once they've disagreed with `compass` by more than CAM_HEADING_WRONG for
        `votes` loops in a row; else None."""
        seen = heading_from_goals(det, self.attack, both_only)
        if seen is not None:
            if abs(wrap_pi(seen - compass)) > CAM_HEADING_WRONG:
                self._cam_votes += 1
                self._cam_heading = seen
            else:
                self._cam_votes = 0
        return self._cam_heading if self._cam_votes >= votes else None

    def on_pause(self):
        """Robot just got paused (picked up and put down somewhere new, or stopped where it is by the referee)."""
        self.world.reset()
        self.mover.reset()
        self.loc.relocalise()
        self._pause_shift = 0.0
        self._free_zero = False
        self._cam_votes = 0

    def shutdown(self):
        print("Shutting down safely...")

        def attempt(what, fn):     # one step failing doesn't stop the rest
            try:
                fn()
            except Exception as e:
                print(f"shutdown: {what} failed: {e!r}")

        # stop motors first
        attempt("stop motors", self.motors.stop)
        if self.kicker is not None:
            attempt("kicker", self.kicker.close)
        for m in self.motors.drivers:
            attempt("clear faults", m.clear_faults)

        # allow motor thread to send stop command
        time.sleep(0.05)

        # stop threads
        for t in (self.loc, self.motors, self.imu, self.pcb):
            t.running = False
        attempt("stop camera", self.vision.stop)
        attempt("stop comms", self.comms.stop)

        # wait for threads
        for t in (self.vision, self.loc, self.motors, self.imu, self.pcb, self.comms):
            attempt("join", lambda: t.join(timeout=1.0))
        attempt("close camera", self.vision.close)

        print("Robot stopped.")


class CameraWatch:
    """No new camera frame for CAMERA_BLIND s (camera frozen / restarting) = blind. The ball, goals and the pose are
    all from the last frame then, relative to where the robot WAS: the brains stand still."""

    def __init__(self):
        self.frame_id = None
        self.t = time.monotonic()

    def blind(self, frame_id):
        """Call every tick with the vision frame_id."""
        now = time.monotonic()
        if frame_id != self.frame_id:
            self.frame_id, self.t = frame_id, now
        return now - self.t > CAMERA_BLIND


class TickGuard:
    """Runs brain.tick() so that one bad loop can't end the game. An exception stops the motors for that loop
    (the next loop sets them again), is printed once (not 100 times a second), and if it keeps happening for
    RESET_AFTER loops in a row the strategy / world / mover are reset in case their state is what's broken."""
    RESET_AFTER = 20          # loops (0.2 s)

    def __init__(self, brain, log=print):
        self.brain = brain
        self.log = log
        self.errors = 0           # bad loops in total
        self.in_row = 0

    def tick(self, paused):
        """Returns True if the loop ran normally."""
        try:
            self.brain.tick(paused)
        except Exception as e:
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
            return False
        if self.in_row:
            self.log(f"tick() working again after {self.in_row} bad loop(s)")
            self.in_row = 0
        return True


def run_robot(hw, brain_cls):
    """The robot program: start everything, then the 100 Hz loop until Ctrl+C. hw = hardware_attack or
    hardware_defense, brain_cls = main_attack.StrikerBrain or main.GoalieBrain."""
    from gpiozero import DigitalInputDevice

    if cfg.REF_MODULE: # comms module OUT1: 3.3 V = PLAY. Pulled down, so unplugged / module off = stopped
        start_pin = DigitalInputDevice(cfg.SWITCH_PIN, pull_up = False)
        def is_paused():
            return not start_pin.is_active
    else: # our own on/off switch, GPIO to ground: closed = paused
        start_pin = DigitalInputDevice(cfg.SWITCH_PIN, pull_up = True)
        def is_paused():
            return start_pin.is_active
    bot = Robot(hw)
    bot.goal_switch = DigitalInputDevice(cfg.GOAL_SWITCH_PIN, pull_up = True) #gpio pin for attack-goal switch
    if not cfg.GOAL_FROM_CAMERA:
        bot.attack = bot.attack_goal(None) #the switch wins over the goal saved before a restart
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
            next_loop = keep_rate(next_loop) # Maintain a fixed 100 Hz loop

    except KeyboardInterrupt:
        print("User stopped.")
    except Exception as e:      # outside tick(): give up, systemd starts the program again
        print(f"Unexpected error: {e!r}")
        traceback.print_exc()
        failed = True
    finally:
        bot.shutdown()
    if failed:
        raise SystemExit(1)


class LedCalibrator: #keeps the brightest LDR reading just above the line threshold
    def __init__(self, pcb):
        self.pcb = pcb
        self.brightness = cfg.LED_BRIGHTNESS_START
        pcb.set_brightness(self.brightness)

    def step(self, colours):
        if max(colours) - 3000 > cfg.LDR_LINE_THRESHOLD: # calibrate pcb leds
            self.brightness += 50
        elif max(colours) < cfg.LDR_LINE_THRESHOLD:
            self.brightness -= 50
        self.brightness = max(min(self.brightness,65535),0)
        self.pcb.set_brightness(self.brightness)


def goal_colour_from_switch(closed):
    """Goal switch closed (pin pulled to ground) = cfg.GOAL_SWITCH_CLOSED, open = the other goal."""
    other = "blue" if cfg.GOAL_SWITCH_CLOSED == "yellow" else "yellow"
    return cfg.GOAL_SWITCH_CLOSED if closed else other


def goal_colour_from_view(det, current):
    """While standing still facing forward: the goal in front of us is the one we attack."""
    if det is None:
        return current
    yellow, blue = det.goals["yellow"], det.goals["blue"]
    if yellow is not None:
        return "yellow" if yellow.centre[1] > 0 else "blue"
    if blue is not None:
        return "blue" if blue.centre[1] > 0 else "yellow"
    return current


def heading_from_goals(det, attack, both_only=False):
    """Our heading (0 = facing the goal we attack, CCW +) from where the goals are in the camera, without the IMU, or
    None. Both goals seen: exact, from the line own goal -> attack goal. One goal: from the direction to it - off by up
    to ~40 deg when we're off to the side of it, so a heading near 0 / 180 deg is snapped to that."""
    if det is None:
        return None
    own = "blue" if attack == "yellow" else "yellow"
    a, o = det.goals.get(attack), det.goals.get(own)
    if a is not None and o is not None:
        vx, vy = a.centre[0] - o.centre[0], a.centre[1] - o.centre[1]
        return math.atan2(vx, vy) if vx * vx + vy * vy > 50.0 ** 2 else None
    if both_only or (a is None and o is None):
        return None
    if a is not None:
        h = math.atan2(a.centre[0], a.centre[1])
    else:
        h = wrap_pi(math.atan2(o.centre[0], o.centre[1]) + math.pi)
    if abs(h) < CAM_HEADING_SNAP:
        return 0.0
    if abs(wrap_pi(h - math.pi)) < CAM_HEADING_SNAP:
        return math.pi
    return h


def own_goal_ahead(det, attack):
    """Our own goal (by its colour, so even with a wrong compass) within OWN_GOAL_NO_KICK of straight ahead."""
    if det is None:
        return False
    o = det.goals.get("blue" if attack == "yellow" else "yellow")
    return o is not None and o.centre[1] > 0 and abs(math.atan2(o.centre[0], o.centre[1])) < OWN_GOAL_NO_KICK


def keep_rate(next_loop):
    """Sleep until the next 100 Hz tick. Returns the new next_loop."""
    next_loop += cfg.CONTROL_PERIOD
    sleep_time = next_loop - time.monotonic()
    if sleep_time > 0:
        time.sleep(sleep_time)
        return next_loop
    return time.monotonic()

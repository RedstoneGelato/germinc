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
import os
import threading
import time
import traceback

import robot_config as cfg
from lines import LineFusion
from motion import Mover
from perception import World

# ---- TUNE
CAMERA_BLIND = 0.3        # s without a new camera frame -> the robot stands still until frames come again
VISION_STALL = 1.5        # s without a new camera frame -> restart the camera
COMMS_STALL = 1.0         # s without the comms thread going round its loop -> restart it
READY_TIMEOUT = 10.0      # s to wait for the sensors at start-up before running anyway
STATE_FILE = "/dev/shm/germinc_state.json"   # RAM disk: survives a program restart, cleared when the Pi reboots
STATE_MAX_AGE = 30.0      # s: older saved state is from an earlier run, ignore it


class Robot:
    """Every thread + helper object one robot needs. hw = hardware_attack or hardware_defense."""

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

        self.attack = "yellow" # goal we shoot at, set while paused
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
        self.leds.brightness = state["brightness"]
        self.pcb.set_brightness(self.leds.brightness)

    def actuate(self, speeds, dribbler_on, kick):
        """Send one loop's decisions to the hardware: 4 wheel speeds, dribbler on/off, kick (ignored without a
        kicker or while it recharges)."""
        self.motors.set(speeds)
        self.motors.set_dribbler(dribbler_on)
        if kick and self.kicker is not None:
            self.kicker.kick()

    def standby(self, colours, det):
        """One loop while paused: motors off, calibrate goal colour, LEDs and heading."""
        self.motors.stop()
        self.attack = goal_colour_from_view(det, self.attack) #calibrate goal colour
        self.leds.step(colours) #calibrate pcb leds
        self.imu.zero() #calibrate imu heading: facing forward = facing the goal we attack

    def on_pause(self):
        """Robot just got paused (it's about to be picked up and put down somewhere new)."""
        self.world.reset()
        self.mover.reset()
        self.loc.relocalise()

    def shutdown(self):
        print("Shutting down safely...")

        def attempt(what, fn):     # one step failing mustn't stop the rest (above all the motors stopping)
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
    all from the last frame then, and they're relative to where the robot WAS, so driving on them pushes the ball
    the wrong way (in the simulator with faults.py it scored own goals): the brains stand still instead."""

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

    script_activate_pin = DigitalInputDevice(cfg.SWITCH_PIN, pull_up = True) #gpio pin for on/off switch
    bot = Robot(hw)
    guard = TickGuard(brain_cls(bot))
    bot.wait_ready()

    print("Waiting for signal")
    failed = False
    try:
        next_loop = time.monotonic()
        while True:
            paused = script_activate_pin.is_active
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


def keep_rate(next_loop):
    """Sleep until the next 100 Hz tick. Returns the new next_loop."""
    next_loop += cfg.CONTROL_PERIOD
    sleep_time = next_loop - time.monotonic()
    if sleep_time > 0:
        time.sleep(sleep_time)
        return next_loop
    return time.monotonic()

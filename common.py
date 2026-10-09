"""
common.py - start-up / standby / shutdown pieces both main.py (goalie) and main_attack.py (striker) use.
"""
import time

import robot_config as cfg
from lines import LineFusion
from motion import Mover
from perception import World


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

    def wait_ready(self):
        print("Waiting for sensors...")
        while not (self.imu.ready and self.vision.ready and self.pcb.ready):
            time.sleep(0.05)

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

        # stop motors first
        self.motors.stop()
        if self.kicker is not None:
            self.kicker.close()
        for m in self.motors.drivers:
            m.clear_faults()

        # allow motor thread to send stop command
        time.sleep(0.05)

        # stop threads
        for t in (self.loc, self.motors, self.imu, self.pcb):
            t.running = False
        self.vision.stop()
        self.comms.stop()

        # wait for threads
        for t in (self.vision, self.loc, self.motors, self.imu, self.pcb, self.comms):
            t.join(timeout=1.0)

        print("Robot stopped.")


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

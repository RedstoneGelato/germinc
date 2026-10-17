"""
faults.py - break our robots on purpose in the simulator, to check they keep playing through it.

    python simulator.py --faults 1                  # window, faults at the default rates (2 = twice as often)
    python simulator.py --headless 60 --faults 1
    python arena.py --games 20 --faults 1           # compare with the same games without --faults

What can go wrong on the real robot, and what the robot code does about it (common.py, vision.py, hardware.py):

    camera freeze  the capture hangs or the camera thread dies: no new frames. The watchdog restarts the camera
                   after common.VISION_STALL s; until the new camera runs the robot only has its last frame (old ball,
                   old goals) and localisation gets nothing new. Simulated: the robot sees its last frame for a while.
    comms drop     wifi / the comms thread drops out: no teammate messages (the watchdog restarts a dead thread).
    imu jump       the BNO08x resets and its heading jumps. utils.HeadingFilter removes jumps faster than the robot
                   can turn; smaller ones get through and stay until the robot is paused (standby re-zeroes it).
                   Simulated: the same filter decides which jumps get through.
    tick crash     an exception somewhere in tick(). common.TickGuard stops the motors for that loop and carries on.
    restart        the whole program dies (segfault, an error outside tick()): systemd starts it again
                   (deploy/germinc.service). The robot stands still for a few seconds, then starts again running,
                   with everything forgotten except the heading / attack goal saved in common.STATE_FILE.

Only our robots get faults (the opponents play normally). Every fault is counted in sim.events ("fault: ...").
"""
import math

import numpy as np

from common import TickGuard
from utils import HeadingFilter

# ---- TUNE: name -> (mean seconds between two of them on one robot at --faults 1, how long one lasts in s)
FAULTS = {
    "camera freeze": (30.0, (1.5, 4.0)),    # watchdog notices after 1.5 s + the camera takes a while to reopen
    "comms drop": (20.0, (1.0, 6.0)),
    "imu jump": (30.0, None),
    "tick crash": (10.0, None),
    "restart": (90.0, (4.0, 8.0)),           # python + opencv imports + camera start + waiting for the sensors
}
IMU_JUMP = (0.1, math.pi)                    # rad, either way


class FrozenVision:
    """Stands in for a robot's SimVision while its camera is frozen: the simulator keeps writing new frames into
    it, the robot keeps seeing the frame from when it froze."""

    def __init__(self, real):
        self.real = real
        self.rv, self.fps = real.rv, real.fps
        self._id, self._det = real.frame_id, real.detections

    frame_id = property(lambda self: self._id, lambda self, v: None)
    detections = property(lambda self: self._det, lambda self, v: None)
    last_res = property(lambda self: None, lambda self, v: None)

    def latest(self):
        return self._det


class FrozenLoc:
    """Stands in for a robot's localisation while its camera is frozen: no new frames, so its pose stays put."""

    def __init__(self, real):
        self.real = real
        self.pose = real.get()

    def process(self, det):
        pass

    def get(self):
        return self.pose

    def __getattr__(self, name):
        return getattr(self.real, name)


class FaultInjector:
    def __init__(self, sim, rate=1.0, seed=0):
        self.sim = sim
        self.rate = rate
        self.rng = np.random.default_rng(1000 + seed)
        self.guards = {r.name: TickGuard(r.brain, log=self._logger(r)) for r in sim.robots}
        self.until = {r.name: {} for r in sim.robots}    # fault -> sim time it ends
        self.crash_next = set()                         # robots whose next tick raises
        self.imu = {r.name: (HeadingFilter(), [0.0]) for r in sim.robots}   # filter + the raw jumps so far
        self.tick_errors = 0

    def _logger(self, r):
        sim = self.sim

        def log(msg):
            sim.log(f"{r.name}: {msg.splitlines()[0]}")
        return log

    def step(self):
        """Call at the start of every simulator step: start and end faults."""
        sim, t = self.sim, self.sim.t
        dt = 0.01
        for r in sim.robots:
            active = self.until[r.name]
            for name, end in list(active.items()):
                if t >= end:
                    del active[name]
                    self._end(r, name)
            if "restart" in active:
                continue
            for name, (every, dur) in FAULTS.items():
                if self.rng.random() >= dt * self.rate / every or name in active or (
                        name == "tick crash" and r.name in self.crash_next):
                    continue
                sim.count(f"fault: {name}")
                if dur is not None:
                    active[name] = t + self.rng.uniform(*dur)
                self._start(r, name)

    def _start(self, r, name):
        bot = r.bot
        if name == "camera freeze":
            bot.vision = FrozenVision(bot.vision)
            bot.loc = FrozenLoc(bot.loc)
        elif name == "comms drop":
            bot.comms.enabled = False
        elif name == "imu jump":
            filt, raw = self.imu[r.name]
            before = filt.update(raw[0], self.sim.t)
            raw[0] += self.rng.choice((-1, 1)) * self.rng.uniform(*IMU_JUMP)
            got_through = filt.update(raw[0], self.sim.t + 0.01) - before   # next IMU sample (100 Hz)
            bot.imu.drift += got_through
            if abs(got_through) > 1e-9:
                self.sim.count("fault: imu jump got through the filter")
        elif name == "tick crash":
            self.crash_next.add(r.name)
        elif name == "restart":
            for other in list(self.until[r.name]):       # the restart ends every other fault
                if other != "restart":
                    del self.until[r.name][other]
                    self._end(r, other)
            bot.motors.stop()
            bot.comms.enabled = False
            self.sim.log(f"{r.name}: program crashed - systemd restarting it")

    def _end(self, r, name):
        bot = r.bot
        if name == "camera freeze":
            bot.vision = bot.vision.real
            bot.loc = bot.loc.real
        elif name == "comms drop":
            bot.comms.enabled = "restart" not in self.until[r.name]
        elif name == "restart":      # a new program: everything forgotten except what common.load_state() restores
            bot.comms.enabled = "comms drop" not in self.until[r.name]
            bot.world.reset()
            bot.mover.reset()
            bot.lines = type(bot.lines)()
            bot.loc.relocalise()
            log = r.brain.log
            r.brain = type(r.brain)(bot, log=log)
            self.guards[r.name] = TickGuard(r.brain, log=self._logger(r))
            self.sim.log(f"{r.name}: restarted")

    def tick(self, r):
        """Run one robot's tick() (instead of r.brain.tick) with the faults applied."""
        guard = self.guards.get(r.name)
        if guard is None:                       # an opponent
            r.brain.tick(r.paused)
            return
        if "restart" in self.until[r.name]:     # program not running: motors off
            r.bot.motors.stop()
            return
        world = r.bot.world
        if r.name in self.crash_next:       # (stays pending while paused: a paused tick doesn't get that far)
            def broken(*args, **kwargs):
                raise RuntimeError("fault injected by faults.py")
            world.update = broken
        try:
            if not guard.tick(r.paused):
                self.crash_next.discard(r.name)
                self.tick_errors += 1
                self.sim.count("tick errors (robot carried on)")
        finally:
            world.__dict__.pop("update", None)

    def summary(self):
        ev = self.sim.events
        return "faults: " + ", ".join(f"{k[7:]} {v:.0f}" for k, v in sorted(ev.items()) if k.startswith("fault: "))

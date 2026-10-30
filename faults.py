import math

import numpy as np

from common import TickGuard
from utils import HeadingFilter

# ---- TUNE: name -> (mean seconds between two of them on one robot at --faults 1, how long one lasts in s)
FAULTS = {
    "camera freeze": (30.0, (1.5, 4.0)),
    "comms drop": (20.0, (1.0, 6.0)),
    "imu jump": (30.0, None),
    "tick crash": (10.0, None),
    "restart": (90.0, (4.0, 8.0)),
}
IMU_JUMP = (0.1, math.pi)


class FrozenVision:

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
        self.until = {r.name: {} for r in sim.robots}
        self.crash_next = set()
        self.imu = {r.name: (HeadingFilter(), [0.0]) for r in sim.robots}
        self.tick_errors = 0

    def _logger(self, r):
        sim = self.sim

        def log(msg):
            sim.log(f"{r.name}: {msg.splitlines()[0]}")
        return log

    def step(self):
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
            got_through = filt.update(raw[0], self.sim.t + 0.01) - before
            bot.imu.drift += got_through
            if abs(got_through) > 1e-9:
                self.sim.count("fault: imu jump got through the filter")
        elif name == "tick crash":
            self.crash_next.add(r.name)
        elif name == "restart":
            for other in list(self.until[r.name]):
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
        elif name == "restart":
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
        guard = self.guards.get(r.name)
        if guard is None:
            r.brain.tick(r.paused)
            return
        if "restart" in self.until[r.name]:
            r.bot.motors.stop()
            return
        world = r.bot.world
        if r.name in self.crash_next:
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

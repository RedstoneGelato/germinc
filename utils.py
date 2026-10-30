import math
import time

import numpy as np


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def angdiff(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


def circular_mean(angles):
    return math.atan2(sum(math.sin(a) for a in angles), sum(math.cos(a) for a in angles))


def rotate(v, angle):
    c, s = math.cos(angle), math.sin(angle)
    return [v[0] * c - v[1] * s, v[0] * s + v[1] * c]


def rotate_points(pts, angle):
    if len(pts) == 0:
        return pts
    c, s = math.cos(angle), math.sin(angle)
    return pts @ np.array([[c, s], [-s, c]], dtype=pts.dtype)


def other_goal(colour):
    return "blue" if colour == "yellow" else "yellow"


class Hysteresis:
    def __init__(self, hold_time, instant_enter=None):
        self.hold_time = hold_time
        self.instant_enter = instant_enter
        self.current = None
        self._pending = None
        self._pending_since = None

    def update(self, raw_value):
        now = time.monotonic()
        if self.current is None:
            self.current = raw_value
        elif raw_value == self.current:
            self._pending = None
        elif self.instant_enter and self.instant_enter(raw_value):
            self.current = raw_value
            self._pending = None
        elif raw_value != self._pending:
            self._pending = raw_value
            self._pending_since = now
        elif now - self._pending_since >= self.hold_time:
            self.current = raw_value
            self._pending = None
        return self.current

    def reset(self):
        self.current = None
        self._pending = None
        self._pending_since = None


class PointTracker:
    def __init__(self, history=6, tolerance=25.0, lost_time=0.3):
        self.history, self.tolerance, self.lost_time = history, tolerance, lost_time
        self.points = []
        self.jumps_in_row = 0
        self.last_seen = 0.0

    def update(self, p, now=None):
        now = time.monotonic() if now is None else now
        if p is None:
            return self.value(now)
        self.last_seen = now
        if self.points:
            mx, my = self.mean()
            if math.hypot(p[0] - mx, p[1] - my) > self.tolerance:
                self.jumps_in_row += 1
                if self.jumps_in_row > self.history // 2:
                    self.points = [p]
                    self.jumps_in_row = 0
                return self.value(now)
        self.jumps_in_row = 0
        self.points.append(p)
        if len(self.points) > self.history:
            self.points.pop(0)
        return self.value(now)

    def mean(self):
        n = len(self.points)
        return [sum(q[0] for q in self.points) / n, sum(q[1] for q in self.points) / n]

    def value(self, now=None):
        now = time.monotonic() if now is None else now
        if not self.points or now - self.last_seen > self.lost_time:
            self.points = []
            return None
        return self.mean()

    def reset(self):
        self.points = []
        self.jumps_in_row = 0
        self.last_seen = 0.0


class HeadingFilter:
    MAX_SPIN = 20.0
    MARGIN = 0.3

    def __init__(self):
        self.offset = 0.0
        self.last = None
        self.jumps = 0
        self._reanchor = False

    def reanchor(self):
        self._reanchor = True

    def continue_from(self, heading):
        self.last = (-1e9, heading)
        self._reanchor = True

    def update(self, raw, t):
        h = wrap_pi(raw + self.offset)
        if self._reanchor and self.last is not None:
            self._reanchor = False
            self.offset = wrap_pi(self.offset + angdiff(self.last[1], h))
            h = self.last[1]
        elif self.last is not None:
            last_t, last_h = self.last
            step = angdiff(h, last_h)
            if abs(step) > self.MAX_SPIN * max(t - last_t, 0.0) + self.MARGIN:
                self.offset = wrap_pi(self.offset - step)
                h = last_h
                self.jumps += 1
        self.last = (t, h)
        return h

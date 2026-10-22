"""
utils.py - small maths helpers and smoothing used by several modules.
"""
import math
import time

import numpy as np


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def angdiff(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))  # wraps correctly through +-pi


def circular_mean(angles):
    return math.atan2(sum(math.sin(a) for a in angles), sum(math.cos(a) for a in angles))


def rotate(v, angle):
    """Rotate a 2D vector counter-clockwise by angle (radians). Robot frame -> relative: rotate(v, compass)."""
    c, s = math.cos(angle), math.sin(angle)
    return [v[0] * c - v[1] * s, v[0] * s + v[1] * c]


def rotate_points(pts, angle):
    """Same as rotate() for an (N, 2) numpy array."""
    if len(pts) == 0:
        return pts
    c, s = math.cos(angle), math.sin(angle)
    return pts @ np.array([[c, s], [-s, c]], dtype=pts.dtype)


class Hysteresis: #used for decision smoothing and ignore flickers, instant enter to instantly switch
    def __init__(self, hold_time, instant_enter=None):
        self.hold_time = hold_time
        self.instant_enter = instant_enter
        self.current = None
        self._pending = None
        self._pending_since = None

    def update(self, raw_value):
        now = time.monotonic()

        if self.current is None: #first call - nothing to debounce yet
            self.current = raw_value
            return self.current

        if raw_value == self.current: #still agrees - clear any pending change
            self._pending = None
            return self.current

        if self.instant_enter and self.instant_enter(raw_value): #instantly swap with no delay
            self.current = raw_value
            self._pending = None
            return self.current

        if raw_value != self._pending: #new candidate - start timing it
            self._pending = raw_value
            self._pending_since = now
            return self.current

        if now - self._pending_since >= self.hold_time: #held long enough - commit it
            self.current = raw_value
            self._pending = None

        return self.current

    def reset(self): #clear state so that unpause doesnt jitter
        self.current = None
        self._pending = None
        self._pending_since = None


class PointTracker:
    """Smooths a 2D position that is sometimes missing.
    - keeps a short history and averages it (like the old GoalTracker)
    - a reading far from the average counts as 'unconcordant'; enough of those in a row = it really moved
    - keeps the last value for `lost_time` seconds after it disappears, then reports None"""

    def __init__(self, history=6, tolerance=25.0, lost_time=0.3):
        self.history, self.tolerance, self.lost_time = history, tolerance, lost_time
        self.points = []
        self.unconcordant = 0
        self.last_seen = 0.0

    def update(self, p, now=None):
        now = time.monotonic() if now is None else now
        if p is not None:
            self.last_seen = now
            if self.points:
                mx, my = self.mean()
                if math.hypot(p[0] - mx, p[1] - my) > self.tolerance:
                    self.unconcordant += 1
                    if self.unconcordant > self.history // 2:   # moved for real
                        self.points = [p]
                        self.unconcordant = 0
                    return self.value(now)
            self.unconcordant = 0
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
        self.unconcordant = 0
        self.last_seen = 0.0

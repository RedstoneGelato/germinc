"""
strategy.py (past_versions/comp) - the old COMPETITION logic as a simulator opponent: every decision comes from the
comp-style state machines in strategy_fallback.py (relative to the robot, no localisation used for decisions), with
today's perception / motion / hardware around them. Same interface as the real strategy.py.
"""
from strategy_fallback import GoalieFallback, StrikerFallback


class GoalieStrategy:
    def __init__(self, has_dribbler=True):
        self.f = GoalieFallback()
        self.kick, self.speed, self.chasing, self.botstate, self.phase = False, {}, False, 0, "comp"

    def reset(self):
        self.f.reset()

    def update(self, world, compass, mate, line_touches):
        out = self.f.update(world, compass, mate.get("command") if mate else None,
                            mate.get("bot active") if mate else None, line_touches)
        self.speed, self.chasing, self.botstate = self.f.speed, self.f.chasing, self.f.botstate
        return out


class StrikerStrategy:
    def __init__(self):
        self.f = StrikerFallback()
        self.kick, self.speed, self.chasing, self.botstate, self.phase, self.command = False, {}, False, 2, "comp", 0

    def reset(self):
        self.f.reset()

    def update(self, world, compass, mate, line_touches):
        out = self.f.update(world, compass, mate.get("bot active") if mate else None, line_touches)
        self.speed, self.command, self.botstate = self.f.speed, self.f.command, self.f.botstate
        self.chasing = self.f.botstate in (1, 2)
        return out

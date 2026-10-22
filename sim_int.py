#!/usr/bin/env python3
"""
sim_int.py - simulator.py / arena.py on the INTERNATIONAL field (field_int.py) and rules, with our robots running
strategy_int.py. simulator.py and arena.py themselves are unchanged (still RCJA).

    python sim_int.py                                   # the simulator window (same keys / options as simulator.py)
    python sim_int.py --opp v88 --noise
    python sim_int.py --arena --games 20 --fast --opp v88     # arena.py's many games (same options as arena.py)
    python sim_int.py --rcja-strategy --arena ...       # our RCJA strategy.py on this field instead, to compare
    python sim_int.py --opp-as-is --arena --opp ba ...  # opponents NOT kept legal in the penalty areas (see below)

The referee follows the RCJ Soccer 2026 rules instead of the RCJA ones:
    2.5  the ball is never out at the line: it's live up to the walls
    2.7  lack of progress -> nearest free neutral spot (after NO_PROGRESS_TIME)
    2.8  a robot touching a wall or COMPLETELY inside a penalty area is out: off for 1 minute (or till the next
         kickoff), then back on the free neutral spot furthest from the ball, facing its own goal. Pushed out by an
         opponent (touched it just now, and it wasn't driving that way itself or the opponent was driving it there)
         = not taken off, nudged straight back in (pushed_out())
    2.6  multiple defense: two robots of a team even partly in a penalty area (any time) -> the one further from
         the ball goes to the free neutral spot furthest from the ball
Opponents loaded with --opp play on this field too (their own strategy, with MAX_OUT for these walls). Their moves
also go through strategy_int's penalty area guard (legal_opponent()), as if they'd been adapted to these rules: a
team that won here doesn't park its goalie completely inside the area. Without that (--opp-as-is) the RCJA-built
opponents spend much of each game a robot down (ba: ~3.5 outs + ~3 multiple defences a game).
"""
import math
import os
import sys

if "--rcja-strategy" in sys.argv:          # (an environment variable, so arena's game processes see it too)
    os.environ["GERMINC_RCJA_STRATEGY"] = "1"
RCJA_STRATEGY = os.environ.get("GERMINC_RCJA_STRATEGY") == "1"
if "--opp-as-is" in sys.argv:
    os.environ["GERMINC_OPP_AS_IS"] = "1"
OPP_LEGAL = os.environ.get("GERMINC_OPP_AS_IS") != "1"

import field_int  # noqa: E402

field_int.install()                        # before anything else imports field
if not RCJA_STRATEGY:
    import strategy_int  # noqa: E402
    sys.modules["strategy"] = strategy_int # main.py / main_attack.py's `from strategy import ...` get strategy_int

import simulator as S  # noqa: E402

field = field_int
NO_PROGRESS_TIME = 5.0      # s the ball hasn't moved 5 cm -> lack of progress (rule 2.7: "a reasonable period")
PUSHING_TIME = 0.5          # s an attacker and a defender keep shoving over the ball near an area -> pushing (2.6)
PUSHER_GAP = 2.0            # cm: an opponent this close (edge to edge)...
PUSHER_OUT_SPEED = 10.0     # ...driving at least this fast the way our robot went out = it pushed it out
PUSH_NEAR = 5.0             # cm: 'near the area' for the both-on-the-ball push (robot edge within this of the area)
PUSH_GOAL_WINDOW = 0.5      # s: a goal this soon after such a push doesn't count
S.DAMAGED_TIME = 60.0       # out of bounds: 1 minute off (rule 2.8)
KICKOFF_CIRCLE_R = 30.0     # centre circle radius (60 cm diameter)
KICKOFF_BLOCK = os.environ.get("GERMINC_KICKOFF_BLOCK", "1") == "1"   # the team NOT kicking off stands its striker in
                            # front of the centre circle, on the line to its goal (both teams) - see kickoff()


def load_code(folder):
    """simulator.load_code(), but the other version's modules get field_int as their field (and MAX_OUT for these
    walls, and the white penalty lines in lines.py) - so they play on this field too."""
    import importlib
    folder = S.code_folder(folder)
    names = [k for k in S.CODE_MODULES if k != "field"]
    saved = {k: sys.modules.pop(k) for k in names if k in sys.modules}
    sys.path.insert(0, os.path.abspath(folder))
    try:
        mods = {}
        for k in names:                    # (in CODE_MODULES order: robot_config before everything that uses it)
            if not os.path.isfile(os.path.join(folder, k + ".py")):
                continue
            mods[k] = importlib.import_module(k)
            if k == "robot_config":
                mods[k].MAX_OUT = field.max_out(mods[k].ROBOT_RADIUS)
            elif k == "lines":
                mods[k].IGNORE_INNER_LINES = True
    finally:
        sys.path.pop(0)
        for k in names:
            sys.modules.pop(k, None)
        sys.modules.update(saved)
    if OPP_LEGAL and "strategy" in mods:
        legal_opponent(mods["strategy"])
    return S.Code(os.path.basename(os.path.normpath(folder)), mods)


def legal_opponent(mod):
    """Wrap another version's GoalieStrategy / StrikerStrategy update() in strategy_int's penalty area guard: never
    completely inside an area, and (attacking, teammate on) not in its own area at all. A robot with a .role (the
    Bodensee port switches roles) defends when its role is "defender"."""
    import strategy_int as SI
    for name, goalie in (("GoalieStrategy", True), ("StrikerStrategy", False)):
        cls = getattr(mod, name, None)
        if cls is None or getattr(cls.update, "int_legal", False):
            continue
        orig = cls.update

        def update(self, world, compass, mate, line_touches, _orig=orig, _goalie=goalie):
            out = _orig(self, world, compass, mate, line_touches)
            if not hasattr(self, "_int_guard"):
                self._int_guard = SI.BoxGuard(brake=not _goalie)
            role = getattr(self, "role", None)
            defending = role == "defender" if role is not None else _goalie
            mate_on = bool(mate) and mate.get("bot active") == 1
            own = SI.box_limit(partly_ok=defending or not mate_on)
            return self._int_guard(out, world.pose, {-1: own, 1: SI.box_limit()})
        update.int_legal = True
        cls.update = update


S.load_code = load_code


# ==================================== referee (international rules) ====================================

def _sgn_of(team):
    """Which end a team's own goal is at."""
    return -1.0 if team == "us" else 1.0


def _free_spots(sim, ignore=None):
    bodies = [b for b in sim.on_field if b is not ignore]
    free = [p for p in field.NEUTRAL_SPOTS if all(math.hypot(p[0] - b.x, p[1] - b.y) > b.r + 10 for b in bodies)]
    return free or list(field.NEUTRAL_SPOTS)


def _furthest_spot(sim, ignore=None):
    bx, by = sim.ball if sim.ball_away_until is None else sim.ball_out_at
    return max(_free_spots(sim, ignore), key=lambda p: math.hypot(p[0] - bx, p[1] - by))


def _out_reason(b):
    if field.touches_wall(b.x, b.y, b.r, 0.3):
        return "wall"
    for sgn in (1.0, -1.0):
        if field.fully_in_penalty(b.x, b.y, b.r, sgn):
            return "penalty area"
    return None


def _out_dir(b, why):
    """Unit arrow the robot went out along (towards the wall / deeper into the area)."""
    if why == "wall":
        gx = field.WALL_W / 2 - abs(b.x)
        gy = field.WALL_L / 2 - abs(b.y)
        return (math.copysign(1.0, b.x), 0.0) if gx < gy else (0.0, math.copysign(1.0, b.y))
    sgn = max((1.0, -1.0), key=lambda s: field.penalty_depth(b.x, b.y, s))
    e = 0.5
    gx = field.penalty_depth(b.x + e, b.y, sgn) - field.penalty_depth(b.x - e, b.y, sgn)
    gy = field.penalty_depth(b.x, b.y + e, sgn) - field.penalty_depth(b.x, b.y - e, sgn)
    n = math.hypot(gx, gy) or 1.0
    return (gx / n, gy / n)


def pushed_out(self, b, why):
    """Pushed out by an opponent: it touched an opponent within
    PUSHED_TIME, and either it wasn't driving out itself (PUSHED_OUT_SPEED) or an opponent touching it is driving it
    that way (PUSHER_OUT_SPEED) - shoved out while both pushed used to count as going out on its own."""
    if self.t - b.opp_contact_t > S.PUSHED_TIME:
        return False
    n = _out_dir(b, why)
    if b.vx * n[0] + b.vy * n[1] < S.PUSHED_OUT_SPEED:
        return True
    return any(o.team != b.team and math.hypot(o.x - b.x, o.y - b.y) < o.r + b.r + PUSHER_GAP
               and o.vx * n[0] + o.vy * n[1] > PUSHER_OUT_SPEED for o in self.on_field)


def push_back_in(self, b, why):
    """Pushed out by an opponent: straight back in, just far enough."""
    if why == "wall":
        b.x = max(-(field.WALL_W / 2 - b.r - 3), min(field.WALL_W / 2 - b.r - 3, b.x))
        b.y = max(-(field.WALL_L / 2 - b.r - 3), min(field.WALL_L / 2 - b.r - 3, b.y))
    else:
        n = _out_dir(b, why)
        sgn = max((1.0, -1.0), key=lambda s: field.penalty_depth(b.x, b.y, s))
        while field.penalty_depth(b.x, b.y, sgn) > b.r - 3:
            b.x -= n[0]
            b.y -= n[1]
    self.count(f"{b.name} pushed out, back at once")
    self.log(f"{b.name} pushed out - put straight back")


def put_back(self, b):
    """Rule 2.8: on the free neutral spot furthest from the ball, facing its own goal."""
    x, y = _furthest_spot(self, ignore=b)
    b.removed_until = None
    b.place(x, y, math.pi if b.team == "us" else 0.0)
    if isinstance(b, S.SimRobot):
        b.paused = True
        b.return_at = self.t + 0.5
    self.log(f"{b.name} back on the field")


def place_ball(self, spot, why):
    if self.held_by is not None:
        self.release("ball replaced")
    self.ball, self.ball_v = list(spot), [0.0, 0.0]
    self.last_restart = self.t
    self.ball_out_since, self.progress = None, (list(spot), self.t)
    self.log(f"ball {why} -> neutral spot ({spot[0]:.0f},{spot[1]:.0f})")


def multiple_defence(self):
    """Rule 2.6: two robots of a team at least partly in a penalty area (either one, any time) -> the one further from
    the ball to the free neutral spot furthest from the ball."""
    bx, by = self.ball
    for team in ("us", "them"):
        for sgn in (1.0, -1.0):
            mine = [b for b in self.on_field if b.team == team and field.partly_in_penalty(b.x, b.y, b.r, sgn)]
            if len(mine) < 2:
                continue
            far = max(mine, key=lambda b: math.hypot(b.x - bx, b.y - by))
            if self.held_by is far:
                self.release("multiple defence")
            x, y = _furthest_spot(self, ignore=far)
            far.place(x, y)
            self.count(f"{far.name} moved (multiple defence)")
            self.log(f"{far.name}: multiple defence - moved to a neutral spot")


def _ball_push(self):
    """An attacker and a defender touching each other, both touching the ball, one of them
    near the defender's own area (its edge within PUSH_NEAR of it). Returns the defending team or None."""
    bx, by = self.ball
    bodies = self.on_field

    def on_ball(c):
        return self.held_by is c or math.hypot(c.x - bx, c.y - by) < c.r + S.BALL_RADIUS + 1.5

    for i, a in enumerate(bodies):
        for b in bodies[i + 1:]:
            if a.team == b.team or math.hypot(a.x - b.x, a.y - b.y) > a.r + b.r + 0.5:
                continue
            if not (on_ball(a) and on_ball(b)):
                continue
            for d in (a, b):
                sgn = -1.0 if d.team == "us" else 1.0            # d defending its own area
                if any(field.partly_in_penalty(c.x, c.y, c.r + PUSH_NEAR, sgn) for c in (a, b)):
                    return d.team
    return None


def pushing(self):
    """Rule 2.6: an attacker and a defender touching, one of them partly in a penalty area and one of them touching
    the ball, for PUSHING_TIME -> ball to the free neutral spot furthest from it."""
    if self.ball_away_until is not None:
        return
    who = _ball_push(self)
    if who is not None:
        self._push_since = None
        self._ball_push_t = self.t
        self.count(f"pushing on the ball in {who} area")
        place_ball(self, _furthest_spot(self), "pushing on the ball")
        return
    bx, by = self.ball
    found = None
    bodies = self.on_field
    for i, a in enumerate(bodies):
        for b in bodies[i + 1:]:
            if a.team == b.team or math.hypot(a.x - b.x, a.y - b.y) > a.r + b.r + 0.5:
                continue
            on_ball = any(self.held_by is c or math.hypot(c.x - bx, c.y - by) < c.r + S.BALL_RADIUS + 1.5
                          for c in (a, b))
            if not on_ball:
                continue
            for sgn in (1.0, -1.0):
                if field.partly_in_penalty(a.x, a.y, a.r, sgn) or field.partly_in_penalty(b.x, b.y, b.r, sgn):
                    found = "us" if sgn < 0 else "them"          # whose area
    if found is None:
        self._push_since = None
        return
    if self._push_since is None:
        self._push_since = self.t
    elif self.t - self._push_since >= PUSHING_TIME:
        self._push_since = None
        self.count(f"pushing in {found} area")
        place_ball(self, _furthest_spot(self), "pushing")


def bookkeeping(self):
    """simulator.Sim.bookkeeping() with the international referee (see the top of this file)."""
    bx, by = self.ball
    in_goal = (abs(bx) < field.GOAL_W / 2 and          # inside a goal, fully over the goal line
               field.ATTACK_GOAL_Y + S.BALL_RADIUS < abs(by) < field.GOAL_BACK_Y)
    if self.goal_timer is None and in_goal and (self.t - getattr(self, "_ball_push_t", -1e9) < PUSH_GOAL_WINDOW
                                                or _ball_push(self) is not None):
        self._ball_push_t = self.t                     # pushed in: no goal, ball to the furthest spot, play on
        self.count("goal disallowed (pushing)")
        place_ball(self, _furthest_spot(self), "goal disallowed (pushing)")
        bx, by = self.ball
        in_goal = False
    if self.goal_timer is None and in_goal:
        who = "us" if by > 0 else "them"
        self.score[who] += 1
        for sh in reversed(self.shots):
            if sh["team"] == who and self.t - sh["t"] < 1.5:
                sh["goal"] = True
                break
        self.count(f"goal {who} " + ("kicked" if self.t - self.last_kick < 1.5 else "pushed"))
        self.log(f"GOAL for {who}  ({self.score['us']}-{self.score['them']})")
        self.last_scorer = who
        self.goal_timer = self.t + 1.0
    if self.goal_timer is not None and self.t >= self.goal_timer:
        self.kickoff()
        return
    if self.goal_timer is None:
        # rule 2.7: lack of progress -> nearest free neutral spot (the ball is never out: rule 2.5)
        if math.hypot(bx - self.progress[0][0], by - self.progress[0][1]) > 5:
            self.progress = ([bx, by], self.t)
        if self.t - self.progress[1] > NO_PROGRESS_TIME:
            self.count("ball no progress")
            self.place_ball_neutral("no progress")
    for b in self.robots + self.opponents:
        if b.removed_until is not None:
            if self.t >= b.removed_until and self.goal_timer is None:
                self.put_back(b)
            continue
        why = _out_reason(b)                 # rule 2.8: touching a wall / completely inside a penalty area
        if why and not b.was_out:
            b.outs += 1
            if pushed_out(self, b, why):
                push_back_in(self, b, why)
                why = None
            else:
                self.count(f"{b.name} out ({why})")
                self.take_off(b)
        b.was_out = bool(why)
    self.multiple_defence()
    pushing(self)
    for r in self.players:
        if r.return_at is not None and self.t >= r.return_at:
            r.return_at = None
            if self.start_at is None:
                r.paused = False
        pose = r.bot.loc.get()
        if not r.paused and pose.confident and r.name in self.loc_err:
            self.loc_err[r.name].append(math.hypot(pose.x - r.x, pose.y - r.y))


_kickoff = S.Sim.kickoff


def kickoff(self):
    """simulator's kickoff, then the striker of the team that isn't kicking off (it just scored) stands just outside
    the centre circle on the line to its own goal (KICKOFF_BLOCK: against Bodensee, shots straight from their kickoff
    were the most common goal against), or else at least out of its own penalty area (RCJA: touching its front - with
    its goalie partly in there too that's multiple defense here, rule 2.6)."""
    self._push_since = None
    _kickoff(self)
    for b in self.robots + self.opponents:
        if "goalie" in b.name or not isinstance(b, S.SimRobot):
            continue
        sgn = _sgn_of(b.team)
        if KICKOFF_BLOCK and self.last_scorer == b.team:
            # the other team kicks off: stand on the line from the centre spot to our goal, just outside the centre
            # circle (rule 2.3.1: 30 cm from the ball, outside the circle) - in the way of a shot straight from it
            jitter = b.x - math.copysign(S.KICKOFF_SIDE, b.x)       # (kickoff() placed it KICKOFF_SIDE to the side)
            b.place(jitter, sgn * (KICKOFF_CIRCLE_R + b.r + 2))
        elif field.partly_in_penalty(b.x, b.y, b.r, sgn):
            b.place(b.x, sgn * (field.ATTACK_GOAL_Y - field.PENALTY_D - b.r - 3))


S.Sim.bookkeeping = bookkeeping
S.Sim.multiple_defence = multiple_defence
S.Sim.put_back = put_back
S.Sim.kickoff = kickoff


def main():
    args = [a for a in sys.argv[1:] if a not in ("--rcja-strategy", "--opp-as-is")]
    if args and args[0] == "--arena":
        import arena
        sys.argv = ["arena.py"] + args[1:]
        return arena.main()
    sys.argv = ["simulator.py"] + args
    return S.main()


if __name__ == "__main__":
    sys.exit(main())

"""
strategy.py - what each robot decides to do, in FIELD coordinates (uses the localisation).

    GoalieStrategy   (main.py)         guard an arc in front of our goal, clear the ball when it's ours to take
    StrikerStrategy  (main_attack.py)  get behind the ball, carry it with the dribbler, aim at the open part of the
                                       goal and kick; hand the ball to the goalie when the goalie is closer

While the localisation isn't confident, both hand over to the comp-style relative logic in strategy_fallback.py.

Interface (used by the brains in main.py / main_attack.py):
    update(world, compass, mate, line_touches) -> (desired_pos, desired_heading, dribbler_on)
        desired_pos      relative vector (field axes, cm) to drive along
        desired_heading  compass heading to turn to (radians, 0 = facing the attack goal, CCW +)
    .kick      True = fire the kicker this loop (ignored without a kicker / while it recharges)
    .speed     speed profile for motion.Mover.step
    .botstate  number for printing / the simulator
    .chasing   True = this robot is going for the ball (sent to the teammate)
    .command   striker only: 1 = goalie take the ball, 0 = goalie stay in goal (sent to the goalie)

Who goes for the ball: both robots know both positions and the ball (comms), and the striker decides with the
same cost for both (distance to the ball, plus a penalty for being on the wrong side of it).
"""
import math
import time

import field
import robot_config as cfg
from strategy_fallback import GoalieFallback, StrikerFallback
from utils import Hysteresis

R = cfg.ROBOT_RADIUS

# ---- TUNE (cm, radians)
FIELD_MARGIN = R + 4.0            # targets are kept this far inside the white line
BEHIND_DIST = R + 12.0            # line up this far behind the ball before pushing it
OUT_MARGIN = -cfg.MAX_OUT          # getting behind the ball / going round things: the robot may be partly past the
                                  # line (centre up to MAX_OUT past its outer edge, part of it still on the line)
ORBIT_DIST = R + 14.0             # go round the ball at this distance when on the wrong side of it
ALIGN_TOL = 6.0                   # lined up when within this of the ball->aim line (+ ALIGN_TOL_PER_CM * distance)
ALIGN_TOL_PER_CM = 0.2
PUSH_DIST = 25.0                  # once lined up, aim this far past the ball
DRIBBLER_RANGE = 25.0             # dribbler on when the ball is this close
AVOID_DIST = 2 * R + 8.0          # centre-to-centre distance to keep from obstacles when driving past them
AIM_MARGIN = 5.0                  # aim points are on the goal line, at least ball radius + this inside the posts
POST_MARGIN = 5.0                 # kick only if the robot's heading crosses the goal line ball radius + this inside a post
AIM_POINTS = 7                    # candidate aim points across the goal
KICK_RANGE = 90.0                 # kick when the aim point is closer than this...
KICK_ANGLE = math.radians(8)      # ...and the robot faces it within this (goalie clears; the striker instead checks
                                  # that its actual heading would put the ball between the posts)...
KICK_CLEARANCE = 8.0              # ...and no obstacle's edge is closer than this to the ball's path
PRESSURE_DIST = 2 * R + 12.0      # an opponent's centre this close = about to lose the ball: shoot as long as the
                                  # ball's path misses every robot (no KICK_CLEARANCE margin), better than losing it
CARRY_MULTI = 0.8                 # speed multiplier while carrying the ball (on top of motion.py's dribbler slow-down
                                  # for moving sideways / backwards, robot_config.DRIBBLE_SPEED_MIN)
SHOT_GRID = 12.0                  # cm between candidate shooting spots
SHOT_MIN_CLEAR = KICK_CLEARANCE + 6.0   # a shooting spot needs at least this much room around the shot
SHOT_TRAVEL_COST = 0.6            # score lost per cm of driving to a shooting spot
SHOT_BLOCKED_COST = 40.0          # extra score lost if an obstacle is in the way of getting there
SHOT_SWITCH = 10.0                # only switch to another shooting spot if it scores this much better...
SHOT_HOLD = 1.5                   # ...and the current one has been the plan for at least this long (s)
SHOT_ARRIVED = 8.0                # cm: at the shooting spot -> turn to the aim and shoot
SHOT_KEEP_SLACK = 5.0             # cm of clearance / range a spot we're already going to may lose before we give up
                                  # on it (far obstacles' positions wobble by a few cm from frame to frame)
SHOT_DIST_COST = 0.2              # score lost per cm of shot length (short shots are more reliable)
CARRY_BACK_COST = 3.0             # carrying the ball: cost per cm a shooting spot / dodge waypoint is behind the robot
CARRY_BACK_ALLOW = 2.0            # carrying the ball: never drive more than this (cm) backwards - go sideways instead
LEAD_TIME_MAX = 0.6               # s: go for where a moving ball will be in up to this long...
LEAD_SPEED = 100.0                # ...(= distance to it / this speed)
CATCH_MULTI_MIN = 0.3             # pushing a free ball: speed multiplier right at the ball (gentle, so it's caught
CATCH_MULTI_PER_CM = 0.015        # instead of knocked away), rising this much per cm the robot's edge is from it
RACE_MARGIN = 15.0                # ...unless an opponent is less than this much further from the ball than us: then
                                  # full speed (PUSH_MULTI), getting there first matters more than catching it cleanly

GUARD_RADIUS = min(field.PENALTY_D - 5.0, 25.0)   # goalie: distance from the goal centre it guards at
GUARD_FACE_MAX = math.radians(50)                   # goalie: turn towards the ball at most this much
GUARD_GAIN = 15.0                 # goalie guarding / blocking: full speed when this far (cm) from its spot, slower
                                  # in proportion closer (the default speed curve crawls over short distances)
GUARD_SMOOTH = 0.3                # goalie guarding / blocking: motion.Mover smoothing (default 0.1 lags ~0.1 s)
BLOCK_SPEED = 20.0                # goalie: ball rolling towards our goal line faster than this (cm/s)...
BLOCK_TIME = 1.5                  # ...and crossing our goal line within this (s), between the posts (+ BLOCK_WIDEN
BLOCK_WIDEN = 8.0                 # cm) -> stand where it will cross the goalie's line, whatever else it was doing
DANGER_DIST = 35.0                # goalie: ball in our penalty area and this close -> clear it...
OPP_ON_BALL = 12.0                # ...unless an opponent has it (its centre within R + this of the ball): then guard
CLEAR_RETURN_Y = field.OWN_GOAL_Y + 55.0   # goalie: stop clearing once the ball is past this
CLEAR_AWAY_Y = field.OWN_GOAL_Y + 50.0     # goalie: ball closer to our goal line than this -> push it straight away
                                           # from the goal centre (so it stays between ball and goal), not at their goal
STRIKER_CLEAR_Y = 0.0                      # striker: ball in our half and the striker goal-side of it -> clear it
                                           # up-field from there (clear_aim) instead of going round it
HANDOVER_Y = field.OWN_GOAL_Y + 60.0       # striker: ball behind this and goalie closer -> goalie takes it
HANDOVER_MARGIN = 10.0            # goalie must be this much closer (cost) to take the ball...
CONTEST_DIST = 8.0                # ...and no opponent's edge within this of the ball (contested: both robots go)
PUSH_MULTI = 1.0                  # speed multiplier when pushing a CONTESTED ball (full speed: win pushing contests;
                                  # a free ball is approached normally, or it gets knocked away instead of caught)
WRONG_SIDE_COST = 30.0            # cost penalty for being on the attack side of the ball
SUPPORT_OFFSET = 45.0             # striker: wait this far up-field of the ball while the goalie clears
WAIT_POS = (0.0, -10.0)           # striker: no ball known anywhere -> wait here...
SEARCH_TIME = 3.0                 # ...after first spending this long (s) looking where the ball was last seen
BALL_OUT_MARGIN = 5.0             # ball centre this far past the outer edge of the white line = clearly out
NEUTRAL_WAIT = R + 15.0           # ball out: striker waits this far behind the neutral spot it'll come back to
                                  # (not on it: an occupied spot isn't used, the ball would go somewhere else)


# ==================================== helpers ====================================
def unit(v):
    n = math.hypot(v[0], v[1])
    return (0.0, 0.0) if n < 1e-9 else (v[0] / n, v[1] / n)


def heading_for(d):
    """Direction (field axes) -> compass heading that faces it."""
    return math.atan2(-d[0], d[1])


def clamp_in_field(p, margin=FIELD_MARGIN):
    hx = field.PLAY_W / 2 - margin
    hy = field.PLAY_L / 2 - margin
    return [min(max(p[0], -hx), hx), min(max(p[1], -hy), hy)]


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def seg_dist(p, a, b):
    """Distance from p to the segment a-b, and how far along it (0..1) the closest point is."""
    ab = (b[0] - a[0], b[1] - a[1])
    L2 = ab[0] ** 2 + ab[1] ** 2
    t = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / L2))
    c = (a[0] + t * ab[0], a[1] + t * ab[1])
    return dist(p, c), t


def ball_cost(pos, ball):
    """How bad a robot at pos is placed to take the ball (cm): distance, + penalty if it's up-field of the ball."""
    if pos is None:
        return float("inf")
    return dist(pos, ball) + (WRONG_SIDE_COST if pos[1] > ball[1] + 5 else 0.0)


def obstacle_centres(me, obstacles):
    """Obstacles are stored as the point nearest to us (their front edge as we see it); their centre is about one
    robot radius further away."""
    out = []
    for o, size in obstacles:
        u = unit((o[0] - me[0], o[1] - me[1]))
        out.append((o[0] + u[0] * R, o[1] + u[1] * R))
    return out


def best_aim(ball, centres, max_range=None):
    """Point in the attack goal with the most room around the ball's path to it (only points within max_range).
    centres = obstacle centres. Returns (aim, clearance = gap between the path and the nearest obstacle's edge);
    clearance is -1e9 if no aim point is in range."""
    half = field.GOAL_W / 2 - cfg.BALL_RADIUS - AIM_MARGIN
    y = field.ATTACK_GOAL_Y                     # on the goal line: a ball through these clears the posts
    best, best_score, best_clear = (0.0, y), -1e9, -1e9
    for i in range(AIM_POINTS):
        x = -half + 2 * half * i / (AIM_POINTS - 1)
        if max_range is not None and dist(ball, (x, y)) > max_range:
            continue
        clear = clearance(ball, (x, y), centres)
        score = min(clear, 40.0) - 0.05 * abs(x)      # prefer room, then the middle
        if score > best_score:
            best, best_score, best_clear = (x, y), score, clear
    return best, best_clear


def clearance(a, b, centres):
    """Gap between the path a -> b and the nearest obstacle's edge."""
    return min((seg_dist(c, a, b)[0] - R for c in centres), default=100.0)


def path_blocked(a, b, centres, ignore_near=None):
    """First obstacle centre on the way from a to b (closer than AVOID_DIST to the path), or None."""
    worst = None
    for c in centres:
        if ignore_near is not None and dist(c, ignore_near) < 15 + R:
            continue
        d, t = seg_dist(c, a, b)
        if d < AVOID_DIST and 0 < t < 1 and (worst is None or t < worst[1]):
            worst = (c, t)
    return None if worst is None else worst[0]


class Avoider:
    """Drives round obstacles. If the straight path is blocked it tries a waypoint beside each obstacle (both
    sides), keeps the ones where BOTH legs (robot -> waypoint -> target) are clear of every obstacle, and takes the
    shortest. It keeps its waypoint unless one is clearly shorter (otherwise it re-decides every loop and shuffles
    left-right). If nothing is clear it falls back to going round the first obstacle on the path."""
    STICK = 15.0       # cm: a new route must be this much shorter to switch to it

    def __init__(self):
        self.wp = None
        self.obstacle = None
        self.side = 1.0

    def __call__(self, robot, target, centres, ignore_near=None, back_cost=0.0):
        """back_cost: extra cost per cm a waypoint is behind the robot (towards our goal) - used when carrying."""
        if path_blocked(robot, target, centres, ignore_near) is None:
            self.wp = self.obstacle = None
            return target
        used = [c for c in centres if ignore_near is None or dist(c, ignore_near) >= 15 + R]
        best, best_cost = None, None
        for c in used:
            for wp in (self.beside(c, target, 1.0), self.beside(c, target, -1.0),
                       self.beside_seen(c, robot, 1.0), self.beside_seen(c, robot, -1.0)):
                if wp is None or not field.inside_play_area(wp[0], wp[1], OUT_MARGIN):
                    continue
                if path_blocked(robot, wp, used) is not None or path_blocked(wp, target, used) is not None:
                    continue
                cost = dist(robot, wp) + dist(wp, target) + back_cost * max(0.0, robot[1] - wp[1])
                if self.wp is not None and dist(wp, self.wp) < 10:
                    cost -= self.STICK          # this is (about) the route we're already on
                if best_cost is None or cost < best_cost:
                    best, best_cost = wp, cost
        if best is not None:
            self.wp = best
            return clamp_in_field(best, OUT_MARGIN)
        self.wp = None
        return self.fallback(robot, target, path_blocked(robot, target, centres, ignore_near))

    @staticmethod
    def beside(c, target, side):
        """Point to the side of obstacle c, far enough out that the line from there to the target clears it too
        (a point just AVOID_DIST to the side would still clip it when the target is close behind it)."""
        u = unit((target[0] - c[0], target[1] - c[1]))
        n = (-u[1], u[0])
        L = dist(c, target)
        need = AVOID_DIST + 3
        if L <= need + 1:
            return None                          # the target is right next to the obstacle
        off = min(need * L / math.sqrt(L * L - need * need), 80.0)
        return [c[0] + n[0] * side * off, c[1] + n[1] * side * off]

    @staticmethod
    def beside_seen(c, robot, side):
        """Point to the side of obstacle c as seen from the robot (straight across its line of sight)."""
        u = unit((c[0] - robot[0], c[1] - robot[1]))
        off = AVOID_DIST + 3
        return [c[0] - u[1] * side * off, c[1] + u[0] * side * off]

    def fallback(self, robot, target, c):
        """Nothing fully clear: go round the first obstacle on the path, keeping the side we picked."""
        u = unit((target[0] - c[0], target[1] - c[1]))
        n = (-u[1], u[0])
        if self.obstacle is None or dist(c, self.obstacle) > 20:
            self.side = 1.0 if (robot[0] - c[0]) * n[0] + (robot[1] - c[1]) * n[1] >= 0 else -1.0
        self.obstacle = c
        wp = self.beside(c, target, self.side) or [c[0] + n[0] * self.side * (AVOID_DIST + 3),
                                                   c[1] + n[1] * self.side * (AVOID_DIST + 3)]
        if not field.inside_play_area(wp[0], wp[1], OUT_MARGIN):     # no room that side: go round the other way
            self.side = -self.side
            wp = [2 * c[0] - wp[0], 2 * c[1] - wp[1]]
        return clamp_in_field(wp, OUT_MARGIN)


class ShotPlanner:
    """Picks a spot to shoot from: in kick range, a clear line to some part of the goal, cheap to get to.
    Sticks with its spot unless another one is clearly better or it stops being usable."""

    def __init__(self):
        self.spot = None
        self.since = 0.0

    def reset(self):
        self.spot = None

    def score(self, p, me, centres, min_clear=SHOT_MIN_CLEAR, max_range=KICK_RANGE):
        aim, clear = best_aim(p, centres, max_range)
        if clear < min_clear:
            return None, aim
        travel = dist(me, p)
        blocked = path_blocked(me, p, centres) is not None
        sc = (min(clear, 40.0) - SHOT_TRAVEL_COST * travel - (SHOT_BLOCKED_COST if blocked else 0.0)
              - SHOT_DIST_COST * dist(p, aim) - 0.1 * abs(p[0])
              - CARRY_BACK_COST * max(0.0, me[1] - p[1]))      # don't carry the ball backwards to get there
        return sc, aim

    def plan(self, me, centres):
        """Returns (spot, aim) or (None, None) if nowhere works."""
        hx = field.PLAY_W / 2 - FIELD_MARGIN
        y_hi = field.ATTACK_GOAL_Y - R - 5
        y_lo = max(field.ATTACK_GOAL_Y - KICK_RANGE, -field.PLAY_L / 2 + FIELD_MARGIN)
        best = (None, None, None)
        nx = int(2 * hx // SHOT_GRID) + 1
        ny = int((y_hi - y_lo) // SHOT_GRID) + 1
        for i in range(nx):
            for j in range(ny):
                p = (-hx + i * SHOT_GRID, y_lo + j * SHOT_GRID)
                if any(dist(p, c) < AVOID_DIST for c in centres):
                    continue
                sc, aim = self.score(p, me, centres)
                if sc is not None and (best[0] is None or sc > best[0]):
                    best = (sc, p, aim)
        if self.spot is not None:
            # keep the current spot while it's still usable (lower bar than for picking a new one, so small
            # wobbles in where the obstacles seem to be don't make it hop between spots), unless clearly beaten
            sc, aim = self.score(self.spot, me, centres, min_clear=KICK_CLEARANCE - SHOT_KEEP_SLACK,
                                 max_range=KICK_RANGE + SHOT_KEEP_SLACK)
            young = time.monotonic() - self.since < SHOT_HOLD
            if sc is not None and (young or best[0] is None or sc + SHOT_SWITCH >= best[0]):
                return self.spot, aim
        self.spot, self.since = best[1], time.monotonic()
        return best[1], best[2]


def choose_push_aim(ball, goal_aim, defending=False):
    """Which way to push the ball. Normally at goal_aim, but if getting behind it for that would mean standing past
    the white line (ball hugging the line), push it somewhere reachable instead: straight up-field, or inwards.
    defending: the goalie clearing near our goal - the fallbacks are sideways, away from the goal mouth."""
    if defending:
        side = math.copysign(1.0, ball[0] or 1.0)
        candidates = [goal_aim, (side * field.PLAY_W / 2, ball[1] + 40.0), (side * field.PLAY_W / 2, ball[1])]
    else:
        candidates = [goal_aim, (ball[0] * 0.5, field.ATTACK_GOAL_Y), (0.0, field.ATTACK_GOAL_Y), (0.0, 0.0)]
    best, best_cost = goal_aim, None
    for k, a in enumerate(candidates):
        u = unit((a[0] - ball[0], a[1] - ball[1]))
        behind = (ball[0] - u[0] * BEHIND_DIST, ball[1] - u[1] * BEHIND_DIST)
        cost = 2.0 * dist(behind, clamp_in_field(behind, OUT_MARGIN)) + 8.0 * k   # unreachable, then less goal-ward
        if best_cost is None or cost < best_cost:
            best, best_cost = a, cost
    return best


def approach(robot, ball, aim):
    """Where to go to push the ball towards aim. Returns (target, heading, phase) in field coordinates.
    phase: "orbit" (wrong side, going round), "line up" (behind it, not lined up), "push"."""
    u = unit((aim[0] - ball[0], aim[1] - ball[1]))
    n = (-u[1], u[0])
    r = (robot[0] - ball[0], robot[1] - ball[1])
    along = r[0] * u[0] + r[1] * u[1]              # < 0 = behind the ball (good)
    perp = r[0] * n[0] + r[1] * n[1]
    heading = heading_for(u)
    behind = [ball[0] - u[0] * BEHIND_DIST, ball[1] - u[1] * BEHIND_DIST]
    # going straight to the line-up point would clip the ball (and maybe knock it the wrong way): go round instead
    clips = (abs(perp) > ALIGN_TOL + ALIGN_TOL_PER_CM * (-along)
             and seg_dist(ball, robot, behind)[0] < R + cfg.BALL_RADIUS + 3)
    if along > -R * 0.6 or clips:
        side = 1.0 if perp >= 0 else -1.0
        for s in (side, -side):
            wp = [ball[0] - u[0] * ORBIT_DIST * 0.3 + n[0] * s * ORBIT_DIST,
                  ball[1] - u[1] * ORBIT_DIST * 0.3 + n[1] * s * ORBIT_DIST]
            if field.inside_play_area(wp[0], wp[1], OUT_MARGIN):
                break
        return clamp_in_field(wp, OUT_MARGIN), heading, "orbit"
    if abs(perp) > ALIGN_TOL + ALIGN_TOL_PER_CM * (-along):
        return clamp_in_field(behind, OUT_MARGIN), heading, "line up"
    return [ball[0] + u[0] * PUSH_DIST, ball[1] + u[1] * PUSH_DIST], heading, "push"


def around_ball(me, target, ball):
    """If driving straight from me to target would hit the ball, go via a point beside it instead (the side that's
    shorter and inside the field). Used whenever the robot isn't meant to touch the ball: guarding, going round it,
    lining up - knocking it on the way there was putting it in our own goal."""
    clear = R + cfg.BALL_RADIUS + 3
    d, t = seg_dist(ball, me, target)
    if d >= clear or t <= 0.0 or t >= 1.0 or dist(me, ball) < clear:
        return target
    u = unit((target[0] - me[0], target[1] - me[1]))
    best = None
    for side in (1.0, -1.0):
        wp = [ball[0] - u[1] * side * (clear + 3), ball[1] + u[0] * side * (clear + 3)]
        if not field.inside_play_area(wp[0], wp[1], OUT_MARGIN):
            continue
        cost = dist(me, wp) + dist(wp, target)
        if best is None or cost < best[0]:
            best = (cost, wp)
    return target if best is None else best[1]


def clear_aim(ball):
    """Where to push a ball near our goal: away from the goal centre, half way to straight up-field (so it doesn't
    go out next to our goal, which puts it back on a neutral spot right in front of it)."""
    away = unit((ball[0], ball[1] - field.OWN_GOAL_Y))
    away = unit((away[0], away[1] + 1.0))
    return (ball[0] + away[0] * 100.0, ball[1] + away[1] * 100.0)


def lead_ball(me, ball, vel):
    """Where the ball will be when we get there (roughly): it keeps rolling at vel (field frame cm/s)."""
    t = min(dist(me, ball) / LEAD_SPEED, LEAD_TIME_MAX)
    return clamp_in_field([ball[0] + vel[0] * t, ball[1] + vel[1] * t], -cfg.BALL_RADIUS)


def catch_speed(me, ball):
    """Speed multiplier for running onto a free ball: slow right at it, faster further away."""
    return min(1.0, CATCH_MULTI_MIN + CATCH_MULTI_PER_CM * max(0.0, dist(me, ball) - R))


def guard_speed(me, target):
    """Speed profile for the goalie's guarding moves: proportional to how far off its spot it is, and reacting
    faster than the default smoothing (GUARD_SMOOTH)."""
    return {"spd_multi": min(1.0, max(0.1, dist(me, target) / GUARD_GAIN)), "smooth": GUARD_SMOOTH}


def contested(ball, centres):
    """An opponent is right at the ball."""
    return any(dist(c, ball) < CONTEST_DIST + R for c in centres)


def ball_out(ball):
    """Ball clearly outside the white line (the goal notches count as in)."""
    return not field.inside_play_area(ball[0], ball[1], -BALL_OUT_MARGIN)


def restart_spot(ball):
    """Neutral spot the referee will put an out ball on: the one nearest to where it is."""
    return min(field.NEUTRAL_SPOTS, key=lambda p: dist(p, ball))


def rel(p, pose):
    return [p[0] - pose.x, p[1] - pose.y]


def mate_active(mate):
    return bool(mate) and mate.get("bot active") == 1


# ==================================== GOALIE ====================================
class GoalieStrategy:
    """botstate 0 = no ball known  -> guard the middle of the goal
       botstate 1 = ball in dribbler -> carry it up-field and kick
       botstate 2 = clearing       -> go for the ball
       botstate 3 = guarding       -> on the arc between the ball and the goal centre, facing the ball
       botstate 4 = ball out       -> guard as if the ball were already on the neutral spot it'll come back to
       (fallback = strategy_fallback.GoalieFallback while not localised: its botstates are shown + 10)"""

    def __init__(self, has_dribbler=True):
        self.has_dribbler = has_dribbler     # without one, a ball at the front is pushed (botstate 2), not carried
        self.fallback = GoalieFallback()
        self.botstate_hyst = Hysteresis(hold_time=0.11, instant_enter=lambda v: v == 1)
        self.botstate = 0
        self.speed = {}
        self.kick = False
        self.chasing = False
        self.phase = ""
        self.avoid = Avoider()

    def reset(self):
        self.fallback.reset()
        self.botstate_hyst.reset()

    def update(self, world, compass, mate, line_touches):
        pose = world.pose
        self.kick = False
        if pose is None or not pose.confident:
            out = self.fallback.update(world, compass, mate.get("command") if mate else None,
                                       mate.get("bot active") if mate else None, line_touches)
            self.botstate, self.speed, self.chasing, self.phase = 10 + self.fallback.botstate, self.fallback.speed, self.fallback.chasing, "fallback"
            return out

        me = (pose.x, pose.y)
        ball = world.ball_field
        goal = (0.0, field.OWN_GOAL_Y)
        striker_on = mate_active(mate)

        # ---- decide
        if ball is None:
            raw = 0
        elif ball_out(ball):
            raw = 4
        elif world.ball_in_capture and self.has_dribbler:
            raw = 1
        else:
            in_box = abs(ball[0]) < field.PENALTY_W / 2 and ball[1] < field.OWN_GOAL_Y + field.PENALTY_D
            told = striker_on and mate.get("command") == 1
            clearing = self.botstate in (1, 2)
            opp_on_ball = any(dist(c, ball) < R + OPP_ON_BALL for c in obstacle_centres(me, world.obstacles_field))
            if not striker_on:
                raw = 2                                           # alone: go for it (like the comp code)
            elif opp_on_ball:
                raw = 3     # an opponent has it: cover the goal, don't challenge (in the simulator most goals against
                            # came from an opponent shooting past a goalie that had come out for the ball)
            elif told or (in_box and dist(me, ball) < DANGER_DIST):
                raw = 2
            elif clearing and ball[1] < CLEAR_RETURN_Y and not (mate.get("command") == 0 and mate.get("chasing")):
                raw = 2                                           # keep clearing until it's up-field
            else:
                raw = 3
        self.botstate = self.botstate_hyst.update(raw)
        if ball is None and self.botstate != 0:
            self.botstate = 0
        self.chasing = self.botstate in (1, 2)

        # ---- act
        dribbler = False
        self.speed = {}
        block = None
        if ball is not None and self.botstate != 1 and not contested(ball, obstacle_centres(me, world.obstacles_field)):
            block = self.block_point(ball, world.ball_vel)    # only a free ball (an opponent dribbling it: guard)
        if block is not None:          # ball (a shot, or rolling) heading into our goal: get in its way
            target = block
            heading = max(-GUARD_FACE_MAX, min(GUARD_FACE_MAX, heading_for((ball[0] - me[0], ball[1] - me[1]))))
            self.phase = "block"
            self.speed = guard_speed(me, target)
            return rel(target, pose), heading, False
        if self.botstate == 0:
            target, heading = (0.0, goal[1] + GUARD_RADIUS), 0.0
            self.phase = "guard middle"
            self.speed = guard_speed(me, target)
        elif self.botstate in (3, 4):
            b = ball if self.botstate == 3 else restart_spot(ball)   # ball out: guard against where it'll come back
            target, heading = self.guard(me, b)
            self.speed = guard_speed(me, target)
            if self.botstate == 3:
                target = around_ball(me, target, ball)
            self.phase = "guard" if self.botstate == 3 else "ball out"
        else:
            centres = obstacle_centres(me, world.obstacles_field)
            aim, clear = best_aim(ball, centres)
            if self.botstate == 1:
                target, heading = list(aim), heading_for((aim[0] - me[0], aim[1] - me[1]))
                dribbler = True
                self.speed = {"spd_multi": CARRY_MULTI}
                self.kick = self.can_kick(compass, heading, me, aim, clear, kick_range=1e9)   # clearing: kick from anywhere
                self.phase = "carry"
            else:
                defending = ball[1] < CLEAR_AWAY_Y
                if defending:                     # deep in our half: away from our goal, never round the ball
                    aim = clear_aim(ball)
                target, heading, dribbler = self.go_for_ball(me, ball, world.ball_vel, aim, centres, defending)
                # never go round the ball: that leaves the goal open (nearly every goal against us in the simulator) -
                # unless it's already behind the goalie's line, then getting it out sideways is all that's left
                if self.phase == "orbit" and ball[1] > field.OWN_GOAL_Y + R + 3:
                    target, heading = self.guard(me, ball)   # guard until lined up
                    self.speed = guard_speed(me, target)
                    target = around_ball(me, target, ball)
                    self.phase, dribbler = "guard (ball behind)", False
        return rel(target, pose), heading, dribbler

    @staticmethod
    def block_point(ball, vel):
        """Where a ball rolling towards our goal will cross the goalie's line (just in front of the goal), if it will
        cross between the posts soon - else None."""
        if vel[1] > -BLOCK_SPEED:
            return None
        y_line = field.OWN_GOAL_Y + R + 3
        if ball[1] < y_line:
            return None
        t_goal = (field.OWN_GOAL_Y - ball[1]) / vel[1]            # where it crosses the goal line
        if t_goal > BLOCK_TIME or abs(ball[0] + vel[0] * t_goal) > field.GOAL_W / 2 + BLOCK_WIDEN:
            return None
        t = (y_line - ball[1]) / vel[1]                         # stand where it crosses the goalie's line
        x = ball[0] + vel[0] * t
        lim = field.GOAL_W / 2 + R
        return [max(-lim, min(lim, x)), y_line]

    @staticmethod
    def guard(me, b):
        """Guard position for a ball at b: on the arc round our goal centre, between ball and goal, facing the ball.
        (Tried instead: the furthest-back spot on the bisector that still covers the whole goal - the striker drill
        scored more against that, the goalie is never quite on such a deep spot.)"""
        goal = (0.0, field.OWN_GOAL_Y)
        v = unit((b[0] - goal[0], b[1] - goal[1]))
        target = [goal[0] + v[0] * GUARD_RADIUS, max(goal[1] + v[1] * GUARD_RADIUS, goal[1] + R + 3)]
        target[0] = max(-(field.PENALTY_W / 2 - R), min(field.PENALTY_W / 2 - R, target[0]))
        face = heading_for((b[0] - me[0], b[1] - me[1]))
        return target, max(-GUARD_FACE_MAX, min(GUARD_FACE_MAX, face))

    def go_for_ball(self, me, ball, vel, aim, centres, defending=False):
        """Get behind the ball and push it towards aim (both robots). Returns (target, heading, dribbler on)."""
        b = lead_ball(me, ball, vel)
        target, heading, self.phase = approach(me, b, choose_push_aim(b, aim, defending))
        if self.phase != "push":
            target = around_ball(me, target, b)
        if self.phase == "push":
            race = any(dist(c, ball) < dist(me, ball) + RACE_MARGIN for c in centres)   # an opponent could get there first
            self.speed = {"spd_multi": PUSH_MULTI if race else catch_speed(me, ball)}
        # when pushing, the target is a point past the ball: only the way to the ball itself must be clear
        nav = b if self.phase == "push" else target
        wp = self.avoid(me, nav, centres, ignore_near=b)
        target = target if wp is nav else wp
        return target, heading, dist(me, ball) < DRIBBLER_RANGE

    @staticmethod
    def can_shoot(compass, me, centres):
        """Would kicking now score? The robot's actual heading must cross the goal line between the posts (with
        room for the ball), within KICK_RANGE, with nothing in the way. Under pressure (an opponent within
        PRESSURE_DIST) the path only has to miss the robots, without the KICK_CLEARANCE margin."""
        fx, fy = -math.sin(compass), math.cos(compass)
        if fy < 0.2:
            return False
        t = (field.ATTACK_GOAL_Y - me[1]) / fy
        cross = (me[0] + fx * t, field.ATTACK_GOAL_Y)
        pressed = any(dist(c, me) < PRESSURE_DIST for c in centres)
        if not (0 < t < KICK_RANGE and abs(cross[0]) <= field.GOAL_W / 2 - cfg.BALL_RADIUS - POST_MARGIN):
            return False
        return clearance(me, cross, centres) > (0.0 if pressed else KICK_CLEARANCE)

    @staticmethod
    def can_kick(compass, heading, me, aim, clear, kick_range=KICK_RANGE):
        err = abs(math.atan2(math.sin(heading - compass), math.cos(heading - compass)))
        return err < KICK_ANGLE and dist(me, aim) < kick_range and clear > KICK_CLEARANCE


# ==================================== STRIKER ====================================
class StrikerStrategy:
    """botstate 0 = no ball known, goalie on  -> wait in the middle
       botstate 1 = ball in dribbler            -> carry it to the best aim point, kick when lined up
       botstate 2 = ball known                  -> get behind it (orbit / line up / push)
       botstate 3 = no ball known, goalie off   -> stand in front of our goal
       botstate 4 = goalie is taking the ball   -> wait up-field of the ball for a pass
       botstate 5 = ball out                    -> wait just behind the neutral spot it'll come back to, facing
                                                   the attack goal
       (fallback = strategy_fallback.StrikerFallback while not localised: its botstates are shown + 10)"""

    def __init__(self):
        self.fallback = StrikerFallback()
        self.botstate_hyst = Hysteresis(hold_time=0.11, instant_enter=lambda v: v == 1)
        self.botstate = 2
        self.command = 0
        self.speed = {}
        self.kick = False
        self.chasing = False
        self.phase = ""
        self.last_ball = None        # (field position, time) of the last ball we (or the goalie) saw
        self.avoid = Avoider()
        self.shot = ShotPlanner()

    def reset(self):
        self.fallback.reset()
        self.botstate_hyst.reset()
        self.last_ball = None
        self.shot.reset()

    def update(self, world, compass, mate, line_touches):
        pose = world.pose
        self.kick = False
        if pose is None or not pose.confident:
            out = self.fallback.update(world, compass, mate.get("bot active") if mate else None, line_touches)
            self.botstate, self.speed, self.command, self.phase = 10 + self.fallback.botstate, self.fallback.speed, self.fallback.command, "fallback"
            self.chasing = self.fallback.botstate in (1, 2)
            return out

        me = (pose.x, pose.y)
        ball = world.ball_field
        goalie_on = mate_active(mate)
        goalie_pos = mate.get("pos") if goalie_on else None

        # ---- decide (and tell the goalie)
        if ball is None:
            raw = 0 if goalie_on else 3
            self.command = 1 if not goalie_on else 0
        elif ball_out(ball):
            raw, self.command = 5, 0
        elif world.ball_in_capture:
            raw, self.command = 1, 0
        else:
            handover = (goalie_on and not contested(ball, obstacle_centres(me, world.obstacles_field)) and ball[1] < HANDOVER_Y and
                        ball_cost(goalie_pos, ball) + HANDOVER_MARGIN < ball_cost(me, ball))
            if (self.botstate == 4 and goalie_on and mate.get("chasing") and ball[1] < CLEAR_RETURN_Y
                    and not contested(ball, obstacle_centres(me, world.obstacles_field))):
                handover = True                                   # let the goalie finish the clearance
            raw, self.command = (4, 1) if handover else (2, 0)
        self.botstate = self.botstate_hyst.update(raw)
        if ball is None and self.botstate in (2, 4, 5):
            self.botstate = 0 if goalie_on else 3
        if ball is None and self.botstate == 1:      # just lost sight of it while holding it: it's in the dribbler
            fwd = (-math.sin(compass), math.cos(compass))
            ball = (me[0] + fwd[0] * (R + cfg.BALL_RADIUS), me[1] + fwd[1] * (R + cfg.BALL_RADIUS))
        self.chasing = self.botstate in (1, 2)

        # ---- act
        dribbler = False
        self.speed = {}
        heading = 0.0
        if ball is not None:   # remember it (an out ball: remember the spot it'll come back to, to look there)
            self.last_ball = (restart_spot(ball) if ball_out(ball) else ball, time.monotonic())
        centres = obstacle_centres(me, world.obstacles_field)
        if self.botstate != 1:
            self.shot.reset()
        if self.botstate == 5:
            spot = restart_spot(ball)
            target = clamp_in_field([spot[0], spot[1] - NEUTRAL_WAIT], R)     # own-goal side of the spot
            target = self.avoid(me, target, centres)
            self.phase = "ball out"
        elif self.botstate == 0:
            lb = self.last_ball
            if lb is not None and time.monotonic() - lb[1] < SEARCH_TIME:
                target, self.phase = clamp_in_field(lb[0]), "search"    # go and look where it was
                target = self.avoid(me, target, centres)
            else:
                target, self.phase = list(WAIT_POS), "wait"
        elif self.botstate == 3:
            target, self.phase = [0.0, field.OWN_GOAL_Y + GUARD_RADIUS + 10], "defend"
        elif self.botstate == 4:
            target = clamp_in_field([-ball[0] * 0.5, ball[1] + SUPPORT_OFFSET])
            target = self.avoid(me, target, centres)
            heading = heading_for((ball[0] - me[0], ball[1] - me[1]))
            self.phase = "support"
        else:
            aim, clear = best_aim(ball, centres)
            if self.botstate == 1:
                dribbler = True
                self.speed = {"spd_multi": CARRY_MULTI}
                heading = heading_for((aim[0] - me[0], aim[1] - me[1]))
                if GoalieStrategy.can_shoot(compass, me, centres):
                    target, self.kick, self.phase = list(me), True, "shoot"   # clear shot from here: take it
                else:
                    spot, spot_aim = self.shot.plan(me, centres)
                    if spot is None:                       # no good spot anywhere: push towards the goal
                        target, self.phase = self.avoid(me, list(aim), centres, back_cost=CARRY_BACK_COST), "carry"
                    else:
                        # always face the aim point (robot -> ball -> aim in a line) and strafe to the spot,
                        # so the shot is already lined up when we get there
                        heading = heading_for((spot_aim[0] - me[0], spot_aim[1] - me[1]))
                        if dist(me, spot) > SHOT_ARRIVED:
                            target, self.phase = self.avoid(me, list(spot), centres, back_cost=CARRY_BACK_COST), "to spot"
                        else:
                            target, self.phase = list(spot), "line up shot"
                            self.kick = GoalieStrategy.can_shoot(compass, me, centres)
                if target[1] < me[1] - CARRY_BACK_ALLOW:     # never back off with the ball: go sideways instead
                    target = [target[0], me[1]]
            else:
                # ball in our half and we're between it and our goal: clear it from here (like the goalie) instead
                # of going round it, which would leave the opponent a free push at our goal
                defending = ball[1] < STRIKER_CLEAR_Y and me[1] < ball[1] - R
                if defending:
                    aim = clear_aim(ball)
                target, heading, dribbler = GoalieStrategy.go_for_ball(self, me, ball, world.ball_vel, aim, centres,
                                                                       defending)
        return rel(target, pose), heading, dribbler

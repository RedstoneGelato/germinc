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
KICK_RANGE = 50.0                 # kick when the aim point is closer than this (against a goalie that guards well,
                                  # in the simulator < 50 cm scored 3x as often as 50-75 cm)...
PRESSURE_RANGE = 60.0             # ...or this, when about to lose the ball anyway (PRESSURE_DIST)
LONG_RANGE = field.PLAY_L / 2     # ...or this (half the field) at an OPEN goal: no robot within OPEN_GOAL_DIST of the
OPEN_GOAL_DIST = 30.0             # goal's centre (their goalie is out) or the shot passes every robot by OPEN_CLEAR cm
OPEN_CLEAR = 20.0
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
CARRY_MARGIN = R + 16.0           # carrying the ball: robot centre kept this far inside the line (the ball sits in
                                  # front of the robot: going partly past the line with it puts the ball out)
CARRY_BACK_ALLOW = 2.0            # carrying the ball: never drive more than this (cm) backwards - go sideways instead
LEAD_TIME_MAX = 0.6               # s: go for where a moving ball will be in up to this long...
LEAD_SPEED = 100.0                # ...(= distance to it / this speed)
CATCH_MULTI_MIN = 0.3             # pushing a free ball: speed multiplier right at the ball (gentle, so it's caught
CATCH_MULTI_PER_CM = 0.015        # instead of knocked away), rising this much per cm the robot's edge is from it
RACE_MARGIN = 15.0                # ...unless an opponent is less than this much further from the ball than us: then
                                  # full speed (PUSH_MULTI), getting there first matters more than catching it cleanly

GUARD_RADIUS = min(field.PENALTY_D - 5.0, 20.0)   # goalie: distance from the goal centre it guards at
GUARD_LEAD = 0.1                  # goalie guarding: guard where the ball will be this many seconds ahead
GUARD_GAIN = 15.0                 # goalie guarding / blocking: full speed when this far (cm) from its spot, slower
                                  # in proportion closer (the default speed curve crawls over short distances)
GUARD_SMOOTH = 0.3                # goalie guarding / blocking: motion.Mover smoothing (default 0.1 lags ~0.1 s)
BLOCK_SPEED = 20.0                # goalie: ball rolling towards our goal line faster than this (cm/s)...
BLOCK_TIME = 1.5                  # ...and crossing our goal line within this (s), between the posts (+ BLOCK_WIDEN
BLOCK_WIDEN = 8.0                 # cm) -> stand where it will cross the goalie's line, whatever else it was doing
DANGER_DIST = 35.0                # goalie: ball in our penalty area and this close -> clear it...
CARRIER_WEIGHT = 0.0              # goalie, an opponent has the ball: how far to move from the guard spot towards the
CARRIER_WIDEN = 10.0              # line it's facing along (if that line hits the goal +- this): 1 = all the way.
                                  # 0 = off: 0.6 did slightly worse in the simulator (a striker turns before it kicks,
                                  # so the goalie moving onto its current line just opens the other side)
GOALIE_KICK_TURN = math.radians(60)   # goalie with the ball (striker off): kicks it clear facing at most this far off
GOALIE_KICK_CLEAR = 50.0          # up-field, when no robot is in the way for this far (cm)
PASS_SIDE = 30.0                  # goalie with the ball, striker on: kicks it past the striker, aiming this far to one
PASS_AHEAD = 20.0                 # side of it and this far up-field of it (the more open side; both open: the one
PASS_TOL = 0.25                   # nearer the middle), once facing it within this (rad)
EST_SPEED = 120.0                 # rough driving speed (cm/s), turning speed (rad/s) and turning speed with the ball in
EST_TURN = 3.5                    # the dribbler, for choosing between catching the ball from the front or orbiting
CATCH_TURN_WEIGHT = 0.5           # turning back with the ball counts this much (we already have it then: better than
                                  # counting it fully in the simulator)
EST_TURN_DRIBBLE = 2.4            # round it (catch_is_faster) - TUNE on the robot (TURN_MAX, DRIBBLE_TURN_MAX x top spin)
BLOCKER_WIDTH = 2.0               # an opponent's centre within 2R + this of our straight line to the ball = in the way
BLOCKER_UPFIELD = 0.4             # ...unless we'd push the group this up-field (direction's y component)
BLOCKER_CLEAR = 8.0               # going round an opponent + ball: pass this far (cm, robot edges) clear of the group
SOLO_CHASE_Y = 0.0                # goalie on its own (striker off): goes for the ball only in our half (y below this)
SOLO_CLOSER = 15.0                # ...or when it's this much closer to the ball than every opponent; else it guards
DODGE_DIST = 2 * R + 20.0         # striker with the ball, no shot, an opponent's centre this close: sidestep away
DODGE_NEAR = 2 * R + 6.0          # ...that is this close (touching, e.g. a scrum)...
DODGE_CLOSING = 20.0              # ...or coming at us faster than this (cm/s)
PUSHBACK_GAP = 6.0                # an opponent's centre within 2R + this of ours, with the ball between us (in front,
PUSHBACK_ANGLE = math.radians(50) # within this angle of the line to it) = a shoving match: shove back at full speed
PUSHBACK_REACH = 40.0             # (a close target means a slow robot, which just gets pushed back to our goal)
GOALIE_KICK_OUT = True            # goalie in a shoving match with the ball in its dribbler: kick it out over a side line
GOALIE_OUT_UP = 0.3               # (this much up-field per cm sideways), once facing within
GOALIE_KICK_OUT_TOL = 0.4         # this (rad) - an out ball comes back on the halfway line, far from our goal. The side
GOALIE_OUT_CLEAR = 60.0           # with no robot within this (cm) of the kick's path first (kick_out_side()), else the nearer
ORBIT_SPD_MIN = 0.5               # going round the ball / an opponent: at least this speed multiplier (the waypoints are
                                  # close, and motion.py's speed drops with the distance: it crawled round, getting pushed)
CHALLENGE_DIST = 35.0             # goalie: an opponent has the ball this close (cm) to our goal centre -> go and knock it
CHALLENGE_UP = 40.0               # out over the nearer side line (aim this far up-field of the ball, past the line):
                                  # standing guard just gets the goalie shoved back into its goal
DODGE_MARGIN = R + 4.0            # sidestepping: the robot's centre may come this close to the line (CARRY_MARGIN left
                                  # too narrow a lane to get past a robot standing in the middle: it shuffled in front
                                  # of it, side to side, for ever)
DODGE_STEP = 30.0                 # cm the sidestep target is to the side
PUSHER_DIST = 28.0                # ...except an opponent pushing it at our goal (ball this close to the goal centre and
PUSHER_SPEED = 10.0               # rolling towards our goal faster than this): meet it, guarding just gets it pushed in
OPP_ON_BALL = 12.0                # ...unless an opponent has it (its centre within R + this of the ball): then guard
CLEAR_RETURN_Y = field.OWN_GOAL_Y + 55.0   # goalie: stop clearing once the ball is past this
CLEAR_AWAY_Y = field.OWN_GOAL_Y + 50.0     # goalie: ball closer to our goal line than this -> push it straight away
                                           # from the goal centre (so it stays between ball and goal), not at their goal
CATCH_FRONT_Y = field.OWN_GOAL_Y + 25.0     # striker not lined up behind the ball (orbit / line up) and the ball is
CATCH_FRONT_DIST = 999.0                   # past this (and within this): drive straight onto it facing it, catch it
                                           # with the dribbler and turn with it (botstate 1) instead of going round
                                           # it - in the simulator orbiting took 30% of the striker's time, and
                                           # catching from the front beat it every time it was used more
CLEAR_OUT_X = 15.0                         # clearing a ball this far to the side of our goal: push it out over the
                                           # side line (it comes back on the halfway line, far from our goal)
STRIKER_CLEAR_Y = 0.0                      # striker: ball in our half and the striker goal-side of it -> clear it
                                           # up-field from there (clear_aim) instead of going round it
HANDOVER_Y = field.OWN_GOAL_Y + 60.0       # striker: ball behind this and goalie closer -> goalie takes it
HANDOVER_MARGIN = 10.0            # goalie must be this much closer (cost) to take the ball...
CONTEST_DIST = 8.0                # ...and no opponent's edge within this of the ball (contested: both robots go)
PUSH_MULTI = 1.0                  # speed multiplier when pushing a CONTESTED ball (full speed: win pushing contests;
                                  # a free ball is approached normally, or it gets knocked away instead of caught)
WRONG_SIDE_COST = 30.0            # cost penalty for being on the attack side of the ball
SUPPORT_OFFSET = 45.0             # striker: wait this far up-field of the ball while the goalie clears
WAIT_POS = (0.0, -22.0)           # striker: no ball known anywhere -> wait here (behind the centre neutral spot, not
                                  # on it: an occupied spot is skipped when the ball is put back)...
SEARCH_TIME = 1.0                 # ...after first spending this long (s) looking where the ball was last seen
BALL_OUT_MARGIN = 1.0             # ball centre this far past the outer edge of the white line = (about to be) out
                                  # (the referee takes it off once it's fully past: BALL_RADIUS)
OUT_SOON_DIST = 15.0              # ball last seen this close to the line and rolling outwards faster than
OUT_SOON_SPEED = 0.0              # this (cm/s) = it went out
RESTART_WAIT = 4.0                # s the striker waits at the restart spot after the ball disappeared there
RESTART_SIDE = 0.0                # ...diagonally behind it, this far towards the middle (not head-on with their striker)
NEUTRAL_WAIT = R + 11.0           # ball out: striker waits this far behind the neutral spot it'll come back to
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
    """Where to push a ball near our goal: away from the goal centre, half way to straight up-field (up-field rather
    than just sideways, so it actually leaves our end)."""
    if abs(ball[0]) > CLEAR_OUT_X:    # out to the side: an out ball goes back to the halfway line (field.NEUTRAL_SPOTS)
        return (math.copysign(field.PLAY_W / 2 + 30.0, ball[0]), ball[1] + 15.0)
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


def catch_is_faster(me, compass, ball, aim, orbit_target):
    """Would catching the ball from where we are (turn to face it while driving to it, then turn with it until we
    face the aim, slower because of the dribbler) be quicker than going round it (orbit_target, then to the line-up
    point behind it, then onto the ball)? With the ball behind us, turning round and back takes longer than a
    short orbit."""
    to_ball = (ball[0] - me[0], ball[1] - me[1])
    turn_to_ball = abs(math.atan2(math.sin(heading_for(to_ball) - compass), math.cos(heading_for(to_ball) - compass)))
    to_aim = heading_for((aim[0] - ball[0], aim[1] - ball[1]))
    turn_back = abs(math.atan2(math.sin(to_aim - heading_for(to_ball)), math.cos(to_aim - heading_for(to_ball))))
    t_catch = (max(turn_to_ball / EST_TURN, max(0.0, dist(me, ball) - R) / EST_SPEED)
               + CATCH_TURN_WEIGHT * turn_back / EST_TURN_DRIBBLE)
    u = unit((aim[0] - ball[0], aim[1] - ball[1]))
    behind = (ball[0] - u[0] * BEHIND_DIST, ball[1] - u[1] * BEHIND_DIST)
    t_orbit = (dist(me, orbit_target) + dist(orbit_target, behind) + (BEHIND_DIST - R)) / EST_SPEED
    return t_catch < t_orbit


def blocker_between(me, ball, centres):
    """An opponent (centre) in the way between us and the ball: our body would hit it on the straight way to the
    ball, and it's nearer the ball than we are - or None."""
    best = None
    for c in centres:
        d, t = seg_dist(c, me, ball)
        if d < 2 * R + BLOCKER_WIDTH and 0.05 < t and dist(c, ball) < dist(me, ball) and (best is None or t < best[0]):
            best = (t, c)
    return None if best is None else best[1]


def contested(ball, centres):
    """An opponent is right at the ball."""
    return any(dist(c, ball) < CONTEST_DIST + R for c in centres)


def ball_out(ball):
    """Ball clearly outside the white line (the goal notches count as in)."""
    return not field.inside_play_area(ball[0], ball[1], -BALL_OUT_MARGIN)


def going_out(ball, vel):
    """The ball is out, or right at the line and rolling out."""
    if ball_out(ball):
        return True
    if field.inside_play_area(ball[0], ball[1], OUT_SOON_DIST):
        return False
    n = (math.copysign(1.0, ball[0]), 0.0) if abs(ball[0]) > field.PLAY_W / 2 - OUT_SOON_DIST else \
        (0.0, math.copysign(1.0, ball[1]))
    return vel[0] * n[0] + vel[1] * n[1] > OUT_SOON_SPEED


def restart_spot(ball):
    """Neutral spot the referee will put an out ball on: the one nearest to where it is."""
    return min(field.NEUTRAL_SPOTS, key=lambda p: dist(p, ball))


def shover(me, ball, centres):
    """The opponent shoving against us over the ball (touching, ball in between, not pushing it towards our own
    goal), or None."""
    if dist(me, ball) > R + cfg.BALL_RADIUS + 6:
        return None
    to_ball = math.atan2(ball[1] - me[1], ball[0] - me[0])
    for c in centres:
        if dist(c, me) > 2 * R + PUSHBACK_GAP:
            continue
        a = math.atan2(c[1] - me[1], c[0] - me[0])
        if abs(math.atan2(math.sin(a - to_ball), math.cos(a - to_ball))) < PUSHBACK_ANGLE and c[1] - me[1] > -0.3 * dist(c, me):
            return c
    return None


def possession(world, mate):
    """Who has the ball: "us" (in our dribbler or the teammate's), "them" (an opponent's centre right behind it, see
    GoalieStrategy.carrier) or None (free / not seen)."""
    if world.ball_in_capture or (mate and mate.get("has ball")):
        return "us"
    if world.ball_field is not None and GoalieStrategy.carrier(world.ball_field, world.opponents) is not None:
        return "them"
    return None


def knock_out_aim(ball, me):
    """Goalie challenging an opponent that has the ball: knock it out over the nearer side line (the side away from
    the goalie when the ball is in the middle), a bit up-field so the goalie comes at it from our goal's side."""
    side = math.copysign(1.0, (ball[0] - me[0]) if abs(ball[0]) < 10 else ball[0])
    return (side * (field.PLAY_W / 2 + 30.0), ball[1] + CHALLENGE_UP)


def kick_out_side(me, ball, centres, current=None):
    """Goalie kicking the ball out: which side line (+1 / -1). The side with no robot in the way of the kick (path
    clear for GOALIE_OUT_CLEAR cm), the nearer one if both or neither are clear. Keeps `current` while it's clear
    (re-deciding every loop as the robots move would turn it back and forth)."""
    def clear(side):
        u = unit((side, GOALIE_OUT_UP))
        return clearance(me, (me[0] + u[0] * GOALIE_OUT_CLEAR, me[1] + u[1] * GOALIE_OUT_CLEAR), centres) > 0
    if current is not None and clear(current):
        return current
    near = math.copysign(1.0, ball[0] or me[0] or 1.0)
    return near if clear(near) or not clear(-near) else -near


def pass_aim(me, striker, centres):
    """Goalie clearing to the striker: a point beside the striker (PASS_SIDE to one side, PASS_AHEAD up-field), so
    the ball rolls past it up-field and it can take it facing the way it's going (instead of turning round). The side
    with the clear path; both clear (or both blocked): the one nearer the middle of the field (both blocked: the one
    with more room)."""
    cands = []
    for side in (1.0, -1.0):
        p = clamp_in_field([striker[0] + side * PASS_SIDE, striker[1] + PASS_AHEAD])
        cands.append((clearance(me, p, [c for c in centres if dist(c, striker) > R]), p))
    open_ = [c for c in cands if c[0] > KICK_CLEARANCE]
    if len(open_) == 1:
        return open_[0][1]
    if open_:
        return min(open_, key=lambda c: abs(c[1][0]))[1]
    return max(cands, key=lambda c: c[0])[1]


def rel(p, pose):
    return [p[0] - pose.x, p[1] - pose.y]


def mate_active(mate):
    return bool(mate) and mate.get("bot active") == 1


# ==================================== GOALIE ====================================
class GoalieStrategy:
    """botstate 0 = no ball known  -> guard the middle of the goal
       botstate 1 = ball in dribbler -> turn up-field and kick it clear (only with a dribbler)
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
        self.possession = None   # "us" / "them" / None: who has the ball (possession())
        self.out_side = None     # kicking the ball out: over which side line (kick_out_side())
        self.avoid = Avoider()

    def reset(self):
        self.fallback.reset()
        self.botstate_hyst.reset()

    def update(self, world, compass, mate, line_touches):
        pose = world.pose
        self.kick = False
        self.possession = possession(world, mate)
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
        elif world.ball_in_capture and self.has_dribbler and self.carrier(ball, world.opponents) is None:
            raw = 1     # (an opponent pushing the ball into our front isn't "ours": keep guarding, see below)
        else:
            in_box = abs(ball[0]) < field.PENALTY_W / 2 and ball[1] < field.OWN_GOAL_Y + field.PENALTY_D
            told = striker_on and mate.get("command") == 1
            clearing = self.botstate in (1, 2)
            opp_on_ball = self.possession == "them"
            if not striker_on:          # alone: clear what's in our half or clearly ours, otherwise mind the goal
                closest_opp = min((dist(c, ball) for c in world.opponents), default=1e9)
                raw = 2 if ball[1] < SOLO_CHASE_Y or dist(me, ball) + SOLO_CLOSER < closest_opp else 3
            elif opp_on_ball and dist(ball, goal) < CHALLENGE_DIST:
                raw = 2     # ...unless it's bringing it at our goal: go and knock it out over the side line
            elif opp_on_ball and not (dist(ball, goal) < PUSHER_DIST and world.ball_vel[1] < -PUSHER_SPEED):
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

        # ---- an opponent shoving the ball against the goalie: shove straight back (guarding = crawling = getting
        # pushed into our goal), or with the ball in the dribbler, kick it out over the side line
        centres = obstacle_centres(me, world.obstacles_field)
        sh = shover(me, ball, centres) if ball is not None else None
        if sh is None:
            self.out_side = None
        else:
            if GOALIE_KICK_OUT and world.ball_in_capture and self.has_dribbler:
                self.out_side = kick_out_side(me, ball, [c for c in centres if c is not sh], self.out_side)
                heading = heading_for((self.out_side, GOALIE_OUT_UP))
                err = abs(math.atan2(math.sin(heading - compass), math.cos(heading - compass)))
                self.kick, self.speed, self.phase = err < GOALIE_KICK_OUT_TOL, {}, "kick out"
                return rel(list(me), pose), heading, True
            d = unit((sh[0] - me[0], sh[1] - me[1]))     # (pushing off sideways towards the line opened the goal)
            target = clamp_in_field([me[0] + d[0] * PUSHBACK_REACH, me[1] + d[1] * PUSHBACK_REACH], OUT_MARGIN)
            self.speed, self.phase = {"spd_multi": 1.0}, "push back"
            return rel(target, pose), 0.0, False

        # ---- act
        dribbler = False
        self.speed = {}
        block = None
        if ball is not None and self.botstate != 1 and not contested(ball, obstacle_centres(me, world.obstacles_field)):
            block = self.block_point(ball, world.ball_vel)    # only a free ball (an opponent dribbling it: guard)
        if block is not None:          # ball (a shot, or rolling) heading into our goal: get in its way
            target = block
            heading = 0.0                  # no turning: all the motors go into getting there
            self.phase = "block"
            self.speed = guard_speed(me, target)
            return rel(target, pose), heading, False
        if self.botstate == 0:
            target, heading = (0.0, goal[1] + GUARD_RADIUS), 0.0
            self.phase = "guard middle"
            self.speed = guard_speed(me, target)
        elif self.botstate in (3, 4):
            b = ball if self.botstate == 3 else restart_spot(ball)   # ball out: guard against where it'll come back
            if self.botstate == 3:          # guard where the ball will be in a moment, not where it was
                b = (b[0] + world.ball_vel[0] * GUARD_LEAD, b[1] + world.ball_vel[1] * GUARD_LEAD)
            target, heading = self.guard(me, b)
            carrier = self.carrier(ball, world.opponents) if self.botstate == 3 and CARRIER_WEIGHT else None
            if carrier is not None:         # an opponent has it: lean towards the line it would shoot along
                target = self.shot_line_guard(target, b, carrier)
            self.speed = guard_speed(me, target)
            if self.botstate == 3:
                target = around_ball(me, target, ball)
            self.phase = ("guard carrier" if carrier is not None else "guard") if self.botstate == 3 else "ball out"
        else:
            centres = obstacle_centres(me, world.obstacles_field)
            aim, clear = best_aim(ball, centres)
            if self.botstate == 1:
                # ball in the dribbler: stay put, turn up-field (towards the open part of their goal, at most
                # GOALIE_KICK_TURN off straight up-field) and kick it clear as soon as we face up-field with room
                dribbler = True
                target = list(me)
                striker = mate.get("pos") if striker_on else None
                if striker is not None and striker[1] > me[1] + R:
                    # pass it past the striker (up-field of us): to its more open side, kick when facing it
                    p = pass_aim(me, striker, centres)
                    heading = heading_for((p[0] - me[0], p[1] - me[1]))
                    err = abs(math.atan2(math.sin(heading - compass), math.cos(heading - compass)))
                    self.kick = err < PASS_TOL
                    self.phase = "clear pass"
                else:
                    want = heading_for((aim[0] - me[0], aim[1] - me[1]))
                    heading = max(-GOALIE_KICK_TURN, min(GOALIE_KICK_TURN, want))
                    fwd = (-math.sin(compass), math.cos(compass))
                    ahead = (me[0] + fwd[0] * GOALIE_KICK_CLEAR, me[1] + fwd[1] * GOALIE_KICK_CLEAR)
                    self.kick = abs(compass) < GOALIE_KICK_TURN + 0.1 and clearance(me, ahead, centres) > 0
                    self.phase = "clear kick"
            else:
                defending = ball[1] < CLEAR_AWAY_Y
                if self.possession == "them":     # challenging the opponent that has it: knock it out of play
                    aim, defending = knock_out_aim(ball, me), True
                elif defending:                   # deep in our half: away from our goal, never round the ball
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
    def carrier(ball, opponents):
        """The opponent (centre, field frame) that has the ball - its centre right behind the ball - or None.
        Uses both robots' cameras (world.opponents merges the teammate's "opps")."""
        c = min(opponents, key=lambda o: dist(o, ball), default=None)
        return c if c is not None and dist(c, ball) < R + OPP_ON_BALL else None

    @staticmethod
    def shot_line_guard(target, ball, carrier):
        """Shift the guard spot towards the line carrier -> ball (the ball sits in front of the robot holding it, so
        that's where a kick would go), if that line ends up in our goal."""
        d = unit((ball[0] - carrier[0], ball[1] - carrier[1]))
        if d[1] > -0.2:
            return target                                    # not facing our goal
        x_goal = ball[0] + d[0] * (field.OWN_GOAL_Y - ball[1]) / d[1]
        if abs(x_goal) > field.GOAL_W / 2 + CARRIER_WIDEN:
            return target                                    # facing past the posts
        x_line = ball[0] + d[0] * (target[1] - ball[1]) / d[1]   # the line at the guard spot's distance
        x = target[0] + CARRIER_WEIGHT * (x_line - target[0])
        lim = field.PENALTY_W / 2 - R
        return [max(-lim, min(lim, x)), target[1]]

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
        return target, 0.0     # facing up-field: turning towards the ball would starve the sideways moves (in
                               # motion.VelocityToMotor turning and moving share the motors), and the camera sees
                               # all round anyway

    def go_for_ball(self, me, ball, vel, aim, centres, defending=False):
        """Get behind the ball and push it towards aim (both robots). Returns (target, heading, dribbler on)."""
        b = lead_ball(me, ball, vel)
        target, heading, self.phase = approach(me, b, choose_push_aim(b, aim, defending))
        if self.phase != "push":
            target = around_ball(me, target, b)
        if self.phase in ("orbit", "line up"):
            self.speed = {"spd_min": ORBIT_SPD_MIN}
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
        if not (0 < t < LONG_RANGE and abs(cross[0]) <= field.GOAL_W / 2 - cfg.BALL_RADIUS - POST_MARGIN):
            return False
        clear = clearance(me, cross, centres)
        if t >= (PRESSURE_RANGE if pressed else KICK_RANGE):          # a long shot: only at an open goal
            goal_empty = all(dist(c, (0.0, field.ATTACK_GOAL_Y)) > OPEN_GOAL_DIST for c in centres)
            wide = clear > OPEN_CLEAR
            return clear > KICK_CLEARANCE and (goal_empty or wide)
        return clear > (0.0 if pressed else KICK_CLEARANCE)

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
        self.possession = None   # "us" / "them" / None: who has the ball (possession())
        self.speed = {}
        self.kick = False
        self.chasing = False
        self.phase = ""
        self.last_ball = None        # (field position, time) of the last ball we (or the goalie) saw
        self.avoid = Avoider()
        self.shot = ShotPlanner()
        self.group_side = None          # side picked to go round an opponent in front of the ball

    def reset(self):
        self.fallback.reset()
        self.botstate_hyst.reset()
        self.last_ball = None
        self.shot.reset()

    def update(self, world, compass, mate, line_touches):
        pose = world.pose
        self.kick = False
        self.possession = possession(world, mate)
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
            out = going_out(ball, world.ball_vel)
            self.last_ball = (restart_spot(ball) if out else ball, time.monotonic(), out)
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
            if lb is not None and lb[2] and time.monotonic() - lb[1] < RESTART_WAIT:
                # restart play: the ball went out and comes back on that neutral spot - wait just behind it,
                # facing their goal, to take it the moment it's there
                target = [lb[0][0] - math.copysign(RESTART_SIDE, lb[0][0] or 1.0), lb[0][1] - NEUTRAL_WAIT]
                target, self.phase = self.avoid(me, target, centres), "restart"
            elif lb is not None and time.monotonic() - lb[1] < SEARCH_TIME:
                target, self.phase = clamp_in_field(lb[0]), "search"    # go and look where it was
                target = self.avoid(me, target, centres)
            else:
                target, self.phase = list(WAIT_POS), "wait"
        elif self.botstate == 3:
            target, self.phase = [0.0, field.OWN_GOAL_Y + GUARD_RADIUS + 10], "defend"
        elif self.botstate == 4:
            target = clamp_in_field([-ball[0] * 0.5, ball[1] + SUPPORT_OFFSET])
            target = self.avoid(me, target, centres)
            heading = 0.0       # face up-field (facing the goalie's ball meant turning round to take a clearance)
            self.phase = "support"
        else:
            aim, clear = best_aim(ball, centres)
            if self.botstate == 1:
                dribbler = True
                self.speed = {"spd_multi": CARRY_MULTI}
                heading = heading_for((aim[0] - me[0], aim[1] - me[1]))
                threat = self.dodge_threat(me, world)
                if GoalieStrategy.can_shoot(compass, me, centres):
                    target, self.kick, self.phase = list(me), True, "shoot"   # clear shot from here: take it
                elif threat is not None and dist(threat, me) < DODGE_DIST:
                    target, self.phase = self.dodge(me, threat), "dodge"      # about to be tackled: sidestep
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
                # the ball is in front of us: keep it in the field (sidestepping a bit closer to the line)
                target = clamp_in_field(target, DODGE_MARGIN if self.phase == "dodge" else CARRY_MARGIN)
            else:
                # ball in our half and we're between it and our goal: clear it from here (like the goalie) instead
                # of going round it, which would leave the opponent a free push at our goal
                defending = ball[1] < STRIKER_CLEAR_Y and me[1] < ball[1] - R
                if defending:
                    aim = clear_aim(ball)
                target, heading, dribbler = GoalieStrategy.go_for_ball(self, me, ball, world.ball_vel, aim, centres,
                                                                       defending)
                if (self.phase in ("orbit", "line up") and ball[1] > CATCH_FRONT_Y and dist(me, ball) < CATCH_FRONT_DIST
                        and path_blocked(me, ball, centres, ignore_near=ball) is None
                        and catch_is_faster(me, compass, ball, aim, target)):
                    # wrong side of the ball: catch it from here and turn with it instead of going round it
                    target, heading, dribbler = list(ball), heading_for((ball[0] - me[0], ball[1] - me[1])), True
                    self.speed = {"spd_multi": catch_speed(me, ball)}
                    self.phase = "catch"
                blocker = blocker_between(me, ball, centres)
                if blocker is not None and unit((ball[0] - me[0], ball[1] - me[1]))[1] > BLOCKER_UPFIELD:
                    blocker = None      # shoving it up-field (towards their goal) is fine: just go
                if blocker is not None:
                    # an opponent between us and the ball: going straight at the ball would shove it onto the ball
                    # (and maybe into our goal) - go round the opponent and the ball together instead
                    target = self.around_group(me, ball, blocker)
                    heading, dribbler = heading_for((ball[0] - me[0], ball[1] - me[1])), False
                    self.speed, self.phase = {"spd_min": ORBIT_SPD_MIN}, "round blocker"
                else:
                    self.group_side = None
                if not goalie_on and self.solo_defend(me, ball, world.opponents):
                    # on our own and an opponent will get to the ball first in our half: mind the goal instead
                    target, heading = GoalieStrategy.guard(me, ball)
                    block = GoalieStrategy.block_point(ball, world.ball_vel)
                    target = around_ball(me, block or target, ball)
                    self.speed, dribbler, self.phase = guard_speed(me, target), False, "solo guard"
        if self.botstate == 2 and ball is not None:
            # going for the ball and an opponent shoves against us over it (kickoffs): shove back at full speed
            # (with the ball already ours, dodging round it scored more)
            sh = shover(me, ball, centres)
            if sh is not None:
                d = unit((sh[0] - me[0], sh[1] - me[1]))
                target = clamp_in_field([me[0] + d[0] * PUSHBACK_REACH, me[1] + d[1] * PUSHBACK_REACH], OUT_MARGIN)
                heading = heading_for(d)
                self.speed, self.phase = {"spd_multi": 1.0}, "push back"
        return rel(target, pose), heading, dribbler

    def around_group(self, me, ball, blocker):
        """Waypoint beside the opponent + ball as one group, far enough out to clear both. Keeps the side it picked
        (re-deciding every loop shuffles); prefers the side that stays in the field, then the side the ball is on
        (shorter way round to it)."""
        g = ((ball[0] + blocker[0]) / 2, (ball[1] + blocker[1]) / 2)
        reach = dist(ball, blocker) / 2 + R + BLOCKER_CLEAR
        u = unit((g[0] - me[0], g[1] - me[1]))
        n = (-u[1], u[0])

        def wp(side):
            return [g[0] + n[0] * side * reach, g[1] + n[1] * side * reach]

        if self.group_side is None:
            ball_side = 1.0 if (ball[0] - blocker[0]) * n[0] + (ball[1] - blocker[1]) * n[1] >= 0 else -1.0
            self.group_side = min((ball_side, -ball_side),
                                  key=lambda s: dist(wp(s), clamp_in_field(wp(s), OUT_MARGIN)) * 10 + (s != ball_side))
        return clamp_in_field(wp(self.group_side), OUT_MARGIN)

    @staticmethod
    def solo_defend(me, ball, opponents):
        """Goalie off: the ball is in our half and an opponent is clearly closer to it than we are."""
        closest_opp = min((dist(c, ball) for c in opponents), default=1e9)
        return ball[1] < 0.0 and closest_opp + SOLO_CLOSER < dist(me, ball)

    @staticmethod
    def dodge(me, threat):
        """Sidestep away from an opponent about to tackle (keeping the heading, so the shot stays lined up): to the
        side of the line between us, the up-field side if both are in the field. (Tried: ignoring opponents behind
        us, keeping the chosen side for a while, adding forward progress, only dodging opponents moving towards us -
        all worse in self-play.)"""
        d = unit((threat[0] - me[0], threat[1] - me[1]))
        best = None
        for side in (1.0, -1.0):
            p = [me[0] - d[1] * side * DODGE_STEP - d[0] * 10.0, me[1] + d[0] * side * DODGE_STEP - d[1] * 10.0]
            score = p[1] - 3.0 * dist(p, clamp_in_field(p, DODGE_MARGIN))
            if best is None or score > best[0]:
                best = (score, p)
        return best[1]

    @staticmethod
    def dodge_threat(me, world):
        """Opponent to dodge: the nearest one that's either coming at us (DODGE_CLOSING) or already right next to us
        (DODGE_NEAR, e.g. a scrum). One standing still further away is just gone round (dodging it only lost the
        ball sideways: with the opponents standing still the striker shuffled left and right in front of them)."""
        best = None
        for o, v in zip(world.opponents, world.opponent_vel):
            d = dist(o, me)
            to_me = unit((me[0] - o[0], me[1] - o[1]))
            if (d < DODGE_NEAR or v[0] * to_me[0] + v[1] * to_me[1] > DODGE_CLOSING) and (best is None or d < best[0]):
                best = (d, o)
        return None if best is None else best[1]


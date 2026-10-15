"""
strategy.py - what each robot decides to do, every loop.

(New to this file? Read STRATEGY_EXPLAINED.md first: it explains the Python and the maths used here.)

The three "brains" in this file:
    GoalieStrategy   (main.py)         guard the goal, clear the ball when it's ours to take
    StrikerStrategy  (main_attack.py)  chase the ball fast, carry it at their goal and kick

While the robot doesn't know where it is on the field (pose not confident), both hand over to the comp-style
relative logic in strategy_fallback.py.

Every brain has the same update() - main.py / main_attack.py call it once per loop:
    update(world, compass, mate, line_touches) -> (desired_pos, desired_heading, dribbler_on)
        desired_pos      arrow (x, y in cm, field directions) from the robot to where it should go
        desired_heading  compass heading to turn to (radians, 0 = facing the goal we attack, + = anticlockwise)
        dribbler_on      True / False
and sets these for the main loop to read:
    .kick      True = fire the kicker this loop (ignored without a kicker / while it recharges)
    .speed     speed settings for motion.Mover.step, e.g. {"spd_multi": 0.6} = 60% speed
    .botstate  a number for printing / the simulator
    .phase     a word for printing / the simulator ("chase", "carry", "guard", ...)
    .chasing   True = this robot is going for the ball (sent to the teammate)
    .command   striker only: 1 = goalie take the ball, 0 = goalie stay in goal (sent to the goalie)

FIELD COORDINATES (cm): (0, 0) = centre of the field, +y = towards the goal we attack, +x = right when facing it.
A point is a pair (x, y): p[0] = x, p[1] = y.
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
KICK_CLEARANCE = 8.0              # goalie passing: no obstacle's edge closer than this to the ball's path
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
BLOCK_GAIN = 25.0                 # blocking: full speed when this far from the block point, slower in proportion closer
BLOCK_STAY = 5.0                  # ...and no sideways move at all if the ball already comes within this of our centre
                                  # (full speed in, it overshot and the ball hit its edge - deflected into our goal)
DANGER_DIST = 35.0                # goalie: ball in our penalty area and this close -> clear it...
CARRIER_WEIGHT = 0.0              # goalie, an opponent has the ball: how far to move from the guard spot towards the
CARRIER_WIDEN = 10.0              # line it's facing along (if that line hits the goal +- this): 1 = all the way.
                                  # 0 = off: 0.6 did slightly worse in the simulator (a striker turns before it kicks,
                                  # so the goalie moving onto its current line just opens the other side)
GOALIE_FACE_MAX = math.radians(70)    # goalie guarding: face the ball, at most this far off up-field
GOALIE_CLEAR_PATH = 60.0          # goalie kicks: the ball's path must miss every robot for this far (cm) - it kicked
                                  # straight into their striker (clearance up to a robot radius into it was accepted)
GOALIE_KICK_TURN = math.radians(60)   # goalie with the ball (striker off): kicks it clear facing at most this far off
GOALIE_KICK_CLEAR = 50.0          # up-field, when no robot is in the way for this far (cm)
PASS_SIDE = 30.0                  # goalie with the ball, striker on: kicks it past the striker, aiming this far to one
PASS_AHEAD = 20.0                 # side of it and this far up-field of it (the more open side; both open: the one
PASS_TOL = 0.25                   # nearer the middle), once facing it within this (rad)
SOLO_CHASE_Y = 0.0                # goalie on its own (striker off): goes for the ball only in our half (y below this)
SOLO_CLOSER = 15.0                # ...or when it's this much closer to the ball than every opponent; else it guards
PUSHBACK_GAP = 6.0                # an opponent's centre within 2R + this of ours, with the ball between us (in front,
PUSHBACK_ANGLE = math.radians(50) # within this angle of the line to it) = a shoving match: shove back at full speed
GOALIE_STEP_UP = False            # goalie steps up when the ball is deep in their half (off: with the ball back in ~0.5 s
                                  # after going out there, it wasn't home in time - most goals against came that way)
STEP_UP_BALL_Y = 50.0             # goalie: ball further up-field than this and no opponent below STEP_UP_CLEAR_Y ->
STEP_UP_CLEAR_Y = -20.0           # step up to (ball x * STEP_UP_X_SHARE, STEP_UP_Y) behind the striker for rebounds /
STEP_UP_X_SHARE = 0.3             # clearances
STEP_UP_Y = -40.0
GOALIE_LOOK_AT_SPOT = False       # goalie, ball out: come up to look at the spot the striker can't see (off: with the
                                  # ball back in ~0.5 s the goalie wasn't home in time - conceded 2x as many)
LOOK_DIST = 55.0                  # ...standing this far from that spot
GOALIE_ALONE_ATTACKS = True       # striker off: the goalie plays the striker's logic
KICK_RETURN_T = 1.0               # goalie: after kicking / clearing, this long (s) before it goes for the ball again
GOALIE_MAX_OUT = 12.0             # goalie going for the ball: not further than this past the penalty area's front
PUSHBACK_REACH = 40.0             # (a close target means a slow robot, which just gets pushed back to our goal)
GOALIE_KICK_OUT = True            # goalie in a shoving match with the ball in its dribbler: kick it out over a side line
GOALIE_OUT_UP = 0.3               # (this much up-field per cm sideways), once facing within
GOALIE_KICK_OUT_TOL = 0.4         # this (rad) - an out ball comes back on the halfway line, far from our goal. The side
GOALIE_OUT_CLEAR = 60.0           # with no robot within this (cm) of the kick's path first (kick_out_side()), else the nearer
GOALIE_ORBIT_SPD_MIN = 0.5        # goalie going round the ball / lining up: at least this speed multiplier (the waypoints
                                  # are close and motion.py slows down near them; full speed overshot near our goal)
CHALLENGE_DIST = 35.0             # goalie: an opponent has the ball this close (cm) to our goal centre -> go and knock it
CHALLENGE_UP = 40.0               # out over the nearer side line (aim this far up-field of the ball, past the line):
                                  # standing guard just gets the goalie shoved back into its goal
PUSHER_DIST = 28.0                # goalie: an opponent has the ball but is pushing it at our goal (ball this close to the
PUSHER_SPEED = 10.0               # goal centre, rolling towards it faster than this): meet it - guarding just gets it pushed in
OPP_ON_BALL = 12.0                # an opponent "has" the ball: its centre within R + this of the ball
CLEAR_RETURN_Y = field.OWN_GOAL_Y + 55.0   # goalie: stop clearing once the ball is past this
CLEAR_AWAY_Y = field.OWN_GOAL_Y + 50.0     # goalie: ball closer to our goal line than this -> push it straight away
                                           # from the goal centre (so it stays between ball and goal), not at their goal
CLEAR_OUT_X = 15.0                         # clearing a ball this far to the side of our goal: push it out over the
                                           # side line (it comes back on the halfway line, far from our goal)
CONTEST_DIST = 8.0                # an opponent's edge within this of the ball = contested (the goalie doesn't try to block)
PUSH_MULTI = 1.0                  # speed multiplier when pushing a CONTESTED ball (full speed: win pushing contests;
                                  # a free ball is approached normally, or it gets knocked away instead of caught)
WAIT_POS = (0.0, -22.0)           # striker waiting for a lost ball: at least 60% speed while further than 30 cm from here
BALL_OUT_MARGIN = 1.0             # ball centre this far past the outer edge of the white line = (about to be) out
                                  # (the referee takes it off once it's fully past: BALL_RADIUS)
OUT_SOON_DIST = 15.0              # ball last seen this close to the line and rolling outwards faster than
OUT_SOON_SPEED = 0.0              # this (cm/s) = it went out
RESTART_WAIT = 4.0                # s the striker waits at the restart spot after the ball disappeared there
RESTART_SIDE = 0.0                # ...diagonally behind it, this far towards the middle (not head-on with their striker)
RESTART_X_SHARE = 1.0             # ...and this share of the way from the middle across to that spot (1 = right behind
                                  # it): the camera can't see a spot on the far side from behind the other side one, and
                                  # the referee often uses another spot than the nearest
NEUTRAL_WAIT = R + 11.0           # ball out: striker waits this far behind the neutral spot it'll come back to
                                  # (not on it: an occupied spot isn't used, the ball would go somewhere else)


# =====================================================================================================================
#                                                  HELPERS
# Small tools used by the brains further down. Each takes points / numbers and gives back an answer.
# Reminder: a point or an arrow is a pair (x, y). "unit arrow" = an arrow of length 1 (just a direction).
# =====================================================================================================================

# ---------------------------------------------- basic geometry -----------------------------------------------------

def unit(v):
    """The arrow v shortened (or stretched) to length 1 - just its direction. (0, 0) if v has no length."""
    length = math.hypot(v[0], v[1])
    if length < 1e-9:                      # (almost) zero: can't divide by it
        return (0.0, 0.0)
    return (v[0] / length, v[1] / length)


def heading_for(d):
    """Arrow d (field directions) -> the compass heading that faces along it."""
    return math.atan2(-d[0], d[1])


def clamp_in_field(p, margin=FIELD_MARGIN):
    """Point p moved back inside the field (at least `margin` cm inside the white line)."""
    half_x = field.PLAY_W / 2 - margin     # furthest the point may be from the centre, sideways
    half_y = field.PLAY_L / 2 - margin     # ...and up / down the field
    x = min(max(p[0], -half_x), half_x)
    y = min(max(p[1], -half_y), half_y)
    return [x, y]


def dist(a, b):
    """Distance (cm) between points a and b."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def seg_dist(p, a, b):
    """How close point p comes to the straight line from a to b.
    Returns (that distance, t): t = where along the line the closest point is (0 = at a, 1 = at b)."""
    ab = (b[0] - a[0], b[1] - a[1])                  # arrow a -> b
    length_squared = ab[0] ** 2 + ab[1] ** 2
    if length_squared < 1e-9:                        # a and b are the same point
        t = 0.0
    else:
        along = (p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]     # dot product: how far p is along a -> b
        t = max(0.0, min(1.0, along / length_squared))            # ...as a share of the line, kept 0..1
    closest = (a[0] + t * ab[0], a[1] + t * ab[1])
    return dist(p, closest), t


# ------------------------------------------ robots and the ball ----------------------------------------------------

def obstacle_centres(me, obstacles):
    """The camera stores each robot it sees as the point nearest to us (its front edge as we see it). Its centre is
    about one robot radius further away from us - this gives the list of centres."""
    centres = []
    for edge_point, size in obstacles:
        away = unit((edge_point[0] - me[0], edge_point[1] - me[1]))     # direction from us to it
        centres.append((edge_point[0] + away[0] * R, edge_point[1] + away[1] * R))
    return centres


# ----------------------------------------------- shooting ----------------------------------------------------------

def best_aim(ball, centres, max_range=None):
    """The point in their goal with the most room around the ball's path to it (only points within max_range).
    centres = robot centres. Returns (aim point, clearance = gap between the path and the nearest robot's edge);
    clearance is -1e9 if no point is in range."""
    half = field.GOAL_W / 2 - cfg.BALL_RADIUS - AIM_MARGIN    # how far left / right of the goal's centre we may aim
    y = field.ATTACK_GOAL_Y                                   # on the goal line: a ball through these clears the posts
    best, best_score, best_clear = (0.0, y), -1e9, -1e9
    for i in range(AIM_POINTS):                               # AIM_POINTS points spread evenly across the goal
        x = -half + 2 * half * i / (AIM_POINTS - 1)
        if max_range is not None and dist(ball, (x, y)) > max_range:
            continue                                          # too far: skip this point
        clear = clearance(ball, (x, y), centres)
        score = min(clear, 40.0) - 0.05 * abs(x)              # prefer room (up to 40 cm counts), then the middle
        if score > best_score:
            best, best_score, best_clear = (x, y), score, clear
    return best, best_clear


def clearance(a, b, centres):
    """Gap (cm) between the path a -> b and the nearest robot's edge (below 0 = a robot is in the way).
    100 if there are no robots."""
    gaps = [seg_dist(c, a, b)[0] - R for c in centres]
    return min(gaps, default=100.0)


def path_blocked(a, b, centres, ignore_near=None):
    """The first robot (centre) in the way of driving from a to b, or None.
    Robots near `ignore_near` don't count (usually the ball: an opponent on the ball isn't "in the way" of going
    for the ball). A robot we're already level with and not touching doesn't count either (the path's closest
    point to it is within 2R of where we are): counting it sent robots on detours round robots they were just
    passing."""
    worst = None                                     # (robot, how far along the path) of the first one in the way
    for c in centres:
        if ignore_near is not None and dist(c, ignore_near) < 15 + R:
            continue
        d, t = seg_dist(c, a, b)
        closest_point_is_right_here = t * dist(a, b) < 2 * R
        touching_it = dist(a, c) <= 2 * R + 2
        if closest_point_is_right_here and not touching_it:
            continue                                 # we're already past it
        in_the_way = d < AVOID_DIST and 0 < t < 1
        if in_the_way and (worst is None or t < worst[1]):
            worst = (c, t)
    if worst is None:
        return None
    return worst[0]


class Avoider:
    """Drives round robots in the way. Use it like a function: target = self.avoid(me, target, centres).
    If the straight path is blocked it tries a waypoint beside each robot (both sides), keeps the ones where BOTH
    legs (robot -> waypoint -> target) are clear, and takes the shortest. It keeps its waypoint unless another is
    clearly shorter (otherwise it re-decides every loop and shuffles left-right). If nothing is clear it goes round
    the first robot on the path."""
    STICK = 15.0       # cm: a new route must be this much shorter to switch to it

    def __init__(self):
        self.wp = None           # the waypoint we're going via (None = going straight)
        self.obstacle = None     # the robot we're going round (fallback())
        self.side = 1.0          # ...on which side

    def __call__(self, robot, target, centres, ignore_near=None, back_cost=0.0):
        """back_cost: extra cost per cm a waypoint is behind the robot (towards our goal) - used when carrying."""
        # straight path clear: just go
        if path_blocked(robot, target, centres, ignore_near) is None:
            self.wp = None
            self.obstacle = None
            return target

        # try a waypoint each side of each robot, keep the cheapest one with both legs clear
        used = [c for c in centres if ignore_near is None or dist(c, ignore_near) >= 15 + R]
        best, best_cost = None, None
        for c in used:
            candidates = (self.beside(c, target, 1.0), self.beside(c, target, -1.0),
                          self.beside_seen(c, robot, 1.0), self.beside_seen(c, robot, -1.0))
            for wp in candidates:
                if wp is None or not field.inside_play_area(wp[0], wp[1], OUT_MARGIN):
                    continue                             # no such point / outside the field
                if path_blocked(robot, wp, used) is not None or path_blocked(wp, target, used) is not None:
                    continue                             # a leg is blocked
                cost = dist(robot, wp) + dist(wp, target) + back_cost * max(0.0, robot[1] - wp[1])
                if self.wp is not None and dist(wp, self.wp) < 10:
                    cost -= self.STICK                   # this is (about) the route we're already on
                if best_cost is None or cost < best_cost:
                    best, best_cost = wp, cost
        if best is not None:
            self.wp = best
            return clamp_in_field(best, OUT_MARGIN)

        # nothing fully clear: go round the first robot on the path
        self.wp = None
        return self.fallback(robot, target, path_blocked(robot, target, centres, ignore_near))

    @staticmethod
    def beside(c, target, side):
        """Point to the side of robot c, far enough out that the line from there to the target misses it too
        (a point just AVOID_DIST to the side would still clip it when the target is close behind it)."""
        u = unit((target[0] - c[0], target[1] - c[1]))       # direction robot -> target
        n = (-u[1], u[0])                                    # ...turned 90 degrees: sideways
        L = dist(c, target)
        need = AVOID_DIST + 3
        if L <= need + 1:
            return None                                      # the target is right next to the robot
        off = min(need * L / math.sqrt(L * L - need * need), 80.0)
        return [c[0] + n[0] * side * off, c[1] + n[1] * side * off]

    @staticmethod
    def beside_seen(c, robot, side):
        """Point to the side of robot c as seen from our robot (straight across our line of sight)."""
        u = unit((c[0] - robot[0], c[1] - robot[1]))
        off = AVOID_DIST + 3
        return [c[0] - u[1] * side * off, c[1] + u[0] * side * off]

    def fallback(self, robot, target, c):
        """Nothing fully clear: go round the first robot on the path (c), keeping the side we picked."""
        u = unit((target[0] - c[0], target[1] - c[1]))
        n = (-u[1], u[0])
        if self.obstacle is None or dist(c, self.obstacle) > 20:
            # a new robot to go round: pick the side we're already on
            if (robot[0] - c[0]) * n[0] + (robot[1] - c[1]) * n[1] >= 0:
                self.side = 1.0
            else:
                self.side = -1.0
        self.obstacle = c
        wp = self.beside(c, target, self.side) or [c[0] + n[0] * self.side * (AVOID_DIST + 3),
                                                   c[1] + n[1] * self.side * (AVOID_DIST + 3)]
        if not field.inside_play_area(wp[0], wp[1], OUT_MARGIN):     # no room that side: go round the other way
            self.side = -self.side
            wp = [2 * c[0] - wp[0], 2 * c[1] - wp[1]]
        return clamp_in_field(wp, OUT_MARGIN)


# ------------------------------------- going for the ball (get behind it, push) ------------------------------------

def choose_push_aim(ball, goal_aim, defending=False):
    """Which way to push the ball. Normally at goal_aim, but if getting behind it for that would mean standing past
    the white line (ball hugging the line), push it somewhere reachable instead: straight up-field, or inwards.
    defending: the goalie clearing near our goal - the other choices are sideways, away from the goal mouth."""
    if defending:
        side = math.copysign(1.0, ball[0] or 1.0)              # the side of the field the ball is on
        candidates = [goal_aim, (side * field.PLAY_W / 2, ball[1] + 40.0), (side * field.PLAY_W / 2, ball[1])]
    else:
        candidates = [goal_aim, (ball[0] * 0.5, field.ATTACK_GOAL_Y), (0.0, field.ATTACK_GOAL_Y), (0.0, 0.0)]
    best, best_cost = goal_aim, None
    for k, a in enumerate(candidates):                         # k = 0 for the first choice, 1, 2, ... after
        u = unit((a[0] - ball[0], a[1] - ball[1]))
        behind = (ball[0] - u[0] * BEHIND_DIST, ball[1] - u[1] * BEHIND_DIST)   # where we'd stand to push it there
        # cost: how far that spot is outside the field (can't stand there), then a little for each later choice
        cost = 2.0 * dist(behind, clamp_in_field(behind, OUT_MARGIN)) + 8.0 * k
        if best_cost is None or cost < best_cost:
            best, best_cost = a, cost
    return best


def approach(robot, ball, aim):
    """Where to go to push the ball towards aim. Returns (target, heading, phase), field coordinates.
    phase: "orbit" (wrong side, going round), "line up" (behind it, not lined up), "push"."""
    u = unit((aim[0] - ball[0], aim[1] - ball[1]))     # direction ball -> aim
    n = (-u[1], u[0])                                  # ...sideways
    r = (robot[0] - ball[0], robot[1] - ball[1])       # arrow ball -> robot
    along = r[0] * u[0] + r[1] * u[1]                  # how far in front of the ball we are (< 0 = behind it, good)
    perp = r[0] * n[0] + r[1] * n[1]                   # how far off to the side
    heading = heading_for(u)                           # face where the ball should go
    behind = [ball[0] - u[0] * BEHIND_DIST, ball[1] - u[1] * BEHIND_DIST]    # the line-up point
    lined_up_tol = ALIGN_TOL + ALIGN_TOL_PER_CM * (-along)

    # going straight to the line-up point would clip the ball (and maybe knock it the wrong way): go round instead
    clips = abs(perp) > lined_up_tol and seg_dist(ball, robot, behind)[0] < R + cfg.BALL_RADIUS + 3
    if along > -R * 0.6 or clips:
        # "orbit": to a point beside the ball (our side of it first, the other side if that's outside the field)
        side = 1.0 if perp >= 0 else -1.0
        for s in (side, -side):
            wp = [ball[0] - u[0] * ORBIT_DIST * 0.3 + n[0] * s * ORBIT_DIST,
                  ball[1] - u[1] * ORBIT_DIST * 0.3 + n[1] * s * ORBIT_DIST]
            if field.inside_play_area(wp[0], wp[1], OUT_MARGIN):
                break
        return clamp_in_field(wp, OUT_MARGIN), heading, "orbit"
    if abs(perp) > lined_up_tol:
        return clamp_in_field(behind, OUT_MARGIN), heading, "line up"
    # lined up: drive through the ball, aiming PUSH_DIST past it
    return [ball[0] + u[0] * PUSH_DIST, ball[1] + u[1] * PUSH_DIST], heading, "push"


def around_ball(me, target, ball):
    """If driving straight from me to target would hit the ball, go via a point beside it instead (the side that's
    shorter and inside the field). Used whenever the robot isn't meant to touch the ball: guarding, going round it,
    lining up - knocking it on the way there was putting it in our own goal."""
    clear = R + cfg.BALL_RADIUS + 3                    # centre-to-centre distance that just misses the ball
    d, t = seg_dist(ball, me, target)
    if d >= clear or t <= 0.0 or t >= 1.0 or dist(me, ball) < clear:
        return target                                  # the ball isn't in the way (or we're already at it)
    u = unit((target[0] - me[0], target[1] - me[1]))
    best = None                                        # (cost, waypoint)
    for side in (1.0, -1.0):
        wp = [ball[0] - u[1] * side * (clear + 3), ball[1] + u[0] * side * (clear + 3)]
        if not field.inside_play_area(wp[0], wp[1], OUT_MARGIN):
            continue
        cost = dist(me, wp) + dist(wp, target)
        if best is None or cost < best[0]:
            best = (cost, wp)
    if best is None:
        return target
    return best[1]


def clear_aim(ball):
    """Where to push a ball near our goal: away from the goal centre, half way to straight up-field (up-field rather
    than just sideways, so it actually leaves our end)."""
    if abs(ball[0]) > CLEAR_OUT_X:    # out to the side: an out ball goes back to the halfway line (field.NEUTRAL_SPOTS)
        return (math.copysign(field.PLAY_W / 2 + 30.0, ball[0]), ball[1] + 15.0)
    away = unit((ball[0], ball[1] - field.OWN_GOAL_Y))     # direction goal centre -> ball
    away = unit((away[0], away[1] + 1.0))                  # ...turned half way to up-field
    return (ball[0] + away[0] * 100.0, ball[1] + away[1] * 100.0)


def lead_ball(me, ball, vel):
    """Where the ball will be when we get there (roughly): it keeps rolling at vel (cm/s)."""
    t = min(dist(me, ball) / LEAD_SPEED, LEAD_TIME_MAX)    # seconds until we get there (at most LEAD_TIME_MAX)
    return clamp_in_field([ball[0] + vel[0] * t, ball[1] + vel[1] * t], -cfg.BALL_RADIUS)


def catch_speed(me, ball):
    """Speed multiplier for running onto a free ball: slow right at it, faster further away."""
    gap = max(0.0, dist(me, ball) - R)                     # cm between our bumper and the ball
    return min(1.0, CATCH_MULTI_MIN + CATCH_MULTI_PER_CM * gap)


def guard_speed(me, target):
    """Speed settings for the goalie's guarding moves: in proportion to how far off its spot it is, and reacting
    faster than the default smoothing (GUARD_SMOOTH)."""
    multi = min(1.0, max(0.1, dist(me, target) / GUARD_GAIN))
    return {"spd_multi": multi, "smooth": GUARD_SMOOTH}


def angle_diff(a, b):
    """Difference between two angles (radians), wrapped to -pi..pi (= your angdiff)."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


def contested(ball, centres):
    """An opponent is right at the ball."""
    return any(dist(c, ball) < CONTEST_DIST + R for c in centres)


# ------------------------------------------------ ball out ---------------------------------------------------------

def ball_out(ball):
    """Ball clearly outside the white line (the goal notches count as in)."""
    return not field.inside_play_area(ball[0], ball[1], -BALL_OUT_MARGIN)


def going_out(ball, vel):
    """The ball is out, or right at the line and rolling out."""
    if ball_out(ball):
        return True
    if field.inside_play_area(ball[0], ball[1], OUT_SOON_DIST):
        return False                               # not near the line
    # which line it's near: a side line (left / right) or an end line - n = arrow pointing out over that line
    if abs(ball[0]) > field.PLAY_W / 2 - OUT_SOON_DIST:
        n = (math.copysign(1.0, ball[0]), 0.0)
    else:
        n = (0.0, math.copysign(1.0, ball[1]))
    speed_outwards = vel[0] * n[0] + vel[1] * n[1]
    return speed_outwards > OUT_SOON_SPEED


def restart_spot(ball):
    """The neutral spot the referee will put an out ball on: the one nearest to it."""
    return min(field.NEUTRAL_SPOTS, key=lambda spot: dist(spot, ball))


# ----------------------------------------- shoving, possession, kicks ----------------------------------------------

def shover(me, ball, centres):
    """The opponent shoving against us over the ball (touching, ball in between, not pushing it towards our own
    goal), or None."""
    if dist(me, ball) > R + cfg.BALL_RADIUS + 6:
        return None                                # the ball isn't at our front
    to_ball = math.atan2(ball[1] - me[1], ball[0] - me[0])
    for c in centres:
        if dist(c, me) > 2 * R + PUSHBACK_GAP:
            continue                               # not touching us
        to_it = math.atan2(c[1] - me[1], c[0] - me[0])
        same_side_as_ball = abs(angle_diff(to_it, to_ball)) < PUSHBACK_ANGLE
        not_behind_us = c[1] - me[1] > -0.3 * dist(c, me)
        if same_side_as_ball and not_behind_us:
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
    if abs(ball[0]) < 10:
        side = math.copysign(1.0, ball[0] - me[0])
    else:
        side = math.copysign(1.0, ball[0])
    return (side * (field.PLAY_W / 2 + 30.0), ball[1] + CHALLENGE_UP)


def kick_out_side(me, ball, centres, current=None, shover=None):
    """Goalie kicking the ball out: over which side line (+1 = right / -1 = left). The side with no robot in the
    way of the kick (path clear for GOALIE_OUT_CLEAR cm); if both or neither are clear, the side away from the
    robot shoving us (it was kicked straight into it), else the nearer one. Keeps `current` while it's clear
    (re-deciding every loop as the robots move would turn it back and forth)."""
    def clear(side):
        u = unit((side, GOALIE_OUT_UP))
        kick_end = (me[0] + u[0] * GOALIE_OUT_CLEAR, me[1] + u[1] * GOALIE_OUT_CLEAR)
        return clearance(me, kick_end, centres) > 0
    if current is not None and clear(current):
        return current
    if shover is not None and abs(shover[0] - me[0]) > 2.0:
        near = -math.copysign(1.0, shover[0] - me[0])      # away from the robot shoving us
    else:
        near = math.copysign(1.0, ball[0] or me[0] or 1.0)
    if clear(near) or not clear(-near):
        return near
    return -near


def pass_aim(me, striker, centres):
    """Goalie clearing to the striker: a point beside the striker (PASS_SIDE to one side, PASS_AHEAD up-field), so
    the ball rolls past it up-field and it can take it facing the way it's going (instead of turning round). The side
    with the clear path; both clear: the one nearer the middle; both blocked: the one with more room."""
    others = [c for c in centres if dist(c, striker) > R]      # every robot except our striker
    cands = []                                                 # (clearance, point) for each side
    for side in (1.0, -1.0):
        p = clamp_in_field([striker[0] + side * PASS_SIDE, striker[1] + PASS_AHEAD])
        cands.append((clearance(me, p, others), p))
    open_ = [c for c in cands if c[0] > KICK_CLEARANCE]
    if len(open_) == 1:
        return open_[0][1]
    if open_:
        return min(open_, key=lambda c: abs(c[1][0]))[1]       # both open: nearer the middle
    return max(cands, key=lambda c: c[0])[1]                   # neither: more room


# ------------------------------------------------- misc ------------------------------------------------------------

def rel(p, pose):
    """Field point p -> the arrow from the robot to it (what update() has to return)."""
    return [p[0] - pose.x, p[1] - pose.y]


def mate_active(mate):
    """The teammate is on (it sends "bot active": 1)."""
    return bool(mate) and mate.get("bot active") == 1


# =====================================================================================================================
#                                                   GOALIE
# =====================================================================================================================
class GoalieStrategy:
    """botstate 0 = no ball known    -> guard the middle of the goal
       botstate 1 = ball in dribbler -> turn up-field and kick it clear (only with a dribbler)
       botstate 2 = clearing         -> go for the ball
       botstate 3 = guarding         -> on the arc between the ball and the goal centre
       botstate 4 = ball out         -> guard as if the ball were already on the neutral spot it'll come back to
       (fallback = strategy_fallback.GoalieFallback while not localised: its botstates are shown + 10)
       Striker off: plays StrikerStrategy's logic instead (GOALIE_ALONE_ATTACKS)."""

    def __init__(self, has_dribbler=True):
        self.has_dribbler = has_dribbler     # without one, a ball at the front is pushed (botstate 2), not carried
        self.fallback = GoalieFallback()
        # your Hysteresis (utils.py): a new botstate must hold 0.11 s before it's used - except 1 (ball in the
        # dribbler), which switches at once
        self.botstate_hyst = Hysteresis(hold_time=0.11, instant_enter=lambda v: v == 1)
        self.botstate = 0
        self.speed = {}
        self.kick = False
        self.chasing = False
        self.phase = ""
        self.possession = None   # "us" / "them" / None: who has the ball (possession())
        self.out_side = None     # kicking the ball out: over which side line (kick_out_side())
        self.kicked_at = -1e9    # when it last kicked (KICK_RETURN_T)
        self.out_ball = None     # (ball position, time) it was last seen going out
        self.alone = None        # striker off: the StrikerStrategy the goalie plays (GOALIE_ALONE_ATTACKS)
        self.avoid = Avoider()

    def reset(self):
        self.fallback.reset()
        self.botstate_hyst.reset()

    def update(self, world, compass, mate, line_touches):
        pose = world.pose
        self.kick = False
        self.possession = possession(world, mate)

        # ---- not sure where we are on the field: the fallback (relative) logic decides
        if pose is None or not pose.confident:
            command = mate.get("command") if mate else None
            mate_on = mate.get("bot active") if mate else None
            out = self.fallback.update(world, compass, command, mate_on, line_touches)
            self.botstate = 10 + self.fallback.botstate
            self.speed = self.fallback.speed
            self.chasing = self.fallback.chasing
            self.kick = getattr(self.fallback, "kick", False)
            self.phase = "fallback"
            return out

        me = (pose.x, pose.y)
        ball = world.ball_field                  # None = no ball seen
        goal = (0.0, field.OWN_GOAL_Y)           # centre of our goal
        striker_on = mate_active(mate)

        # ---- striker off: play the striker's logic (it holds its own against two opponents; guarding the goal on
        # our own just got overrun)
        if not striker_on and GOALIE_ALONE_ATTACKS:
            if self.alone is None:
                self.alone = StrikerStrategy()
            out = self.alone.update(world, compass, mate, line_touches)
            self.kick = self.alone.kick
            self.speed = self.alone.speed
            self.chasing = self.alone.chasing
            self.botstate = self.alone.botstate
            self.command = self.alone.command
            self.phase = "alone: " + self.alone.phase
            return out
        self.alone = None

        # ---- remember a ball going out (it comes back on a neutral spot: "look at spot" below)
        if ball is not None:
            if going_out(ball, world.ball_vel):
                self.out_ball = (list(ball), time.monotonic())
            elif not ball_out(ball):
                self.out_ball = None

        # ---- DECIDE the botstate (raw = what it would be this loop, before the Hysteresis smoothing)
        if ball is None:
            raw = 0
        elif ball_out(ball):
            raw = 4
        elif world.ball_in_capture and self.has_dribbler and self.carrier(ball, world.opponents) is None:
            raw = 1     # (an opponent pushing the ball into our front isn't "ours": keep guarding, see below)
        else:
            in_box = abs(ball[0]) < field.PENALTY_W / 2 and ball[1] < field.OWN_GOAL_Y + field.PENALTY_D
            told = striker_on and mate.get("command") == 1         # the striker told us to take it
            clearing = self.botstate in (1, 2)                      # we were already going for it
            opp_on_ball = self.possession == "them"
            pushed_at_goal = dist(ball, goal) < PUSHER_DIST and world.ball_vel[1] < -PUSHER_SPEED
            striker_chasing_it = striker_on and mate.get("command") == 0 and mate.get("chasing")
            just_kicked = time.monotonic() - self.kicked_at <= KICK_RETURN_T
            if not striker_on:
                # alone: clear what's in our half or clearly ours, otherwise mind the goal
                closest_opp = min((dist(c, ball) for c in world.opponents), default=1e9)
                if ball[1] < SOLO_CHASE_Y or dist(me, ball) + SOLO_CLOSER < closest_opp:
                    raw = 2
                else:
                    raw = 3
            elif opp_on_ball and dist(ball, goal) < CHALLENGE_DIST:
                raw = 2     # an opponent is bringing it at our goal: go and knock it out over the side line
            elif opp_on_ball and not pushed_at_goal:
                raw = 3     # an opponent has it: cover the goal, don't challenge (in the simulator most goals against
                            # came from an opponent shooting past a goalie that had come out for the ball)
            elif told or (in_box and dist(me, ball) < DANGER_DIST):
                raw = 2
            elif clearing and ball[1] < CLEAR_RETURN_Y and not striker_chasing_it and not just_kicked:
                raw = 2     # keep clearing until it's up-field
            else:
                raw = 3
        too_far_out = me[1] > field.OWN_GOAL_Y + field.PENALTY_D + GOALIE_MAX_OUT
        if raw == 2 and striker_on and too_far_out and not told:
            raw = 3     # it chased a cleared ball and left the goal empty: back to guarding
        self.botstate = self.botstate_hyst.update(raw)
        if ball is None and self.botstate != 0:
            self.botstate = 0
        self.chasing = self.botstate in (1, 2)

        # ---- an opponent shoving the ball against the goalie: shove straight back (guarding = crawling = getting
        # pushed into our goal), or with the ball in the dribbler, kick it out over the side line
        centres = obstacle_centres(me, world.obstacles_field)
        if ball is not None:
            sh = shover(me, ball, centres)
        else:
            sh = None
        if sh is None:
            self.out_side = None
        else:
            if GOALIE_KICK_OUT and world.ball_in_capture and self.has_dribbler:
                other_robots = [c for c in centres if c is not sh]
                self.out_side = kick_out_side(me, ball, other_robots, self.out_side, shover=sh)
                heading = heading_for((self.out_side, GOALIE_OUT_UP))
                err = abs(angle_diff(heading, compass))
                self.kick = err < GOALIE_KICK_OUT_TOL
                self.speed = {}
                self.phase = "kick out"
                return rel(list(me), pose), heading, True
            d = unit((sh[0] - me[0], sh[1] - me[1]))     # (pushing off sideways towards the line opened the goal)
            target = clamp_in_field([me[0] + d[0] * PUSHBACK_REACH, me[1] + d[1] * PUSHBACK_REACH], OUT_MARGIN)
            self.speed = {"spd_multi": 1.0}
            self.phase = "push back"
            return rel(target, pose), 0.0, False

        # ---- ACT
        dribbler = False
        self.speed = {}

        # a free ball (a shot, or rolling) heading into our goal: get in its way (an opponent dribbling it: guard)
        block = None
        if ball is not None and self.botstate != 1 and not contested(ball, obstacle_centres(me, world.obstacles_field)):
            block = self.block_point(ball, world.ball_vel)
        if block is not None:
            target = block
            if abs(block[0] - me[0]) < BLOCK_STAY:
                target = [me[0], block[1]]         # already in its way: don't slide past it
            face = heading_for((ball[0] - me[0], ball[1] - me[1]))
            heading = max(-GOALIE_FACE_MAX, min(GOALIE_FACE_MAX, face))   # facing the incoming ball
            self.phase = "block"
            self.speed = {"spd_multi": min(1.0, max(0.1, dist(me, target) / BLOCK_GAIN)), "smooth": GUARD_SMOOTH}
            return rel(target, pose), heading, False

        # ball out: come up just far enough to see the neutral spot the striker isn't waiting at (from right behind
        # one side spot its camera can't see the other one), facing it - it tells the striker where the ball is
        ball_is_out = ball is None or ball_out(ball)
        if (GOALIE_LOOK_AT_SPOT and self.out_ball is not None and striker_on and ball_is_out
                and time.monotonic() - self.out_ball[1] < RESTART_WAIT):
            p = restart_spot(self.out_ball[0])                     # where the striker waits
            others = [q for q in field.NEUTRAL_SPOTS if dist(q, p) > 1]
            if abs(p[0]) > 1:
                q = max(others, key=lambda q: dist(q, p))          # side spot: look at the far side spot
            else:
                q = min(others, key=lambda q: abs(q[0] - me[0]))   # centre spot: look at the side spot on our side
            u = unit((0.0 - q[0], field.OWN_GOAL_Y - q[1]))        # direction from that spot to our goal
            target = [q[0] + u[0] * LOOK_DIST, q[1] + u[1] * LOOK_DIST]
            heading = heading_for((q[0] - target[0], q[1] - target[1]))
            self.speed = {"spd_multi": 1.0}
            self.phase = "look at spot"
            return rel(target, pose), heading, False

        nobody_back_here = all(o[1] > STEP_UP_CLEAR_Y for o in world.opponents)
        if self.botstate == 0:
            target, heading = (0.0, goal[1] + GUARD_RADIUS), 0.0
            self.phase = "guard middle"
            self.speed = guard_speed(me, target)
        elif GOALIE_STEP_UP and self.botstate == 3 and striker_on and ball[1] > STEP_UP_BALL_Y and nobody_back_here:
            # ball deep in their half and no opponent back here: step up behind the striker for rebounds / clearances
            target, heading = [ball[0] * STEP_UP_X_SHARE, STEP_UP_Y], 0.0
            self.speed = {"spd_multi": 1.0}
            self.phase = "step up"
        elif self.botstate in (3, 4):
            # guard: on the arc round our goal, between it and the ball
            if self.botstate == 3:
                # guard where the ball will be in a moment, not where it was
                b = (ball[0] + world.ball_vel[0] * GUARD_LEAD, ball[1] + world.ball_vel[1] * GUARD_LEAD)
            else:
                b = restart_spot(ball)        # ball out: guard against where it'll come back
            target, heading = self.guard(me, b)
            carrier = None
            if self.botstate == 3 and CARRIER_WEIGHT:      # (CARRIER_WEIGHT = 0: switched off)
                carrier = self.carrier(ball, world.opponents)
            if carrier is not None:         # an opponent has it: lean towards the line it would shoot along
                target = self.shot_line_guard(target, b, carrier)
            self.speed = guard_speed(me, target)
            if self.botstate == 3:
                target = around_ball(me, target, ball)
                self.phase = "guard carrier" if carrier is not None else "guard"
            else:
                self.phase = "ball out"
        else:
            # botstate 1 / 2: the ball is ours to take
            centres = obstacle_centres(me, world.obstacles_field)
            aim, clear = best_aim(ball, centres)
            if self.botstate == 1:
                target, heading = self.clear_ball(world, compass, mate, me, aim, centres, striker_on)
                dribbler = True
            else:
                target, heading, dribbler = self.take_ball(world, me, ball, aim, centres)
        return rel(target, pose), heading, dribbler

    def clear_ball(self, world, compass, mate, me, aim, centres, striker_on):
        """Ball in the dribbler: stay put, turn up-field and kick it clear. Returns (target, heading); sets .kick."""
        target = list(me)
        striker = mate.get("pos") if striker_on else None
        fwd = (-math.sin(compass), math.cos(compass))                 # the way we face now
        pressed = any(dist(o, me) < 2 * R + 20 for o in world.opponents)
        if pressed:
            # an opponent closing in: no lining up a pass - turn to the open up-field side (away from the nearest
            # opponent first) and clear it the first moment the way is really clear
            nearest = min(world.opponents, key=lambda o: dist(o, me), default=None)
            if nearest is not None and abs(nearest[0] - me[0]) > 2.0:
                side = -math.copysign(1.0, nearest[0] - me[0])
            else:
                side = math.copysign(1.0, me[0] or 1.0)

            def open_way(sd):
                u = unit((sd * 0.6, 1.0))
                return clearance(me, (me[0] + u[0] * GOALIE_CLEAR_PATH, me[1] + u[1] * GOALIE_CLEAR_PATH), centres) > 0
            if not open_way(side) and open_way(-side):
                side = -side
            heading = heading_for((side * 0.6, 1.0))
            ahead = (me[0] + fwd[0] * GOALIE_CLEAR_PATH, me[1] + fwd[1] * GOALIE_CLEAR_PATH)
            self.kick = fwd[1] > 0.3 and clearance(me, ahead, centres) > 0
            self.phase = "clear now"
        elif striker is not None and striker[1] > me[1] + R:
            # pass it past the striker (up-field of us): to its more open side, kick when facing it
            p = pass_aim(me, striker, centres)
            heading = heading_for((p[0] - me[0], p[1] - me[1]))
            err = abs(angle_diff(heading, compass))
            others = [c for c in centres if dist(c, striker) > R]          # (our striker is the one we pass to)
            ahead = (me[0] + fwd[0] * GOALIE_CLEAR_PATH, me[1] + fwd[1] * GOALIE_CLEAR_PATH)
            self.kick = err < PASS_TOL and clearance(me, ahead, others) > 0
            self.phase = "clear pass"
        else:
            # turn towards the open part of their goal (at most GOALIE_KICK_TURN off straight up-field), kick as
            # soon as we face up-field with room
            want = heading_for((aim[0] - me[0], aim[1] - me[1]))
            heading = max(-GOALIE_KICK_TURN, min(GOALIE_KICK_TURN, want))
            ahead = (me[0] + fwd[0] * GOALIE_KICK_CLEAR, me[1] + fwd[1] * GOALIE_KICK_CLEAR)
            self.kick = abs(compass) < GOALIE_KICK_TURN + 0.1 and clearance(me, ahead, centres) > 0
            self.phase = "clear kick"
        return target, heading

    def take_ball(self, world, me, ball, aim, centres):
        """Botstate 2: go for the ball. Returns (target, heading, dribbler)."""
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
            self.phase = "guard (ball behind)"
            dribbler = False
        return target, heading, dribbler

    def note_kick(self):
        """Called after each loop (main.py): remember when we kicked (KICK_RETURN_T)."""
        if self.kick:
            self.kicked_at = time.monotonic()

    @staticmethod
    def carrier(ball, opponents):
        """The opponent (centre) that has the ball - the closest one, if its centre is right behind the ball - or None.
        Uses both robots' cameras (world.opponents merges the teammate's "opps")."""
        closest = min(opponents, key=lambda o: dist(o, ball), default=None)
        if closest is not None and dist(closest, ball) < R + OPP_ON_BALL:
            return closest
        return None

    @staticmethod
    def shot_line_guard(target, ball, carrier):
        """(Off: CARRIER_WEIGHT = 0.) Shift the guard spot towards the line carrier -> ball (the ball sits in front of
        the robot holding it, so that's where a kick would go), if that line ends up in our goal."""
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
            return None                                       # not rolling towards our goal (fast enough)
        y_line = field.OWN_GOAL_Y + R + 3                     # the goalie's line
        if ball[1] < y_line:
            return None                                       # already past it
        t_goal = (field.OWN_GOAL_Y - ball[1]) / vel[1]        # seconds until it crosses the goal line...
        x_goal = ball[0] + vel[0] * t_goal                    # ...and where
        if t_goal > BLOCK_TIME or abs(x_goal) > field.GOAL_W / 2 + BLOCK_WIDEN:
            return None                                       # too late / it'll miss the goal
        t = (y_line - ball[1]) / vel[1]                       # stand where it crosses the goalie's line
        x = ball[0] + vel[0] * t
        lim = field.GOAL_W / 2 + R
        return [max(-lim, min(lim, x)), y_line]

    @staticmethod
    def guard(me, b):
        """Guard position for a ball at b: on the arc round our goal centre, between ball and goal. Returns
        (target, heading). (Tried instead: the furthest-back spot on the bisector that still covers the whole goal -
        the striker drill scored more against that, the goalie is never quite on such a deep spot.)"""
        goal = (0.0, field.OWN_GOAL_Y)
        v = unit((b[0] - goal[0], b[1] - goal[1]))                     # direction goal -> ball
        x = goal[0] + v[0] * GUARD_RADIUS
        y = max(goal[1] + v[1] * GUARD_RADIUS, goal[1] + R + 3)        # not behind the goal line
        lim = field.PENALTY_W / 2 - R
        x = max(-lim, min(lim, x))                                     # inside the penalty area
        # face the ball (at most GOALIE_FACE_MAX off up-field - never towards our own goal): ready to take it and
        # kick it out / up-field
        face = heading_for((b[0] - me[0], b[1] - me[1]))
        return [x, y], max(-GOALIE_FACE_MAX, min(GOALIE_FACE_MAX, face))

    def go_for_ball(self, me, ball, vel, aim, centres, defending=False):
        """Get behind the ball and push it towards aim (the goalie). Returns (target, heading,
        dribbler on); sets .phase and .speed."""
        b = lead_ball(me, ball, vel)                       # where the ball will be
        target, heading, self.phase = approach(me, b, choose_push_aim(b, aim, defending))
        if self.phase != "push":
            target = around_ball(me, target, b)            # not pushing yet: don't bump it on the way
        if self.phase in ("orbit", "line up"):
            self.speed = {"spd_min": GOALIE_ORBIT_SPD_MIN}
        if self.phase == "push":
            race = any(dist(c, ball) < dist(me, ball) + RACE_MARGIN for c in centres)   # an opponent could get there first
            self.speed = {"spd_multi": PUSH_MULTI if race else catch_speed(me, ball)}
        # go round robots in the way - when pushing, the target is a point past the ball: only the way to the ball
        # itself must be clear
        nav = b if self.phase == "push" else target
        wp = self.avoid(me, nav, centres, ignore_near=b)
        if wp is not nav:
            target = wp                                    # the Avoider gave a waypoint
        return target, heading, dist(me, ball) < DRIBBLER_RANGE

# =====================================================================================================================
#                                            STRIKER (attacker)
# Built by playing ports of other teams' strategies in the simulator (past_versions/ts, ba, lx, tn): going for the
# ball the way team TS did (fast, straight at it - careful lining up lost every race), carrying it the way Bodensee
# Adler did (straight at the goal, kick once facing it). While the ball is out of play / not seen it waits for the
# restart or waits behind a neutral spot (restarts decide many games: the ball goes out ~20 times a game).
# =====================================================================================================================

CHASE_EXAGGERATE = 1.7      # chase: aim this much further round the ball than its angle, at the closest (90..20 cm)
CHASE_FACE_DIST = 55.0      # chase: turn to face the ball when it's this close...
CHASE_FACE_ANGLE = 60.0     # ...and within this angle (degrees) of up-field
CARRY_FACING = math.radians(10)   # carrying: facing the goal within this...
CARRY_KICK_HOLD = 0.2       # ...for this long (s) -> kick...
CARRY_KICK_RANGE = 70.0     # ...if this close to it (from further the goalie saved it)
QUICK_SHOT_TIME = 1.0       # just won the ball: shoot within this long (s) if the shot is open...
QUICK_SHOT_RANGE = 130.0    # ...from up to this far (before they close us down)
CHARGED_DIST = 50.0         # an opponent this close coming at us faster than CHARGED_SPEED (cm/s) = about to lose
CHARGED_SPEED = 30.0        # the ball: shoot now if the shot is open (OPEN_SHOT_RANGE)
OPEN_SHOT_RANGE = 120.0
HEAD_ON_GAP = 8.0           # an opponent's centre within 2R + this, within HEAD_ON_ANGLE of straight ahead = pressed
HEAD_ON_ANGLE = math.radians(40)   # against the ball head-on (kickoffs): step past it (SLIP_SIDE across,
SLIP_SIDE = 25.0                   # SLIP_FORWARD forwards) - driving into it got us shoved back to our goal
SLIP_FORWARD = 15.0
BOX_KEEP = R + 2.0          # the striker keeps this far out of our penalty area while our goalie is on (both of us in it
                            # with the ball = multiple defence: the goalie gets moved out and the goal is empty)
CARRY_FRONT_CONE = math.radians(60)   # carrying: no opponent within CARRY_CONTACT_RANGE (allowing for
CARRY_CONTACT_RANGE = 60.0            # CARRY_LEAD s of its movement) inside this angle either side of our front
CARRY_LEAD = 0.3                      # (the capture zone): a bump there knocks the ball off us, on the side or
CARRY_TURN_MAX = math.radians(120)    # back it doesn't. Turn up to CARRY_TURN_MAX off the goal for that,
CARRY_TURN_STEP = math.radians(10)    # trying every CARRY_TURN_STEP.
CARRY_CLOSING = 20.0                  # only opponents coming at us faster than this (cm/s)
RESTART_SCREEN = True       # restart ball behind us (we're up-field of it) and their nearest robot within SCREEN_OPP
SCREEN_TIME = 3.0           # cm of it, within SCREEN_TIME s of the ball going out: face up-field and go round the ball
SCREEN_OPP = 60.0           # on that robot's side (bodying it) to get behind the ball - until it's further than
                            # SCREEN_OPP from the ball or the ball is in front of us
RESTART_CATCH = True        # restart orbit: within CATCH_RESTART_TIME s of the ball going out, it's back on a spot,
CATCH_RESTART_TIME = 1.5    # we're up-field of it and an opponent is within CATCH_RESTART_OPP cm of it -> full speed
CATCH_RESTART_OPP = 70.0    # round it on our side to get behind it, facing up-field (never turning round)
RESTART_RUSH = False        # (off: with 0.5 s restarts it cost us - TS 66 -> 73 wins of 88 without it) restart race: the ball has just been put back on a neutral spot after going out ->
RUSH_TIME = 4.0             # (within this many s of it being out, and within RUSH_SPOT cm of a spot)
RUSH_SPOT = 8.0             # ...if we're behind it (within RUSH_BEHIND_X cm sideways) go in at full speed instead of
RUSH_BEHIND_X = 12.0        # slowing down to catch it, and kick it the moment we have it if we face their goal mouth
RUSH_FLICK_RANGE = 200.0    # (from up to this far; robots in the way don't matter). False = off.
RUSH_CLEAR = 50.0           # ...only rush while no opponent is within this of the ball (rushing in with one coming
                            # the other way ran us into it at full speed)
RUSH_TURN_TIME = 1.5        # for this long (s) after a rush, carrying slows down while it turns: full speed up to
RUSH_TURN_FREE = math.radians(30)      # RUSH_TURN_FREE off the way we face, then slower the more it turns, down
RUSH_TURN_SLOWEST = math.radians(60)   # to RUSH_TURN_MIN_SPD at RUSH_TURN_FREE + RUSH_TURN_SLOWEST (it couldn't
RUSH_TURN_MIN_SPD = 0.4                # turn away from an opponent at rush speed)
                            # Simulator, 44 games: TS 26-0 (was 1-0), Bodensee 215-5, lnxrobots 69-7, Technulgy 45-4
                            # (were 212-8, 80-11, 45-5)


BEHIND_ORBIT = True         # chase, ball right behind us (more than BEHIND_ANGLE degrees off up-field):
BEHIND_ANGLE = 150.0        # True: within BEHIND_RANGE cm, orbit - go to a point BEHIND_GAP beside it (our side),
BEHIND_RANGE = 70.0         # facing up-field, then behind it: never backs into it (that pushed it towards our own goal)
BEHIND_GAP = R + cfg.BALL_RADIUS + 10.0   # or turns round. Simulator, 88 games: TS 45 wins, the others 76-85.
BEHIND_CLOSE = 50.0         # False: within BEHIND_CLOSE cm, just head BEHIND_SIDESTEP degrees off up-field on its side
BEHIND_SIDESTEP = 110.0     # instead (still drove into it from further back): TS 55 wins, the others 74-82.


def share(value, start, end):
    """How far value has got from start to end, as 0..1 (0 at start or before, 1 at end or past it).
    e.g. share(d, 90, 20) = 0 when d >= 90 cm, 1 when d <= 20 cm, 0.5 at 55 cm."""
    return min(1.0, max(0.0, (start - value) / (start - end)))


def chase(me, ball):
    """Go for a loose ball (team TS's way): along the ball's direction, aiming further round it the closer it is (so
    the robot curls in behind it), turning to face it once it's close and in front.
    Returns (target field point, heading, dribbler on, speed multiplier)."""
    d = dist(me, ball)
    # the ball's direction from us in DEGREES: 0 = straight up-field, +90 = to our right, -90 = left, 180 = behind
    a = math.degrees(math.atan2(ball[0] - me[0], ball[1] - me[1]))

    def soften(x, B=76.0):
        # about 0 for small angles, about x for big ones (the change happens around B degrees)
        return x * (1 - 1 / (1 + abs(x / B) ** 3))

    # 1) the closer we get (90 -> 20 cm), the further round the ball we aim: from a towards a * 1.7, so the robot
    #    curls round to come in behind it
    t1 = share(d, 90.0, 20.0)
    further_round = max(-180.0, min(180.0, a * CHASE_EXAGGERATE))
    ang = a + (further_round - a) * t1
    # 2) close up (50 -> 20 cm): ball nearly in front -> straight at it; ball beside / behind -> keep going round
    t2 = share(d, 50.0, 20.0)
    ang = ang + (180.0 * soften(ang) / soften(180.0) - ang) * t2
    # (BEHIND_ORBIT off) ball close and right behind us: head off to its side instead of straight at it
    if not BEHIND_ORBIT and d < BEHIND_CLOSE and abs(a) > BEHIND_ANGLE:
        ang = math.copysign(BEHIND_SIDESTEP, a or 1.0)
    # drive towards a point 40 cm away in that direction
    target = [me[0] + math.sin(math.radians(ang)) * 40, me[1] + math.cos(math.radians(ang)) * 40]
    # ball behind us: orbit - to the point beside it on our side (the middle of the field if we're right above it),
    # facing up-field, at full speed; once level with it the normal chase takes it from behind
    if BEHIND_ORBIT and abs(a) > BEHIND_ANGLE and d < BEHIND_RANGE:
        dx = me[0] - ball[0]
        side = math.copysign(1.0, dx if abs(dx) > 2.0 else -ball[0] or 1.0)
        beside = [ball[0] + side * BEHIND_GAP, ball[1]]
        return clamp_in_field(around_ball(me, beside, ball), FIELD_MARGIN), 0.0, False, 1.0

    # 3) facing: up-field, but within CHASE_FACE_DIST and with the ball within CHASE_FACE_ANGLE of up-field, turn
    #    to face it - fully by 22 cm, sooner the more it's to the side. (-a: headings count anticlockwise, a clockwise)
    heading = 0.0
    if d <= CHASE_FACE_DIST and abs(a) <= CHASE_FACE_ANGLE:
        far_away = share(d, 22, 50)                       # 0 at 22 cm, 1 at 50 cm
        straight_ahead = max(0.0, 1 - 1.5 * abs(a) / 90)  # 1 straight ahead, 0 from 60 degrees off
        t = far_away * straight_ahead
        heading = math.radians(-a) * (1 - t)               # t = 0: face the ball fully, t = 1: face up-field

    # 4) speed: full from 45 cm, easing to 80% at 30 cm, and x0.6 under 25 cm (gentle at the ball, to catch it)
    spd = 0.8 + 0.2 * share(d, 30, 45)
    if d < 25:
        spd *= 0.6

    # 5) dribbler on when it's close and in front
    dribbler = d <= 33 and abs(a) <= 45
    return clamp_in_field(target, FIELD_MARGIN), heading, dribbler, spd


def shot_open(compass, me, centres, max_range):
    """Kicking right now would score: our actual heading crosses their goal mouth within max_range (with room for
    the ball inside the posts) and the ball's path misses every robot."""
    fx, fy = -math.sin(compass), math.cos(compass)     # the way we face
    if fy < 0.2:
        return False                                   # not facing up-field
    t = (field.ATTACK_GOAL_Y - me[1]) / fy             # how far along our facing line their goal line is
    cross = (me[0] + fx * t, field.ATTACK_GOAL_Y)      # where we'd cross it
    in_range = 0 < t < max_range
    between_posts = abs(cross[0]) <= field.GOAL_W / 2 - cfg.BALL_RADIUS - POST_MARGIN
    if not (in_range and between_posts):
        return False
    return clearance(me, cross, centres) > 0.0


def head_on(me, compass, opponents):
    """An opponent pressed against our front (centre within 2R + HEAD_ON_GAP, within HEAD_ON_ANGLE of straight
    ahead), or None."""
    fwd = (-math.sin(compass), math.cos(compass))      # the way we face
    for o in opponents:
        d = dist(o, me)
        touching = d < 2 * R + HEAD_ON_GAP
        # dot product of "our front" and "the way to it" = cos(the angle between them)
        cos_angle = ((o[0] - me[0]) * fwd[0] + (o[1] - me[1]) * fwd[1]) / max(d, 1e-6)
        if touching and cos_angle > math.cos(HEAD_ON_ANGLE):
            return o
    return None


def charged(me, opponents, velocities):
    """An opponent within CHARGED_DIST coming at us faster than CHARGED_SPEED?"""
    for o, v in zip(opponents, velocities):
        to_me = unit((me[0] - o[0], me[1] - o[1]))
        closing_speed = v[0] * to_me[0] + v[1] * to_me[1]      # how fast it's coming at us
        if dist(o, me) < CHARGED_DIST and closing_speed > CHARGED_SPEED:
            return True
    return False


def carry_dir(me, goal, opponents, velocities):
    """Carrying the ball: the direction (unit arrow) to carry it in, facing it - the one nearest straight at their
    goal that keeps every nearby opponent coming at us out of our front cone, so if we bump into one it's with the
    side / back of the robot, not the capture zone with the ball (carrying straight at the goal ran the ball into
    an opponent coming the other way). None possible: the one keeping them furthest round to the side."""
    to_goal = math.atan2(goal[1] - me[1], goal[0] - me[0])    # (maths angle, not a compass heading)

    # the angles to the opponents that matter: coming at us, and near us (now or CARRY_LEAD s from now)
    near = []
    for o, v in zip(opponents, velocities):
        soon = (o[0] + v[0] * CARRY_LEAD, o[1] + v[1] * CARRY_LEAD)     # where it'll be in CARRY_LEAD s
        to_me = unit((me[0] - o[0], me[1] - o[1]))
        closing_speed = v[0] * to_me[0] + v[1] * to_me[1]
        if closing_speed < CARRY_CLOSING:
            continue                                  # (one standing still / moving away: carry on past it)
        if dist(soon, me) < CARRY_CONTACT_RANGE or dist(o, me) < CARRY_CONTACT_RANGE:
            near.append(math.atan2(soon[1] - me[1], soon[0] - me[0]))
    if not near:
        return (math.cos(to_goal), math.sin(to_goal))  # nobody coming: straight at the goal

    # try turning 0, -10, +10, -20, +20 ... degrees off the goal (smallest turn first)
    best = None                                       # (how far the nearest opponent is off our front, angle)
    n = int(CARRY_TURN_MAX / CARRY_TURN_STEP + 0.5)   # 12 steps each way
    for k in sorted(range(-n, n + 1), key=abs):
        a = to_goal + k * CARRY_TURN_STEP
        worst = min(abs(angle_diff(b, a)) for b in near)    # the opponent nearest our front (angle)
        if worst >= CARRY_FRONT_CONE:
            return (math.cos(a), math.sin(a))         # all of them outside our front cone: carry this way
        if best is None or worst > best[0]:
            best = (worst, a)
    return (math.cos(best[1]), math.sin(best[1]))


def keep_out_of_box(target):
    """Move a target point out of our penalty area (+ BOX_KEEP): to its front edge."""
    front = field.OWN_GOAL_Y + field.PENALTY_D + BOX_KEEP
    if abs(target[0]) < field.PENALTY_W / 2 + BOX_KEEP and target[1] < front:
        return [target[0], front]
    return target


class StrikerStrategy:
    """Ball in play (seen, not out):
         ball in the dribbler, an opponent pressed against it head-on -> "slip past": step forwards past it
         ball in the dribbler                                          -> "carry": at their goal (carry_dir()), kick
                when facing it (CARRY_*), just after winning it (QUICK_SHOT_*) or when charged (CHARGED_*) - if open
         otherwise                                                     -> "chase" (chase()), or "screen" /
                                                                          "restart orbit" for a restart ball behind us
       Ball out           -> "ball out": wait just behind the neutral spot it'll come back to
       Ball not seen      -> "restart" (it went out a moment ago: wait behind that spot), "wait" (lost it: behind the
                             neutral spot nearest to where it was last seen) - or "defend" in front of our goal
                             while the goalie is off
       Never turns round to face our own goal to get the ball (screen, restart orbit, chase's orbit round a ball
       right behind us).
       Never goes into our penalty area while the goalie is on (keep_out_of_box()).
       botstate: 0 no ball, 1 ball in the dribbler, 2 going for it, 3 defend (goalie off), 5 ball out
       (+10 = fallback while not localised)."""

    def __init__(self):
        self.fallback = StrikerFallback()
        self.avoid = Avoider()
        self.kick = False
        self.speed = {}
        self.chasing = True
        self.botstate = 0
        self.phase = ""
        self.command = 0         # (sent to the goalie: 1 = take the ball - not used any more, always 0)
        self.possession = None   # "us" / "them" / None: who has the ball (possession())
        self.facing_since = None # carrying: since when we've faced their goal (CARRY_KICK_HOLD)
        self.won_at = None       # when we got the ball (QUICK_SHOT_TIME)
        self.out_at = None       # last time the ball was out of play (screen / restart orbit / rush)
        self.rushed_at = None    # last time we rushed a restart ball (RUSH_TURN_TIME)
        self.last_ball = None    # (position, time, went out?) of the last ball seen - for an out ball, the position
                                 # is the neutral spot it'll come back to

    def reset(self):
        self.fallback.reset()
        self.last_ball = None
        self.facing_since = None
        self.won_at = None

    def update(self, world, compass, mate, line_touches):
        pose = world.pose
        self.kick = False
        self.command = 0
        self.possession = possession(world, mate)

        # ---- not sure where we are on the field: the fallback (relative) logic decides
        if pose is None or not pose.confident:
            mate_on = mate.get("bot active") if mate else None
            out = self.fallback.update(world, compass, mate_on, line_touches)
            self.botstate = 10 + self.fallback.botstate
            self.speed = self.fallback.speed
            self.kick = getattr(self.fallback, "kick", False)
            self.phase = "fallback"
            self.chasing = self.fallback.botstate in (1, 2)
            return out

        me = (pose.x, pose.y)
        ball = world.ball_field
        if ball is None and world.ball_in_capture:
            # the camera lost it while it's in our dribbler: it's right in front of us
            fwd = (-math.sin(compass), math.cos(compass))
            ball = (me[0] + fwd[0] * (R + cfg.BALL_RADIUS), me[1] + fwd[1] * (R + cfg.BALL_RADIUS))
        if ball is not None:
            # remember it (a ball going out: remember the spot it'll come back to, to wait there)
            went_out = going_out(ball, world.ball_vel)
            remembered = restart_spot(ball) if went_out else ball
            self.last_ball = (remembered, time.monotonic(), went_out)
        if not world.ball_in_capture:
            self.won_at = None
        centres = obstacle_centres(me, world.obstacles_field)
        goalie_on = mate_active(mate)
        self.speed = {}

        # (a ball just past the line but still there isn't out yet - the referee takes it off once it's fully past:
        # keep playing it till then; giving up on it at once handed it to them)
        gone_out = ball is not None and not field.inside_play_area(ball[0], ball[1], -cfg.BALL_RADIUS - 3.0)
        # coming back for a restart from their half: steer round the neutral spots (the ball may be put back on one
        # and driving across it pushed it backwards)
        spot_obstacles = list(field.NEUTRAL_SPOTS) if me[1] > 0 else []
        if ball is None or gone_out:
            if self.last_ball is not None and self.last_ball[2]:   # it really went out (not just lost from view)
                self.out_at = time.monotonic()
            self.chasing = False
            dribbler = False
            if ball is not None:
                # ---- ball out: wait on our goal's side of the spot it'll come back to
                spot = restart_spot(ball)
                target = clamp_in_field([spot[0] * RESTART_X_SHARE, spot[1] - NEUTRAL_WAIT], R)
                target = self.avoid(me, target, centres + spot_obstacles)
                heading = 0.0
                self.botstate, self.phase = 5, "ball out"
            else:
                # ---- ball not seen
                target, heading = self.no_ball(world, me, centres + spot_obstacles, goalie_on)
        else:
            # ---- ball in play: chase / carry / shoot
            target, heading, dribbler = self.attack(world, compass, me, ball)
            self.chasing = True
            self.botstate = 1 if world.ball_in_capture else 2
            target = clamp_in_field(target, FIELD_MARGIN)

        # ---- goalie on: never into our penalty area
        if goalie_on:
            target = keep_out_of_box(target)
        return rel(target, pose), heading, dribbler

    def no_ball(self, world, me, centres, goalie_on):
        """Ball not seen (by us or the goalie). Returns (target, heading); sets .botstate / .phase / .speed."""
        heading = 0.0
        if not goalie_on:
            # on our own: stand in front of our goal
            self.botstate, self.phase = 3, "defend"
            return [0.0, field.OWN_GOAL_Y + GUARD_RADIUS + 10], heading
        self.botstate = 0
        lb = self.last_ball
        now = time.monotonic()
        if lb is not None and lb[2] and now - lb[1] < RESTART_WAIT:
            # restart: the ball went out and comes back on that neutral spot - wait just behind it, facing their
            # goal, to take it the moment it's there
            spot = lb[0]
            x = spot[0] * RESTART_X_SHARE - math.copysign(RESTART_SIDE, spot[0] or 1.0)
            target = self.avoid(me, [x, spot[1] - NEUTRAL_WAIT], centres)
            self.phase = "restart"
            their_keeper_home = any(dist(o, (0.0, field.ATTACK_GOAL_Y)) < 30 for o in world.opponents)
            if their_keeper_home:
                # face the far corner of their goal from the spot
                far = (-math.copysign(field.GOAL_W / 2 - 8.0, spot[0] or 1.0), field.ATTACK_GOAL_Y)
                heading = heading_for((far[0] - spot[0], far[1] - spot[1]))
        else:
            # lost it: wait behind the neutral spot nearest to where it was last seen (going round robots in the way)
            spot = restart_spot(lb[0]) if lb is not None else (0.0, 0.0)
            target = self.avoid(me, [spot[0] * RESTART_X_SHARE, spot[1] - NEUTRAL_WAIT], centres)
            self.phase = "wait"
            if dist(me, WAIT_POS) > 30:
                self.speed = {"spd_min": 0.6}      # far to go: not slow just because a waypoint is close
        return target, heading

    def restart_ball(self, ball, within=RUSH_TIME):
        """The ball has just been put back on a neutral spot after going out (within `within` s of being out)."""
        if self.out_at is None:
            return False
        recently_out = time.monotonic() - self.out_at < within
        on_a_spot = min(dist(ball, spot) for spot in field.NEUTRAL_SPOTS) < RUSH_SPOT
        return recently_out and on_a_spot

    def attack(self, world, compass, me, ball):
        """Ball in play. Returns (target field point, heading, dribbler); sets .kick / .speed / .phase."""
        goal = (0.0, field.ATTACK_GOAL_Y)                                  # centre of their goal
        heading = heading_for((goal[0] - me[0], goal[1] - me[1]))          # facing their goal

        # ---- no ball in the dribbler: chase it
        if not world.ball_in_capture:
            self.facing_since = None
            target, heading, dribbler, spd = chase(me, ball)
            # restart screen: the ball is just back and behind us, their robot is near it - face up-field and push
            # it away from the ball (not turning round, never touching the ball)
            behind_us = me[1] > ball[1] - 5.0
            nearest = min(world.opponents, key=lambda o: dist(o, ball), default=None)
            if (RESTART_SCREEN and behind_us and nearest is not None and dist(nearest, ball) < SCREEN_OPP
                    and self.restart_ball(ball, SCREEN_TIME)):
                # go round the ball on the side their robot is on (bodying it on the way), to get behind the ball
                side = math.copysign(1.0, (nearest[0] - ball[0]) or (me[0] - ball[0]) or 1.0)
                gap = R + cfg.BALL_RADIUS + 8.0
                if me[1] > ball[1] - gap * 0.5:
                    target = [ball[0] + side * gap, ball[1] - gap * 0.5]       # beside the ball, their side
                else:
                    target = [ball[0], ball[1] - gap]                          # behind it
                self.speed = {"spd_multi": 1.0}
                self.phase = "screen"
                return around_ball(me, target, ball), 0.0, False
            # restart orbit: the ball is just back, we're on the wrong side of it (up-field) and they're coming -
            # full speed round it to get behind it, facing up-field
            wrong_side = me[1] > ball[1] - 5.0
            they_are_coming = any(dist(o, ball) < CATCH_RESTART_OPP for o in world.opponents)
            if RESTART_CATCH and self.restart_ball(ball, CATCH_RESTART_TIME) and wrong_side and they_are_coming:
                # orbit round the ball to get behind it, facing up-field the whole time (turning round to face our own
                # goal to get the ball is never done): beside it on our side, then behind it
                side = math.copysign(1.0, (me[0] - ball[0]) or 1.0)
                gap = R + cfg.BALL_RADIUS + 8.0
                if me[1] > ball[1] - gap * 0.5:
                    target = [ball[0] + side * gap, ball[1] - gap * 0.5]
                else:
                    target = [ball[0], ball[1] - gap]
                self.speed = {"spd_multi": 1.0}
                self.phase = "restart orbit"
                return around_ball(me, target, ball), 0.0, False
            ball_behind_x = abs(me[0] - ball[0]) < RUSH_BEHIND_X
            nobody_near = all(dist(o, ball) > RUSH_CLEAR for o in world.opponents)
            if RESTART_RUSH and self.restart_ball(ball) and me[1] < ball[1] and ball_behind_x and nobody_near:
                spd = 1.0                            # restart race: full speed in (no slowing down to catch it)
                self.phase = "rush"
                self.rushed_at = time.monotonic()
            else:
                self.phase = "chase"
            self.speed = {"spd_multi": spd}
            return target, heading, dribbler

        # ---- ball in the dribbler, an opponent pressed against our front: step past it (SLIP_SIDE across, away
        # from the side it's on, and SLIP_FORWARD forwards)
        ho = head_on(me, compass, world.opponents)
        if ho is not None:
            fwd = (-math.sin(compass), math.cos(compass))
            cross = (ho[0] - me[0]) * fwd[1] - (ho[1] - me[1]) * fwd[0]    # which side of our front it's on
            side = 1.0 if cross < 0 else -1.0
            sideways = (fwd[1] * side, -fwd[0] * side)
            self.speed = {"spd_multi": 1.0}
            self.phase = "slip past"
            target = [me[0] + sideways[0] * SLIP_SIDE + fwd[0] * SLIP_FORWARD,
                      me[1] + sideways[1] * SLIP_SIDE + fwd[1] * SLIP_FORWARD]
            return target, heading, True

        # ---- ball in the dribbler: carry it, kick when there's a chance
        now = time.monotonic()
        centres = obstacle_centres(me, world.obstacles_field)
        if self.won_at is None:
            self.won_at = now
        facing_goal_mouth = shot_open(compass, me, [], RUSH_FLICK_RANGE)   # [] = ignore robots in the way
        if RESTART_RUSH and self.restart_ball(ball) and facing_goal_mouth:
            self.kick = True                                   # restart race: flick it up-field straight away
            flicked = True
        else:
            flicked = False
        just_won_it = now - self.won_at < QUICK_SHOT_TIME
        if just_won_it and shot_open(compass, me, centres, QUICK_SHOT_RANGE):
            self.kick = True                                   # quick shot: before they close us down
        u = carry_dir(me, goal, world.opponents, world.opponent_vel)   # towards the goal, front away from them
        carry_heading = heading_for(u)
        # just after a rush: slow down while turning (at rush speed it couldn't turn away from an opponent)
        carry_spd = 1.0
        just_rushed = self.rushed_at is not None and now - self.rushed_at <= RUSH_TURN_TIME
        turn = abs(angle_diff(carry_heading, compass))          # how far the carry has to turn us
        if just_rushed and turn >= RUSH_TURN_FREE:
            carry_spd = max(RUSH_TURN_MIN_SPD, 1.0 - (turn - RUSH_TURN_FREE) / RUSH_TURN_SLOWEST * 0.6)
        self.speed = {"spd_multi": carry_spd}
        self.phase = "flick" if flicked else "carry"
        if charged(me, world.opponents, world.opponent_vel) and shot_open(compass, me, centres, OPEN_SHOT_RANGE):
            self.kick = True                                   # about to be shoved off it: shoot
        # facing their goal (within CARRY_FACING) for CARRY_KICK_HOLD s, within CARRY_KICK_RANGE: kick
        facing_err = abs(angle_diff(heading, compass))
        if facing_err < CARRY_FACING:
            if not self.facing_since:
                self.facing_since = now
            if now - self.facing_since > CARRY_KICK_HOLD and dist(me, goal) < CARRY_KICK_RANGE:
                self.kick = True
        else:
            self.facing_since = None
        target = [me[0] + u[0] * 40, me[1] + u[1] * 40]        # 40 cm along the carry direction
        return target, carry_heading, True

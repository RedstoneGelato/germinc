"""
strategy_fallback.py - what the robots do while they DON'T know where they are on the field (localisation not
confident). strategy.py hands over to GoalieFallback / StrikerFallback then.

Everything here is RELATIVE: the robot is at (0, 0), x / y are field directions (the compass keeps them pointing the
same way as the field: +y = towards the goal we attack), in cm. What the camera gives us:
    world.ball            the ball (None = not seen)
    world.ball_in_capture the ball is in our capture zone (dribbler)
    world.attack_goal / world.own_goal   Goal(near, centre) or None (not seen)
Both classes: update(...) -> (desired_pos, desired_heading, dribbler_on), and .speed / .kick / .botstate / .chasing.
"""
import math
import time

import field
import robot_config as cfg

R = cfg.ROBOT_RADIUS

# ---- TUNE (cm)
DEFAULT_GOAL = [0.0, 200.0]   # their goal not seen: carry the ball this way (straight up-field)
GOALIE_DIST = 20.0            # goalie: guard this far out from our goal's centre (like strategy.GUARD_RADIUS)
BACKUP_Y = -60.0              # goalie: our goal not seen - back up this way
GOALIE_CLEAR_DIST = 35.0      # goalie: ball this close to our goal's centre and this close to the goalie -> clear it
GOALIE_KICK_TOL = math.radians(35)   # goalie with the ball: kick when facing within this of straight up-field
BLOCK_SPEED = 20.0            # goalie: ball rolling at our goal faster than this (cm/s) -> get in its way
KICK_FACING = math.radians(10)       # striker: kick when facing their goal's centre within this...
KICK_RANGE = 75.0             # ...and the goal centre is closer than this
QUICK_SHOT_TIME = 1.0         # ...or within this long (s) of winning the ball...
QUICK_SHOT_FACING = math.radians(6)  # ...facing it this closely...
QUICK_SHOT_RANGE = 130.0      # ...from up to this far
MIDFIELD_FROM_GOAL = 90.0     # striker, no ball: wait this far in front of their goal (if seen)
ADVANCE_Y = 60.0              # striker, no ball, their goal not seen: drive this way
HOME_DIST = 25.0              # striker on its own, no ball: stand this far in front of our goal
CHASE_MARGIN = R + 4.0        # striker: keep its target this far inside the white line (when a goal tells us roughly
CARRY_MARGIN = R + 16.0       # where we are), and this far while carrying the ball (it sits in front of the robot)
GOAL_CENTRE_Y = field.ATTACK_GOAL_Y + field.GOAL_DEPTH / 2  # where a seen goal's "centre" roughly is (field y)


def rough_position(world):
    """Where we roughly are on the field, worked out from a goal the camera sees (its field position is known), or
    None if neither goal is seen."""
    att, own = world.attack_goal, world.own_goal
    if att is not None:
        return (-att.centre[0], GOAL_CENTRE_Y - att.centre[1])
    if own is not None:
        return (-own.centre[0], -GOAL_CENTRE_Y - own.centre[1])
    return None


def keep_in_field(target, world, margin):
    """A relative target moved back inside the field (margin cm inside the line), if we know roughly where we are."""
    me = rough_position(world)
    if me is None:
        return target
    hx, hy = field.PLAY_W / 2 - margin, field.PLAY_L / 2 - margin
    fx = min(max(me[0] + target[0], -hx), hx)
    fy = min(max(me[1] + target[1], -hy), hy)
    return [fx - me[0], fy - me[1]]


def angdiff(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


def heading_for(d):
    """Arrow d (field directions) -> compass heading that faces along it."""
    return math.atan2(-d[0], d[1])


def unit(v):
    n = math.hypot(v[0], v[1])
    return (0.0, 0.0) if n < 1e-9 else (v[0] / n, v[1] / n)


def share(value, start, end):
    return min(1.0, max(0.0, (start - value) / (start - end)))


def chase_rel(ball):
    """strategy.chase() with the robot at (0, 0): (target, heading, dribbler on, speed multiplier). Ball right behind
    us: orbit to a point beside it instead of backing into it (like strategy.BEHIND_ORBIT)."""
    d = math.hypot(ball[0], ball[1])
    a = math.degrees(math.atan2(ball[0], ball[1]))          # 0 = up-field, + = right

    def soften(x, B=76.0):
        return x * (1 - 1 / (1 + abs(x / B) ** 3))
    if abs(a) > 150.0 and d < 70.0:                         # right behind us: go round beside it, facing up-field
        side = math.copysign(1.0, -ball[0] or 1.0)
        return [ball[0] + side * (R + cfg.BALL_RADIUS + 10.0), ball[1]], 0.0, False, 1.0
    t1 = share(d, 90.0, 20.0)
    ang = a + (max(-180.0, min(180.0, a * 1.7)) - a) * t1
    t2 = share(d, 50.0, 20.0)
    ang = ang + (180.0 * soften(ang) / soften(180.0) - ang) * t2
    target = [math.sin(math.radians(ang)) * 40, math.cos(math.radians(ang)) * 40]
    heading = 0.0
    if d <= 55.0 and abs(a) <= 60.0:
        t = share(d, 22, 50) * max(0.0, 1 - 1.5 * abs(a) / 90)
        heading = math.radians(-a) * (1 - t)
    spd = 0.8 + 0.2 * share(d, 30, 45)
    if d < 25:
        spd *= 0.6
    return target, heading, d <= 33 and abs(a) <= 45, spd


class GoalieFallback:
    """botstate 0 = no ball -> stand in front of our goal
       botstate 1 = ball in the dribbler -> turn up-field and kick it clear
       botstate 2 = ball close to our goal (and to us) -> go for it
       botstate 3 = otherwise -> guard on the arc between our goal and the ball (block it if it's rolling at us)"""

    def __init__(self):
        self.botstate = 3
        self.speed = {}
        self.kick = False
        self.chasing = False

    def reset(self):
        pass

    def update(self, world, compass, comms_command, attack_bot_state, line_touches):
        self.kick = False
        self.speed = {}
        ball, own = world.ball, world.own_goal
        if own is None:                                       # don't know where our goal is: back up towards it
            self.botstate, self.chasing = 0, False
            return [0.0, BACKUP_Y], 0.0, False
        g = own.centre
        if ball is None:
            self.botstate, self.chasing = 0, False
            return [g[0], g[1] + GOALIE_DIST + 6.0], 0.0, False
        if world.ball_in_capture:
            # ball in the dribbler: turn up-field, kick it clear as soon as we face up-field
            self.botstate, self.chasing = 1, True
            self.kick = abs(angdiff(compass, 0.0)) < GOALIE_KICK_TOL
            return [0.0, 0.0], 0.0, True
        to_ball = (ball[0] - g[0], ball[1] - g[1])
        if math.hypot(*to_ball) < GOALIE_CLEAR_DIST and math.hypot(*ball) < GOALIE_CLEAR_DIST:
            # close to our goal and to us: go for it from the goal's side (push it away from the goal)
            self.botstate, self.chasing = 2, True
            u = unit(to_ball)
            self.speed = {"spd_multi": 0.7}
            return [ball[0] + u[0] * 10.0, ball[1] + u[1] * 10.0], 0.0, math.hypot(*ball) < 25.0
        self.botstate, self.chasing = 3, False
        vel = getattr(world, "ball_vel", (0.0, 0.0))
        if vel[1] < -BLOCK_SPEED and ball[1] > g[1]:
            # rolling at our goal: stand where it crosses our guard line
            line_y = g[1] + R + 3.0
            t = (line_y - ball[1]) / vel[1]
            x = ball[0] + vel[0] * t
            if abs(x - g[0]) < 35.0 and t < 1.5:
                target = [x, line_y]
                self.speed = {"spd_multi": min(1.0, max(0.1, math.hypot(*target) / 25.0)), "smooth": 0.3}
                return target, 0.0, False
        u = unit(to_ball)
        target = [g[0] + u[0] * GOALIE_DIST, max(g[1] + u[1] * GOALIE_DIST, g[1] + R + 3.0)]
        self.speed = {"spd_multi": min(1.0, max(0.1, math.hypot(*target) / 15.0)), "smooth": 0.3}
        return target, 0.0, False


class StrikerFallback:
    """botstate 0 = no ball, goalie on    -> wait in front of their goal (or drive up-field)
       botstate 1 = ball in the dribbler  -> carry it at their goal, kick when facing it close enough
       botstate 2 = ball seen             -> chase it (TS-style, like strategy.chase())
       botstate 3 = no ball, goalie off   -> stand in front of our goal
    command (sent to the goalie): always 0 (the goalie decides by itself)."""

    def __init__(self):
        self.botstate = 2
        self.command = 0
        self.speed = {}
        self.kick = False
        self.chasing = False
        self.won_at = None

    def reset(self):
        self.won_at = None

    def update(self, world, compass, goalie_bot_state, line_touches):
        self.kick = False
        self.command = 0
        ball = world.ball
        att, own = world.attack_goal, world.own_goal
        if not world.ball_in_capture:
            self.won_at = None
        if ball is None:
            self.chasing = False
            self.speed = {}
            if goalie_bot_state == 1:
                self.botstate = 0
                if att is not None:
                    return [att.centre[0], att.centre[1] - MIDFIELD_FROM_GOAL], 0.0, False
                return [0.0, ADVANCE_Y], 0.0, False
            self.botstate = 3
            if own is not None:
                return [own.centre[0], own.centre[1] + HOME_DIST + R], 0.0, False
            return [0.0, -50.0], 0.0, False
        self.chasing = True
        if world.ball_in_capture:
            # carry at their goal, facing it; kick when facing it close enough, or right after winning the ball
            self.botstate = 1
            now = time.monotonic()
            if self.won_at is None:
                self.won_at = now
            goal = att.centre if att is not None else DEFAULT_GOAL
            heading = heading_for(goal)
            err = abs(angdiff(heading, compass))
            dist = math.hypot(goal[0], goal[1])
            if att is not None and err < KICK_FACING and dist < KICK_RANGE:
                self.kick = True
            if att is not None and now - self.won_at < QUICK_SHOT_TIME and err < QUICK_SHOT_FACING \
                    and dist < QUICK_SHOT_RANGE:
                self.kick = True
            u = unit(goal)
            self.speed = {"spd_multi": 1.0}
            return keep_in_field([u[0] * 40, u[1] * 40], world, CARRY_MARGIN), heading, True
        self.botstate = 2
        target, heading, dribbler, spd = chase_rel(ball)
        self.speed = {"spd_multi": spd}
        return keep_in_field(target, world, CHASE_MARGIN), heading, dribbler

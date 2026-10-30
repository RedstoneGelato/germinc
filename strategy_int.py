import math
import time

import field_int as field
field.install()
import robot_config as cfg  # noqa: E402  (after install(): it sets cfg.MAX_OUT for these walls)
from strategy_fallback import GoalieFallback, StrikerFallback
from utils import Hysteresis

R = cfg.ROBOT_RADIUS

BOX_POKE = 6.0
BOX_KEEP_INT = 4.0
BRAKE_TIME = 0.08
BRAKE_MAX = 6.0
GOALIE_LINE_D = field.PENALTY_D - R + BOX_POKE

# ---- TUNE (cm, radians)
FIELD_MARGIN = R + 4.0
BEHIND_DIST = R + 12.0
OUT_MARGIN = -cfg.MAX_OUT
ORBIT_DIST = R + 14.0
ALIGN_TOL = 6.0
ALIGN_TOL_PER_CM = 0.2
PUSH_DIST = 25.0
DRIBBLER_RANGE = 25.0
AVOID_DIST = 2 * R + 8.0
AIM_MARGIN = 5.0
POST_MARGIN = 5.0
AIM_POINTS = 7
KICK_CLEARANCE = 8.0
LEAD_TIME_MAX = 0.6
LEAD_SPEED = 100.0
CATCH_MULTI_MIN = 0.3
CATCH_MULTI_PER_CM = 0.015
RACE_MARGIN = 15.0

GUARD_RADIUS = GOALIE_LINE_D
GUARD_OUT_START = 60.0
GUARD_OUT_PER_CM = 0.0
GUARD_OUT_MAX = 40.0
GUARD_BISECTOR = True
GUARD_COVER = 0.6
GUARD_D_MAX = 30.0
GUARD_LEAD = 0.1
GUARD_GAIN = 15.0
GUARD_SMOOTH = 0.3
BLOCK_SPEED = 20.0
BLOCK_TIME = 1.5
BLOCK_WIDEN = 8.0
BLOCK_GAIN = 25.0
BLOCK_STAY = 5.0
DANGER_DIST = 35.0
CARRIER_WEIGHT = 0.0
CARRIER_WIDEN = 10.0
POST_SEAL = True
SEAL_SIDE = True
SEAL_SIDE_X = field.PENALTY_W / 2 - R + BOX_POKE + 1.0
SEAL_X = field.GOAL_W / 2 + 2.0
SEAL_Y = 25.0
SEAL_OPP = 45.0
SEAL_KEEP_X = field.GOAL_W / 2 - 6.0
SEAL_POST_X = field.GOAL_W / 2 - 2.0
SEAL_GAP = 0.0
SEAL_INNER_X = field.GOAL_W / 2 - 6.0
SEAL_INNER_KEEP_X = field.GOAL_W / 2 - 10.0
GOALIE_FACE_MAX = math.radians(70)
SPOT_TAKE = True
SPOT_TAKE_DIST = 60.0
SPOT_TAKE_CLEAR = 60.0
SPOT_TAKE_ON = 5.0
SPOT_TAKE_STILL = 10.0
BOX_TURN = True
BOX_TURN_BEHIND = 3.0
BOX_TURN_MIN_D = 10.0
BOX_TURN_OPP = 25.0
BOX_TURN_SPEED = 0.35
BOX_KICK_DIRS = ((0.0, 1.0), (0.7, 0.7), (1.0, 0.25))
BOX_KICK_TOL = math.radians(15)
GOALIE_CLEAR_PATH = 60.0
GOALIE_KICK_TURN = math.radians(60)
GOALIE_KICK_CLEAR = 50.0
PRESSED_GAP = 45.0
PASS_SIDE = 30.0
PASS_AHEAD = 20.0
PASS_TOL = 0.25
SOLO_CHASE_Y = 0.0
SOLO_CLOSER = 15.0
PUSHBACK_GAP = 6.0
PUSHBACK_ANGLE = math.radians(50)
GOALIE_STEP_UP = False
STEP_UP_BALL_Y = 50.0
STEP_UP_CLEAR_Y = -20.0
STEP_UP_X_SHARE = 0.3
STEP_UP_Y = -40.0
GOALIE_LOOK_AT_SPOT = False
LOOK_DIST = 55.0
GOALIE_ALONE_ATTACKS = True
KICK_RETURN_T = 1.0
CLEAR_SHARE = True
CLEAR_SHARE_AFTER = 0.4
CLEAR_ROLL_SPEED = 250.0
CLEAR_ROLL_DECAY = 0.8
CLEAR_MEET_SPEED = 120.0
CLEAR_MEET_MIN = GOALIE_CLEAR_PATH + 2 * R
CLEAR_MEET_SIDE = R + 6.0
CLEAR_MEET_NEAR = 40.0
DEEP_NO_TOUCH = True
DEEP_GAP = 3.0
DEEP_OPP = 35.0
GOALIE_MAX_OUT = 12.0
PUSHBACK_REACH = 40.0
GOALIE_KICK_OUT = True
GOALIE_OUT_UP = 1.0
GOALIE_KICK_OUT_TOL = 0.4
GOALIE_OUT_CLEAR = 60.0
GOALIE_ORBIT_SPD_MIN = 0.5
CHALLENGE_DIST = 35.0
CHALLENGE_UP = 40.0
PUSHER_DIST = 28.0
PUSHER_SPEED = 10.0
OPP_ON_BALL = 12.0
CLEAR_RETURN_Y = field.OWN_GOAL_Y + 55.0
CLEAR_AWAY_Y = field.OWN_GOAL_Y + 50.0
CLEAR_OUT_X = 15.0
CONTEST_DIST = 8.0
PUSH_MULTI = 1.0
WAIT_POS = (0.0, -22.0)
BALL_OUT_MARGIN = 1.0
OUT_SOON_DIST = 15.0
OUT_SOON_SPEED = 0.0
RESTART_WAIT = 4.0
RESTART_SIDE = 0.0
RESTART_X_SHARE = 1.0
NEUTRAL_WAIT = R + 11.0


def unit(v):
    length = math.hypot(v[0], v[1])
    if length < 1e-9:
        return (0.0, 0.0)
    return (v[0] / length, v[1] / length)


def heading_for(d):
    return math.atan2(-d[0], d[1])


def clamp_in_field(p, margin=FIELD_MARGIN):
    half_x = field.PLAY_W / 2 - margin
    half_y = field.PLAY_L / 2 - margin
    x = min(max(p[0], -half_x), half_x)
    y = min(max(p[1], -half_y), half_y)
    return [x, y]


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def seg_dist(p, a, b):
    ab = (b[0] - a[0], b[1] - a[1])
    length_squared = ab[0] ** 2 + ab[1] ** 2
    if length_squared < 1e-9:
        t = 0.0
    else:
        along = (p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]
        t = max(0.0, min(1.0, along / length_squared))
    closest = (a[0] + t * ab[0], a[1] + t * ab[1])
    return dist(p, closest), t


def obstacle_centres(me, obstacles):
    centres = []
    for edge_point, size in obstacles:
        away = unit((edge_point[0] - me[0], edge_point[1] - me[1]))
        centres.append((edge_point[0] + away[0] * R, edge_point[1] + away[1] * R))
    return centres


def best_aim(ball, centres, max_range=None):
    half = field.GOAL_W / 2 - cfg.BALL_RADIUS - AIM_MARGIN
    y = field.ATTACK_GOAL_Y
    best, best_score, best_clear = (0.0, y), -1e9, -1e9
    for i in range(AIM_POINTS):
        x = -half + 2 * half * i / (AIM_POINTS - 1)
        if max_range is not None and dist(ball, (x, y)) > max_range:
            continue
        clear = clearance(ball, (x, y), centres)
        score = min(clear, 40.0) - 0.05 * abs(x)
        if score > best_score:
            best, best_score, best_clear = (x, y), score, clear
    return best, best_clear


def clearance(a, b, centres):
    gaps = [seg_dist(c, a, b)[0] - R for c in centres]
    return min(gaps, default=100.0)


def path_blocked(a, b, centres, ignore_near=None):
    worst = None
    for c in centres:
        if ignore_near is not None and dist(c, ignore_near) < 15 + R:
            continue
        d, t = seg_dist(c, a, b)
        closest_point_is_right_here = t * dist(a, b) < 2 * R
        touching_it = dist(a, c) <= 2 * R + 2
        if closest_point_is_right_here and not touching_it:
            continue
        in_the_way = d < AVOID_DIST and 0 < t < 1
        if in_the_way and (worst is None or t < worst[1]):
            worst = (c, t)
    if worst is None:
        return None
    return worst[0]


class Avoider:
    STICK = 15.0

    def __init__(self):
        self.wp = None
        self.obstacle = None
        self.side = 1.0

    def __call__(self, robot, target, centres, ignore_near=None, back_cost=0.0):
        if path_blocked(robot, target, centres, ignore_near) is None:
            self.wp = None
            self.obstacle = None
            return target

        used = [c for c in centres if ignore_near is None or dist(c, ignore_near) >= 15 + R]
        best, best_cost = None, None
        for c in used:
            candidates = (self.beside(c, target, 1.0), self.beside(c, target, -1.0),
                          self.beside_seen(c, robot, 1.0), self.beside_seen(c, robot, -1.0))
            for wp in candidates:
                if wp is None or not field.inside_play_area(wp[0], wp[1], OUT_MARGIN):
                    continue
                if path_blocked(robot, wp, used) is not None or path_blocked(wp, target, used) is not None:
                    continue
                cost = dist(robot, wp) + dist(wp, target) + back_cost * max(0.0, robot[1] - wp[1])
                if self.wp is not None and dist(wp, self.wp) < 10:
                    cost -= self.STICK
                if best_cost is None or cost < best_cost:
                    best, best_cost = wp, cost
        if best is not None:
            self.wp = best
            return clamp_in_field(best, OUT_MARGIN)

        self.wp = None
        return self.fallback(robot, target, path_blocked(robot, target, centres, ignore_near))

    @staticmethod
    def beside(c, target, side):
        u = unit((target[0] - c[0], target[1] - c[1]))
        n = (-u[1], u[0])
        L = dist(c, target)
        need = AVOID_DIST + 3
        if L <= need + 1:
            return None
        off = min(need * L / math.sqrt(L * L - need * need), 80.0)
        return [c[0] + n[0] * side * off, c[1] + n[1] * side * off]

    @staticmethod
    def beside_seen(c, robot, side):
        u = unit((c[0] - robot[0], c[1] - robot[1]))
        off = AVOID_DIST + 3
        return [c[0] - u[1] * side * off, c[1] + u[0] * side * off]

    def fallback(self, robot, target, c):
        u = unit((target[0] - c[0], target[1] - c[1]))
        n = (-u[1], u[0])
        if self.obstacle is None or dist(c, self.obstacle) > 20:
            if (robot[0] - c[0]) * n[0] + (robot[1] - c[1]) * n[1] >= 0:
                self.side = 1.0
            else:
                self.side = -1.0
        self.obstacle = c
        wp = self.beside(c, target, self.side) or [c[0] + n[0] * self.side * (AVOID_DIST + 3),
                                                   c[1] + n[1] * self.side * (AVOID_DIST + 3)]
        if not field.inside_play_area(wp[0], wp[1], OUT_MARGIN):
            self.side = -self.side
            wp = [2 * c[0] - wp[0], 2 * c[1] - wp[1]]
        return clamp_in_field(wp, OUT_MARGIN)


def choose_push_aim(ball, goal_aim, defending=False):
    if defending:
        side = math.copysign(1.0, ball[0] or 1.0)
        candidates = [goal_aim, (side * field.PLAY_W / 2, ball[1] + 40.0), (side * field.PLAY_W / 2, ball[1])]
    else:
        candidates = [goal_aim, (ball[0] * 0.5, field.ATTACK_GOAL_Y), (0.0, field.ATTACK_GOAL_Y), (0.0, 0.0)]
    best, best_cost = goal_aim, None
    for k, a in enumerate(candidates):
        u = unit((a[0] - ball[0], a[1] - ball[1]))
        behind = (ball[0] - u[0] * BEHIND_DIST, ball[1] - u[1] * BEHIND_DIST)
        cost = 2.0 * dist(behind, clamp_in_field(behind, OUT_MARGIN)) + 8.0 * k
        if best_cost is None or cost < best_cost:
            best, best_cost = a, cost
    return best


def approach(robot, ball, aim):
    u = unit((aim[0] - ball[0], aim[1] - ball[1]))
    n = (-u[1], u[0])
    r = (robot[0] - ball[0], robot[1] - ball[1])
    along = r[0] * u[0] + r[1] * u[1]
    perp = r[0] * n[0] + r[1] * n[1]
    heading = heading_for(u)
    behind = [ball[0] - u[0] * BEHIND_DIST, ball[1] - u[1] * BEHIND_DIST]
    lined_up_tol = ALIGN_TOL + ALIGN_TOL_PER_CM * (-along)

    clips = abs(perp) > lined_up_tol and seg_dist(ball, robot, behind)[0] < R + cfg.BALL_RADIUS + 3
    if along > -R * 0.6 or clips:
        side = 1.0 if perp >= 0 else -1.0
        for s in (side, -side):
            wp = [ball[0] - u[0] * ORBIT_DIST * 0.3 + n[0] * s * ORBIT_DIST,
                  ball[1] - u[1] * ORBIT_DIST * 0.3 + n[1] * s * ORBIT_DIST]
            if field.inside_play_area(wp[0], wp[1], OUT_MARGIN):
                break
        return clamp_in_field(wp, OUT_MARGIN), heading, "orbit"
    if abs(perp) > lined_up_tol:
        return clamp_in_field(behind, OUT_MARGIN), heading, "line up"
    return [ball[0] + u[0] * PUSH_DIST, ball[1] + u[1] * PUSH_DIST], heading, "push"


def around_ball(me, target, ball):
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
    if best is None:
        return target
    return best[1]


def deep_no_touch(me, target, ball, opponents=()):
    if not DEEP_NO_TOUCH or ball is None or any(dist(o, ball) < DEEP_OPP for o in opponents):
        return target
    if ball[1] - field.OWN_GOAL_Y >= GOALIE_LINE_D + R or abs(ball[0]) >= field.GOAL_W / 2 + R or me[1] <= ball[1]:
        return target
    clear = R + cfg.BALL_RADIUS + DEEP_GAP
    d_now = dist(me, ball)
    if d_now < clear:
        u = unit((me[0] - ball[0], me[1] - ball[1]))
        return [ball[0] + u[0] * clear, ball[1] + u[1] * clear]
    d, t = seg_dist(ball, me, target)
    if d >= clear:
        return target
    v = (target[0] - me[0], target[1] - me[1])
    lo, hi = 0.0, t
    for _ in range(12):
        mid = (lo + hi) / 2
        if dist((me[0] + v[0] * mid, me[1] + v[1] * mid), ball) >= clear:
            lo = mid
        else:
            hi = mid
    return [me[0] + v[0] * lo, me[1] + v[1] * lo]


def clear_aim(ball):
    if abs(ball[0]) > CLEAR_OUT_X:
        return (math.copysign(field.PLAY_W / 2, ball[0]), ball[1] + 60.0)
    away = unit((ball[0], ball[1] - field.OWN_GOAL_Y))
    away = unit((away[0], away[1] + 1.0))
    return (ball[0] + away[0] * 100.0, ball[1] + away[1] * 100.0)


def lead_ball(me, ball, vel):
    t = min(dist(me, ball) / LEAD_SPEED, LEAD_TIME_MAX)
    return clamp_in_field([ball[0] + vel[0] * t, ball[1] + vel[1] * t], -cfg.BALL_RADIUS)


def catch_speed(me, ball):
    gap = max(0.0, dist(me, ball) - R)
    return min(1.0, CATCH_MULTI_MIN + CATCH_MULTI_PER_CM * gap)


def guard_speed(me, target):
    multi = min(1.0, max(0.1, dist(me, target) / GUARD_GAIN))
    return {"spd_multi": multi, "smooth": GUARD_SMOOTH}


def angle_diff(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


def contested(ball, centres):
    return any(dist(c, ball) < CONTEST_DIST + R for c in centres)


def ball_out(ball):
    if not field.BALL_OUT_AT_LINE:
        return False
    return not field.inside_play_area(ball[0], ball[1], -BALL_OUT_MARGIN)


def going_out(ball, vel):
    if not field.BALL_OUT_AT_LINE:
        return False
    if ball_out(ball):
        return True
    if field.inside_play_area(ball[0], ball[1], OUT_SOON_DIST):
        return False
    if abs(ball[0]) > field.PLAY_W / 2 - OUT_SOON_DIST:
        n = (math.copysign(1.0, ball[0]), 0.0)
    else:
        n = (0.0, math.copysign(1.0, ball[1]))
    speed_outwards = vel[0] * n[0] + vel[1] * n[1]
    return speed_outwards > OUT_SOON_SPEED


def restart_spot(ball):
    return min(field.NEUTRAL_SPOTS, key=lambda spot: dist(spot, ball))


def shover(me, ball, centres):
    if dist(me, ball) > R + cfg.BALL_RADIUS + 6:
        return None
    to_ball = math.atan2(ball[1] - me[1], ball[0] - me[0])
    for c in centres:
        if dist(c, me) > 2 * R + PUSHBACK_GAP:
            continue
        to_it = math.atan2(c[1] - me[1], c[0] - me[0])
        same_side_as_ball = abs(angle_diff(to_it, to_ball)) < PUSHBACK_ANGLE
        not_behind_us = c[1] - me[1] > -0.3 * dist(c, me)
        if same_side_as_ball and not_behind_us:
            return c
    return None


def possession(world, mate):
    if world.ball_in_capture or (mate and mate.get("has ball")):
        return "us"
    if world.ball_field is not None and GoalieStrategy.carrier(world.ball_field, world.opponents) is not None:
        return "them"
    return None


def knock_out_aim(ball, me):
    if abs(ball[0]) < 10:
        side = math.copysign(1.0, ball[0] - me[0])
    else:
        side = math.copysign(1.0, ball[0])
    return (side * field.PLAY_W / 2, ball[1] + 2 * CHALLENGE_UP)


def kick_out_side(me, ball, centres, current=None, shover=None):
    def clear(side):
        u = unit((side, GOALIE_OUT_UP))
        kick_end = (me[0] + u[0] * GOALIE_OUT_CLEAR, me[1] + u[1] * GOALIE_OUT_CLEAR)
        return clearance(me, kick_end, centres) > 0
    if current is not None and clear(current):
        return current
    if shover is not None and abs(shover[0] - me[0]) > 2.0:
        near = -math.copysign(1.0, shover[0] - me[0])
    else:
        near = math.copysign(1.0, ball[0] or me[0] or 1.0)
    if clear(near) or not clear(-near):
        return near
    return -near


def pass_aim(me, striker, centres):
    others = [c for c in centres if dist(c, striker) > R]
    cands = []
    for side in (1.0, -1.0):
        p = clamp_in_field([striker[0] + side * PASS_SIDE, striker[1] + PASS_AHEAD])
        cands.append((clearance(me, p, others), p))
    open_ = [c for c in cands if c[0] > KICK_CLEARANCE]
    if len(open_) == 1:
        return open_[0][1]
    if open_:
        return min(open_, key=lambda c: abs(c[1][0]))[1]
    return max(cands, key=lambda c: c[0])[1]


def rel(p, pose):
    return [p[0] - pose.x, p[1] - pose.y]


COMMS_GRACE = 10.0


class MateMemory:

    def __init__(self):
        self.on_at = -1e9

    def __call__(self, mate):
        now = time.monotonic()
        if mate:
            if mate.get("bot active") == 1:
                self.on_at = now
            return mate
        if now - self.on_at < COMMS_GRACE:
            return {"bot active": 1, "command": 0}
        return mate


def mate_active(mate):
    return bool(mate) and mate.get("bot active") == 1


def box_limit(partly_ok=True):
    return R - BOX_POKE if partly_ok else -(R + BOX_KEEP_INT)


def box_out_dir(p, sgn):
    e = 0.5
    gx = field.penalty_depth(p[0] + e, p[1], sgn) - field.penalty_depth(p[0] - e, p[1], sgn)
    gy = field.penalty_depth(p[0], p[1] + e, sgn) - field.penalty_depth(p[0], p[1] - e, sgn)
    d = unit((-gx, -gy))
    return d if d != (0.0, 0.0) else (0.0, float(-sgn))


BOX_DETOUR = 10.0


def box_guard(me, target, limits, brake=0.0):
    here = {s: field.penalty_depth(me[0], me[1], s) for s in limits}

    def depth_over(p):
        over = -1e9
        for s, lim in limits.items():
            dep = field.penalty_depth(p[0], p[1], s)
            b = brake if dep > here[s] + 0.5 else 0.0
            over = max(over, dep - lim + b)
        return over

    for s, lim in limits.items():
        over = field.penalty_depth(me[0], me[1], s) - lim
        if over > 0:
            d = box_out_dir(me, s)
            return [me[0] + d[0] * (over + 8.0), me[1] + d[1] * (over + 8.0)]
    target = list(target)
    for _ in range(3):
        worst = max(limits, key=lambda s: field.penalty_depth(target[0], target[1], s) - limits[s])
        over = field.penalty_depth(target[0], target[1], worst) - limits[worst]
        if over <= 0:
            break
        d = box_out_dir(target, worst)
        target = [target[0] + d[0] * (over + 0.5), target[1] + d[1] * (over + 0.5)]
    n = int(dist(me, target))
    last = list(me)
    for i in range(1, n + 1):
        p = [me[0] + (target[0] - me[0]) * i / n, me[1] + (target[1] - me[1]) * i / n]
        if depth_over(p) > 0:
            if dist(last, me) < 2.0:
                s = max(limits, key=lambda s: field.penalty_depth(p[0], p[1], s) - limits[s])
                return [me[0], me[1] - s * BOX_DETOUR]
            return last
        last = p
    return target


class BoxGuard:
    BRAKE_WINDOW = 0.15

    def __init__(self, brake=True):
        self.brake = brake
        self.poses = []
        self.speed = 0.0

    def __call__(self, out, pose, limits):
        if pose is None or not pose.confident:
            self.poses = []
            return out
        now = time.monotonic()
        self.poses.append((now, pose.x, pose.y))
        while len(self.poses) > 2 and now - self.poses[1][0] >= self.BRAKE_WINDOW:
            self.poses.pop(0)
        t0, x0, y0 = self.poses[0]
        if now - t0 >= self.BRAKE_WINDOW * 0.5:
            self.speed = math.hypot(pose.x - x0, pose.y - y0) / (now - t0)
        move, heading, dribbler = out
        me = (pose.x, pose.y)
        brake = min(BRAKE_MAX, self.speed * BRAKE_TIME) if self.brake else 0.0
        target = box_guard(me, (me[0] + move[0], me[1] + move[1]), limits, brake)
        return rel(target, pose), heading, dribbler


class GoalieStrategy:

    def __init__(self, has_dribbler=True):
        self.has_dribbler = has_dribbler
        self.fallback = GoalieFallback()
        self.botstate_hyst = Hysteresis(hold_time=0.11, instant_enter=lambda v: v == 1)
        self.botstate = 0
        self.speed = {}
        self.kick = False
        self.chasing = False
        self.phase = ""
        self.possession = None
        self.out_side = None
        self.kicked_at = -1e9
        self.out_ball = None
        self.sealing = False
        self.alone = None
        self.avoid = Avoider()
        self.mate_mem = MateMemory()
        self.box_guard = BoxGuard(brake=False)
        self.clear_plan = None
        self.clear_plan_at = -1e9

    def reset(self):
        self.fallback.reset()
        self.botstate_hyst.reset()

    def update(self, world, compass, mate, line_touches):
        planned_at = self.clear_plan_at
        out = self._update(world, compass, mate, line_touches)
        if self.clear_plan is not None and self.clear_plan_at == planned_at:
            now = time.monotonic()
            if not (self.kicked_at >= self.clear_plan_at and now - self.kicked_at < CLEAR_SHARE_AFTER):
                self.clear_plan = None
        return self.box_guard(out, world.pose, {-1: box_limit(), 1: box_limit()})

    def _update(self, world, compass, mate, line_touches):
        mate = self.mate_mem(mate)
        pose = world.pose
        self.kick = False
        self.possession = possession(world, mate)

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
        ball = world.ball_field
        goal = (0.0, field.OWN_GOAL_Y)
        striker_on = mate_active(mate)

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

        if ball is not None:
            if going_out(ball, world.ball_vel):
                self.out_ball = (list(ball), time.monotonic())
            elif not ball_out(ball):
                self.out_ball = None

        if ball is None:
            raw = 0
        elif ball_out(ball):
            raw = 4
        elif world.ball_in_capture and self.has_dribbler and self.carrier(ball, world.opponents) is None:
            raw = 1
        else:
            in_box = abs(ball[0]) < field.PENALTY_W / 2 and ball[1] < field.OWN_GOAL_Y + field.PENALTY_D
            told = striker_on and mate.get("command") == 1
            clearing = self.botstate in (1, 2)
            opp_on_ball = self.possession == "them"
            pushed_at_goal = dist(ball, goal) < PUSHER_DIST and world.ball_vel[1] < -PUSHER_SPEED
            striker_chasing_it = striker_on and mate.get("command") == 0 and mate.get("chasing")
            just_kicked = time.monotonic() - self.kicked_at <= KICK_RETURN_T
            if not striker_on:
                closest_opp = min((dist(c, ball) for c in world.opponents), default=1e9)
                if ball[1] < SOLO_CHASE_Y or dist(me, ball) + SOLO_CLOSER < closest_opp:
                    raw = 2
                else:
                    raw = 3
            elif opp_on_ball and dist(ball, goal) < CHALLENGE_DIST:
                raw = 2
            elif opp_on_ball and not pushed_at_goal:
                raw = 3
            elif told or (in_box and dist(me, ball) < DANGER_DIST):
                raw = 2
            elif clearing and ball[1] < CLEAR_RETURN_Y and not striker_chasing_it and not just_kicked:
                raw = 2
            else:
                raw = 3
        too_far_out = me[1] > field.OWN_GOAL_Y + field.PENALTY_D + GOALIE_MAX_OUT
        free_spot_ball = False
        if SPOT_TAKE and ball is not None and not world.ball_in_capture and ball[1] < 0:
            on_spot = min(dist(ball, q) for q in field.NEUTRAL_SPOTS) < SPOT_TAKE_ON
            still = math.hypot(*world.ball_vel) < SPOT_TAKE_STILL
            others = list(world.opponents) + ([mate["pos"]] if striker_on and mate.get("pos") else [])
            nobody = all(dist(o, ball) > SPOT_TAKE_CLEAR for o in others)
            free_spot_ball = on_spot and still and nobody and dist(me, ball) < SPOT_TAKE_DIST
            if free_spot_ball:
                raw = 2
        if raw == 2 and striker_on and too_far_out and not told and not free_spot_ball:
            raw = 3
        self.botstate = self.botstate_hyst.update(raw)
        if ball is None and self.botstate != 0:
            self.botstate = 0
        self.chasing = self.botstate in (1, 2)

        centres = obstacle_centres(me, world.obstacles_field)
        self.sealing = (ball is not None and not world.ball_in_capture
                        and self.wide_threat(ball, centres, SEAL_KEEP_X if self.sealing else SEAL_X))
        if self.sealing:
            target = self.seal_spot(ball)
            face = heading_for((ball[0] - me[0], ball[1] - me[1]))
            heading = max(-GOALIE_FACE_MAX, min(GOALIE_FACE_MAX, face))
            target = around_ball(me, target, ball)
            target = deep_no_touch(me, target, ball, world.opponents)
            self.speed = {"spd_multi": 1.0, "smooth": GUARD_SMOOTH}
            self.phase = "seal post"
            return rel(target, pose), heading, False

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
            d = unit((sh[0] - me[0], sh[1] - me[1]))
            target = clamp_in_field([me[0] + d[0] * PUSHBACK_REACH, me[1] + d[1] * PUSHBACK_REACH], OUT_MARGIN)
            self.speed = {"spd_multi": 1.0}
            self.phase = "push back"
            return rel(target, pose), 0.0, False

        dribbler = False
        self.speed = {}

        block = None
        if ball is not None and self.botstate != 1 and not contested(ball, obstacle_centres(me, world.obstacles_field)):
            block = self.block_point(ball, world.ball_vel)
        if block is not None:
            target = block
            if abs(block[0] - me[0]) < BLOCK_STAY:
                target = [me[0], block[1]]
            face = heading_for((ball[0] - me[0], ball[1] - me[1]))
            heading = max(-GOALIE_FACE_MAX, min(GOALIE_FACE_MAX, face))
            self.phase = "block"
            self.speed = {"spd_multi": min(1.0, max(0.1, dist(me, target) / BLOCK_GAIN)), "smooth": GUARD_SMOOTH}
            return rel(target, pose), heading, False

        ball_is_out = ball is None or ball_out(ball)
        if (GOALIE_LOOK_AT_SPOT and self.out_ball is not None and striker_on and ball_is_out
                and time.monotonic() - self.out_ball[1] < RESTART_WAIT):
            p = restart_spot(self.out_ball[0])
            others = [q for q in field.NEUTRAL_SPOTS if dist(q, p) > 1]
            if abs(p[0]) > 1:
                q = max(others, key=lambda q: dist(q, p))
            else:
                q = min(others, key=lambda q: abs(q[0] - me[0]))
            u = unit((0.0 - q[0], field.OWN_GOAL_Y - q[1]))
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
            target, heading = [ball[0] * STEP_UP_X_SHARE, STEP_UP_Y], 0.0
            self.speed = {"spd_multi": 1.0}
            self.phase = "step up"
        elif self.botstate in (3, 4):
            if self.botstate == 3:
                b = (ball[0] + world.ball_vel[0] * GUARD_LEAD, ball[1] + world.ball_vel[1] * GUARD_LEAD)
            else:
                b = restart_spot(ball)
            target, heading = self.guard(me, b)
            carrier = None
            if self.botstate == 3 and CARRIER_WEIGHT:
                carrier = self.carrier(ball, world.opponents)
            if carrier is not None:
                target = self.shot_line_guard(target, b, carrier)
            self.speed = guard_speed(me, target)
            if self.botstate == 3:
                target = deep_no_touch(me, around_ball(me, target, ball), ball, world.opponents)
                self.phase = "guard carrier" if carrier is not None else "guard"
            else:
                self.phase = "ball out"
        else:
            centres = obstacle_centres(me, world.obstacles_field)
            aim, clear = best_aim(ball, centres)
            if self.botstate == 1:
                target, heading = self.clear_ball(world, compass, mate, me, aim, centres, striker_on)
                dribbler = True
                if CLEAR_SHARE:
                    self.clear_plan = [round(me[0]), round(me[1]), round(heading, 2)]
                    self.clear_plan_at = time.monotonic()
            else:
                target, heading, dribbler = self.take_ball(world, me, ball, aim, centres)
        return rel(target, pose), heading, dribbler

    def clear_ball(self, world, compass, mate, me, aim, centres, striker_on):
        if BOX_TURN and me[1] < field.OWN_GOAL_Y + field.PENALTY_D + R:
            return self.box_clear(compass, me, centres)
        target = list(me)
        striker = mate.get("pos") if striker_on else None
        fwd = (-math.sin(compass), math.cos(compass))
        pressed = any(dist(o, me) < 2 * R + PRESSED_GAP for o in world.opponents)
        if pressed:
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
            p = pass_aim(me, striker, centres)
            heading = heading_for((p[0] - me[0], p[1] - me[1]))
            err = abs(angle_diff(heading, compass))
            others = [c for c in centres if dist(c, striker) > R]
            ahead = (me[0] + fwd[0] * GOALIE_CLEAR_PATH, me[1] + fwd[1] * GOALIE_CLEAR_PATH)
            self.kick = err < PASS_TOL and clearance(me, ahead, others) > 0
            self.phase = "clear pass"
        else:
            want = heading_for((aim[0] - me[0], aim[1] - me[1]))
            heading = max(-GOALIE_KICK_TURN, min(GOALIE_KICK_TURN, want))
            ahead = (me[0] + fwd[0] * GOALIE_KICK_CLEAR, me[1] + fwd[1] * GOALIE_KICK_CLEAR)
            self.kick = abs(compass) < GOALIE_KICK_TURN + 0.1 and clearance(me, ahead, centres) > 0
            self.phase = "clear kick"
        return target, heading

    def box_clear(self, compass, me, centres):
        side = math.copysign(1.0, me[0] or 1.0)

        def path(dvec):
            u = unit(dvec)
            return clearance(me, (me[0] + u[0] * GOALIE_CLEAR_PATH, me[1] + u[1] * GOALIE_CLEAR_PATH), centres)
        cands = []
        for dx, dy in BOX_KICK_DIRS:
            for sd in ((side, -side) if dx else (1.0,)):
                cands.append((dx * sd, dy))
        keep = getattr(self, "box_dir", None) if self.phase == "box clear" else None
        if keep is not None and path(keep) > 0:
            best = keep
        else:
            open_ = [c for c in cands if path(c) > 0]
            best = open_[0] if open_ else max(cands, key=path)
        self.box_dir = best
        heading = heading_for(best)
        fwd = (-math.sin(compass), math.cos(compass))
        ahead = (me[0] + fwd[0] * GOALIE_CLEAR_PATH, me[1] + fwd[1] * GOALIE_CLEAR_PATH)
        self.kick = (abs(angle_diff(heading, compass)) < BOX_KICK_TOL and fwd[1] > 0.0
                     and clearance(me, ahead, centres) > 0)
        self.phase = "box clear"
        return list(me), heading

    def take_ball(self, world, me, ball, aim, centres):
        in_box = abs(ball[0]) < field.PENALTY_W / 2 + cfg.BALL_RADIUS and ball[1] < field.OWN_GOAL_Y + field.PENALTY_D
        too_deep = abs(ball[0]) < field.GOAL_W / 2 + R and ball[1] - field.OWN_GOAL_Y < BOX_TURN_MIN_D
        contested = any(dist(o, ball) < BOX_TURN_OPP for o in world.opponents)
        if BOX_TURN and in_box and ball[1] < me[1] - BOX_TURN_BEHIND and not too_deep and not contested:
            u = unit((me[0] - ball[0], me[1] - ball[1]))
            reach = R + cfg.BALL_RADIUS - 2.0
            target = [ball[0] + u[0] * reach, ball[1] + u[1] * reach]
            heading = heading_for((ball[0] - me[0], ball[1] - me[1]))
            d = dist(me, ball)
            self.speed = {"spd_multi": BOX_TURN_SPEED if d < R + 20 else catch_speed(me, ball)}
            self.phase = "box turn"
            return target, heading, True
        defending = ball[1] < CLEAR_AWAY_Y
        if self.possession == "them":
            aim, defending = knock_out_aim(ball, me), True
        elif defending:
            aim = clear_aim(ball)
        target, heading, dribbler = self.go_for_ball(me, ball, world.ball_vel, aim, centres, defending)
        if self.phase == "orbit" and ball[1] > field.OWN_GOAL_Y + R + 3:
            target, heading = self.guard(me, ball)
            self.speed = guard_speed(me, target)
            target = around_ball(me, target, ball)
            self.phase = "guard (ball behind)"
            dribbler = False
        return deep_no_touch(me, target, ball, world.opponents), heading, dribbler

    def note_kick(self):
        if self.kick:
            self.kicked_at = time.monotonic()

    @staticmethod
    def carrier(ball, opponents):
        closest = min(opponents, key=lambda o: dist(o, ball), default=None)
        if closest is not None and dist(closest, ball) < R + OPP_ON_BALL:
            return closest
        return None

    @staticmethod
    def shot_line_guard(target, ball, carrier):
        d = unit((ball[0] - carrier[0], ball[1] - carrier[1]))
        if d[1] > -0.2:
            return target
        x_goal = ball[0] + d[0] * (field.OWN_GOAL_Y - ball[1]) / d[1]
        if abs(x_goal) > field.GOAL_W / 2 + CARRIER_WIDEN:
            return target
        x_line = ball[0] + d[0] * (target[1] - ball[1]) / d[1]
        x = target[0] + CARRIER_WEIGHT * (x_line - target[0])
        lim = field.PENALTY_W / 2 - R
        return [max(-lim, min(lim, x)), target[1]]

    @staticmethod
    def wide_threat(ball, centres, min_x=SEAL_X):
        if not POST_SEAL or ball[1] > field.OWN_GOAL_Y + SEAL_Y:
            return False
        inner_x = SEAL_INNER_KEEP_X if min_x < SEAL_X else SEAL_INNER_X
        if abs(ball[0]) < min(min_x, inner_x):
            return False
        return any(dist(c, ball) < SEAL_OPP for c in centres)

    @staticmethod
    def seal_spot(ball):
        if SEAL_SIDE and abs(ball[0]) - 2.0 >= SEAL_SIDE_X:
            return [math.copysign(SEAL_SIDE_X, ball[0]), field.OWN_GOAL_Y + R + SEAL_GAP]
        return [math.copysign(min(SEAL_POST_X, abs(ball[0]) - 2.0), ball[0]), field.OWN_GOAL_Y + R + SEAL_GAP]

    @staticmethod
    def block_point(ball, vel):
        if vel[1] > -BLOCK_SPEED:
            return None
        y_line = field.OWN_GOAL_Y + GOALIE_LINE_D
        if ball[1] < y_line:
            return None
        t_goal = (field.OWN_GOAL_Y - ball[1]) / vel[1]
        x_goal = ball[0] + vel[0] * t_goal
        if t_goal > BLOCK_TIME or abs(x_goal) > field.GOAL_W / 2 + BLOCK_WIDEN:
            return None
        t = (y_line - ball[1]) / vel[1]
        x = ball[0] + vel[0] * t
        lim = field.GOAL_W / 2 + R
        return [max(-lim, min(lim, x)), y_line]

    @staticmethod
    def guard(me, b):
        goal = (0.0, field.OWN_GOAL_Y)
        face = heading_for((b[0] - me[0], b[1] - me[1]))
        bd = b[1] - goal[1]
        if GUARD_BISECTOR and bd > GOALIE_LINE_D + R:
            half = field.GOAL_W / 2 - cfg.BALL_RADIUS
            u1, u2 = unit((-half - b[0], -bd)), unit((half - b[0], -bd))
            u = unit((u1[0] + u2[0], u1[1] + u2[1]))
            d = max(GOALIE_LINE_D, min(GUARD_COVER * bd, GUARD_D_MAX, bd - R - 5.0))
            t = (bd - d) / max(-u[1], 1e-3)
            lim = field.PENALTY_W / 2 - R
            x = max(-lim, min(lim, b[0] + u[0] * t))
            return [x, goal[1] + d], max(-GOALIE_FACE_MAX, min(GOALIE_FACE_MAX, face))
        v = unit((b[0] - goal[0], b[1] - goal[1]))
        out = min(max(0.0, (dist(b, goal) - GUARD_OUT_START) * GUARD_OUT_PER_CM), GUARD_OUT_MAX - GUARD_RADIUS)
        radius = GUARD_RADIUS + out
        x = goal[0] + v[0] * radius
        y = max(goal[1] + v[1] * radius, goal[1] + R + 3)
        lim = field.PENALTY_W / 2 - R
        x = max(-lim, min(lim, x))
        face = heading_for((b[0] - me[0], b[1] - me[1]))
        return [x, y], max(-GOALIE_FACE_MAX, min(GOALIE_FACE_MAX, face))

    def go_for_ball(self, me, ball, vel, aim, centres, defending=False):
        b = lead_ball(me, ball, vel)
        target, heading, self.phase = approach(me, b, choose_push_aim(b, aim, defending))
        if self.phase != "push":
            target = around_ball(me, target, b)
        if self.phase in ("orbit", "line up"):
            self.speed = {"spd_min": GOALIE_ORBIT_SPD_MIN}
        if self.phase == "push":
            race = any(dist(c, ball) < dist(me, ball) + RACE_MARGIN for c in centres)
            self.speed = {"spd_multi": PUSH_MULTI if race else catch_speed(me, ball)}
        nav = b if self.phase == "push" else target
        wp = self.avoid(me, nav, centres, ignore_near=b)
        if wp is not nav:
            target = wp
        return target, heading, dist(me, ball) < DRIBBLER_RANGE


CHASE_EXAGGERATE = 1.7
CHASE_FACE_DIST = 55.0
CHASE_FACE_ANGLE = 60.0
BANK_SHOT = False
BANK_RANGE = 150.0
BANK_ROOM = 4.0
BANK_BOUNCE = 0.5
BANK_MIN_FY = 0.3
FACE_BALL = True
FACE_BALL_ALL = False
OUR_CORNER_X = field.GOAL_W / 2 + 5.0
OUR_CORNER_Y = 45.0
TURN_AWAY = True
TURN_AWAY_Y = 60.0
TURN_AWAY_FY = -0.2
TURN_AWAY_SPEED = 0.4
TURN_AWAY_DONE = math.radians(30)
TURN_AWAY_SIDE_FY = -0.3
TURN_AFTER_EDGE = True
TURN_EDGE_TIME = 0.5
FACE_BALL_SPEED = 0.5
EDGE_CAPTURE = True
EDGE_CAPTURE_RANGE = 45.0
EDGE_CAPTURE_SPEED = 0.6
EDGE_SKIP_CORNERS = True
EDGE_CORNER = 10.0
EDGE_OUR_CORNERS = False
EDGE_WALL_GAP = 2.0
CARRY_FACING = math.radians(10)
CARRY_KICK_HOLD = 0.2
CARRY_KICK_RANGE = 70.0
QUICK_SHOT_TIME = 1.0
QUICK_SHOT_RANGE = 130.0
CHARGED_DIST = 50.0
CHARGED_SPEED = 30.0
OPEN_SHOT_RANGE = 120.0
LONG_SHOT_FROM = 70.0
LONG_SHOT_ROOM = cfg.BALL_RADIUS + 4.0
HEAD_ON_GAP = 8.0
HEAD_ON_ANGLE = math.radians(40)
SLIP_SIDE = 25.0
SLIP_FORWARD = 15.0
BOX_KEEP = R + 2.0
CARRY_FRONT_CONE = math.radians(60)
CARRY_CONTACT_RANGE = 60.0
CARRY_LEAD = 0.3
CARRY_TURN_MAX = math.radians(120)
CARRY_TURN_STEP = math.radians(10)
CARRY_CLOSING = 20.0
SHIELD = False
SHIELD_IN = R
SHIELD_FACE_BACK = 0.0
SHIELD_PUSH = 15.0
SHIELD_PUSH_ACROSS = 0.6
SHIELD_SHOT_CLEAR = 4.0
SHIELD_SHOT_KEEP = 0.0
SHIELD_SHOT_SPD = 0.3
SHIELD_KICK_CLEAR = 3.0
SHIELD_MAX_TIME = 3.0
CARRY_TO_CORNER = False
CORNER_INSET = 5.0
CORNER_REACHED = 8.0
CORNER_RANGE = 50.0
CORNER_SPD = 0.35
CORNER_TURN_SHARE = 0.3
CARRY_AIM_CORNER = True
SHIELD_STAY = 6.0
BASE_SHIELD = False
BASE_SHIELD_Y = 40.0
BASE_SHIELD_EDGE = 12.0
BASE_SHIELD_T = 0.3
FEINT = True
FEINT_RANGE = 120.0
FEINT_KEEPER_D = 50.0
FEINT_MID = 4.0
FEINT_AIM_IN = 10.0
FEINT_ROOM = 2.0
FEINT_SPD = 0.7
FEINT_MAX_TIME = 1.5
RESTART_SCREEN = True
SCREEN_TIME = 3.0
SCREEN_OPP = 60.0
RESTART_CATCH = True
CATCH_RESTART_TIME = 1.5
CATCH_RESTART_OPP = 70.0
RESTART_RUSH = False
RUSH_TIME = 4.0
RUSH_SPOT = 8.0
RUSH_BEHIND_X = 12.0
RUSH_FLICK_RANGE = 200.0
RUSH_CLEAR = 50.0
RUSH_TURN_TIME = 1.5
RUSH_TURN_FREE = math.radians(30)
RUSH_TURN_SLOWEST = math.radians(60)
RUSH_TURN_MIN_SPD = 0.4


BEHIND_ORBIT = True
BEHIND_ANGLE = 150.0
BEHIND_RANGE = 70.0
BEHIND_GAP = R + cfg.BALL_RADIUS + 10.0
BEHIND_CLOSE = 50.0
BEHIND_SIDESTEP = 110.0


def share(value, start, end):
    return min(1.0, max(0.0, (start - value) / (start - end)))


def chase(me, ball):
    d = dist(me, ball)
    a = math.degrees(math.atan2(ball[0] - me[0], ball[1] - me[1]))

    def soften(x, B=76.0):
        return x * (1 - 1 / (1 + abs(x / B) ** 3))

    t1 = share(d, 90.0, 20.0)
    further_round = max(-180.0, min(180.0, a * CHASE_EXAGGERATE))
    ang = a + (further_round - a) * t1
    t2 = share(d, 50.0, 20.0)
    ang = ang + (180.0 * soften(ang) / soften(180.0) - ang) * t2
    if not BEHIND_ORBIT and d < BEHIND_CLOSE and abs(a) > BEHIND_ANGLE:
        ang = math.copysign(BEHIND_SIDESTEP, a or 1.0)
    target = [me[0] + math.sin(math.radians(ang)) * 40, me[1] + math.cos(math.radians(ang)) * 40]
    in_our_corner = abs(ball[0]) > OUR_CORNER_X and ball[1] < field.OWN_GOAL_Y + OUR_CORNER_Y
    if FACE_BALL and (FACE_BALL_ALL or in_our_corner) and d <= CHASE_FACE_DIST and abs(a) > CHASE_FACE_ANGLE:
        u = unit((ball[0] - me[0], ball[1] - me[1]))
        reach = R + cfg.BALL_RADIUS - 2.0
        target = [ball[0] - u[0] * reach, ball[1] - u[1] * reach]
        spd = FACE_BALL_SPEED if d < 25 else 0.9
        return clamp_in_field(target, FIELD_MARGIN), math.radians(-a), d <= 33, spd
    if BEHIND_ORBIT and abs(a) > BEHIND_ANGLE and d < BEHIND_RANGE:
        dx = me[0] - ball[0]
        side = math.copysign(1.0, dx if abs(dx) > 2.0 else -ball[0] or 1.0)
        beside = [ball[0] + side * BEHIND_GAP, ball[1]]
        return clamp_in_field(around_ball(me, beside, ball), FIELD_MARGIN), 0.0, False, 1.0

    heading = 0.0
    if d <= CHASE_FACE_DIST and abs(a) <= CHASE_FACE_ANGLE:
        far_away = share(d, 22, 50)
        straight_ahead = max(0.0, 1 - 1.5 * abs(a) / 90)
        t = far_away * straight_ahead
        heading = math.radians(-a) * (1 - t)

    spd = 0.8 + 0.2 * share(d, 30, 45)
    if d < 25:
        spd *= 0.6

    dribbler = d <= 33 and abs(a) <= 45
    return clamp_in_field(target, FIELD_MARGIN), heading, dribbler, spd


def edge_capture(me, ball):
    if not EDGE_CAPTURE or dist(me, ball) > EDGE_CAPTURE_RANGE:
        return None
    on_line = cfg.BALL_RADIUS + field.LINE_W
    over_x = abs(ball[0]) - (field.PLAY_W / 2 - on_line)
    over_y = abs(ball[1]) - (field.PLAY_L / 2 - on_line)
    if abs(ball[0]) <= field.GOAL_W / 2:
        over_y = -1.0
    if over_x <= 0 and over_y <= 0:
        return None
    if EDGE_SKIP_CORNERS and over_x > -EDGE_CORNER and over_y > -EDGE_CORNER and not (EDGE_OUR_CORNERS and ball[1] < 0):
        return None
    n = unit((math.copysign(1.0, ball[0]) if over_x > 0 else 0.0, math.copysign(1.0, ball[1]) if over_y > 0 else 0.0))
    reach = R + cfg.BALL_RADIUS - 1.0
    target = clamp_in_field([ball[0] - n[0] * reach, ball[1] - n[1] * reach], OUT_MARGIN + EDGE_WALL_GAP)
    heading = heading_for((ball[0] - me[0], ball[1] - me[1]))
    spd = EDGE_CAPTURE_SPEED if dist(me, ball) < 25 else 0.9
    return around_ball(me, target, ball), heading, True, spd


def shot_open(compass, me, centres, max_range, room=0.0):
    fx, fy = -math.sin(compass), math.cos(compass)
    if fy < 0.2:
        return False
    t = (field.ATTACK_GOAL_Y - me[1]) / fy
    cross = (me[0] + fx * t, field.ATTACK_GOAL_Y)
    in_range = 0 < t < max_range
    between_posts = abs(cross[0]) <= field.GOAL_W / 2 - cfg.BALL_RADIUS - POST_MARGIN
    if not (in_range and between_posts):
        return False
    return clearance(me, cross, centres) > room


def bank_shot_open(compass, me, centres, max_range=BANK_RANGE, room=BANK_ROOM):
    if not BANK_SHOT:
        return False
    fx, fy = -math.sin(compass), math.cos(compass)
    if fy < BANK_MIN_FY or abs(fx) < 0.05:
        return False
    xw = math.copysign(field.WALL_W / 2 - cfg.BALL_RADIUS, fx)
    t1 = (xw - me[0]) / fx
    wall = (xw, me[1] + fy * t1)
    if t1 <= 0 or wall[1] >= field.ATTACK_GOAL_Y:
        return False
    t2 = (field.ATTACK_GOAL_Y - wall[1]) / fy
    cross = (wall[0] - BANK_BOUNCE * fx * t2, field.ATTACK_GOAL_Y)
    if t1 + t2 > max_range or abs(cross[0]) > field.GOAL_W / 2 - cfg.BALL_RADIUS - POST_MARGIN:
        return False
    return clearance(me, wall, centres) > room and clearance(wall, cross, centres) > room


def clear_shot(compass, me, centres, max_range):
    far = dist(me, (0.0, field.ATTACK_GOAL_Y)) > LONG_SHOT_FROM
    return shot_open(compass, me, centres, max_range, LONG_SHOT_ROOM if far else 0.0)


def head_on(me, compass, opponents):
    fwd = (-math.sin(compass), math.cos(compass))
    for o in opponents:
        d = dist(o, me)
        touching = d < 2 * R + HEAD_ON_GAP
        cos_angle = ((o[0] - me[0]) * fwd[0] + (o[1] - me[1]) * fwd[1]) / max(d, 1e-6)
        if touching and cos_angle > math.cos(HEAD_ON_ANGLE):
            return o
    return None


def charged(me, opponents, velocities):
    for o, v in zip(opponents, velocities):
        to_me = unit((me[0] - o[0], me[1] - o[1]))
        closing_speed = v[0] * to_me[0] + v[1] * to_me[1]
        if dist(o, me) < CHARGED_DIST and closing_speed > CHARGED_SPEED:
            return True
    return False


def carry_dir(me, goal, opponents, velocities):
    to_goal = math.atan2(goal[1] - me[1], goal[0] - me[0])

    near = []
    for o, v in zip(opponents, velocities):
        soon = (o[0] + v[0] * CARRY_LEAD, o[1] + v[1] * CARRY_LEAD)
        to_me = unit((me[0] - o[0], me[1] - o[1]))
        closing_speed = v[0] * to_me[0] + v[1] * to_me[1]
        if closing_speed < CARRY_CLOSING:
            continue
        if dist(soon, me) < CARRY_CONTACT_RANGE or dist(o, me) < CARRY_CONTACT_RANGE:
            near.append(math.atan2(soon[1] - me[1], soon[0] - me[0]))
    if not near:
        return (math.cos(to_goal), math.sin(to_goal))

    best = None
    n = int(CARRY_TURN_MAX / CARRY_TURN_STEP + 0.5)
    for k in sorted(range(-n, n + 1), key=abs):
        a = to_goal + k * CARRY_TURN_STEP
        worst = min(abs(angle_diff(b, a)) for b in near)
        if worst >= CARRY_FRONT_CONE:
            return (math.cos(a), math.sin(a))
        if best is None or worst > best[0]:
            best = (worst, a)
    return (math.cos(best[1]), math.sin(best[1]))


def keep_out_of_box(target):
    front = field.OWN_GOAL_Y + field.PENALTY_D + BOX_KEEP
    if abs(target[0]) < field.PENALTY_W / 2 + BOX_KEEP and target[1] < front:
        return [target[0], front]
    return target


def meet_clear(me, plan):
    src = (plan[0], plan[1])
    d = (-math.sin(plan[2]), math.cos(plan[2]))
    roll = CLEAR_ROLL_SPEED / CLEAR_ROLL_DECAY * 0.95
    half_x, half_y = field.PLAY_W / 2 - FIELD_MARGIN, field.PLAY_L / 2 - FIELD_MARGIN
    if abs(d[0]) > 1e-6:
        roll = min(roll, max(0.0, (math.copysign(half_x, d[0]) - src[0]) / d[0]))
    if abs(d[1]) > 1e-6:
        roll = min(roll, max(0.0, (math.copysign(half_y, d[1]) - src[1]) / d[1]))
    s = roll
    for i in range(int(CLEAR_MEET_MIN), int(roll), 5):
        p = (src[0] + d[0] * i, src[1] + d[1] * i)
        ball_t = -math.log(1.0 - i * CLEAR_ROLL_DECAY / CLEAR_ROLL_SPEED) / CLEAR_ROLL_DECAY
        if dist(me, p) / CLEAR_MEET_SPEED <= ball_t:
            s = i
            break
    p = (src[0] + d[0] * s, src[1] + d[1] * s)
    cross = (me[0] - src[0]) * d[1] - (me[1] - src[1]) * d[0]
    side = 1.0 if cross >= 0 else -1.0
    target = [p[0] + d[1] * side * CLEAR_MEET_SIDE, p[1] - d[0] * side * CLEAR_MEET_SIDE]
    return clamp_in_field(target, FIELD_MARGIN)


class StrikerStrategy:

    def __init__(self):
        self.fallback = StrikerFallback()
        self.avoid = Avoider()
        self.kick = False
        self.speed = {}
        self.chasing = True
        self.botstate = 0
        self.phase = ""
        self.command = 0
        self.possession = None
        self.facing_since = None
        self.won_at = None
        self.out_at = None
        self.rushed_at = None
        self.shield_side = None
        self.shield_since = None
        self.shield_shot = False
        self.shield_base = False
        self.feint_since = None
        self.feint_side = None
        self.turn_away_side = None
        self.edge_at = None
        self.last_ball = None
        self.mate_mem = MateMemory()
        self.goalie_on = False
        self.box_guard = BoxGuard()

    def reset(self):
        self.fallback.reset()
        self.last_ball = None
        self.facing_since = None
        self.won_at = None

    def update(self, world, compass, mate, line_touches):
        out = self._update(world, compass, mate, line_touches)
        return self.box_guard(out, world.pose, {-1: box_limit(partly_ok=not self.goalie_on), 1: box_limit()})

    def _update(self, world, compass, mate, line_touches):
        mate = self.mate_mem(mate)
        pose = world.pose
        self.kick = False
        self.command = 0
        self.possession = possession(world, mate)

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
            fwd = (-math.sin(compass), math.cos(compass))
            ball = (me[0] + fwd[0] * (R + cfg.BALL_RADIUS), me[1] + fwd[1] * (R + cfg.BALL_RADIUS))
        if ball is not None:
            went_out = going_out(ball, world.ball_vel)
            remembered = restart_spot(ball) if went_out else ball
            self.last_ball = (remembered, time.monotonic(), went_out)
        if not world.ball_in_capture:
            self.won_at = None
        centres = obstacle_centres(me, world.obstacles_field)
        goalie_on = mate_active(mate)
        self.goalie_on = goalie_on
        self.speed = {}

        gone_out = (field.BALL_OUT_AT_LINE and ball is not None
                    and not field.inside_play_area(ball[0], ball[1], -cfg.BALL_RADIUS - 3.0))
        spot_obstacles = list(field.NEUTRAL_SPOTS) if me[1] > 0 else []
        if ball is None or gone_out:
            if self.last_ball is not None and self.last_ball[2]:
                self.out_at = time.monotonic()
            self.chasing = False
            dribbler = False
            if ball is not None:
                spot = restart_spot(ball)
                target = clamp_in_field([spot[0] * RESTART_X_SHARE, spot[1] - NEUTRAL_WAIT], R)
                target = self.avoid(me, target, centres + spot_obstacles)
                heading = 0.0
                self.botstate, self.phase = 5, "ball out"
            else:
                target, heading = self.no_ball(world, me, centres + spot_obstacles, goalie_on)
        elif self.meeting_clear(world, mate, me, ball, goalie_on):
            target = self.avoid(me, meet_clear(me, mate["clear"]), centres)
            heading = heading_for((0.0 - me[0], field.ATTACK_GOAL_Y - me[1]))
            dribbler = False
            self.chasing = True
            self.speed = {"spd_multi": 1.0}
            self.botstate, self.phase = 2, "meet clear"
        else:
            target, heading, dribbler = self.attack(world, compass, me, ball)
            self.chasing = True
            self.botstate = 1 if world.ball_in_capture else 2
            if self.phase not in ("edge capture", "turn away"):
                target = clamp_in_field(target, FIELD_MARGIN)

        if goalie_on:
            target = keep_out_of_box(target)
        return rel(target, pose), heading, dribbler

    @staticmethod
    def meeting_clear(world, mate, me, ball, goalie_on):
        if not (CLEAR_SHARE and goalie_on and mate and mate.get("clear")) or world.ball_in_capture:
            return False
        return bool(mate.get("has ball")) or dist(me, ball) > CLEAR_MEET_NEAR

    def no_ball(self, world, me, centres, goalie_on):
        heading = 0.0
        if not goalie_on:
            self.botstate, self.phase = 3, "defend"
            return [0.0, field.OWN_GOAL_Y + GUARD_RADIUS + 10], heading
        self.botstate = 0
        lb = self.last_ball
        now = time.monotonic()
        if lb is not None and lb[2] and now - lb[1] < RESTART_WAIT:
            spot = lb[0]
            x = spot[0] * RESTART_X_SHARE - math.copysign(RESTART_SIDE, spot[0] or 1.0)
            target = self.avoid(me, [x, spot[1] - NEUTRAL_WAIT], centres)
            self.phase = "restart"
            their_keeper_home = any(dist(o, (0.0, field.ATTACK_GOAL_Y)) < 30 for o in world.opponents)
            if their_keeper_home:
                far = (-math.copysign(field.GOAL_W / 2 - 8.0, spot[0] or 1.0), field.ATTACK_GOAL_Y)
                heading = heading_for((far[0] - spot[0], far[1] - spot[1]))
        else:
            spot = restart_spot(lb[0]) if lb is not None else (0.0, 0.0)
            target = self.avoid(me, [spot[0] * RESTART_X_SHARE, spot[1] - NEUTRAL_WAIT], centres)
            self.phase = "wait"
            if dist(me, WAIT_POS) > 30:
                self.speed = {"spd_min": 0.6}
        return target, heading

    def restart_ball(self, ball, within=RUSH_TIME):
        if self.out_at is None:
            return False
        recently_out = time.monotonic() - self.out_at < within
        on_a_spot = min(dist(ball, spot) for spot in field.NEUTRAL_SPOTS) < RUSH_SPOT
        return recently_out and on_a_spot

    @staticmethod
    def in_their_box(p, margin=0.0):
        return abs(p[0]) <= field.PENALTY_W / 2 - margin and p[1] >= field.ATTACK_GOAL_Y - field.PENALTY_D + margin

    @classmethod
    def box_opponents(cls, world, me):
        if not cls.in_their_box(me):
            return list(world.opponents)
        return [o for o in world.opponents if cls.in_their_box(o)]

    @classmethod
    def box_centres(cls, world, me):
        centres = obstacle_centres(me, world.obstacles_field)
        if not cls.in_their_box(me):
            return centres
        return [c for c in centres if cls.in_their_box(c)]

    @staticmethod
    def box_corner(me):
        side = math.copysign(1.0, me[0] or 1.0)
        return [side * (field.PENALTY_W / 2 - 10.0), field.ATTACK_GOAL_Y - GOALIE_LINE_D]

    def carry_target(self, me, goal):
        corner = self.box_corner(me)
        if CARRY_AIM_CORNER and me[1] < corner[1] - CORNER_REACHED:
            return corner
        return goal

    def corner_setup(self, world, me, goal):
        if not world.opponents or self.in_their_box(me):
            return False
        keeper = min(world.opponents, key=lambda o: dist(o, goal))
        if not self.in_their_box(keeper):
            return False
        others = [o for o in world.opponents if o is not keeper]
        return all(o[1] < me[1] for o in others)

    def base_line_out(self, world, me, ball, keeper, staying):
        stay = SHIELD_STAY if staying else 0.0
        if ball[1] < field.ATTACK_GOAL_Y - BASE_SHIELD_Y - stay or abs(ball[0]) < field.GOAL_W / 2 - stay:
            return False
        if not self.in_their_box(keeper, -stay) or self.in_their_box(me, SHIELD_IN):
            return False
        if staying:
            return True
        if field.inside_play_area(ball[0], ball[1], BASE_SHIELD_EDGE):
            return False
        vel = world.ball_vel
        ahead = (ball[0] + vel[0] * BASE_SHIELD_T, ball[1] + vel[1] * BASE_SHIELD_T)
        return not field.inside_play_area(ahead[0], ahead[1])

    def shield(self, world, compass, me, ball, goal):
        now = time.monotonic()
        opponents = self.box_opponents(world, me)
        if self.shield_side == 0 or not opponents:
            return None
        keeper = min(opponents, key=lambda o: dist(o, goal))
        margin = SHIELD_IN if self.shield_side is None else SHIELD_IN - SHIELD_STAY
        in_box = self.in_their_box(me, margin) and self.in_their_box(keeper, margin)
        staying = self.shield_side is not None and self.shield_base
        base = BASE_SHIELD and not in_box and self.base_line_out(world, me, ball, keeper, staying)
        if not (in_box or base):
            self.shield_side = None
            return None
        if self.shield_side is None:
            self.shield_base = base
            dx = me[0] if abs(me[0]) > 3.0 else me[0] - keeper[0]
            self.shield_side = math.copysign(1.0, dx or 1.0)
            self.shield_since = now
            self.shield_shot = False
        elif now - self.shield_since > SHIELD_MAX_TIME:
            self.shield_side = 0
            return None
        self.facing_since = None
        side = self.shield_side
        centres = self.box_centres(world, me)

        aim, clear = best_aim(ball, centres)
        self.shield_shot = clear >= (SHIELD_SHOT_KEEP if self.shield_shot else SHIELD_SHOT_CLEAR)
        if self.shield_shot:
            if shot_open(compass, me, centres, field.PENALTY_D + 20.0, SHIELD_KICK_CLEAR):
                self.kick = True
            to_aim = unit((aim[0] - me[0], aim[1] - me[1]))
            self.speed = {"spd_multi": SHIELD_SHOT_SPD}
            self.phase = "shield shot"
            return [me[0] + to_aim[0] * 20, me[1] + to_aim[1] * 20], heading_for(to_aim), True

        face = (side * math.cos(SHIELD_FACE_BACK), -math.sin(SHIELD_FACE_BACK))
        heading = heading_for(face)
        away = unit((keeper[0] - me[0], keeper[1] - me[1]))
        push = unit((away[0] - side * SHIELD_PUSH_ACROSS, away[1]))
        target = [keeper[0] + push[0] * SHIELD_PUSH, keeper[1] + push[1] * SHIELD_PUSH]
        self.speed = {"spd_multi": 1.0}
        self.phase = "shield"
        return target, heading, True

    def feint(self, world, compass, me, goal, centres, now):
        if self.feint_since == 0 or not world.opponents or dist(me, goal) > FEINT_RANGE:
            return None
        keeper = min(world.opponents, key=lambda o: dist(o, goal))
        if dist(keeper, goal) > FEINT_KEEPER_D or keeper[1] < me[1] + R:
            return None
        if self.feint_since is None:
            self.feint_since = now
            mid = abs(keeper[0]) < FEINT_MID
            self.feint_side = math.copysign(1.0, (me[0] if mid else keeper[0]) or 1.0)
        elif now - self.feint_since > FEINT_MAX_TIME:
            self.feint_since = 0
            return None
        side = self.feint_side
        aim = (-side * (field.GOAL_W / 2 - FEINT_AIM_IN), field.ATTACK_GOAL_Y)
        heading = heading_for((aim[0] - me[0], aim[1] - me[1]))
        if shot_open(compass, me, centres, FEINT_RANGE + 20.0, FEINT_ROOM):
            self.kick = True
        post = (side * field.GOAL_W / 2, field.ATTACK_GOAL_Y)
        u = unit((post[0] - me[0], post[1] - me[1]))
        self.speed = {"spd_multi": FEINT_SPD}
        self.phase = "feint"
        return [me[0] + u[0] * 40, me[1] + u[1] * 40], heading, True

    def attack(self, world, compass, me, ball):
        goal = (0.0, field.ATTACK_GOAL_Y)
        heading = heading_for((goal[0] - me[0], goal[1] - me[1]))

        if not world.ball_in_capture:
            self.facing_since = None
            self.shield_side = None
            self.feint_since = None
            self.turn_away_side = None
            target, heading, dribbler, spd = chase(me, ball)
            edge = edge_capture(me, ball)
            if edge is not None:
                target, heading, dribbler, spd = edge
                self.speed = {"spd_multi": spd}
                self.phase = "edge capture"
                self.edge_at = time.monotonic()
                return target, heading, dribbler
            behind_us = me[1] > ball[1] - 5.0
            nearest = min(world.opponents, key=lambda o: dist(o, ball), default=None)
            if (RESTART_SCREEN and behind_us and nearest is not None and dist(nearest, ball) < SCREEN_OPP
                    and self.restart_ball(ball, SCREEN_TIME)):
                side = math.copysign(1.0, (nearest[0] - ball[0]) or (me[0] - ball[0]) or 1.0)
                gap = R + cfg.BALL_RADIUS + 8.0
                if me[1] > ball[1] - gap * 0.5:
                    target = [ball[0] + side * gap, ball[1] - gap * 0.5]
                else:
                    target = [ball[0], ball[1] - gap]
                self.speed = {"spd_multi": 1.0}
                self.phase = "screen"
                return around_ball(me, target, ball), 0.0, False
            wrong_side = me[1] > ball[1] - 5.0
            they_are_coming = any(dist(o, ball) < CATCH_RESTART_OPP for o in world.opponents)
            if RESTART_CATCH and self.restart_ball(ball, CATCH_RESTART_TIME) and wrong_side and they_are_coming:
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
                spd = 1.0
                self.phase = "rush"
                self.rushed_at = time.monotonic()
            else:
                self.phase = "chase"
            self.speed = {"spd_multi": spd}
            return target, heading, dribbler

        fwd = (-math.sin(compass), math.cos(compass))
        just_edge = self.edge_at is not None and time.monotonic() - self.edge_at < TURN_EDGE_TIME
        self.edge_at = None
        if self.turn_away_side is None and ((TURN_AWAY and fwd[1] < TURN_AWAY_FY and
                                             me[1] < field.OWN_GOAL_Y + TURN_AWAY_Y) or
                                            (TURN_AFTER_EDGE and just_edge)):
            self.turn_away_side = math.copysign(1.0, me[0] or 1.0)
        if self.turn_away_side is not None:
            if abs(angle_diff(heading, compass)) > TURN_AWAY_DONE:
                side = self.turn_away_side
                face = heading_for((side, 0.0)) if fwd[1] < TURN_AWAY_SIDE_FY else heading
                self.speed = {"spd_multi": TURN_AWAY_SPEED}
                self.phase = "turn away"
                return [me[0], me[1]], face, True
            self.turn_away_side = None

        if SHIELD:
            out = self.shield(world, compass, me, ball, goal)
            if out is not None:
                return out

        ho = head_on(me, compass, self.box_opponents(world, me))
        if ho is not None:
            fwd = (-math.sin(compass), math.cos(compass))
            cross = (ho[0] - me[0]) * fwd[1] - (ho[1] - me[1]) * fwd[0]
            side = 1.0 if cross < 0 else -1.0
            sideways = (fwd[1] * side, -fwd[0] * side)
            self.speed = {"spd_multi": 1.0}
            self.phase = "slip past"
            target = [me[0] + sideways[0] * SLIP_SIDE + fwd[0] * SLIP_FORWARD,
                      me[1] + sideways[1] * SLIP_SIDE + fwd[1] * SLIP_FORWARD]
            return target, heading, True

        now = time.monotonic()
        centres = self.box_centres(world, me)
        if FEINT:
            out = self.feint(world, compass, me, goal, centres, now)
            if out is not None:
                return out
        if self.won_at is None:
            self.won_at = now
        facing_goal_mouth = shot_open(compass, me, [], RUSH_FLICK_RANGE)
        if RESTART_RUSH and self.restart_ball(ball) and facing_goal_mouth:
            self.kick = True
            flicked = True
        else:
            flicked = False
        just_won_it = now - self.won_at < QUICK_SHOT_TIME
        if just_won_it and clear_shot(compass, me, centres, QUICK_SHOT_RANGE):
            self.kick = True
        pairs = [(o, v) for o, v in zip(world.opponents, world.opponent_vel)
                 if not self.in_their_box(me) or self.in_their_box(o)]
        opps, vels = [o for o, _ in pairs], [v for _, v in pairs]
        u = carry_dir(me, self.carry_target(me, goal), opps, vels)
        carry_heading = heading_for(u)
        carry_spd = 1.0
        just_rushed = self.rushed_at is not None and now - self.rushed_at <= RUSH_TURN_TIME
        turn = abs(angle_diff(carry_heading, compass))
        if just_rushed and turn >= RUSH_TURN_FREE:
            carry_spd = max(RUSH_TURN_MIN_SPD, 1.0 - (turn - RUSH_TURN_FREE) / RUSH_TURN_SLOWEST * 0.6)
        self.speed = {"spd_multi": carry_spd}
        self.phase = "flick" if flicked else "carry"
        if charged(me, opps, vels) and clear_shot(compass, me, centres, OPEN_SHOT_RANGE):
            self.kick = True
        if bank_shot_open(compass, me, centres) and not clear_shot(compass, me, centres, OPEN_SHOT_RANGE):
            self.kick = True
            self.phase = "bank shot"
        facing_err = abs(angle_diff(heading, compass))
        if facing_err < CARRY_FACING:
            if not self.facing_since:
                self.facing_since = now
            if now - self.facing_since > CARRY_KICK_HOLD and dist(me, goal) < CARRY_KICK_RANGE:
                self.kick = True
        else:
            self.facing_since = None
        target = [me[0] + u[0] * 40, me[1] + u[1] * 40]
        if SHIELD and CARRY_TO_CORNER and self.shield_side != 0 and self.corner_setup(world, me, goal):
            side = math.copysign(1.0, me[0] or 1.0)
            corner = self.box_corner(me)
            if CORNER_REACHED < dist(me, corner) < CORNER_RANGE:
                corner_heading = heading_for((side, 0.0))
                off = abs(angle_diff(corner_heading, compass))
                self.speed = {"spd_multi": CORNER_SPD * max(CORNER_TURN_SHARE, math.cos(min(off, math.pi / 2)))}
                self.phase = "carry corner"
                return corner, corner_heading, True
        return target, carry_heading, True

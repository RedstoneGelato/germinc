import math
import time

import field
import robot_config as cfg
from strategy_fallback import GoalieFallback, StrikerFallback
from utils import Hysteresis

R = cfg.ROBOT_RADIUS

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

GUARD_RADIUS = min(field.PENALTY_D - 5.0, 20.0)
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
SEAL_X = field.GOAL_W / 2 + 2.0
SEAL_Y = 25.0
SEAL_OPP = 45.0
SEAL_KEEP_X = field.GOAL_W / 2 - 6.0
SEAL_POST_X = field.GOAL_W / 2 - 2.0
SEAL_GAP = 0.0
SEAL_INNER_X = field.GOAL_W / 2 - 6.0
SEAL_INNER_KEEP_X = field.GOAL_W / 2 - 10.0
GOALIE_FACE_MAX = math.radians(70)
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
GOALIE_MAX_OUT = 12.0
PUSHBACK_REACH = 40.0
GOALIE_KICK_OUT = True
GOALIE_OUT_UP = 0.3
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


def clear_aim(ball):
    if abs(ball[0]) > CLEAR_OUT_X:
        return (math.copysign(field.PLAY_W / 2 + 30.0, ball[0]), ball[1] + 15.0)
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
    return not field.inside_play_area(ball[0], ball[1], -BALL_OUT_MARGIN)


def going_out(ball, vel):
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
    return (side * (field.PLAY_W / 2 + 30.0), ball[1] + CHALLENGE_UP)


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

    def reset(self):
        self.fallback.reset()
        self.botstate_hyst.reset()

    def update(self, world, compass, mate, line_touches):
        mate = self.mate_mem(mate)
        pose = world.pose
        self.kick = False
        self.possession = possession(world, mate)
        if pose is None or not pose.confident:
            return self._use_fallback(world, compass, mate, line_touches)

        me = (pose.x, pose.y)
        ball = world.ball_field
        goal = (0.0, field.OWN_GOAL_Y)
        striker_on = mate_active(mate)
        if not striker_on and GOALIE_ALONE_ATTACKS:
            return self._play_alone(world, compass, mate, line_touches)
        self.alone = None

        self._remember_ball_going_out(ball, world.ball_vel)
        self._update_botstate(world, mate, me, ball, goal, striker_on)
        centres = obstacle_centres(me, world.obstacles_field)
        action = self._seal_post(world, me, ball, centres, pose) or self._handle_shover(world, compass, me, ball, centres, pose)
        if action:
            return action
        self.speed = {}
        return (self._block_shot(world, me, ball, centres, pose)
                or self._look_at_restart_spot(me, ball, striker_on, pose)
                or self._play_botstate(world, compass, mate, me, ball, goal, centres, striker_on, pose))

    def _use_fallback(self, world, compass, mate, line_touches):
        command = mate.get("command") if mate else None
        mate_on = mate.get("bot active") if mate else None
        out = self.fallback.update(world, compass, command, mate_on, line_touches)
        self.botstate = 10 + self.fallback.botstate
        self.speed = self.fallback.speed
        self.chasing = self.fallback.chasing
        self.kick = getattr(self.fallback, "kick", False)
        self.phase = "fallback"
        return out

    def _play_alone(self, world, compass, mate, line_touches):
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

    def _remember_ball_going_out(self, ball, vel):
        if ball is None:
            return
        if going_out(ball, vel):
            self.out_ball = (list(ball), time.monotonic())
        elif not ball_out(ball):
            self.out_ball = None

    def _raw_botstate(self, world, mate, me, ball, goal, striker_on):
        if ball is None:
            return 0
        if ball_out(ball):
            return 4
        if world.ball_in_capture and self.has_dribbler and self.carrier(ball, world.opponents) is None:
            return 1
        if not striker_on:
            closest_opp = min((dist(c, ball) for c in world.opponents), default=1e9)
            return 2 if ball[1] < SOLO_CHASE_Y or dist(me, ball) + SOLO_CLOSER < closest_opp else 3
        in_box = abs(ball[0]) < field.PENALTY_W / 2 and ball[1] < field.OWN_GOAL_Y + field.PENALTY_D
        told = mate.get("command") == 1
        clearing = self.botstate in (1, 2)
        opp_on_ball = self.possession == "them"
        pushed_at_goal = dist(ball, goal) < PUSHER_DIST and world.ball_vel[1] < -PUSHER_SPEED
        striker_chasing_it = mate.get("command") == 0 and mate.get("chasing")
        just_kicked = time.monotonic() - self.kicked_at <= KICK_RETURN_T
        if opp_on_ball and dist(ball, goal) < CHALLENGE_DIST:
            return 2
        if opp_on_ball and not pushed_at_goal:
            return 3
        if told or (in_box and dist(me, ball) < DANGER_DIST):
            return 2
        if clearing and ball[1] < CLEAR_RETURN_Y and not striker_chasing_it and not just_kicked:
            return 2
        return 3

    def _update_botstate(self, world, mate, me, ball, goal, striker_on):
        raw = self._raw_botstate(world, mate, me, ball, goal, striker_on)
        told = striker_on and mate.get("command") == 1
        too_far_out = me[1] > field.OWN_GOAL_Y + field.PENALTY_D + GOALIE_MAX_OUT
        if raw == 2 and striker_on and too_far_out and not told:
            raw = 3
        self.botstate = self.botstate_hyst.update(raw)
        if ball is None and self.botstate != 0:
            self.botstate = 0
        self.chasing = self.botstate in (1, 2)

    def _face_ball(self, me, ball):
        face = heading_for((ball[0] - me[0], ball[1] - me[1]))
        return max(-GOALIE_FACE_MAX, min(GOALIE_FACE_MAX, face))

    def _seal_post(self, world, me, ball, centres, pose):
        self.sealing = (ball is not None and not world.ball_in_capture
                        and self.wide_threat(ball, centres, SEAL_KEEP_X if self.sealing else SEAL_X))
        if not self.sealing:
            return None
        heading = self._face_ball(me, ball)
        target = around_ball(me, self.seal_spot(ball), ball)
        self.speed = {"spd_multi": 1.0, "smooth": GUARD_SMOOTH}
        self.phase = "seal post"
        return rel(target, pose), heading, False

    def _handle_shover(self, world, compass, me, ball, centres, pose):
        pusher = shover(me, ball, centres) if ball is not None else None
        if pusher is None:
            self.out_side = None
            return None
        if GOALIE_KICK_OUT and world.ball_in_capture and self.has_dribbler:
            other_robots = [c for c in centres if c is not pusher]
            self.out_side = kick_out_side(me, ball, other_robots, self.out_side, shover=pusher)
            heading = heading_for((self.out_side, GOALIE_OUT_UP))
            self.kick = abs(angle_diff(heading, compass)) < GOALIE_KICK_OUT_TOL
            self.speed = {}
            self.phase = "kick out"
            return rel(list(me), pose), heading, True
        d = unit((pusher[0] - me[0], pusher[1] - me[1]))
        target = clamp_in_field([me[0] + d[0] * PUSHBACK_REACH, me[1] + d[1] * PUSHBACK_REACH], OUT_MARGIN)
        self.speed = {"spd_multi": 1.0}
        self.phase = "push back"
        return rel(target, pose), 0.0, False

    def _block_shot(self, world, me, ball, centres, pose):
        if ball is None or self.botstate == 1 or contested(ball, centres):
            return None
        block = self.block_point(ball, world.ball_vel)
        if block is None:
            return None
        target = [me[0], block[1]] if abs(block[0] - me[0]) < BLOCK_STAY else block
        heading = self._face_ball(me, ball)
        self.phase = "block"
        self.speed = {"spd_multi": min(1.0, max(0.1, dist(me, target) / BLOCK_GAIN)), "smooth": GUARD_SMOOTH}
        return rel(target, pose), heading, False

    def _look_at_restart_spot(self, me, ball, striker_on, pose):
        ball_is_out = ball is None or ball_out(ball)
        if not (GOALIE_LOOK_AT_SPOT and self.out_ball is not None and striker_on and ball_is_out
                and time.monotonic() - self.out_ball[1] < RESTART_WAIT):
            return None
        spot = restart_spot(self.out_ball[0])
        others = [q for q in field.NEUTRAL_SPOTS if dist(q, spot) > 1]
        if abs(spot[0]) > 1:
            watch = max(others, key=lambda q: dist(q, spot))
        else:
            watch = min(others, key=lambda q: abs(q[0] - me[0]))
        u = unit((0.0 - watch[0], field.OWN_GOAL_Y - watch[1]))
        target = [watch[0] + u[0] * LOOK_DIST, watch[1] + u[1] * LOOK_DIST]
        heading = heading_for((watch[0] - target[0], watch[1] - target[1]))
        self.speed = {"spd_multi": 1.0}
        self.phase = "look at spot"
        return rel(target, pose), heading, False

    def _play_botstate(self, world, compass, mate, me, ball, goal, centres, striker_on, pose):
        dribbler = False
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
            target, heading = self._guard_position(world, me, ball)
        else:
            aim, _ = best_aim(ball, centres)
            if self.botstate == 1:
                target, heading = self.clear_ball(world, compass, mate, me, aim, centres, striker_on)
                dribbler = True
            else:
                target, heading, dribbler = self.take_ball(world, me, ball, aim, centres)
        return rel(target, pose), heading, dribbler

    def _guard_position(self, world, me, ball):
        if self.botstate == 3:
            watched = (ball[0] + world.ball_vel[0] * GUARD_LEAD, ball[1] + world.ball_vel[1] * GUARD_LEAD)
        else:
            watched = restart_spot(ball)
        target, heading = self.guard(me, watched)
        carrier = self.carrier(ball, world.opponents) if self.botstate == 3 and CARRIER_WEIGHT else None
        if carrier is not None:
            target = self.shot_line_guard(target, watched, carrier)
        self.speed = guard_speed(me, target)
        if self.botstate == 3:
            target = around_ball(me, target, ball)
            self.phase = "guard carrier" if carrier is not None else "guard"
        else:
            self.phase = "ball out"
        return target, heading

    def clear_ball(self, world, compass, mate, me, aim, centres, striker_on):
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

    def take_ball(self, world, me, ball, aim, centres):
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
        return target, heading, dribbler

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
        return [math.copysign(min(SEAL_POST_X, abs(ball[0]) - 2.0), ball[0]), field.OWN_GOAL_Y + R + SEAL_GAP]

    @staticmethod
    def block_point(ball, vel):
        if vel[1] > -BLOCK_SPEED:
            return None
        y_line = field.OWN_GOAL_Y + R + 3
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
        v = unit((b[0] - goal[0], b[1] - goal[1]))
        x = goal[0] + v[0] * GUARD_RADIUS
        y = max(goal[1] + v[1] * GUARD_RADIUS, goal[1] + R + 3)
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
CORNER_INSET = R + 5.0
CORNER_REACHED = 8.0
CARRY_AIM_CORNER = True
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
        self.feint_since = None
        self.feint_side = None
        self.last_ball = None
        self.mate_mem = MateMemory()

    def reset(self):
        self.fallback.reset()
        self.last_ball = None
        self.facing_since = None
        self.won_at = None

    def update(self, world, compass, mate, line_touches):
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
        self.speed = {}

        gone_out = ball is not None and not field.inside_play_area(ball[0], ball[1], -cfg.BALL_RADIUS - 3.0)
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
        else:
            target, heading, dribbler = self.attack(world, compass, me, ball)
            self.chasing = True
            self.botstate = 1 if world.ball_in_capture else 2
            target = clamp_in_field(target, FIELD_MARGIN)

        if goalie_on:
            target = keep_out_of_box(target)
        return rel(target, pose), heading, dribbler

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
        inset = CORNER_INSET
        return [side * (field.PENALTY_W / 2 - inset), field.ATTACK_GOAL_Y - field.PENALTY_D + inset]

    def carry_target(self, me, goal):
        corner = self.box_corner(me)
        if CARRY_AIM_CORNER and me[1] < corner[1] - CORNER_REACHED:
            return corner
        return goal

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
        if not world.ball_in_capture:
            return self._go_for_ball(world, me, ball)
        return self._carry_ball(world, compass, me, ball, goal)

    def _orbit_restart_ball(self, me, ball, side, phase):
        gap = R + cfg.BALL_RADIUS + 8.0
        if me[1] > ball[1] - gap * 0.5:
            target = [ball[0] + side * gap, ball[1] - gap * 0.5]
        else:
            target = [ball[0], ball[1] - gap]
        self.speed = {"spd_multi": 1.0}
        self.phase = phase
        return around_ball(me, target, ball), 0.0, False

    def _go_for_ball(self, world, me, ball):
        self.facing_since = None
        self.feint_since = None
        target, heading, dribbler, spd = chase(me, ball)
        behind_ball = me[1] > ball[1] - 5.0
        nearest = min(world.opponents, key=lambda o: dist(o, ball), default=None)
        if (RESTART_SCREEN and behind_ball and nearest is not None and dist(nearest, ball) < SCREEN_OPP
                and self.restart_ball(ball, SCREEN_TIME)):
            side = math.copysign(1.0, (nearest[0] - ball[0]) or (me[0] - ball[0]) or 1.0)
            return self._orbit_restart_ball(me, ball, side, "screen")
        they_are_coming = any(dist(o, ball) < CATCH_RESTART_OPP for o in world.opponents)
        if RESTART_CATCH and self.restart_ball(ball, CATCH_RESTART_TIME) and behind_ball and they_are_coming:
            side = math.copysign(1.0, (me[0] - ball[0]) or 1.0)
            return self._orbit_restart_ball(me, ball, side, "restart orbit")
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

    def _slip_past(self, compass, me, blocker, heading):
        fwd = (-math.sin(compass), math.cos(compass))
        cross = (blocker[0] - me[0]) * fwd[1] - (blocker[1] - me[1]) * fwd[0]
        side = 1.0 if cross < 0 else -1.0
        sideways = (fwd[1] * side, -fwd[0] * side)
        self.speed = {"spd_multi": 1.0}
        self.phase = "slip past"
        target = [me[0] + sideways[0] * SLIP_SIDE + fwd[0] * SLIP_FORWARD,
                  me[1] + sideways[1] * SLIP_SIDE + fwd[1] * SLIP_FORWARD]
        return target, heading, True

    def _carry_speed(self, carry_heading, compass, now):
        just_rushed = self.rushed_at is not None and now - self.rushed_at <= RUSH_TURN_TIME
        turn = abs(angle_diff(carry_heading, compass))
        if just_rushed and turn >= RUSH_TURN_FREE:
            return max(RUSH_TURN_MIN_SPD, 1.0 - (turn - RUSH_TURN_FREE) / RUSH_TURN_SLOWEST * 0.6)
        return 1.0

    def _kick_when_lined_up(self, compass, me, goal, heading, now):
        if abs(angle_diff(heading, compass)) >= CARRY_FACING:
            self.facing_since = None
            return
        if not self.facing_since:
            self.facing_since = now
        if now - self.facing_since > CARRY_KICK_HOLD and dist(me, goal) < CARRY_KICK_RANGE:
            self.kick = True

    def _carry_ball(self, world, compass, me, ball, goal):
        heading = heading_for((goal[0] - me[0], goal[1] - me[1]))
        blocker = head_on(me, compass, self.box_opponents(world, me))
        if blocker is not None:
            return self._slip_past(compass, me, blocker, heading)

        now = time.monotonic()
        centres = self.box_centres(world, me)
        if FEINT:
            out = self.feint(world, compass, me, goal, centres, now)
            if out is not None:
                return out
        if self.won_at is None:
            self.won_at = now
        flicked = RESTART_RUSH and self.restart_ball(ball) and shot_open(compass, me, [], RUSH_FLICK_RANGE)
        if flicked:
            self.kick = True
        if now - self.won_at < QUICK_SHOT_TIME and clear_shot(compass, me, centres, QUICK_SHOT_RANGE):
            self.kick = True
        pairs = [(o, v) for o, v in zip(world.opponents, world.opponent_vel)
                 if not self.in_their_box(me) or self.in_their_box(o)]
        opps, vels = [o for o, _ in pairs], [v for _, v in pairs]
        u = carry_dir(me, self.carry_target(me, goal), opps, vels)
        carry_heading = heading_for(u)
        self.speed = {"spd_multi": self._carry_speed(carry_heading, compass, now)}
        self.phase = "flick" if flicked else "carry"
        if charged(me, opps, vels) and clear_shot(compass, me, centres, OPEN_SHOT_RANGE):
            self.kick = True
        self._kick_when_lined_up(compass, me, goal, heading, now)
        return [me[0] + u[0] * 40, me[1] + u[1] * 40], carry_heading, True


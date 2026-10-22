"""
strategy_int.py - strategy.py for the INTERNATIONAL RoboCupJunior Soccer rules (RCJ Soccer 2026, the rules RCJ
Singapore plays), on the international field (field_int.py). Run it with main_int.py / main_attack_int.py, and in the
simulator with sim_int.py. strategy.py stays the RCJA version.

It is strategy.py with these changes - each one is marked "INT" in the code, with the rule that needs it:
    rules 2.6 / 2.8   no robot may be COMPLETELY inside a penalty area (out of bounds, 1 minute off). RCJA goalies
                      live in the area. Both brains now finish with box_guard(): every move is cut short where it
                      would leave less than BOX_POKE cm of the robot outside an area's front / side lines (and
                      already too deep = straight back out). The goalie guards on a line GOALIE_LINE_D in front of
                      the goal line (GUARD_RADIUS), and blocks shots there (block_point()).
    rule 2.6          multiple defense: two robots of a team even PARTLY in a penalty area, at ANY time (RCJA: only
                      when it affects play) -> the one further from the ball is moved. The striker's box_guard()
                      keeps it fully out of our area while our goalie is on (BOX_KEEP_INT).
    rule 2.6          pushing: attacker and defender touching with one of them partly in an area and either touching
                      the ball -> ball to the furthest free neutral spot, no goal. So the shield (shoving their
                      goalie off the near post) can't score: SHIELD, BASE_SHIELD and CARRY_TO_CORNER are off. (They
                      also needed our robot fully inside their area.)
    rule 2.5          the ball is never out at the line - it's live up to the walls. ball_out() / going_out() are
                      always False (no "ball out" waiting, restarts only come from lack of progress).
    rules 2.5 / 2.7   a ball knocked out over the side line doesn't come back on the halfway line any more: it stays
                      by the wall or goes to the NEAREST free neutral spot (the corner spots are 45 cm from our end).
                      The goalie's clears / knock-outs / kick-outs go up-field instead of straight out
                      (clear_aim(), knock_out_aim(), GOALIE_OUT_UP).
                      A ball on / past the line (not in a corner) gets caught from the field side: partly over the
                      line, dribbler facing it (edge_capture(); ba /40 98-17 -> 127-16, v88 150-34 -> 147-42).
    rule 2.8          out of bounds = touching a wall, not crossing the white line. With the 12 cm outer area our
                      22 cm robot can't get past the line without touching the wall, so cfg.MAX_OUT (field_int
                      .install()) is negative: OUT_MARGIN keeps us WALL_GAP off the walls.
    rules 2.6 / 2.8   the goalie never gets nearer our goal line than GOALIE_LINE_D in front of the goal, so it's always
                      up-field of a loose ball down there: it doesn't move into one (deep_no_touch(), unless an
                      opponent is near it), and for a wide ball it seals the post from the side of the area, where it
                      can get down to the goal line (SEAL_SIDE). ba /40 127-16 -> 125-10, v88 147-42 -> 152-35.
    striker, facing   in our corners it faces the ball and goes straight at it
                      (FACE_BALL; ba /40 145-9 -> 163-23, v88 199-9 -> 176-13). Caught facing our goal, it turns
                      round towards the near side line, away from the goal (TURN_AWAY: ba 165-20, v88 180-12),
                      on the spot until it faces their goal, then drives (not turning while moving). Same turn
                      on the spot right after an edge capture (TURN_AFTER_EDGE).
    goalie, spots     a ball still on one of our end's spots, nobody else near it: goalie takes it (SPOT_TAKE).
    striker, feint    carrying in with their goalie in front of their goal: drive at the goalie's side, face
                      the open side, shoot when it's open (FEINT; ba 304-28 -> 314-26, v88 391-27 -> 463-21 /80).
    bank shot         off a side wall (BANK_SHOT), off until the real wall's bounce is measured (BANK_BOUNCE).
    clear share       goalie with the ball, about to clear it: sends the striker where it'll kick it ("clear"), and
                      the striker goes straight to where it can meet that kick (CLEAR_SHARE, meet_clear()), not
                      back and forth round the goalie. Untested in the simulator (Jerold, 19 Oct 2026).
    goalie, box       a ball in our area between the goalie and the goal line (in reach, no opponent on it): turn
                      round, catch it, kick it out the clearest way, up-field first (BOX_TURN, box_clear()).
                      ba /40 141-8 -> 145-9, v88 203-13 -> 199-9 (with the on-the-ball pushing referee).
    60 cm goal        the goalie guards on the bisector of the angle the ball sees the goal under, stepping out to
                      30 cm so it covers the whole mouth (GUARD_BISECTOR): balls from the side were going in at the
                      near post past a goalie on the line to the goal centre.
    field spec        the carry's corner aim point (box_corner()) is just outside their area's front corner (RCJA:
                      16 cm inside it). Everything else that uses the field's sizes follows field_int.py by itself.
Kick-off positions (rule 2.3.1: 30 cm from the ball, outside the centre circle) are where we place the robots, so
nothing here changes for them.

(strategy.py's description follows.)

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

import field_int as field
field.install()      # INT: every other module (strategy_fallback, lines, ...) gets field_int as its `field` too
import robot_config as cfg  # noqa: E402  (after install(): it sets cfg.MAX_OUT for these walls)
from strategy_fallback import GoalieFallback, StrikerFallback
from utils import Hysteresis

R = cfg.ROBOT_RADIUS

# ---- INT: international rules (see the list at the top)
BOX_POKE = 6.0                    # INT (rules 2.6 / 2.8): never completely inside a penalty area - at least this much
                                  # of our robot (cm) stays outside its front / side lines (box_guard())
BOX_KEEP_INT = 4.0                # INT (rule 2.6): striker, goalie on: our robot's edge stays this far out of our area
BRAKE_TIME = 0.08                 # INT: box_guard() stops this many s of our speed short of the limit (the motors
BRAKE_MAX = 6.0                   # lag: carrying in at 60 cm/s it ran 4 cm past) - at most BRAKE_MAX cm
GOALIE_LINE_D = field.PENALTY_D - R + BOX_POKE   # INT: the goalie's centre is never closer than this to our goal line
                                                 # in front of the goal (25 - 11 + 6 = 20 cm)

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

GUARD_RADIUS = GOALIE_LINE_D      # goalie: distance from the goal centre it guards at (INT, rules 2.6 / 2.8: RCJA
                                  # was min(PENALTY_D - 5, 20) - deeper than that is completely inside the area)
GUARD_OUT_START = 60.0            # INT, only without GUARD_BISECTOR: ball further than this from our goal centre ->
GUARD_OUT_PER_CM = 0.0            # guard further out, this many cm per cm further, up to GUARD_OUT_MAX from the goal
GUARD_OUT_MAX = 40.0              # centre (0.3 made no difference against Bodensee)
GUARD_BISECTOR = True             # INT (60 cm goal): guard on the line halving the angle the ball sees the goal mouth
GUARD_COVER = 0.6                 # under (not the line to the goal centre: a ball from the side went in at the near
GUARD_D_MAX = 30.0                # post), at depth GUARD_COVER x the ball's (the depth where our 22 cm covers the whole
                                  # mouth as the ball sees it), between GOALIE_LINE_D and GUARD_D_MAX from the goal line
                                  # sim_int (fast, 60 s, opponents kept legal): Bodensee /160 393-103 -> 392-66 (131W
                                  # 16D 13L), v88 /80 234-81 -> 304-64. GUARD_D_MAX 20: ba 179-37, 40: 334-62 /160
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
POST_SEAL = True                  # goalie: ball beside our goal (past a post sideways, close to the goal line - the
SEAL_SIDE = True                  # INT: ball out past SEAL_SIDE_X + 2: seal at the side of our area, down on the goal
SEAL_SIDE_X = field.PENALTY_W / 2 - R + BOX_POKE + 1.0   # line (seal_spot()); |x| 36: BOX_POKE + 1 cm out of the side
SEAL_X = field.GOAL_W / 2 + 2.0   # shots "from a large angle") with an opponent at it -> stand on the goal line at
SEAL_Y = 25.0                     # the near post and stay there. Going for it / shoving back took the goalie up-field
SEAL_OPP = 45.0                   # round the ball and it was rolled in along the line behind it: most wide goals
SEAL_KEEP_X = field.GOAL_W / 2 - 6.0   # against v88. Ball |x| past SEAL_X (once sealing: past SEAL_KEEP_X), less than
SEAL_POST_X = field.GOAL_W / 2 - 2.0   # SEAL_Y above the goal line, an opponent's centre within SEAL_OPP of it; the
SEAL_GAP = 0.0                         # goalie stands at |x| SEAL_POST_X, its edge SEAL_GAP above the goal line.
SEAL_INNER_X = field.GOAL_W / 2 - 6.0  # also seal for a ball inside the post (|x| past this, once sealing past
SEAL_INNER_KEEP_X = field.GOAL_W / 2 - 10.0  # SEAL_INNER_KEEP_X), standing just inside the ball: TS dribbled along
                                       # the goal line from there while the goalie orbited. 160 fast games vs TS:
                                       # 33 -> 22 goals against (v88 14 -> 18, within noise)
                                       # Simulator, 80 fast games vs v88: 69-152 before, 102-44 with it (wide goals
                                       # against 56 -> 4). SEAL_GAP 3 (the guard line) let most of them back in: 91-95
GOALIE_FACE_MAX = math.radians(70)    # goalie guarding: face the ball, at most this far off up-field
SPOT_TAKE = True                  # INT: ball still on one of our end's neutral spots, within SPOT_TAKE_DIST
SPOT_TAKE_DIST = 60.0             # of the goalie and every other robot (both theirs, our striker) further than
SPOT_TAKE_CLEAR = 60.0            # SPOT_TAKE_CLEAR from it: the goalie goes up, takes it and kicks it up-field
SPOT_TAKE_ON = 5.0                # ...'on the spot': within this
SPOT_TAKE_STILL = 10.0            # ...and slower than this (cm/s)
BOX_TURN = True                   # goalie: ball in our penalty area on the goal side of us -> turn round to face it
                                  # (GOALIE_FACE_MAX doesn't apply), catch it in the dribbler, then turn and kick it out the
                                  # clearest way, up-field first (box_clear()). Going round it isn't possible here: the
                                  # goalie may not get nearer the goal line than GOALIE_LINE_D
BOX_TURN_BEHIND = 3.0             # ...when the ball is this much nearer our goal line than we are
BOX_TURN_MIN_D = 10.0             # ...not for a ball in front of the goal nearer the goal line than this (out of reach:
                                  # touching it would only push it in)
BOX_TURN_OPP = 25.0               # ...nor with an opponent this close to the ball (they'd shove it in past us)
BOX_TURN_SPEED = 0.35             # ...going at it this slowly under 20 cm (knocking it on would put it in our goal)
BOX_KICK_DIRS = ((0.0, 1.0), (0.7, 0.7), (1.0, 0.25))   # kick directions in order of preference (x mirrored: our
                                  # side first); never back towards our goal
BOX_KICK_TOL = math.radians(15)   # ...kick when facing one within this
GOALIE_CLEAR_PATH = 60.0          # goalie kicks: the ball's path must miss every robot for this far (cm) - it kicked
                                  # straight into their striker (clearance up to a robot radius into it was accepted)
GOALIE_KICK_TURN = math.radians(60)   # goalie with the ball (striker off): kicks it clear facing at most this far off
GOALIE_KICK_CLEAR = 50.0          # up-field, when no robot is in the way for this far (cm)
PRESSED_GAP = 45.0                # goalie with the ball: an opponent's centre within 2R + this = pressed, clear it now
                                  # (20: lining up a pass, TS's striker drove in and took it - 80 fast games vs TS
                                  # 20 -> 10 goals against)
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
CLEAR_SHARE = True                # INT: goalie with the ball, about to clear it: sends where it'll kick it ("clear":
                                  # [x, y, heading]) and the striker goes straight to where it can meet that kick
                                  # (meet_clear()) instead of chasing the ball back and forth round the goalie (Jerold,
                                  # 19 Oct 2026)
CLEAR_SHARE_AFTER = 0.4           # ...still sent this long (s) after the kick, while the ball is on its way
CLEAR_ROLL_SPEED = 250.0          # ball speed (cm/s) straight off the kicker (simulator KICK_SPEED) - MEASURE
CLEAR_ROLL_DECAY = 0.8            # ...slowing down by this share per second (simulator BALL_FRICTION) - MEASURE
CLEAR_MEET_SPEED = 120.0          # striker's average speed (cm/s) for working out where it gets to before the ball
CLEAR_MEET_MIN = GOALIE_CLEAR_PATH + 2 * R   # never wait closer to the goalie than this along the kick (the goalie
                                  # only kicks with GOALIE_CLEAR_PATH clear: standing there would stop the kick)
CLEAR_MEET_SIDE = R + 6.0         # wait this far beside the kick's line (our side of it), so the ball rolls past us
                                  # instead of into our back, then chase it as usual
CLEAR_MEET_NEAR = 40.0            # ball within this of the striker after the kick: stop waiting, chase it
DEEP_NO_TOUCH = True              # INT: loose ball in front of our goal, nearer the goal line than the goalie's line
                                  # (+ R): don't move into it from up-field (deep_no_touch())
DEEP_GAP = 3.0                    # ...keep this far clear of touching it
DEEP_OPP = 35.0                   # ...unless an opponent is within this of the ball
GOALIE_MAX_OUT = 12.0             # goalie going for the ball: not further than this past the penalty area's front
PUSHBACK_REACH = 40.0             # (a close target means a slow robot, which just gets pushed back to our goal)
GOALIE_KICK_OUT = True            # goalie in a shoving match with the ball in its dribbler: kick it out over a side line
GOALIE_OUT_UP = 1.0               # (this much up-field per cm sideways), once facing within    (INT, rules 2.5 /
                                  # 2.7: RCJA 0.3 - out over the side line isn't a halfway restart any more, so 45 deg
                                  # up-field, at the side wall well away from our goal)
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


def deep_no_touch(me, target, ball, opponents=()):
    """INT: the ball is loose right in front of our goal, nearer the goal line than the goalie may go (GOALIE_LINE_D +
    R): the goalie is up-field of it, so touching it knocks it towards the goal (going round it, lining up or sealing,
    it was clipping slow balls rolling out of the goal mouth and turning them in: half the goals against Bodensee).
    If the move me -> target would touch it, stop where it would start to (backing off if we're already too close).
    Not with an opponent within DEEP_OPP of it: it'd shoot it in anyway, so we go for it as usual."""
    if not DEEP_NO_TOUCH or ball is None or any(dist(o, ball) < DEEP_OPP for o in opponents):
        return target
    if ball[1] - field.OWN_GOAL_Y >= GOALIE_LINE_D + R or abs(ball[0]) >= field.GOAL_W / 2 + R or me[1] <= ball[1]:
        return target
    clear = R + cfg.BALL_RADIUS + DEEP_GAP
    d_now = dist(me, ball)
    if d_now < clear:
        u = unit((me[0] - ball[0], me[1] - ball[1]))
        return [ball[0] + u[0] * clear, ball[1] + u[1] * clear]                 # too close already: back off
    d, t = seg_dist(ball, me, target)
    if d >= clear:
        return target
    # walk along the move till we'd get within `clear` of the ball
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
    """Where to push a ball near our goal: away from the goal centre, half way to straight up-field (up-field rather
    than just sideways, so it actually leaves our end)."""
    if abs(ball[0]) > CLEAR_OUT_X:    # out to the side: at the side wall, well up-field
        # (INT, rules 2.5 / 2.7: RCJA pushed it straight out - an out ball came back on the halfway line. Here it
        # stays live by the wall, or goes to the nearest corner spot, right in front of our goal)
        return (math.copysign(field.PLAY_W / 2, ball[0]), ball[1] + 60.0)
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
    """Ball clearly outside the white line (inside a goal counts as in).
    INT (rule 2.5): never - the ball is live everywhere inside the walls."""
    if not field.BALL_OUT_AT_LINE:
        return False
    return not field.inside_play_area(ball[0], ball[1], -BALL_OUT_MARGIN)


def going_out(ball, vel):
    """The ball is out, or right at the line and rolling out. INT (rule 2.5): never."""
    if not field.BALL_OUT_AT_LINE:
        return False
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
    # (INT, rules 2.5 / 2.7: RCJA aimed 30 cm past the side line, 40 up-field - an out ball came back on the
    # halfway line. Here it stays live, so knock it at the side wall well up-field)
    return (side * field.PLAY_W / 2, ball[1] + 2 * CHALLENGE_UP)


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


COMMS_GRACE = 10.0   # s: no messages from the teammate (comms dropped) within this long of it saying it's on -> assume
                     # it still is. A paused / taken-off robot sends "bot active": 0, so only lost comms goes silent -
                     # without this the goalie played striker 0.5 s into every wifi drop and left the goal empty
                     # (arena with --faults 1, 160 games: 164 -> 132 goals against). 0 = off.


class MateMemory:
    """Stands in for the teammate's message while comms are down (COMMS_GRACE)."""

    def __init__(self):
        self.on_at = -1e9        # when the teammate last said "bot active": 1

    def __call__(self, mate):
        now = time.monotonic()
        if mate:
            if mate.get("bot active") == 1:
                self.on_at = now
            return mate
        if now - self.on_at < COMMS_GRACE:
            return {"bot active": 1, "command": 0}    # (nothing else: its old position / ball would be stale)
        return mate


def mate_active(mate):
    """The teammate is on (it sends "bot active": 1)."""
    return bool(mate) and mate.get("bot active") == 1


# ------------------------------------ INT: penalty areas (rules 2.6 / 2.8) -----------------------------------------

def box_limit(partly_ok=True):
    """How deep (field.penalty_depth()) our centre may go into a penalty area: not completely inside it (BOX_POKE of
    the robot still outside), or with partly_ok False not touching it at all (BOX_KEEP_INT outside)."""
    return R - BOX_POKE if partly_ok else -(R + BOX_KEEP_INT)


def box_out_dir(p, sgn):
    """Unit arrow from p out of that penalty area the shortest way (downhill on field.penalty_depth())."""
    e = 0.5
    gx = field.penalty_depth(p[0] + e, p[1], sgn) - field.penalty_depth(p[0] - e, p[1], sgn)
    gy = field.penalty_depth(p[0], p[1] + e, sgn) - field.penalty_depth(p[0], p[1] - e, sgn)
    d = unit((-gx, -gy))
    return d if d != (0.0, 0.0) else (0.0, float(-sgn))


BOX_DETOUR = 10.0                 # INT: cm box_guard() steps up-field when the straight way is blocked from the start


def box_guard(me, target, limits, brake=0.0):
    """INT (rules 2.6 / 2.8): the move me -> target, changed so it never takes us deeper into a penalty area than
    limits = {sgn: deepest allowed (box_limit())}, sgn = -1 our area, +1 theirs.
        already too deep          -> straight out the shortest way
        target too deep           -> the nearest allowed point to it instead (slides along the area's edge)
        the way there goes too deep -> stop where it would start to (the motion re-plans every loop)
    brake (cm): moving fast, stop on the way this much short of the limit (we'd overshoot it) - only when heading
    deeper in than we are (sliding along the limit is fine)."""
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
    for _ in range(3):                           # target too deep: move it out (a corner can take two steps)
        worst = max(limits, key=lambda s: field.penalty_depth(target[0], target[1], s) - limits[s])
        over = field.penalty_depth(target[0], target[1], worst) - limits[worst]
        if over <= 0:
            break
        d = box_out_dir(target, worst)
        target = [target[0] + d[0] * (over + 0.5), target[1] + d[1] * (over + 0.5)]
    n = int(dist(me, target))                    # walk the way there in ~1 cm steps
    last = list(me)
    for i in range(1, n + 1):
        p = [me[0] + (target[0] - me[0]) * i / n, me[1] + (target[1] - me[1]) * i / n]
        if depth_over(p) > 0:
            if dist(last, me) < 2.0:             # blocked right where we stand (beside our area, target across the
                # front of the goal): stopping here would be for good - step up-field out of the way, then re-plan
                s = max(limits, key=lambda s: field.penalty_depth(p[0], p[1], s) - limits[s])
                return [me[0], me[1] - s * BOX_DETOUR]
            return last
        last = p
    return target


class BoxGuard:
    """INT: update()'s (arrow to target, heading, dribbler) after box_guard(), braking for our speed (from the pose
    BRAKE_WINDOW s ago: the pose jitters a few cm every loop). Without a confident pose (the fallback's relative
    logic) there's nothing to check it against."""
    BRAKE_WINDOW = 0.15

    def __init__(self, brake=True):
        self.brake = brake       # (off for the goalie: stopping short of its spots left gaps - 40 sim games vs v88
                                 # 98-120 with, 104-43 without)
        self.poses = []          # recent (time, x, y)
        self.speed = 0.0         # cm/s

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
        self.sealing = False     # sealing the near post (POST_SEAL)
        self.alone = None        # striker off: the StrikerStrategy the goalie plays (GOALIE_ALONE_ATTACKS)
        self.avoid = Avoider()
        self.mate_mem = MateMemory()
        self.box_guard = BoxGuard(brake=False)   # (INT)
        self.clear_plan = None   # INT (CLEAR_SHARE): [x, y, heading] of the clear kick we're about to do (sent to the
        self.clear_plan_at = -1e9   # striker), and when it was last set

    def reset(self):
        self.fallback.reset()
        self.botstate_hyst.reset()

    def update(self, world, compass, mate, line_touches):
        """INT: the RCJA logic (_update()), then box_guard() - never completely inside a penalty area."""
        planned_at = self.clear_plan_at
        out = self._update(world, compass, mate, line_touches)
        if self.clear_plan is not None and self.clear_plan_at == planned_at:
            # not clearing this loop: keep telling the striker only just after the kick (ball on its way), else drop it
            now = time.monotonic()
            if not (self.kicked_at >= self.clear_plan_at and now - self.kicked_at < CLEAR_SHARE_AFTER):
                self.clear_plan = None
        return self.box_guard(out, world.pose, {-1: box_limit(), 1: box_limit()})

    def _update(self, world, compass, mate, line_touches):
        mate = self.mate_mem(mate)       # comms dropped: the teammate is still on (COMMS_GRACE)
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
        # INT: a ball sitting on one of our end's neutral spots (just put back) with nobody else near it at
        # all: go up and take it, kick it up-field (clear_ball()), then back
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
            raw = 3     # it chased a cleared ball and left the goal empty: back to guarding
        self.botstate = self.botstate_hyst.update(raw)
        if ball is None and self.botstate != 0:
            self.botstate = 0
        self.chasing = self.botstate in (1, 2)

        # ---- ball beside our goal with an opponent at it: seal the near post (POST_SEAL)
        centres = obstacle_centres(me, world.obstacles_field)
        self.sealing = (ball is not None and not world.ball_in_capture
                        and self.wide_threat(ball, centres, SEAL_KEEP_X if self.sealing else SEAL_X))
        if self.sealing:
            target = self.seal_spot(ball)
            face = heading_for((ball[0] - me[0], ball[1] - me[1]))
            heading = max(-GOALIE_FACE_MAX, min(GOALIE_FACE_MAX, face))
            target = around_ball(me, target, ball)
            target = deep_no_touch(me, target, ball, world.opponents)
            self.speed = {"spd_multi": 1.0, "smooth": GUARD_SMOOTH}   # full speed: it gets shoved off the spot otherwise
            self.phase = "seal post"
            return rel(target, pose), heading, False

        # ---- an opponent shoving the ball against the goalie: shove straight back (guarding = crawling = getting
        # pushed into our goal), or with the ball in the dribbler, kick it out over the side line
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
                target = deep_no_touch(me, around_ball(me, target, ball), ball, world.opponents)
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
                if CLEAR_SHARE:     # INT: tell the striker where this kick goes (heading = the way we'll face)
                    self.clear_plan = [round(me[0]), round(me[1]), round(heading, 2)]
                    self.clear_plan_at = time.monotonic()
            else:
                target, heading, dribbler = self.take_ball(world, me, ball, aim, centres)
        return rel(target, pose), heading, dribbler

    def clear_ball(self, world, compass, mate, me, aim, centres, striker_on):
        """Ball in the dribbler: stay put, turn up-field and kick it clear. Returns (target, heading); sets .kick."""
        if BOX_TURN and me[1] < field.OWN_GOAL_Y + field.PENALTY_D + R:
            return self.box_clear(compass, me, centres)
        target = list(me)
        striker = mate.get("pos") if striker_on else None
        fwd = (-math.sin(compass), math.cos(compass))                 # the way we face now
        pressed = any(dist(o, me) < 2 * R + PRESSED_GAP for o in world.opponents)
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

    def box_clear(self, compass, me, centres):
        """Ball in the dribbler at our area: turn (with it) to the clearest kick direction - up-field if that's open,
        else diagonally, else across towards a side line, never back at our goal - and kick once facing it with the
        way clear. Keeps a direction while it stays clear (no turning back and forth). Returns (target, heading)."""
        side = math.copysign(1.0, me[0] or 1.0)                       # our half of the field: that side first

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
        """Botstate 2: go for the ball. Returns (target, heading, dribbler)."""
        in_box = abs(ball[0]) < field.PENALTY_W / 2 + cfg.BALL_RADIUS and ball[1] < field.OWN_GOAL_Y + field.PENALTY_D
        too_deep = abs(ball[0]) < field.GOAL_W / 2 + R and ball[1] - field.OWN_GOAL_Y < BOX_TURN_MIN_D
        contested = any(dist(o, ball) < BOX_TURN_OPP for o in world.opponents)
        if BOX_TURN and in_box and ball[1] < me[1] - BOX_TURN_BEHIND and not too_deep and not contested:
            # ball in our area, between us and the goal line: face it and catch it (box_clear() then kicks it out)
            u = unit((me[0] - ball[0], me[1] - ball[1]))
            reach = R + cfg.BALL_RADIUS - 2.0
            target = [ball[0] + u[0] * reach, ball[1] + u[1] * reach]
            heading = heading_for((ball[0] - me[0], ball[1] - me[1]))
            d = dist(me, ball)
            self.speed = {"spd_multi": BOX_TURN_SPEED if d < R + 20 else catch_speed(me, ball)}
            self.phase = "box turn"
            return target, heading, True
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
        return deep_no_touch(me, target, ball, world.opponents), heading, dribbler

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
    def wide_threat(ball, centres, min_x=SEAL_X):
        """Ball beside our goal (past a post sideways, close to the goal line) with an opponent at it (POST_SEAL)."""
        if not POST_SEAL or ball[1] > field.OWN_GOAL_Y + SEAL_Y:
            return False
        inner_x = SEAL_INNER_KEEP_X if min_x < SEAL_X else SEAL_INNER_X    # (min_x < SEAL_X: already sealing)
        if abs(ball[0]) < min(min_x, inner_x):
            return False
        return any(dist(c, ball) < SEAL_OPP for c in centres)

    @staticmethod
    def seal_spot(ball):
        """On the goal line at the near post: blocks a ball rolled in along the line and a shot across the goal.
        INT: in front of the goal the goalie can't get down to the line (GOALIE_LINE_D: the box guard held it 20 cm up
        and wide balls went in underneath it, most of v88's goals). At the side of the area it can (only BOX_POKE of
        it has to stick out sideways), so for a ball out past that it stands there instead, right at the post."""
        if SEAL_SIDE and abs(ball[0]) - 2.0 >= SEAL_SIDE_X:
            return [math.copysign(SEAL_SIDE_X, ball[0]), field.OWN_GOAL_Y + R + SEAL_GAP]
        return [math.copysign(min(SEAL_POST_X, abs(ball[0]) - 2.0), ball[0]), field.OWN_GOAL_Y + R + SEAL_GAP]

    @staticmethod
    def block_point(ball, vel):
        """Where a ball rolling towards our goal will cross the goalie's line (just in front of the goal), if it will
        cross between the posts soon - else None."""
        if vel[1] > -BLOCK_SPEED:
            return None                                       # not rolling towards our goal (fast enough)
        y_line = field.OWN_GOAL_Y + GOALIE_LINE_D             # the goalie's line (INT, rules 2.6 / 2.8: RCJA R + 3
                                                              # from the goal line = completely inside the area)
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
        face = heading_for((b[0] - me[0], b[1] - me[1]))
        bd = b[1] - goal[1]                                            # the ball's depth (in front of our goal line)
        if GUARD_BISECTOR and bd > GOALIE_LINE_D + R:
            # INT: on the bisector of the angle ball -> left post / right post (ball centre lines that just go in)
            half = field.GOAL_W / 2 - cfg.BALL_RADIUS
            u1, u2 = unit((-half - b[0], -bd)), unit((half - b[0], -bd))
            u = unit((u1[0] + u2[0], u1[1] + u2[1]))                   # from the ball towards the goal
            d = max(GOALIE_LINE_D, min(GUARD_COVER * bd, GUARD_D_MAX, bd - R - 5.0))
            t = (bd - d) / max(-u[1], 1e-3)
            lim = field.PENALTY_W / 2 - R
            x = max(-lim, min(lim, b[0] + u[0] * t))
            return [x, goal[1] + d], max(-GOALIE_FACE_MAX, min(GOALIE_FACE_MAX, face))
        v = unit((b[0] - goal[0], b[1] - goal[1]))                     # direction goal -> ball
        out = min(max(0.0, (dist(b, goal) - GUARD_OUT_START) * GUARD_OUT_PER_CM), GUARD_OUT_MAX - GUARD_RADIUS)
        radius = GUARD_RADIUS + out                                    # (INT: further out for a far ball)
        x = goal[0] + v[0] * radius
        y = max(goal[1] + v[1] * radius, goal[1] + R + 3)              # not behind the goal line
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
BANK_SHOT = False           # INT: kick when facing a shot off a side wall into their goal (bank_shot_open()).
                            # Off: with the simulator's wall (BALL_BOUNCE 0.5) no bank shot can go in from anywhere;
                            # measure the real wall's bounce into BANK_BOUNCE (0.7: possible from ~30 spots) and test it
BANK_RANGE = 150.0          # ...both legs together at most this long
BANK_ROOM = 4.0             # ...clear of every robot by this
BANK_BOUNCE = 0.5           # ...share of the ball's sideways speed the wall gives back (simulator: BALL_BOUNCE) - MEASURE
BANK_MIN_FY = 0.3           # ...facing at least this much up-field
FACE_BALL = True            # INT: no facing up-field to get the ball - within CHASE_FACE_DIST of a ball further
                            # round than CHASE_FACE_ANGLE, turn to face it and go straight at it, dribbler on
FACE_BALL_ALL = False       # ...anywhere (False: only for a ball in our corners, OUR_CORNER_*). Anywhere was worse:
                            # ba /40 145-9 -> 133-22, v88 199-9 -> 186-30
OUR_CORNER_X = field.GOAL_W / 2 + 5.0   # 'our corner': ball past the goal posts sideways...
OUR_CORNER_Y = 45.0                     # ...and within this of our goal line
TURN_AWAY = True            # INT: with the ball, facing back (forward part under TURN_AWAY_FY) within
TURN_AWAY_Y = 60.0          # TURN_AWAY_Y of our goal line: turn to face the near side line first, so the ball swings
TURN_AWAY_FY = -0.2         # away from our goal (turning the short way it went across the goal mouth)
TURN_AWAY_SPEED = 0.4
TURN_AWAY_DONE = math.radians(30)   # ...turning on the spot (not moving) via the side line until within this of
                                    # facing their goal, then the normal carry drives off
TURN_AWAY_SIDE_FY = -0.3   # ...facing the side line first while the forward part is under this
TURN_AFTER_EDGE = True      # INT: ball caught by edge_capture() (within TURN_EDGE_TIME s of it): the same turn on the
TURN_EDGE_TIME = 0.5        # spot to face their goal before driving (Jerold, 19 Oct 2026)
FACE_BALL_SPEED = 0.5       # ...going at it this fast under 25 cm (catch it, don't knock it on)
EDGE_CAPTURE = True         # INT: ball on / past the white line (live up to the walls): go partly over the line beside it
                            # and face it with the dribbler to pull it back in (edge_capture())
EDGE_CAPTURE_RANGE = 45.0   # ...once it's this close
EDGE_CAPTURE_SPEED = 0.6    # ...at this speed under 25 cm (gentle, so the dribbler catches it instead of knocking it on)
EDGE_SKIP_CORNERS = True    # ...but not in a corner (both lines within EDGE_CORNER): that's where it kept touching walls
EDGE_CORNER = 10.0
EDGE_OUR_CORNERS = False    # ...True: except our own corners (more wall touches; ba /40 156-21, v88 180-20)
EDGE_WALL_GAP = 2.0         # ...stopping this much further off the wall than the furthest out we may go (OUT_MARGIN)
CARRY_FACING = math.radians(10)   # carrying: facing the goal within this...
CARRY_KICK_HOLD = 0.2       # ...for this long (s) -> kick...
CARRY_KICK_RANGE = 70.0     # ...if this close to it (from further the goalie saved it)
QUICK_SHOT_TIME = 1.0       # just won the ball: shoot within this long (s) if the shot is open...
QUICK_SHOT_RANGE = 130.0    # ...from up to this far (before they close us down)
CHARGED_DIST = 50.0         # an opponent this close coming at us faster than CHARGED_SPEED (cm/s) = about to lose
CHARGED_SPEED = 30.0        # the ball: shoot now if the shot is open (OPEN_SHOT_RANGE)
OPEN_SHOT_RANGE = 120.0
LONG_SHOT_FROM = 70.0       # quick / charged shots from further than this from their goal's centre: only with a real
LONG_SHOT_ROOM = cfg.BALL_RADIUS + 4.0   # opening - the ball's path this far (cm) past every robot (only just
                            # crossing the mouth / brushing their goalie, they hardly ever scored: 16 goals from 184
                            # in 80 games). Simulator, 80 games with CARRY_AIM_CORNER: v88 109-4, TS 43-4, Bodensee
                            # 94-2, lnxrobots 108-1, Technulgy 143-0 (were 70-4, 10-4, 52-6, 49-2, 105-0)
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
SHIELD = False              # (INT, rule 2.6 pushing: shoving their goalie in its area gives no goal, the ball goes
                            # to the furthest neutral spot = ours. Rules 2.6 / 2.8: it needed us fully in the area)
                            # in their penalty area with the ball, their goalie in there too: shield the ball. Starts
SHIELD_IN = R               # when our robot and their goalie are both fully inside it (centre this far inside its
                            # lines; their goalie = the opponent nearest their goal), and stops when either leaves.
SHIELD_FACE_BACK = 0.0      # shielding: face the nearer side line (90 deg off facing their goal), turned this much more away from their
                            # goal (ball on the far side of us from the goalie), and drive our body into the goalie:
SHIELD_PUSH = 15.0          # aim this far past its centre, SHIELD_PUSH_ACROSS of that towards the far post - shoving
SHIELD_PUSH_ACROSS = 0.6    # it off the near half of the goal to open a shot for us
SHIELD_SHOT_CLEAR = 4.0     # a path from the ball into the goal with this much room past every robot (in again at
SHIELD_SHOT_KEEP = 0.0      # SHIELD_SHOT_CLEAR, out below SHIELD_SHOT_KEEP: the gap flickers with the camera noise):
SHIELD_SHOT_SPD = 0.3       # turn to it (this slowly forwards) and kick once facing it with
SHIELD_KICK_CLEAR = 3.0     # this much room. While shoving it, the dribbler limits keep the ball (sideways / backwards)
SHIELD_MAX_TIME = 3.0       # s: then give up shielding until we lose the ball and kick / carry as usual
CARRY_TO_CORNER = False     # (INT: sets up the shield, off with it)
                            # carrying, their goalie in their penalty area and their striker behind us: carry to the
CORNER_INSET = 5.0          # nearer front corner of the area (SHIELD_IN + CORNER_INSET inside it) facing the nearer
CORNER_REACHED = 8.0        # side line, from CORNER_RANGE cm away from it (further: the normal carry - carrying
CORNER_RANGE = 50.0         # sideways the whole way was slow and lost the ball) till within CORNER_REACHED of it
CORNER_SPD = 0.35           # carrying to the corner: speed multiplier (x this share while still turning to face the
CORNER_TURN_SHARE = 0.3     # side line: down to CORNER_TURN_SHARE facing 90 degrees off it)
CARRY_AIM_CORNER = True     # the normal carry (carry_dir()) heads for that same front corner of their penalty area
                            # instead of their goal's centre, till level with it (within CORNER_REACHED), then the goal
SHIELD_STAY = 6.0           # shielding: carries on while both robots are within SHIELD_IN - this inside the area
BASE_SHIELD = False         # (INT: off with the shield; also there's no "about to go out" at the line, rule 2.5)
                            # also start the shield outside the area when about to dribble the ball
BASE_SHIELD_Y = 40.0        # out near their goal line: ball within BASE_SHIELD_Y of their goal line, wide of their goal
BASE_SHIELD_EDGE = 12.0     # (|x| >= GOAL_W / 2), within BASE_SHIELD_EDGE of the side or goal line and heading out
BASE_SHIELD_T = 0.3         # over it within BASE_SHIELD_T s at its speed (the goal mouth counts as in), their goalie in
                            # their penalty area. Carries on while the ball is within BASE_SHIELD_Y + SHIELD_STAY of
                            # the line and the goalie within SHIELD_STAY of the area.
                            # Simulator (fast, 60 s, seeds 1-40): TS 103-5 -> 109-5 (losses 2 -> 0), v88 281-9 ->
                            # 247-10. Doesn't catch TS seed 0 186.6 s (their goalie knocks it off before the line);
                            # starting it on the way up the side lane there (goalie within 2R+15-30, or ball heading
                            # out within 0.4 s) kept that ball but lost badly: v88 29-17, TS 21-1 /40.
FEINT = True               # INT: carrying the ball within FEINT_RANGE of their goal, their goalie (the opponent
FEINT_RANGE = 120.0        # nearest their goal, within FEINT_KEEPER_D of its centre) between us and it: drive towards
FEINT_KEEPER_D = 50.0      # the side of the goal it's on (it follows us further that way, opening the other side) while
FEINT_MID = 4.0            # turning to face the open side (FEINT_AIM_IN inside the far post), and kick once that shot
FEINT_AIM_IN = 10.0        # is open by FEINT_ROOM. Goalie within FEINT_MID of the middle: its side = ours. At most
FEINT_ROOM = 2.0           # FEINT_MAX_TIME s, then the normal carry till we lose the ball.
FEINT_SPD = 0.7
FEINT_MAX_TIME = 1.5        # Simulator (fast, 60 s, seeds 1-80): ba 304-28 -> 314-26, v88 391-27 -> 463-21
                           # (range 90: ba 169-11, v88 218-15 /40 seeds 1-40; FEINT_SPD 1.0 / 3 s: no better)
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
    # INT: ball well round from up-field and close: face it and go straight at it (no orbiting round to face up-field)
    in_our_corner = abs(ball[0]) > OUR_CORNER_X and ball[1] < field.OWN_GOAL_Y + OUR_CORNER_Y
    if FACE_BALL and (FACE_BALL_ALL or in_our_corner) and d <= CHASE_FACE_DIST and abs(a) > CHASE_FACE_ANGLE:
        u = unit((ball[0] - me[0], ball[1] - me[1]))
        reach = R + cfg.BALL_RADIUS - 2.0
        target = [ball[0] - u[0] * reach, ball[1] - u[1] * reach]
        spd = FACE_BALL_SPEED if d < 25 else 0.9
        return clamp_in_field(target, FIELD_MARGIN), math.radians(-a), d <= 33, spd
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


def edge_capture(me, ball):
    """INT: the ball is on or past the white line (rule 2.5: still live, up to the walls) and close. Go partly over the
    line, to the field side of the ball, facing it with the dribbler on, to catch it and bring it back in. None when the
    ball is well inside the field (or in a goal) or still far away.
    (Every other target is kept FIELD_MARGIN inside the line, so without this the striker stopped 15 cm short of a ball
    by the wall and just followed it up and down. Letting all chase / carry targets out to the walls instead got it
    taken off for touching a wall ~0.5 times a game.)
    Returns (target field point, heading, dribbler on, speed multiplier) like chase()."""
    if not EDGE_CAPTURE or dist(me, ball) > EDGE_CAPTURE_RANGE:
        return None
    on_line = cfg.BALL_RADIUS + field.LINE_W          # ball centre within this of the line's outer edge = touching it
    over_x = abs(ball[0]) - (field.PLAY_W / 2 - on_line)
    over_y = abs(ball[1]) - (field.PLAY_L / 2 - on_line)
    if abs(ball[0]) <= field.GOAL_W / 2:
        over_y = -1.0                                 # in front of / in a goal: not an edge ball
    if over_x <= 0 and over_y <= 0:
        return None
    if EDGE_SKIP_CORNERS and over_x > -EDGE_CORNER and over_y > -EDGE_CORNER and not (EDGE_OUR_CORNERS and ball[1] < 0):
        return None                                   # in a corner: no room to get beside it without touching a wall
    # come at it from the field side: along the outward normal of the line(s) it's on (a corner: both)
    n = unit((math.copysign(1.0, ball[0]) if over_x > 0 else 0.0, math.copysign(1.0, ball[1]) if over_y > 0 else 0.0))
    reach = R + cfg.BALL_RADIUS - 1.0                 # robot centre this far from the ball: ball in the dribbler
    target = clamp_in_field([ball[0] - n[0] * reach, ball[1] - n[1] * reach], OUT_MARGIN + EDGE_WALL_GAP)
    heading = heading_for((ball[0] - me[0], ball[1] - me[1]))     # dribbler towards the ball
    spd = EDGE_CAPTURE_SPEED if dist(me, ball) < 25 else 0.9
    return around_ball(me, target, ball), heading, True, spd


def shot_open(compass, me, centres, max_range, room=0.0):
    """Kicking right now would score: our actual heading crosses their goal mouth within max_range (with room for
    the ball inside the posts) and the ball's path misses every robot (by more than `room` cm)."""
    fx, fy = -math.sin(compass), math.cos(compass)     # the way we face
    if fy < 0.2:
        return False                                   # not facing up-field
    t = (field.ATTACK_GOAL_Y - me[1]) / fy             # how far along our facing line their goal line is
    cross = (me[0] + fx * t, field.ATTACK_GOAL_Y)      # where we'd cross it
    in_range = 0 < t < max_range
    between_posts = abs(cross[0]) <= field.GOAL_W / 2 - cfg.BALL_RADIUS - POST_MARGIN
    if not (in_range and between_posts):
        return False
    return clearance(me, cross, centres) > room


def bank_shot_open(compass, me, centres, max_range=BANK_RANGE, room=BANK_ROOM):
    """INT: kicking right now would score off a side wall: our facing line hits a side wall before their goal
    line, and the bounce (angle in = angle out) crosses the goal mouth, both legs within max_range in all and clear of
    every robot by `room`. The wall keeps BANK_BOUNCE of the sideways speed (the up-field speed stays). (The ball is
    live up to the walls here, so a bank shot is fair.)"""
    if not BANK_SHOT:
        return False
    fx, fy = -math.sin(compass), math.cos(compass)
    if fy < BANK_MIN_FY or abs(fx) < 0.05:
        return False
    xw = math.copysign(field.WALL_W / 2 - cfg.BALL_RADIUS, fx)    # where the ball's centre turns at the wall
    t1 = (xw - me[0]) / fx
    wall = (xw, me[1] + fy * t1)
    if t1 <= 0 or wall[1] >= field.ATTACK_GOAL_Y:
        return False                                              # reaches their goal line before the wall
    t2 = (field.ATTACK_GOAL_Y - wall[1]) / fy
    cross = (wall[0] - BANK_BOUNCE * fx * t2, field.ATTACK_GOAL_Y)  # bounced: sideways speed flips, x BANK_BOUNCE
    if t1 + t2 > max_range or abs(cross[0]) > field.GOAL_W / 2 - cfg.BALL_RADIUS - POST_MARGIN:
        return False
    return clearance(me, wall, centres) > room and clearance(wall, cross, centres) > room


def clear_shot(compass, me, centres, max_range):
    """shot_open(), from further than LONG_SHOT_FROM only with a real opening (LONG_SHOT_ROOM past every robot)."""
    far = dist(me, (0.0, field.ATTACK_GOAL_Y)) > LONG_SHOT_FROM
    return shot_open(compass, me, centres, max_range, LONG_SHOT_ROOM if far else 0.0)


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


def meet_clear(me, plan):
    """INT (CLEAR_SHARE): where the striker waits for the goalie's clear kick. plan = [x, y, heading] the goalie
    sent: the ball rolls from there along that heading, slowing down (CLEAR_ROLL_*), until it stops or reaches the
    edge of the field. The first point on that path we can get to before the ball does (at CLEAR_MEET_SPEED, from
    CLEAR_MEET_MIN on), else where it stops - moved CLEAR_MEET_SIDE to our side of the line, inside the field."""
    src = (plan[0], plan[1])
    d = (-math.sin(plan[2]), math.cos(plan[2]))                  # the way the ball goes
    # how far it rolls: till it stops, or to the edge of the field
    roll = CLEAR_ROLL_SPEED / CLEAR_ROLL_DECAY * 0.95             # (it never quite stops: 95% of the way)
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
    # beside the line, on the side we're on
    cross = (me[0] - src[0]) * d[1] - (me[1] - src[1]) * d[0]   # > 0: we're to the right of the kick
    side = 1.0 if cross >= 0 else -1.0
    target = [p[0] + d[1] * side * CLEAR_MEET_SIDE, p[1] - d[0] * side * CLEAR_MEET_SIDE]
    return clamp_in_field(target, FIELD_MARGIN)


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
        self.shield_side = None  # shielding: +1 facing the right side line, -1 the left, 0 = gave up (till
        self.shield_since = None # we lose the ball), None = not shielding; and since when
        self.shield_shot = False # shielding: turning to an open shot
        self.shield_base = False # shielding started near their goal line, outside the area (BASE_SHIELD)
        self.feint_since = None  # FEINT: since when (0 = gave up till we lose the ball), and the goalie's side
        self.feint_side = None
        self.turn_away_side = None  # TURN_AWAY: turning round on the spot towards this side line (+1 / -1), or None
        self.edge_at = None      # last time we were going for an edge ball (edge_capture(), TURN_AFTER_EDGE)
        self.last_ball = None    # (position, time, went out?) of the last ball seen - for an out ball, the position
                                 # is the neutral spot it'll come back to
        self.mate_mem = MateMemory()
        self.goalie_on = False   # (INT: box_guard())
        self.box_guard = BoxGuard()

    def reset(self):
        self.fallback.reset()
        self.last_ball = None
        self.facing_since = None
        self.won_at = None

    def update(self, world, compass, mate, line_touches):
        """INT: the RCJA logic (_update()), then box_guard(): never completely inside a penalty area, and not in our
        own at all while our goalie is on (multiple defense)."""
        out = self._update(world, compass, mate, line_touches)
        return self.box_guard(out, world.pose, {-1: box_limit(partly_ok=not self.goalie_on), 1: box_limit()})

    def _update(self, world, compass, mate, line_touches):
        mate = self.mate_mem(mate)       # comms dropped: the teammate is still on (COMMS_GRACE)
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
        self.goalie_on = goalie_on       # (INT: for box_guard() in update())
        self.speed = {}

        # (a ball just past the line but still there isn't out yet - the referee takes it off once it's fully past:
        # keep playing it till then; giving up on it at once handed it to them)
        gone_out = (field.BALL_OUT_AT_LINE and ball is not None      # (INT, rule 2.5: never)
                    and not field.inside_play_area(ball[0], ball[1], -cfg.BALL_RADIUS - 3.0))
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
        elif self.meeting_clear(world, mate, me, ball, goalie_on):
            # ---- INT (CLEAR_SHARE): our goalie is about to clear it (or just has): straight to where we meet the kick
            target = self.avoid(me, meet_clear(me, mate["clear"]), centres)
            heading = heading_for((0.0 - me[0], field.ATTACK_GOAL_Y - me[1]))    # facing their goal
            dribbler = False
            self.chasing = True
            self.speed = {"spd_multi": 1.0}
            self.botstate, self.phase = 2, "meet clear"
        else:
            # ---- ball in play: chase / carry / shoot
            target, heading, dribbler = self.attack(world, compass, me, ball)
            self.chasing = True
            self.botstate = 1 if world.ball_in_capture else 2
            if self.phase not in ("edge capture", "turn away"):   # INT: edge_capture() goes partly over the line
                target = clamp_in_field(target, FIELD_MARGIN)

        # ---- goalie on: never into our penalty area
        if goalie_on:
            target = keep_out_of_box(target)
        return rel(target, pose), heading, dribbler

    @staticmethod
    def meeting_clear(world, mate, me, ball, goalie_on):
        """INT (CLEAR_SHARE): go to meet our goalie's clear kick - it has the ball and has told us where it'll kick
        it, or kicked it a moment ago and the ball isn't near us yet."""
        if not (CLEAR_SHARE and goalie_on and mate and mate.get("clear")) or world.ball_in_capture:
            return False
        return bool(mate.get("has ball")) or dist(me, ball) > CLEAR_MEET_NEAR

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

    @staticmethod
    def in_their_box(p, margin=0.0):
        """Point p at least `margin` cm inside their penalty area's front and side lines. (The goal line end doesn't
        count: their goalie standing with its back over the goal line is still in the area - requiring it fully
        inside there meant the shield never started with their goalie at home.)"""
        return abs(p[0]) <= field.PENALTY_W / 2 - margin and p[1] >= field.ATTACK_GOAL_Y - field.PENALTY_D + margin

    @classmethod
    def box_opponents(cls, world, me):
        """The opponents that matter: with us in their penalty area, only the ones in it too (the others can't get in
        to interfere); otherwise all of them."""
        if not cls.in_their_box(me):
            return list(world.opponents)
        return [o for o in world.opponents if cls.in_their_box(o)]

    @classmethod
    def box_centres(cls, world, me):
        """obstacle_centres(), with the same rule as box_opponents()."""
        centres = obstacle_centres(me, world.obstacles_field)
        if not cls.in_their_box(me):
            return centres
        return [c for c in centres if cls.in_their_box(c)]

    @staticmethod
    def box_corner(me):
        """The nearer front corner of their penalty area: just outside it, the deepest our robot may stand there.
        (INT, rules 2.6 / 2.8: RCJA used SHIELD_IN + CORNER_INSET inside it = completely inside the area.)"""
        side = math.copysign(1.0, me[0] or 1.0)
        return [side * (field.PENALTY_W / 2 - 10.0), field.ATTACK_GOAL_Y - GOALIE_LINE_D]

    def carry_target(self, me, goal):
        """Where the normal carry heads: the nearer front corner of their penalty area till we're level with it
        (CARRY_AIM_CORNER), then their goal."""
        corner = self.box_corner(me)
        if CARRY_AIM_CORNER and me[1] < corner[1] - CORNER_REACHED:
            return corner
        return goal

    def corner_setup(self, world, me, goal):
        """Their goalie is in their penalty area and their striker is behind us (further from their goal than us, or
        not seen) - CARRY_TO_CORNER."""
        if not world.opponents or self.in_their_box(me):
            return False                        # (already in their area: shield() / bodying, never back to a corner)
        keeper = min(world.opponents, key=lambda o: dist(o, goal))
        if not self.in_their_box(keeper):
            return False
        others = [o for o in world.opponents if o is not keeper]
        return all(o[1] < me[1] for o in others)

    def base_line_out(self, world, me, ball, keeper, staying):
        """Near their goal line, wide of their goal, their goalie in their penalty area, and the ball at the side /
        goal line about to go out (BASE_SHIELD)."""
        stay = SHIELD_STAY if staying else 0.0
        if ball[1] < field.ATTACK_GOAL_Y - BASE_SHIELD_Y - stay or abs(ball[0]) < field.GOAL_W / 2 - stay:
            return False
        if not self.in_their_box(keeper, -stay) or self.in_their_box(me, SHIELD_IN):
            return False                        # (fully in the area: the normal shield)
        if staying:
            return True
        if field.inside_play_area(ball[0], ball[1], BASE_SHIELD_EDGE):
            return False                        # not at the side / goal line yet
        vel = world.ball_vel
        ahead = (ball[0] + vel[0] * BASE_SHIELD_T, ball[1] + vel[1] * BASE_SHIELD_T)
        return not field.inside_play_area(ahead[0], ahead[1])

    def shield(self, world, compass, me, ball, goal):
        """Ball in our dribbler, our robot and their goalie both fully inside their penalty area: face the nearer
        side line (the ball away from the goalie and the goal), shove the goalie off the near half of the goal with
        our body, then turn to the gap and shoot (SHIELD_*).
        Returns (target, heading, dribbler) like attack(), or None when not shielding."""
        now = time.monotonic()
        opponents = self.box_opponents(world, me)
        if self.shield_side == 0 or not opponents:
            return None                                        # gave up on this possession (SHIELD_MAX_TIME)
        keeper = min(opponents, key=lambda o: dist(o, goal))
        margin = SHIELD_IN if self.shield_side is None else SHIELD_IN - SHIELD_STAY   # (in: fully inside; stays on
        in_box = self.in_their_box(me, margin) and self.in_their_box(keeper, margin)  # till SHIELD_STAY further out)
        # (or about to dribble it out near their goal line - BASE_SHIELD)
        staying = self.shield_side is not None and self.shield_base
        base = BASE_SHIELD and not in_box and self.base_line_out(world, me, ball, keeper, staying)
        if not (in_box or base):
            self.shield_side = None
            return None
        if self.shield_side is None:
            self.shield_base = base
            # the nearer side line (in the middle: the side away from their goalie)
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

        # a gap into the goal: turn to it and kick once facing it
        aim, clear = best_aim(ball, centres)
        self.shield_shot = clear >= (SHIELD_SHOT_KEEP if self.shield_shot else SHIELD_SHOT_CLEAR)
        if self.shield_shot:
            if shot_open(compass, me, centres, field.PENALTY_D + 20.0, SHIELD_KICK_CLEAR):
                self.kick = True
            to_aim = unit((aim[0] - me[0], aim[1] - me[1]))
            self.speed = {"spd_multi": SHIELD_SHOT_SPD}
            self.phase = "shield shot"
            return [me[0] + to_aim[0] * 20, me[1] + to_aim[1] * 20], heading_for(to_aim), True

        # shield: face the nearer side line, turned a bit away from their goal (heading +90 deg = facing -x)
        face = (side * math.cos(SHIELD_FACE_BACK), -math.sin(SHIELD_FACE_BACK))
        heading = heading_for(face)
        # ...and shove their goalie with our body: drive at a point past it, towards the far post
        away = unit((keeper[0] - me[0], keeper[1] - me[1]))
        push = unit((away[0] - side * SHIELD_PUSH_ACROSS, away[1]))
        target = [keeper[0] + push[0] * SHIELD_PUSH, keeper[1] + push[1] * SHIELD_PUSH]
        self.speed = {"spd_multi": 1.0}
        self.phase = "shield"
        return target, heading, True

    def feint(self, world, compass, me, goal, centres, now):
        """FEINT: drive at the goalie's side of their goal, face the open side, kick when it's open.
        Returns (target, heading, dribbler) or None (not now: the normal carry)."""
        if self.feint_since == 0 or not world.opponents or dist(me, goal) > FEINT_RANGE:
            return None
        keeper = min(world.opponents, key=lambda o: dist(o, goal))
        if dist(keeper, goal) > FEINT_KEEPER_D or keeper[1] < me[1] + R:
            return None                                      # no goalie between us and their goal
        if self.feint_since is None:
            self.feint_since = now
            mid = abs(keeper[0]) < FEINT_MID
            self.feint_side = math.copysign(1.0, (me[0] if mid else keeper[0]) or 1.0)
        elif now - self.feint_since > FEINT_MAX_TIME:
            self.feint_since = 0                             # gave up: the normal carry till we lose the ball
            return None
        side = self.feint_side
        aim = (-side * (field.GOAL_W / 2 - FEINT_AIM_IN), field.ATTACK_GOAL_Y)    # the open side
        heading = heading_for((aim[0] - me[0], aim[1] - me[1]))
        if shot_open(compass, me, centres, FEINT_RANGE + 20.0, FEINT_ROOM):
            self.kick = True
        post = (side * field.GOAL_W / 2, field.ATTACK_GOAL_Y)                     # the goalie's side
        u = unit((post[0] - me[0], post[1] - me[1]))
        self.speed = {"spd_multi": FEINT_SPD}
        self.phase = "feint"
        return [me[0] + u[0] * 40, me[1] + u[1] * 40], heading, True

    def attack(self, world, compass, me, ball):
        """Ball in play. Returns (target field point, heading, dribbler); sets .kick / .speed / .phase."""
        goal = (0.0, field.ATTACK_GOAL_Y)                                  # centre of their goal
        heading = heading_for((goal[0] - me[0], goal[1] - me[1]))          # facing their goal

        # ---- no ball in the dribbler: chase it
        if not world.ball_in_capture:
            self.facing_since = None
            self.shield_side = None
            self.feint_since = None
            self.turn_away_side = None
            target, heading, dribbler, spd = chase(me, ball)
            edge = edge_capture(me, ball)                                 # INT: ball on / past the line
            if edge is not None:
                target, heading, dribbler, spd = edge
                self.speed = {"spd_multi": spd}
                self.phase = "edge capture"
                self.edge_at = time.monotonic()
                return target, heading, dribbler
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

        # ---- INT: ball caught facing back towards our goal, near it (FACE_BALL in our corners), or just caught off
        # the line (TURN_AFTER_EDGE): turn round
        # the way that swings the ball away from our goal (face the near side line first), never across the goal mouth.
        # It turns fully round on the spot (via the side line, then on to their goal) before driving off (Jerold,
        # 19 Oct 2026: better than turning while moving)
        fwd = (-math.sin(compass), math.cos(compass))
        just_edge = self.edge_at is not None and time.monotonic() - self.edge_at < TURN_EDGE_TIME
        self.edge_at = None
        if self.turn_away_side is None and ((TURN_AWAY and fwd[1] < TURN_AWAY_FY and
                                             me[1] < field.OWN_GOAL_Y + TURN_AWAY_Y) or
                                            (TURN_AFTER_EDGE and just_edge)):
            self.turn_away_side = math.copysign(1.0, me[0] or 1.0)    # the side line on our side
        if self.turn_away_side is not None:
            if abs(angle_diff(heading, compass)) > TURN_AWAY_DONE:
                side = self.turn_away_side
                # still facing well back: the side line first (so it turns the outside way), then their goal
                # (from TURN_AWAY_SIDE_FY on: the short way to their goal is already round the outside)
                face = heading_for((side, 0.0)) if fwd[1] < TURN_AWAY_SIDE_FY else heading
                self.speed = {"spd_multi": TURN_AWAY_SPEED}
                self.phase = "turn away"
                return [me[0], me[1]], face, True
            self.turn_away_side = None                                  # turned round: carry as usual

        # ---- ball in the dribbler near their goal, their robot in the way: shield it and go round them
        if SHIELD:
            out = self.shield(world, compass, me, ball, goal)
            if out is not None:
                return out

        # ---- ball in the dribbler, an opponent pressed against our front: step past it (SLIP_SIDE across, away
        # from the side it's on, and SLIP_FORWARD forwards)
        ho = head_on(me, compass, self.box_opponents(world, me))
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
        centres = self.box_centres(world, me)
        if FEINT:
            out = self.feint(world, compass, me, goal, centres, now)
            if out is not None:
                return out
        if self.won_at is None:
            self.won_at = now
        facing_goal_mouth = shot_open(compass, me, [], RUSH_FLICK_RANGE)   # [] = ignore robots in the way
        if RESTART_RUSH and self.restart_ball(ball) and facing_goal_mouth:
            self.kick = True                                   # restart race: flick it up-field straight away
            flicked = True
        else:
            flicked = False
        just_won_it = now - self.won_at < QUICK_SHOT_TIME
        if just_won_it and clear_shot(compass, me, centres, QUICK_SHOT_RANGE):
            self.kick = True                                   # quick shot: before they close us down
        # (in their penalty area: only the opponents in it count - box_opponents())
        pairs = [(o, v) for o, v in zip(world.opponents, world.opponent_vel)
                 if not self.in_their_box(me) or self.in_their_box(o)]
        opps, vels = [o for o, _ in pairs], [v for _, v in pairs]
        u = carry_dir(me, self.carry_target(me, goal), opps, vels)   # towards the goal (or the corner of their
                                                                       # penalty area on the way), front away from them
        carry_heading = heading_for(u)
        # just after a rush: slow down while turning (at rush speed it couldn't turn away from an opponent)
        carry_spd = 1.0
        just_rushed = self.rushed_at is not None and now - self.rushed_at <= RUSH_TURN_TIME
        turn = abs(angle_diff(carry_heading, compass))          # how far the carry has to turn us
        if just_rushed and turn >= RUSH_TURN_FREE:
            carry_spd = max(RUSH_TURN_MIN_SPD, 1.0 - (turn - RUSH_TURN_FREE) / RUSH_TURN_SLOWEST * 0.6)
        self.speed = {"spd_multi": carry_spd}
        self.phase = "flick" if flicked else "carry"
        if charged(me, opps, vels) and clear_shot(compass, me, centres, OPEN_SHOT_RANGE):
            self.kick = True                                   # about to be shoved off it: shoot
        if bank_shot_open(compass, me, centres) and not clear_shot(compass, me, centres, OPEN_SHOT_RANGE):
            self.kick = True                                   # INT: off the side wall into the goal
            self.phase = "bank shot"
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
        # set up the shield: their goalie in their penalty area and their striker behind us (or not seen) -> carry
        # it to the nearer front corner of their penalty area facing the nearer side line, the ball hidden from the
        # goalie, so we arrive in the area already in the shield (shield())
        # (once there, shield() takes over - or if it doesn't start, the normal carry)
        if SHIELD and CARRY_TO_CORNER and self.shield_side != 0 and self.corner_setup(world, me, goal):
            side = math.copysign(1.0, me[0] or 1.0)
            corner = self.box_corner(me)
            if CORNER_REACHED < dist(me, corner) < CORNER_RANGE:
                # slowly (it's all sideways to the way we face), and slower still while turning: too fast lost the
                # ball (the dribbler only holds it so fast sideways)
                corner_heading = heading_for((side, 0.0))
                off = abs(angle_diff(corner_heading, compass))
                self.speed = {"spd_multi": CORNER_SPD * max(CORNER_TURN_SHARE, math.cos(min(off, math.pi / 2)))}
                self.phase = "carry corner"
                return corner, corner_heading, True
        return target, carry_heading, True

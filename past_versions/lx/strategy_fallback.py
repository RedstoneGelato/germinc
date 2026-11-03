"""
strategy_fallback.py - the comp state machines (relative coordinates), used by strategy.py only while the robot
doesn't know where it is on the field (localisation not confident). GoalieFallback / StrikerFallback.
Both have update(...) -> (desired_pos, desired_heading, dribbler_on) and a `speed` dict for motion.Mover.step().

Same states as before; the inputs are now camera based and in cm (relative = field axes, robot at the origin):
    world.ball            ball (was the IR ballpos)
    world.ball_in_capture ball in the dribbler (was "front IR sensors distance == 3")
    world.attack_goal / own_goal   Goal(near, centre) (was goalpos / own_goalpos in camera pixels)

All the distances below were converted from the old units by eye - TUNE them on the field.
"""
import math

import field
import robot_config as cfg
from utils import Hysteresis, rotate, wrap_pi

# ---- TUNE (cm)
GOALIE_DIST = 20.0           # botstate 0: stand so the own goal's near point is this far behind the robot centre
GOALIE_DIST_BALL = 15.0      # botstate 3: same, while tracking the ball sideways
GOALIE_MAX_X = field.PENALTY_W / 2 - 5.0   # botstate 3: don't follow the ball further sideways than this (needs a confident pose)
BACKUP_Y = -60.0             # own goal not known at all: back up this way
BALL_CLOSE = 30.0            # botstate 2 trigger: ball this close...
BALL_CLOSE_ANGLE = math.radians(75)   # ...and within this angle of the robot's front
BEHIND_Y = 12.0              # ball further back than this (relative y) = need to get behind it (was 50)
BEHIND_Y_HYST = 20.0         # (was 80)
FAR_BEHIND_Y = -35.0         # (was -150)
WRAP_X = 25.0                # (was 110)
WRAP_OFFSET = 40.0           # (was 200)
BALL_AHEAD = 5.0             # (was 20)
APPROACH_SIDE_X = 20.0       # (was 80)
APPROACH_BEHIND = 15.0       # (was 60)
DRIBBLER_RANGE = 35.0        # (was 150)
DEFAULT_GOAL = [0.0, 200.0]  # aim here if the attack goal is unknown


def heading_to(goal):
    return wrap_pi(math.atan2(goal[1], goal[0] * 1.6) - math.pi / 2)


class GoalieFallback:
    def __init__(self):
        self.botstate_hyst = Hysteresis(hold_time=0.11)
        self.substate1_hyst = Hysteresis(hold_time=0.11, instant_enter=lambda v: v == 1)
        self.substate2_hyst = Hysteresis(hold_time=0.11, instant_enter=lambda v: v == 1)
        self.botstate = 3
        self.substate1 = 4
        self.substate2 = 4
        self.speed = {}   # default speed profile (see motion.Mover.step)
        self.chasing = False

    def reset(self):
        self.botstate_hyst.reset()
        self.substate1_hyst.reset()
        self.substate2_hyst.reset()

    @staticmethod
    def ball_close_in_front(world, compass):
        if world.ball is None:
            return False
        b = rotate(world.ball, -compass)   # robot frame
        return math.hypot(*b) < BALL_CLOSE and abs(math.atan2(b[0], b[1])) < BALL_CLOSE_ANGLE

    def stay_in_goal(self, world, x_target, dist):
        """Position in front of our own goal. Returns desired_pos."""
        own, att = world.own_goal, world.attack_goal
        if own is not None:
            x = x_target if x_target is not None else own.centre[0]
            pose = world.pose
            if pose is not None and pose.confident:   # don't wander out of the penalty area sideways
                x = min(max(x, -GOALIE_MAX_X - pose.x), GOALIE_MAX_X - pose.x)
            return [x, own.near[1] + dist]
        return [att.centre[0] if att is not None else 0.0, BACKUP_Y]

    def chase(self, world, sub, line_touches, shoot):
        """Go for the ball. shoot=True: score (botstate 1), False: pass/clear (botstate 2).
        Returns (desired_pos, desired_heading, dribbler_on)."""
        ball = world.ball
        goal = world.attack_goal.centre if world.attack_goal is not None else DEFAULT_GOAL
        if sub == 1:
            if shoot:
                return [goal[0] * 1.5, goal[1]], heading_to(goal), True
            return list(goal), heading_to(goal), True
        if sub == 2:
            return (list(ball) if shoot else [0, -WRAP_OFFSET]), 0, False
        if sub == 3:
            if abs(ball[0]) < WRAP_X and ball[1] < 0: #wrap around ball
                if line_touches > 1: #touched line
                    pos = [-WRAP_OFFSET, 0] if ball[0] < 0 else [WRAP_OFFSET, 0] #go other way
                else:
                    pos = [-WRAP_OFFSET, 0] if ball[0] > 0 else [WRAP_OFFSET, 0] #wrap around base on ball position
            elif shoot:
                pos = [0, -WRAP_OFFSET] if ball[1] > BALL_AHEAD else [ball[0], -WRAP_OFFSET]
            else:
                pos = [0, -WRAP_OFFSET]
            return pos, 0, False
        # sub == 4: pathfind to ball
        heading = heading_to(goal) if shoot else 0
        if ball[1] < BEHIND_Y_HYST and abs(ball[0]) > APPROACH_SIDE_X:
            pos = [ball[0], -3]
        else:
            pos = [ball[0], ball[1] - APPROACH_BEHIND]
        return pos, heading, abs(pos[0]) + abs(pos[1]) < DRIBBLER_RANGE

    def substate(self, world, current):
        ball = world.ball
        if world.ball_in_capture:
            return 1  #ball in bcz
        if ball[1] < (BEHIND_Y if current in (1, 4) else BEHIND_Y_HYST): #2 far backup, 3 close backup
            return 2 if ball[1] < FAR_BEHIND_Y else 3
        return 4  #pathfind to ball

    def update(self, world, compass, comms_command, attack_bot_state, line_touches):
        """Returns (desired_pos, desired_heading, dribbler_on)."""
        # ---- determine states
        if comms_command == 0: #signal from attack to chill
            raw_botstate = 3
        elif world.ball is None: #doesnt see ball
            raw_botstate = 0
        elif attack_bot_state == 0 or attack_bot_state is None: #attack bot is off
            raw_botstate = 1
        elif comms_command == 1 or self.ball_close_in_front(world, compass): #signal from other bot to go get ball
            raw_botstate = 2
        else: #chill in goals
            raw_botstate = 3
        self.botstate = self.botstate_hyst.update(raw_botstate) #smoothing
        self.chasing = self.botstate in (1, 2)

        # hysteresis can hold a ball state for a moment after the ball is gone
        if world.ball is None and self.botstate in (1, 2):
            self.botstate = 0

        # ---- state machine
        if self.botstate == 0: #do not see ball
            return self.stay_in_goal(world, None, GOALIE_DIST), 0, False
        if self.botstate == 1: #go for ball then score
            self.substate1 = self.substate1_hyst.update(self.substate(world, self.substate1))
            return self.chase(world, self.substate1, line_touches, shoot=True)
        if self.botstate == 2: # go for ball then pass
            self.substate2 = self.substate2_hyst.update(self.substate(world, self.substate2))
            return self.chase(world, self.substate2, line_touches, shoot=False)
        # botstate 3: chill in goals, follow the ball sideways
        return self.stay_in_goal(world, world.ball[0] if world.ball else None, GOALIE_DIST_BALL), 0, False


# ==================================== STRIKER ====================================
# Ported from the comp 1attack.py. Old values in brackets - TUNE (cm).
S_BEHIND_Y = 12.0            # ball further back than this = need to get behind it (50)
S_BEHIND_Y_HYST = 20.0       # (80)
S_FAR_BEHIND_Y = -40.0       # (-160)
S_WRAP_X = 25.0              # (110)
S_WRAP_OFFSET = 40.0         # (200)
S_APPROACH_SIDE_X = 20.0     # (80)
S_APPROACH_BEHIND = 15.0     # (60)
S_DRIBBLER_RANGE = 35.0      # (150)
S_MIDFIELD_FROM_GOAL = 90.0  # botstate 0: wait this far in front of the attack goal (180)
S_ADVANCE_Y = 60.0           # botstate 0, attack goal unknown: drive this way (250)
S_HOME_DIST = 25.0           # botstate 3: stand this far in front of our own goal (100)
S_HOME_CLOSE = -8.0          # botstate 3: own goal nearer than this behind us = already home (-20)
S_BACKUP_Y = -50.0           # botstate 3, own goal unknown: back up this way (-200)
S_SHOOT_MULTI = 1.5          # speed multiplier while carrying the ball to the goal


class StrikerFallback:
    """botstate 0 = no ball, goalie on    -> wait at midfield
       botstate 1 = ball in capture zone  -> drive at the goal
       botstate 2 = ball seen             -> get behind it and take it (substates 2/3/4 like the goalie)
       botstate 3 = no ball, goalie off   -> go home and defend
    command (sent to the goalie): 1 = go get the ball, 0 = stay in goal."""

    def __init__(self):
        self.botstate_hyst = Hysteresis(hold_time=0.11, instant_enter=lambda v: v == 1)
        self.substate_hyst = Hysteresis(hold_time=0.11)
        self.botstate = 2
        self.substate = 4
        self.command = 1
        self.ingoalspd = cfg.BASE_SPEED // 3
        self.speed = {}
        self.chasing = False

    def reset(self):
        self.botstate_hyst.reset()
        self.substate_hyst.reset()

    def update(self, world, compass, goalie_bot_state, line_touches):
        """Returns (desired_pos, desired_heading, dribbler_on). Also sets self.command and self.speed."""
        ball = world.ball
        att, own = world.attack_goal, world.own_goal
        goal = att.centre if att is not None else DEFAULT_GOAL

        # ---- determine states
        if ball is None: #doesnt see ball
            raw_botstate = 0 if goalie_bot_state == 1 else 3
        elif world.ball_in_capture: # ball in ball capture zone
            raw_botstate = 1 #try to shoot
        else:
            raw_botstate = 2 #try to get possession of ball
        self.botstate = self.botstate_hyst.update(raw_botstate)
        if ball is None and self.botstate == 2:   # hysteresis holding a ball state with no ball
            self.botstate = 0 if goalie_bot_state == 1 else 3

        # ---- state machine
        dribbler_on = False
        desired_heading = 0
        if self.botstate == 0: # do not see ball
            self.command = 1
            if att is not None:
                desired_pos = [att.centre[0], att.centre[1] - S_MIDFIELD_FROM_GOAL] # go midfield
                self.ingoalspd = int(cfg.BASE_SPEED / 5)
            else:
                desired_pos = [own.centre[0] if own is not None else 0.0, S_ADVANCE_Y]
                self.ingoalspd = cfg.BASE_SPEED

        elif self.botstate == 1: # shoot
            self.command = 0
            desired_pos = [goal[0] * 1.5, goal[1]]
            desired_heading = heading_to(goal)
            dribbler_on = True

        elif self.botstate == 2: # go for ball
            if ball[1] < (S_BEHIND_Y if self.substate in (1, 4) else S_BEHIND_Y_HYST):
                raw_substate = 2 if ball[1] < S_FAR_BEHIND_Y else 3  # far vs near backup
            else:
                raw_substate = 4 # just go for ball
            self.substate = self.substate_hyst.update(raw_substate)

            if self.substate == 2:
                desired_pos = list(ball)
                self.command = 1 #send goalie to get ball
            elif self.substate == 3:
                if abs(ball[0]) < S_WRAP_X and ball[1] < 0:
                    if line_touches > 1:
                        desired_pos = [-S_WRAP_OFFSET, 0] if ball[0] < 0 else [S_WRAP_OFFSET, 0]
                    else:
                        desired_pos = [-S_WRAP_OFFSET, 0] if ball[0] > 0 else [S_WRAP_OFFSET, 0]
                else:
                    desired_pos = [0, -S_WRAP_OFFSET]
            else: # just go for ball
                self.command = 0
                desired_heading = heading_to(goal)
                if ball[1] < S_BEHIND_Y_HYST and abs(ball[0]) > S_APPROACH_SIDE_X:
                    desired_pos = [ball[0], -3]
                else:
                    desired_pos = [ball[0], ball[1] - S_APPROACH_BEHIND]
                dribbler_on = abs(desired_pos[0]) + abs(desired_pos[1]) < S_DRIBBLER_RANGE

        else: # botstate 3: goalie is off, go home
            self.command = 1
            if own is not None: #align middle and go backwards
                desired_pos = [own.centre[0], own.near[1] + S_HOME_DIST] if own.near[1] < S_HOME_CLOSE else [own.centre[0], 0]
            else:
                desired_pos = [att.centre[0] if att is not None else 0.0, S_BACKUP_Y]
                self.ingoalspd = cfg.BASE_SPEED

        # speed profile (same rules as the comp code)
        self.speed = {"spd_min": 0.4,
                      "base_speed": cfg.BASE_SPEED if self.botstate in (1, 2) else self.ingoalspd,
                      "spd_multi": S_SHOOT_MULTI if self.botstate == 1 else None}
        return desired_pos, desired_heading, dribbler_on

"""
motion.py - desired movement (relative vector + heading) -> four motor speeds.
Same maths as the old main loop; distances are now cm instead of camera pixels / IR units.
"""
import math

import robot_config as cfg
from lines import limit_outward
from utils import wrap_pi


def VelocityToMotor(xvel, yvel, rot, maxspd): #convert variables into specific motor speed values
    motor1 = xvel*math.cos(math.pi/4) + yvel*math.sin(math.pi/4) - rot
    motor2 = xvel*math.cos(3*math.pi/4) + yvel*math.sin(3*math.pi/4) - rot
    motor3 = xvel*math.cos(5*math.pi/4) + yvel*math.sin(5*math.pi/4) - rot
    motor4 = xvel*math.cos(7*math.pi/4) + yvel*math.sin(7*math.pi/4) - rot

    scale = maxspd/max(abs(motor1), abs(motor2), abs(motor3), abs(motor4), 1)
    motor1 *= scale
    motor2 *= scale
    motor3 *= scale
    motor4 *= scale

    return int(motor1),int(motor2),int(motor3),int(motor4)


def dribble_speed_multi(move, compass):
    """1 when moving the way the robot faces, down to cfg.DRIBBLE_SPEED_MIN moving straight backwards (linear in
    the angle between them). move = relative movement vector, compass = robot heading."""
    if math.hypot(move[0], move[1]) < 1e-6:
        return 1.0
    angle = abs(wrap_pi(math.atan2(-move[0], move[1]) - compass))   # 0 = straight ahead, pi = straight back
    return 1.0 - (1.0 - cfg.DRIBBLE_SPEED_MIN) * angle / math.pi


class Mover:
    def __init__(self):
        self.reset()

    def reset(self):
        self.new_maxspd = 0
        self.new_desired_pos = [0, 0]
        self.last_speeds = (0, 0, 0, 0)

    def step(self, desired_pos, desired_heading, compass, line, pose,
             base_speed=cfg.BASE_SPEED, spd_min=0.2, spd_multi=None, dribbler_on=False, smooth=0.1):
        """desired_pos = relative cm vector to drive towards, desired_heading = radians (0 = facing attack goal).
        line = LineState, pose = localisation Pose (or None). Returns 4 motor speeds.
        base_speed / spd_min / spd_multi come from the strategy's `speed` dict:
            base_speed  speed at full distance, spd_min  slowest multiplier when close,
            spd_multi   if set, use this multiplier instead of the distance-based one (striker shooting),
            smooth      how much of each new command is taken per loop (0.1 = smooth but ~0.1 s behind; the goalie
                        uses more to react to shots).
        dribbler_on: slow down the more the movement direction is off the facing direction (cfg.DRIBBLE_SPEED_MIN)."""
        heading_error = wrap_pi(desired_heading - compass) #angle difference between desired and actual
        spin_weight = cfg.BASE_SPIN * max(0.1,min(abs(heading_error),2)) if heading_error != 0 else cfg.BASE_SPIN #scale spd based on how much angle difference
        if abs(heading_error) < 0.05: #dont spin if difference too small
            rot = 0
        else:
            rot = spin_weight * heading_error

        desired_pos = limit_outward(desired_pos, pose) #slow down before the boundary (needs a confident pose)

        spd_scale_helper = max(min(abs(desired_pos[0]) + abs(desired_pos[1]),110),0) # cm (was 220 old units)
        if spd_multi is None:
            spd_multi = 0.00004 * (spd_scale_helper ** 2) + 0.004 * spd_scale_helper + 0.1 #quadratic scaling of speed, further the bot wants to go, faster itll go
            spd_multi = max(min(spd_multi,1),spd_min)
        maxspd = round(base_speed * (1 + (abs(rot) / 160)) * spd_multi)
        maxspd *= line.speed_multi
        if dribbler_on:
            maxspd *= dribble_speed_multi(desired_pos, compass)
        self.new_maxspd = self.new_maxspd * (1 - smooth) + maxspd * smooth #alpha beta smoothing

        self.new_desired_pos = [desired_pos[0] * smooth + self.new_desired_pos[0] * (1 - smooth), desired_pos[1] * smooth + self.new_desired_pos[1] * (1 - smooth)] #alpha beta smoothing of desired position
        if line.on_line and line.escape is not None:
            self.new_desired_pos = [line.escape[0] * 100, line.escape[1] * 100]  # straight away from the line
            self.new_maxspd = cfg.LINE_ESCAPE_SPEED

        xvel = self.new_desired_pos[0]
        yvel = self.new_desired_pos[1]
        x_field = -yvel
        y_field = xvel
        angle = -compass
        x_robot = x_field * math.cos(angle) - y_field * math.sin(angle)
        y_robot = x_field * math.sin(angle) + y_field * math.cos(angle)

        speeds = VelocityToMotor(x_robot,y_robot,rot,self.new_maxspd) #convert variables into motor speed
        if dribbler_on and not line.on_line: #limit how fast the wheel speeds change so the ball isn't jerked out of the dribbler
            step = cfg.DRIBBLE_MAX_ACCEL * cfg.BASE_SPEED * cfg.CONTROL_PERIOD
            change = [new - old for new, old in zip(speeds, self.last_speeds)]
            biggest = max(abs(c) for c in change)
            if biggest > step: #scale all 4 changes together so the direction stays the same
                speeds = tuple(int(old + c * step / biggest) for old, c in zip(self.last_speeds, change))
        self.last_speeds = speeds
        return speeds

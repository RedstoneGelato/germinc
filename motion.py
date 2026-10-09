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


class Mover:
    def __init__(self):
        self.reset()

    def reset(self):
        self.new_maxspd = 0
        self.new_desired_pos = [0, 0]

    def step(self, desired_pos, desired_heading, compass, line, pose,
             base_speed=cfg.BASE_SPEED, spd_min=0.2, spd_multi=None):
        """desired_pos = relative cm vector to drive towards, desired_heading = radians (0 = facing attack goal).
        line = LineState, pose = localisation Pose (or None). Returns 4 motor speeds.
        base_speed / spd_min / spd_multi come from the strategy's `speed` dict:
            base_speed  speed at full distance, spd_min  slowest multiplier when close,
            spd_multi   if set, use this multiplier instead of the distance-based one (striker shooting)."""
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
        self.new_maxspd = self.new_maxspd * 0.9 + maxspd * 0.1 #alpha beta smoothing

        self.new_desired_pos = [desired_pos[0] * 0.1 + self.new_desired_pos[0] * 0.9, desired_pos[1] * 0.1 + self.new_desired_pos[1] * 0.9] #alpha beta smoothing of desired position
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

        return VelocityToMotor(x_robot,y_robot,rot,self.new_maxspd) #convert variables into motor speed

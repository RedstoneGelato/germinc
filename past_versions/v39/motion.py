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


BRAKE_SMOOTH = 0.5   # velocity smoothing when the new direction is the opposite of the current one (blends from
                     # `smooth` when it's the same): sharp changes of direction / slowing down react much faster
TURN_DEADBAND = 0.03 # rad: closer than this to the wanted heading = don't turn


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
        self.vel = [0.0, 0.0]          # smoothed velocity command, relative (field axes), motor speed units
        self.last_speeds = (0, 0, 0, 0)

    def step(self, desired_pos, desired_heading, compass, line, pose,
             base_speed=cfg.BASE_SPEED, spd_min=0.2, spd_multi=None, dribbler_on=False, smooth=0.1):
        """desired_pos = relative cm vector to drive towards, desired_heading = radians (0 = facing attack goal).
        line = LineState, pose = localisation Pose (or None). Returns 4 motor speeds.
        base_speed / spd_min / spd_multi come from the strategy's `speed` dict:
            base_speed  speed at full distance, spd_min  slowest multiplier when close,
            spd_multi   if set, use this multiplier instead of the distance-based one (striker shooting),
            smooth      how much of each new velocity is taken per loop when going the same way (0.1 = smooth but
                        ~0.1 s behind; the goalie uses more to react to shots). Changing direction / slowing down
                        takes more of it (up to BRAKE_SMOOTH), so the robot doesn't drift on sharp turns.
        Turning has its own speed, proportional to the heading error (cfg.TURN_MAX, cfg.TURN_FULL_ERR), so a robot
        that moves slowly still turns quickly.
        dribbler_on: slow down the more the movement direction is off the facing direction (cfg.DRIBBLE_SPEED_MIN),
        turn at most cfg.DRIBBLE_TURN_MAX and change wheel speeds gently (cfg.DRIBBLE_MAX_ACCEL)."""
        # ---- turning: faster the further off the wanted heading
        heading_error = wrap_pi(desired_heading - compass) #angle difference between desired and actual
        turn_max = cfg.DRIBBLE_TURN_MAX if dribbler_on else cfg.TURN_MAX
        if abs(heading_error) < TURN_DEADBAND: #dont spin if difference too small
            rot = 0.0
        else:
            rot = cfg.BASE_SPEED * turn_max * max(-1.0, min(1.0, heading_error / cfg.TURN_FULL_ERR))

        # ---- moving
        desired_pos = limit_outward(desired_pos, pose) #slow down before the boundary (needs a confident pose)
        spd_scale_helper = max(min(abs(desired_pos[0]) + abs(desired_pos[1]),110),0) # cm (was 220 old units)
        if spd_multi is None:
            spd_multi = 0.00004 * (spd_scale_helper ** 2) + 0.004 * spd_scale_helper + 0.1 #quadratic scaling of speed, further the bot wants to go, faster itll go
            spd_multi = max(min(spd_multi,1),spd_min)
        speed = base_speed * spd_multi * line.speed_multi
        if dribbler_on:
            speed *= dribble_speed_multi(desired_pos, compass)
        d = math.hypot(desired_pos[0], desired_pos[1])
        want = [desired_pos[0] / d * speed, desired_pos[1] / d * speed] if d > 0.5 else [0.0, 0.0]

        if line.on_line and line.escape is not None:
            self.vel = [line.escape[0] * cfg.LINE_ESCAPE_SPEED, line.escape[1] * cfg.LINE_ESCAPE_SPEED]  # straight away from the line
        else:
            v = self.vel
            vm, wm = math.hypot(v[0], v[1]), math.hypot(want[0], want[1])
            cos = (v[0] * want[0] + v[1] * want[1]) / (vm * wm) if vm > 1 and wm > 1 else 1.0
            a = smooth + (BRAKE_SMOOTH - smooth) * (1.0 - cos) / 2.0   # same direction: smooth, opposite: brake hard
            if wm < vm:
                a = max(a, 2 * smooth)                                  # slowing down is quicker than speeding up
            self.vel = [v[0] + (want[0] - v[0]) * a, v[1] + (want[1] - v[1]) * a] #alpha beta smoothing

        # relative (field axes) -> robot frame, the axes VelocityToMotor uses
        x_field = -self.vel[1]
        y_field = self.vel[0]
        angle = -compass
        x_robot = x_field * math.cos(angle) - y_field * math.sin(angle)
        y_robot = x_field * math.sin(angle) + y_field * math.cos(angle)
        trans = VelocityToMotor(x_robot, y_robot, 0, math.hypot(x_robot, y_robot))   # biggest wheel = the speed
        speeds = [m - rot for m in trans]                                          # + turning on top
        biggest = max(abs(m) for m in speeds)
        if biggest > cfg.MOTOR_MAX:                                                # too much: scale everything down
            speeds = [m * cfg.MOTOR_MAX / biggest for m in speeds]
        speeds = tuple(int(m) for m in speeds)
        if dribbler_on and not line.on_line: #limit how fast the wheel speeds change so the ball isn't jerked out of the dribbler
            step = cfg.DRIBBLE_MAX_ACCEL * cfg.BASE_SPEED * cfg.CONTROL_PERIOD
            change = [new - old for new, old in zip(speeds, self.last_speeds)]
            biggest = max(abs(c) for c in change)
            if biggest > step: #scale all 4 changes together so the direction stays the same
                speeds = tuple(int(old + c * step / biggest) for old, c in zip(self.last_speeds, change))
        self.last_speeds = speeds
        return speeds

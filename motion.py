import math

import robot_config as cfg
from lines import limit_outward
from utils import wrap_pi

BRAKE_SMOOTH = 0.5
TURN_DEADBAND = 0.03
TURN_RESUME = 0.10
PARK_ENTER = 2.0
PARK_RESUME = 5.0
BRAKE_LEAD = 12.0    # TUNE: cm the robot still travels at full speed (BASE_SPEED) before slowing down takes effect
SPEED_CURVE_A, SPEED_CURVE_B, SPEED_CURVE_C = 0.00004, 0.004, 0.1
SPEED_CURVE_MAX_CM = 110
STOP_DISTANCE = 0.5
STILL_FRACTION = 0.02
WHEEL_ANGLES = tuple(k * math.pi / 4 for k in (1, 3, 5, 7))


def wheel_speeds(x, y, max_speed):
    wheels = [x * math.cos(a) + y * math.sin(a) for a in WHEEL_ANGLES]
    scale = max_speed / max(abs(wheels[0]), abs(wheels[1]), abs(wheels[2]), abs(wheels[3]), 1)
    return tuple(int(w * scale) for w in wheels)


def dribble_speed_multi(move, compass):
    if math.hypot(move[0], move[1]) < 1e-6:
        return 1.0
    angle = wrap_pi(math.atan2(-move[0], move[1]) - compass)
    side, back = abs(math.sin(angle)), max(0.0, -math.cos(angle))
    return min(1.0, cfg.DRIBBLE_SPEED_MIN / max(side, back, 1e-6))


def distance_speed_multi(target, minimum):
    distance = max(min(abs(target[0]) + abs(target[1]), SPEED_CURVE_MAX_CM), 0)
    multi = SPEED_CURVE_A * (distance ** 2) + SPEED_CURVE_B * distance + SPEED_CURVE_C
    return max(min(multi, 1), minimum)


def field_to_robot(vel, compass):
    x_field, y_field = -vel[1], vel[0]
    angle = -compass
    return (x_field * math.cos(angle) - y_field * math.sin(angle),
            x_field * math.sin(angle) + y_field * math.cos(angle))


class Mover:
    def __init__(self):
        self.reset()

    def reset(self):
        self.vel = [0.0, 0.0]
        self.last_speeds = (0, 0, 0, 0)
        self.parked = False
        self.turn_parked = False

    def step(self, desired_pos, desired_heading, compass, line, pose,
             base_speed=cfg.BASE_SPEED, spd_min=0.2, spd_multi=None, dribbler_on=False, smooth=0.1):
        rot = self._rotation(wrap_pi(desired_heading - compass), dribbler_on)
        if spd_multi is None:
            spd_multi = distance_speed_multi(desired_pos, spd_min)
        speed = base_speed * spd_multi * line.speed_multi
        if dribbler_on:
            speed *= dribble_speed_multi(desired_pos, compass)
        want = self._wanted_velocity(desired_pos, pose, speed, dribbler_on)

        if line.on_line and line.escape is not None:
            self.vel = [line.escape[0] * cfg.LINE_ESCAPE_SPEED, line.escape[1] * cfg.LINE_ESCAPE_SPEED]
        else:
            self._smooth_towards(want, smooth)

        x_robot, y_robot = field_to_robot(self.vel, compass)
        speeds = [m - rot for m in wheel_speeds(x_robot, y_robot, math.hypot(x_robot, y_robot))]
        biggest = max(abs(m) for m in speeds)
        if biggest > cfg.MOTOR_MAX:
            speeds = [m * cfg.MOTOR_MAX / biggest for m in speeds]
        speeds = tuple(int(m) for m in speeds)
        if dribbler_on and not line.on_line:
            speeds = self._limit_acceleration(speeds)
        self.last_speeds = speeds
        return speeds

    def _rotation(self, heading_error, dribbler_on):
        if abs(heading_error) < TURN_DEADBAND:
            self.turn_parked = True
        elif abs(heading_error) > TURN_RESUME or not self.parked or dribbler_on:
            self.turn_parked = False
        if self.turn_parked:
            return 0.0
        turn_max = cfg.DRIBBLE_TURN_MAX if dribbler_on else cfg.TURN_MAX
        return cfg.BASE_SPEED * turn_max * max(-1.0, min(1.0, heading_error / cfg.TURN_FULL_ERR))

    def _wanted_velocity(self, desired_pos, pose, speed, dribbler_on):
        reachable = limit_outward(desired_pos, pose)
        reachable_distance = math.hypot(reachable[0], reachable[1])
        if reachable_distance < PARK_ENTER and not dribbler_on:
            self.parked = True
        elif reachable_distance > PARK_RESUME or dribbler_on:
            self.parked = False
        if self.parked or reachable_distance < STOP_DISTANCE:
            want = [0.0, 0.0]
        else:
            distance = math.hypot(desired_pos[0], desired_pos[1])
            want = [desired_pos[0] / distance * speed, desired_pos[1] / distance * speed]
        lead = [v / cfg.BASE_SPEED * BRAKE_LEAD for v in self.vel]
        return limit_outward(want, pose, lead)

    def _smooth_towards(self, want, smooth):
        v = self.vel
        current, wanted = math.hypot(v[0], v[1]), math.hypot(want[0], want[1])
        cos = (v[0] * want[0] + v[1] * want[1]) / (current * wanted) if current > 1 and wanted > 1 else 1.0
        rate = smooth + (BRAKE_SMOOTH - smooth) * (1.0 - cos) / 2.0
        if wanted < current:
            rate = max(rate, 2 * smooth)
        self.vel = [v[0] + (want[0] - v[0]) * rate, v[1] + (want[1] - v[1]) * rate]
        if self.parked and math.hypot(self.vel[0], self.vel[1]) < STILL_FRACTION * cfg.BASE_SPEED:
            self.vel = [0.0, 0.0]

    def _limit_acceleration(self, speeds):
        max_step = cfg.DRIBBLE_MAX_ACCEL * cfg.BASE_SPEED * cfg.CONTROL_PERIOD
        change = [new - old for new, old in zip(speeds, self.last_speeds)]
        biggest = max(abs(c) for c in change)
        if biggest <= max_step:
            return speeds
        return tuple(int(old + c * max_step / biggest) for old, c in zip(self.last_speeds, change))

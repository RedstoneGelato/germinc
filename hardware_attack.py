from collections import namedtuple

Motor = namedtuple("Motor", "i2c_address elec_angle_offset sincos_centre")

ROBOT_ID = 1

DRIVE_MOTORS = [
    Motor(26, 1559051008, 1258),
    Motor(25, 1349926656, 1247),
    Motor(27, 1769756160, 1247),
    Motor(28, 1790054912, 1227),
]
DRIBBLER = Motor(32, 1431223552, 1245)
DRIBBLER_SPEED = 200000000   # TUNE: flip the sign if it spins the ball away

KICKER_PIN = 17
KICK_PULSE = 0.05            # TUNE carefully: seconds the solenoid is powered
KICK_COOLDOWN = 1.0


def make_motors():
    from hardware import MotorThread
    return MotorThread(DRIVE_MOTORS, DRIBBLER, DRIBBLER_SPEED)


def make_kicker():
    from hardware import Kicker
    return None if KICKER_PIN is None else Kicker(KICKER_PIN, KICK_PULSE, KICK_COOLDOWN)

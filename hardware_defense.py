from hardware_attack import Motor

ROBOT_ID = 2

DRIVE_MOTORS = [
    Motor(26, 1168928512, 1240),
    Motor(32, 1226029824, 1257),
    Motor(28, 1317619456, 1236),
    Motor(25, 1392997120, 1225),
]
DRIBBLER = Motor(27, 1431223552, 1245)   # TUNE: placeholder copied from the striker - calibrate before use
DRIBBLER_SPEED = 200000000

KICKER_PIN = 17              # TUNE: check the pin on this robot (test_solenoid.py)
KICK_PULSE = 0.05
KICK_COOLDOWN = 1.0


def make_motors():
    from hardware import MotorThread
    return MotorThread(DRIVE_MOTORS, DRIBBLER, DRIBBLER_SPEED)


def make_kicker():
    from hardware import Kicker
    return None if KICKER_PIN is None else Kicker(KICKER_PIN, KICK_PULSE, KICK_COOLDOWN)

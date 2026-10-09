"""
hardware_attack.py - everything that is different about the ATTACK (striker) robot's hardware.

If you swap a motor or driver board: change its row here AND in STOP_attack.py.
ELECANGLEOFFSET / SINCOSCENTRE come from the motor driver calibration and belong to that physical motor + driver.
This file is plain data (the Pi libraries are only imported inside the make_ functions) so simulator.py can read it.
"""
ROBOT_ID = 1   # striker

# (i2c address, ELECANGLEOFFSET, SINCOSCENTRE) for motor1..motor4, in VelocityToMotor order
DRIVE_MOTORS = [
    (26, 1559051008, 1258),   # motor1
    (25, 1349926656, 1247),   # motor2
    (27, 1769756160, 1247),   # motor3
    (28, 1790054912, 1227),   # motor4
]

# dribbler: (i2c address, ELECANGLEOFFSET, SINCOSCENTRE), or None if this robot has no dribbler
DRIBBLER = (32, 1431223552, 1245)   # calibration from the Oct 3 code - CHECK it's still this driver
DRIBBLER_SPEED = 200000000          # speed while on; flip the sign if it spins the ball away (the old code flipped it a few times)

# kicker: solenoid GPIO pin (test_solenoid.py), or None if this robot has no kicker
KICKER_PIN = 17
KICK_PULSE = 0.05                   # s the solenoid is powered (adjust carefully, same as test_solenoid.py)
KICK_COOLDOWN = 1.0                 # s between kicks (capacitor recharge / solenoid heat)


def make_motors():
    from hardware import MotorThread
    return MotorThread(DRIVE_MOTORS, DRIBBLER, DRIBBLER_SPEED)


def make_kicker():
    from hardware import Kicker
    return None if KICKER_PIN is None else Kicker(KICKER_PIN, KICK_PULSE, KICK_COOLDOWN)

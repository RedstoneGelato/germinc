"""
hardware_defense.py - everything that is different about the DEFENSE (goalie) robot's hardware.

If you swap a motor or driver board: change its row here AND in STOP_defense.py.
ELECANGLEOFFSET / SINCOSCENTRE come from the motor driver calibration and belong to that physical motor + driver.
This file is plain data (the Pi libraries are only imported inside the make_ functions) so simulator.py can read it.
"""
ROBOT_ID = 2   # goalie

# (i2c address, ELECANGLEOFFSET, SINCOSCENTRE) for motor1..motor4, in VelocityToMotor order
DRIVE_MOTORS = [
    (26, 1168928512, 1240),   # motor1
    (32, 1226029824, 1257),   # motor2
    (28, 1317619456, 1236),   # motor3
    (25, 1392997120, 1225),   # motor4
]

# dribbler: (i2c address, ELECANGLEOFFSET, SINCOSCENTRE), or None if this robot has no dribbler.
# The goalie's old dribbler (Oct 3) was address 25, which is now motor4, so its current address (27?) and
# calibration are unknown: calibrate it, fill this in, and it's used everywhere (robot, simulator, STOP_defense.py).
DRIBBLER = None                     # e.g. (27, <ELECANGLEOFFSET>, <SINCOSCENTRE>)
DRIBBLER_SPEED = 200000000

# kicker: solenoid GPIO pin, or None if this robot has no kicker
KICKER_PIN = None                   # set to 17 (or whichever pin) if the goalie has a solenoid
KICK_PULSE = 0.05
KICK_COOLDOWN = 1.0


def make_motors():
    from hardware import MotorThread
    return MotorThread(DRIVE_MOTORS, DRIBBLER, DRIBBLER_SPEED)


def make_kicker():
    from hardware import Kicker
    return None if KICKER_PIN is None else Kicker(KICKER_PIN, KICK_PULSE, KICK_COOLDOWN)

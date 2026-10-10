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
# The goalie has the same dribbler + kicker as the striker. !! PLACEHOLDER VALUES: the address (27?) and the
# calibration below are NOT this robot's - calibrate its dribbler driver and put the real numbers here (and in
# STOP_defense.py) before running it. Set DRIBBLER = None to run without one.
DRIBBLER = (27, 1431223552, 1245)   # VERIFY address + calibration (placeholder copied from the striker's)
DRIBBLER_SPEED = 200000000

# kicker: solenoid GPIO pin, or None if this robot has no kicker
KICKER_PIN = 17                     # VERIFY the pin on this robot (test_solenoid.py)
KICK_PULSE = 0.05
KICK_COOLDOWN = 1.0


def make_motors():
    from hardware import MotorThread
    return MotorThread(DRIVE_MOTORS, DRIBBLER, DRIBBLER_SPEED)


def make_kicker():
    from hardware import Kicker
    return None if KICKER_PIN is None else Kicker(KICKER_PIN, KICK_PULSE, KICK_COOLDOWN)

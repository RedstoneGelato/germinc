"""
hardware_defense.py - everything that is different about the DEFENSE (goalie) robot's hardware.

If you swap a motor or driver board: change its row here AND in STOP_defense.py.
ELECANGLEOFFSET / SINCOSCENTRE come from the motor driver calibration and belong to that physical motor + driver.
"""
from hardware import MotorThread

ROBOT_ID = 2   # goalie

# (i2c address, ELECANGLEOFFSET, SINCOSCENTRE) for motor1..motor4, in VelocityToMotor order
DRIVE_MOTORS = [
    (26, 1168928512, 1240),   # motor1
    (32, 1226029824, 1257),   # motor2
    (28, 1317619456, 1236),   # motor3
    (25, 1392997120, 1225),   # motor4
]
DRIBBLER_ADDR = 27            # CHECK: the old STOP_defense.py listed 26, 32, 28, 27, 25 - 27 is the one not driving
                              # a wheel. Not driven by the code yet (only stopped by STOP_defense.py)


def make_motors():
    return MotorThread(DRIVE_MOTORS)

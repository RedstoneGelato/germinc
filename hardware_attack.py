"""
hardware_attack.py - everything that is different about the ATTACK (striker) robot's hardware.

If you swap a motor or driver board: change its row here AND in STOP_attack.py.
ELECANGLEOFFSET / SINCOSCENTRE come from the motor driver calibration and belong to that physical motor + driver.
"""
from hardware import MotorThread

ROBOT_ID = 1   # striker

# (i2c address, ELECANGLEOFFSET, SINCOSCENTRE) for motor1..motor4, in VelocityToMotor order
DRIVE_MOTORS = [
    (26, 1559051008, 1258),   # motor1
    (25, 1349926656, 1247),   # motor2
    (27, 1769756160, 1247),   # motor3
    (28, 1790054912, 1227),   # motor4
]
DRIBBLER_ADDR = 32            # not driven by the code yet (only stopped by STOP_attack.py)


def make_motors():
    return MotorThread(DRIVE_MOTORS)

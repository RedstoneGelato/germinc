import math
import time
import busio
import board
import adafruit_bno08x
from adafruit_bno08x.i2c import BNO08X_I2C

i2c = busio.I2C(board.SCL, board.SDA, frequency = 400000)
imu = BNO08X_I2C(i2c)
imu.enable_feature(adafruit_bno08x.BNO_REPORT_GAME_ROTATION_VECTOR)

heading = 0
alpha = 0.2
ready = False

while True:
    quat = imu.game_quaternion  # (x, y, z, w)
    if quat is not None:
        ready = True

        x, y, z, w = quat

        # convert quaternion → yaw (heading)
        heading = math.atan2(
            2*(w*z + x*y),
            1 - 2*(y*y + z*z)
        )

        # smooth
        heading = (heading * (1 - alpha)) + (heading * alpha)
        print(heading)
    time.sleep(0.01)
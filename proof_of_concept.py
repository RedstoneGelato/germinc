import adafruit_bitbangio as bbi
import board
import threading
import adafruit_bno08x.i2c
from adafruit_bno08x.i2c import BNO08X_I2C
import time
import math
from steelbar_powerful_bldc_driver import PowerfulBLDCDriver

i2c = bbi.I2C(board.D6, board.D5, frequency=400000)

def make_motor(addr):
    try:
        i2c.unlock()
    except:
        pass
    m = PowerfulBLDCDriver(i2c, addr)
    i2c.try_lock()
    m.set_current_limit_foc(262144)  # max 8 amps is 524288
    m.set_id_pid_constants(1500, 200)
    m.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
    m.set_position_pid_constants(275, 0, 0)
    m.set_position_region_boundary(250000)
    m.set_ELECANGLEOFFSET(1559051008)
    m.set_SINCOSCENTRE(1258)
    m.set_speed_limit(500_000_000)
    m.configure_operating_mode_and_sensor(3, 1)
    m.configure_command_mode(12)
    i2c.unlock()
    return m

def get_heading(imu):
    quat = imu.game_quaternion  # (x, y, z, w)
    if quat is not None:
        x, y, z, w = quat
        # convert quaternion -> yaw (heading)
        heading = math.atan2(
            2*(w*z + x*y),
            1 - 2*(y*y + z*z)
        )
        return heading

def VelocityToMotor(xvel, yvel, rot, maxspd):
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

def main():
    m25 = make_motor(25)
    m32 = make_motor(32)
    m26 = make_motor(26)
    m28 = make_motor(28)

    imu = BNO08X_I2C(i2c)
    imu.enable_feature(adafruit_bno08x.BNO_REPORT_GAME_ROTATION_VECTOR)
    error = get_heading(imu)
    try:
        while True:
            compass = get_heading(imu) - error
            i2c.try_lock()
            spd1, spd2, spd3, spd4 = VelocityToMotor(0,0,compass,10000000)
            m26.set_speed(spd1)
            m32.set_speed(spd2)
            m28.set_speed(spd3)
            m25.set_speed(spd4)
            i2c.unlock()
            print(f"imu = {compass} spd = {spd1,spd2,spd3,spd4}")

    except KeyboardInterrupt:
        i2c.try_lock()
        m26.set_speed(0)
        m32.set_speed(0)
        m28.set_speed(0)
        m25.set_speed(0)
        m26.clear_faults()
        m32.clear_faults()
        m28.clear_faults()
        m25.clear_faults()
        i2c.unlock()

    except Exception as e:
        print(e)
        i2c.try_lock()
        m26.set_speed(0)
        m32.set_speed(0)
        m28.set_speed(0)
        m25.set_speed(0)
        m26.clear_faults()
        m32.clear_faults()
        m28.clear_faults()
        m25.clear_faults()
        i2c.unlock()

main()
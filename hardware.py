"""
hardware.py - threads that talk to the IMU, the PCB (LDR line ring) and the motor drivers. Shared by both robots;
the motor addresses / calibration for each robot are in hardware_attack.py and hardware_defense.py.

The PCB's IR ball sensors are gone: the ball comes from the camera now (vision.py / detection.py).
The PCB only does the 32 LDRs + bottom LEDs (commands 0x01 read colours, 0x03 set brightness).

Ultrasonic ring (later): add an UltrasonicThread here that keeps `distances` (8 values in cm, None = no echo)
and `angles` (each sensor's direction in the robot frame, radians, 0 = robot's right, CCW). main.py then hands
them to LocalisationThread.walls as (distances, [a + compass for a in angles]).
"""
import math
import threading
import time

import adafruit_bno08x
import board
import busio
from adafruit_bno08x.i2c import BNO08X_I2C
from smbus2 import SMBus, i2c_msg
from steelbar_powerful_bldc_driver import PowerfulBLDCDriver

import robot_config as cfg
from utils import wrap_pi


class IMUThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True

        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.imu = BNO08X_I2C(self.i2c)
        self.imu.enable_feature(adafruit_bno08x.BNO_REPORT_GAME_ROTATION_VECTOR)

        self.heading = 0
        self.heading_offset = 0
        self.ready = False

    def run(self):
        while self.running:
            quat = self.imu.game_quaternion  # (x, y, z, w)
            if quat is not None:
                self.ready = True

                x, y, z, w = quat

                # convert quaternion -> yaw (heading)
                self.heading = math.atan2(
                    2*(w*z + x*y),
                    1 - 2*(y*y + z*z)
                )
            time.sleep(0.01)

    def zero(self):
        """Current heading = facing the goal we attack."""
        self.heading_offset = self.heading

    def compass(self):
        """Robot heading in radians, CCW positive, 0 = facing the goal we attack."""
        return wrap_pi(self.heading - self.heading_offset)


class PCBThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True

        self.ready = False
        self.I2C_BUS = 1
        self.I2C_ADDR = 0x64
        self.CMD_READ_COLOURS = 0x01
        self.CMD_SET_BRIGHTNESS = 0x03
        self.COLOUR_SENSOR_COUNT = cfg.LDR_COUNT
        self.COLOUR_PACKET_SIZE = self.COLOUR_SENSOR_COUNT * 2
        self.CMD_TO_RESPONSE_DELAY = 0.02
        self.READ_RETRIES = 3
        self.RETRY_DELAY = 0.02
        self.bus = SMBus(self.I2C_BUS)
        self.lock = threading.Lock()
        self.bus_lock = threading.Lock()

        self.colours = [0] * self.COLOUR_SENSOR_COUNT
        self.brightness_target = 0
        self._brightness_sent = None

    def _send_command(self, cmd): #low level helper
        self.bus.write_byte(self.I2C_ADDR, cmd)

    def _read_raw(self, length): #low level helper
        msg = i2c_msg.read(self.I2C_ADDR, length)
        self.bus.i2c_rdwr(msg)
        return bytes(msg)

    def _read_packet(self, cmd, length): #low level helper
        last_err = None

        for _ in range(self.READ_RETRIES):
            try:
                with self.bus_lock:
                    self._send_command(cmd)
                    time.sleep(self.CMD_TO_RESPONSE_DELAY)
                    data = self._read_raw(length)
                if len(data) == length:
                    return data

            except OSError as e:
                last_err = e
                time.sleep(self.RETRY_DELAY)

        raise IOError(f"Failed to read packet: {last_err}")

    def _read_colours(self): #read colour sensors
        data = self._read_packet(self.CMD_READ_COLOURS, self.COLOUR_PACKET_SIZE)

        values = []
        for i in range(self.COLOUR_SENSOR_COUNT):
            lo = data[2*i]
            hi = data[2*i + 1]
            values.append(lo | (hi << 8))

        return values

    def set_brightness(self, value: float):  # called from main: just records the request
        self.brightness_target = int(max(0.0, min(65535.0, value)))

    def _write_brightness(self, val):  # called only from the PCB thread
        lo = val & 0xFF
        hi = (val >> 8) & 0xFF
        for _ in range(self.READ_RETRIES):
            try:
                with self.bus_lock:
                    self.bus.write_i2c_block_data(self.I2C_ADDR, self.CMD_SET_BRIGHTNESS, [lo, hi])
                self._brightness_sent = val
                return
            except OSError:
                time.sleep(self.RETRY_DELAY)

    def run(self):
        while self.running:
            try:
                target = self.brightness_target
                if target != self._brightness_sent:
                    self._write_brightness(target)
                new_colours = self._read_colours()
                with self.lock:
                    self.colours = new_colours
                self.ready = True
            except IOError as e:
                print(f"PCB I2C error: {e}")
                self.ready = False
            time.sleep(0.02)  # only one packet per loop now that IR is gone, so poll a bit faster

        self.bus.close()

    def snapshot(self):
        with self.lock:
            return list(self.colours)


# settings every drive motor gets (both robots)
MOTOR_CURRENT_LIMIT = 262144          # max 8 amps is 524288, this is 4 amps
MOTOR_SPEED_LIMIT = 546133333         # max spd


class MotorThread(threading.Thread): #setup motors with motor drivers
    """motors = list of 4 (i2c address, ELECANGLEOFFSET, SINCOSCENTRE), in VelocityToMotor order (motor1..motor4).
    Each robot passes its own list: see hardware_attack.py / hardware_defense.py."""

    def __init__(self, motors):
        super().__init__()
        self.daemon = True
        self.running = True

        self.speedlimit = MOTOR_SPEED_LIMIT
        self.motorspeed1 = 0
        self.motorspeed2 = 0
        self.motorspeed3 = 0
        self.motorspeed4 = 0

        self.i2c = busio.I2C(board.SCL, board.SDA)

        self.drivers = []
        for addr, elec_offset, sincos_centre in motors:
            m = PowerfulBLDCDriver(self.i2c, addr)
            m.set_current_limit_foc(MOTOR_CURRENT_LIMIT)
            m.set_id_pid_constants(1500, 200)
            m.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
            m.set_position_pid_constants(275, 0, 0)
            m.set_position_region_boundary(250000)
            m.set_ELECANGLEOFFSET(elec_offset)   # per-motor calibration
            m.set_SINCOSCENTRE(sincos_centre)    # per-motor calibration
            m.set_speed_limit(self.speedlimit)
            m.configure_operating_mode_and_sensor(3, 1)
            m.configure_command_mode(12)
            self.drivers.append(m)
        self.motor1, self.motor2, self.motor3, self.motor4 = self.drivers

    def run(self):
        while self.running:
            self.motor1.set_speed(int(-self.motorspeed1))
            self.motor2.set_speed(int(-self.motorspeed2))
            self.motor3.set_speed(int(-self.motorspeed3))
            self.motor4.set_speed(int(-self.motorspeed4))
            time.sleep(0.005)

    def set(self, speeds):
        self.motorspeed1, self.motorspeed2, self.motorspeed3, self.motorspeed4 = speeds

    def stop(self):
        self.set((0, 0, 0, 0))

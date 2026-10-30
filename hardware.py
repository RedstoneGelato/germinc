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
from utils import HeadingFilter, wrap_pi


class IMUThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True

        self.i2c = busio.I2C(board.SCL, board.SDA)
        self._open()

        self.heading = 0
        self.heading_offset = 0
        self.filter = HeadingFilter()
        self.ready = False

    def _open(self):
        self.imu = BNO08X_I2C(self.i2c)
        self.imu.enable_feature(adafruit_bno08x.BNO_REPORT_GAME_ROTATION_VECTOR)

    def run(self):
        errors = 0
        while self.running:
            try:
                quat = self.imu.game_quaternion
                errors = 0
            except Exception as e:
                quat = None
                errors += 1
                if errors == 1:
                    print(f"IMU read error: {e!r}")
                if errors % 20 == 0:
                    try:
                        self._open()
                        self.filter.reanchor()
                        print("IMU re-opened")
                    except Exception as e2:
                        print(f"IMU re-open failed: {e2!r}")
            if quat is not None:
                self.ready = True

                x, y, z, w = quat

                self.heading = self.filter.update(math.atan2(
                    2*(w*z + x*y),
                    1 - 2*(y*y + z*z)
                ), time.monotonic())
            time.sleep(0.01)

    def continue_from(self, compass):
        self.heading_offset = 0
        self.filter.continue_from(compass)

    def zero(self, facing=0.0):
        self.heading_offset = self.heading - facing

    def compass(self):
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
        self.CMD_READ_IR = 0x02
        self.COLOUR_SENSOR_COUNT = cfg.LDR_COUNT
        self.COLOUR_PACKET_SIZE = self.COLOUR_SENSOR_COUNT * 2
        self.CMD_TO_RESPONSE_DELAY = 0.02
        self.READ_RETRIES = 3
        self.RETRY_DELAY = 0.02
        self.bus = SMBus(self.I2C_BUS)
        self.lock = threading.Lock()
        self.bus_lock = threading.Lock()

        self.colours = [0] * self.COLOUR_SENSOR_COUNT
        self.ir = None
        self.brightness_target = 0
        self._brightness_sent = None

    def _send_command(self, cmd):
        self.bus.write_byte(self.I2C_ADDR, cmd)

    def _read_raw(self, length):
        msg = i2c_msg.read(self.I2C_ADDR, length)
        self.bus.i2c_rdwr(msg)
        return bytes(msg)

    def _read_packet(self, cmd, length):
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

    def _read_colours(self):
        data = self._read_packet(self.CMD_READ_COLOURS, self.COLOUR_PACKET_SIZE)

        values = []
        for i in range(self.COLOUR_SENSOR_COUNT):
            lo = data[2*i]
            hi = data[2*i + 1]
            values.append(lo | (hi << 8))

        return values

    def _read_ir(self):
        data = self._read_packet(self.CMD_READ_IR, cfg.IR_COUNT * 2)
        return [min(1.0, (data[2*i] | (data[2*i + 1] << 8)) / cfg.IR_RAW_MAX) for i in range(cfg.IR_COUNT)]

    def set_brightness(self, value: float):
        self.brightness_target = int(max(0.0, min(65535.0, value)))

    def _write_brightness(self, val):
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
                new_ir = self._read_ir() if cfg.IR_ON_PCB else None
                with self.lock:
                    self.colours = new_colours
                    self.ir = new_ir
                self.ready = True
            except Exception as e:
                print(f"PCB I2C error: {e!r}")
                self.ready = False
            time.sleep(0.02)

        self.bus.close()

    def snapshot(self):
        with self.lock:
            return list(self.colours)

    def ir_snapshot(self):
        with self.lock:
            return None if self.ir is None else list(self.ir)


MOTOR_CURRENT_LIMIT = 262144
MOTOR_SPEED_LIMIT = 546133333


class MotorThread(threading.Thread):

    def __init__(self, motors, dribbler=None, dribbler_speed=0):
        super().__init__()
        self.daemon = True
        self.running = True

        self.speedlimit = MOTOR_SPEED_LIMIT
        self.speeds = (0, 0, 0, 0)
        self.dribbler_speed = 0
        self.dribbler_on_speed = dribbler_speed

        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.wheels = [self._make_driver(*m) for m in motors]
        self.drivers = list(self.wheels)
        self.dribbler = None
        self.has_dribbler = dribbler is not None
        if dribbler is not None:
            self.dribbler = self._make_driver(*dribbler)
            self.drivers.append(self.dribbler)

    def _make_driver(self, addr, elec_offset, sincos_centre):
        m = PowerfulBLDCDriver(self.i2c, addr)
        m.set_current_limit_foc(MOTOR_CURRENT_LIMIT)
        m.set_id_pid_constants(1500, 200)
        m.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
        m.set_position_pid_constants(275, 0, 0)
        m.set_position_region_boundary(250000)
        m.set_ELECANGLEOFFSET(elec_offset)
        m.set_SINCOSCENTRE(sincos_centre)
        m.set_speed_limit(self.speedlimit)
        m.configure_operating_mode_and_sensor(3, 1)
        m.configure_command_mode(12)
        return m

    def run(self):
        errors = 0
        while self.running:
            try:
                for wheel, speed in zip(self.wheels, self.speeds):
                    wheel.set_speed(int(-speed))
                if self.dribbler is not None:
                    self.dribbler.set_speed(int(-self.dribbler_speed))
                errors = 0
            except Exception as e:
                errors += 1
                if errors == 1 or errors % 1000 == 0:
                    print(f"Motor I2C error ({errors} in a row): {e!r}")
            time.sleep(0.005)

    def set(self, speeds):
        self.speeds = tuple(speeds)

    def set_dribbler(self, on):
        self.dribbler_speed = self.dribbler_on_speed if on else 0

    def stop(self):
        self.set((0, 0, 0, 0))
        self.dribbler_speed = 0


class Kicker:

    def __init__(self, pin, pulse, cooldown):
        from gpiozero import OutputDevice
        self.out = OutputDevice(pin, active_high=True, initial_value=False)
        self.pulse, self.cooldown = pulse, cooldown
        self.last = -1e9

    def ready(self):
        return time.monotonic() - self.last >= self.cooldown

    def kick(self):
        if not self.ready():
            return False
        self.last = time.monotonic()
        self.out.on()
        threading.Timer(self.pulse, self.out.off).start()
        return True

    def close(self):
        self.out.off()
        self.out.close()

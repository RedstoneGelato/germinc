"""Test one part of the robot on its own, without the robot program. Run on the Pi:

    python3 test_parts.py imu         print the compass heading (radians) 100 times a second
    python3 test_parts.py pcb         print the 32 line sensor readings; type a number + Enter to set the LED brightness
    python3 test_parts.py solenoid    fire the kicker each time you press Enter
    python3 test_parts.py motor       calibrate motor drivers and spin them by typed commands

Stop the normal robot program first (sudo systemctl stop germinc). Ctrl+C quits every test.
To drive the whole robot by hand use test_drive.py; for the camera use test_camera.py and test_localisation.py.
"""
import select
import sys
import time

# ---- solenoid
SOLENOID_PIN = 17
SOLENOID_PULSE = 0.05        # seconds the solenoid is powered. Change carefully: long pulses overheat it

# ---- pcb (the STM32 board with the line sensors and LEDs)
PCB_I2C_BUS = 1
PCB_I2C_ADDR = 0x64
PCB_READ_COLOURS = 0x01
PCB_SET_BRIGHTNESS = 0x03
PCB_SENSOR_COUNT = 32
PCB_REPLY_DELAY = 0.02       # seconds between asking and reading the answer
PCB_READ_RETRIES = 3

# ---- motor
MOTOR_FIRMWARE = 3           # the driver firmware version this script knows


def typed_line():
    """The line the user typed since the last call, or None. Does not wait."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip()
    return None


# ==================================== imu ====================================
def imu():
    import math
    import adafruit_bno08x
    import board
    import busio
    from adafruit_bno08x.i2c import BNO08X_I2C

    sensor = BNO08X_I2C(busio.I2C(board.SCL, board.SDA))
    sensor.enable_feature(adafruit_bno08x.BNO_REPORT_GAME_ROTATION_VECTOR)
    while True:
        quat = sensor.game_quaternion
        if quat is not None:
            x, y, z, w = quat
            print(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        time.sleep(0.01)


# ==================================== pcb ====================================
def pcb_read_colours(bus):
    from smbus2 import i2c_msg
    last_error = None
    for _ in range(PCB_READ_RETRIES):
        try:
            bus.write_byte(PCB_I2C_ADDR, PCB_READ_COLOURS)
            time.sleep(PCB_REPLY_DELAY)
            reply = i2c_msg.read(PCB_I2C_ADDR, PCB_SENSOR_COUNT * 2)
            bus.i2c_rdwr(reply)
            data = bytes(reply)
            if len(data) == PCB_SENSOR_COUNT * 2:       # two bytes per sensor, low byte first
                return [data[2 * i] | (data[2 * i + 1] << 8) for i in range(PCB_SENSOR_COUNT)]
        except OSError as e:
            last_error = e
            time.sleep(PCB_REPLY_DELAY)
    raise IOError(f"PCB did not answer after {PCB_READ_RETRIES} tries: {last_error}")


def pcb_set_brightness(bus, value):
    value = int(max(0, min(65535, value)))
    bus.write_i2c_block_data(PCB_I2C_ADDR, PCB_SET_BRIGHTNESS, [value & 0xFF, (value >> 8) & 0xFF])


def pcb():
    from smbus2 import SMBus
    try:
        bus = SMBus(PCB_I2C_BUS)
    except FileNotFoundError:
        sys.exit(f"Could not open /dev/i2c-{PCB_I2C_BUS}.")
    print(f"Reading the PCB at 0x{PCB_I2C_ADDR:02X} on I2C bus {PCB_I2C_BUS}. Type a brightness (0-65535) + Enter.\n")
    try:
        while True:
            typed = typed_line()
            if typed is not None and typed.isnumeric():
                pcb_set_brightness(bus, int(typed))
            print(pcb_read_colours(bus))
            print("------------------------------")
            time.sleep(0.05)
    finally:
        bus.close()


# ==================================== solenoid ====================================
def solenoid():
    from gpiozero import OutputDevice
    pin = OutputDevice(SOLENOID_PIN, active_high=True, initial_value=False)
    print("Press ENTER to fire the solenoid.")
    try:
        while True:
            input()
            print("KICK")
            pin.on()
            time.sleep(SOLENOID_PULSE)
            pin.off()
    finally:
        pin.off()
        pin.close()


# ==================================== motor ====================================
MOTOR_HELP = """Commands (letter then a number, then Enter):
  n 1        choose motor number 1 (the first one is 0)
  m 12       command mode: 12 = speed, 13 = position, 2 = torque
  s 2000000  speed (in speed mode), also sets the speed limit
  p 1.5      position (in position mode)
  c 50000    current limit; in torque mode also the torque
  k 275      position PID constant
  b 250000   position region boundary
  f          clear faults
  d 500      wait 500 ms"""


def motor_setup(i2c, number):
    """Asks for one driver's address, calibrates it and leaves it in speed mode."""
    from steelbar_powerful_bldc_driver import PowerfulBLDCDriver
    address = int(input(f"Please enter the i2c address of motor driver number {number}:\n"))
    if address <= 7 or address >= 120:
        sys.exit("Error: invalid i2c address.")
    driver = PowerfulBLDCDriver(i2c, address)
    version = driver.get_firmware_version()
    print(f"The firmware version of motor driver number {number} is: {version}")
    if version != MOTOR_FIRMWARE:
        sys.exit("Error: unsupported motor driver version. Check for updates, the wiring and the i2c settings.")

    driver.set_current_limit_foc(65536)
    driver.set_id_pid_constants(1500, 200)
    driver.set_iq_pid_constants(1500, 200)
    driver.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
    driver.set_position_pid_constants(275, 0, 0)
    driver.set_position_region_boundary(250000)
    driver.set_speed_limit(10000000)
    driver.configure_operating_mode_and_sensor(15, 1)
    driver.configure_command_mode(15)
    driver.set_calibration_options(300, 2097152, 50000, 500000)

    driver.start_calibration()
    print(f"Starting calibration of motor {number}")
    while not driver.is_calibration_finished():
        print(".", end="", flush=True)
        time.sleep(0.5)
    print()
    # copy these two numbers into ATTACK_ROBOT / DEFENSE_ROBOT in robot_config.py
    print(f"ELECANGLEOFFSET: {driver.get_calibration_ELECANGLEOFFSET()}")
    print(f"SINCOSCENTRE: {driver.get_calibration_SINCOSCENTRE()}")

    driver.configure_operating_mode_and_sensor(3, 1)
    driver.configure_command_mode(12)
    return driver


def motor_command(drivers, modes, selected, command, value):
    """Runs one typed command. Returns the motor number that is selected afterwards."""
    driver = drivers[selected]
    if command == "f":
        driver.clear_faults()
        print("Clear Faults")
    elif value == "":
        print("Unknown command or missing parameter")
    elif command == "n":
        if int(value) not in range(len(drivers)):
            raise ValueError("no such motor")
        selected = int(value)
        print(f"Selected motor number {selected}")
    elif command == "m":
        if int(value) not in (2, 12, 13):
            raise ValueError("no such command mode")
        driver.configure_command_mode(int(value))
        modes[selected] = int(value)
        print(f"Command Mode {int(value)}")
    elif command == "s":
        speed = int(value)
        driver.set_speed_limit(abs(speed))
        if modes[selected] == 12:
            driver.set_speed(speed)
        print(f"Speed {speed}")
    elif command == "p":
        position = float(value)
        if modes[selected] == 13:
            driver.set_position(int(position), int((position * 256) % 256))
        else:
            print("Motor is not in position mode")
        print(f"Position {position}")
    elif command == "c":
        current = int(value)
        driver.set_current_limit_foc(abs(current))
        if modes[selected] == 2:
            driver.set_torque(current)
        print(f"Current (Torque) {current}")
    elif command == "k":
        driver.set_position_pid_constants(float(value), 0, 0)
    elif command == "b":
        driver.set_position_region_boundary(float(value))
    elif command == "d":
        print(f"Delay {int(value)}")
        time.sleep(int(value) / 1000)
    else:
        print("Unknown command or missing parameter")
    return selected


def motor():
    import board
    import busio

    count = int(input("Please enter the number of motor drivers you want to control:\n"))
    if count < 1 or count > 8:
        sys.exit("Error: motor count out of range.")
    i2c = busio.I2C(board.SCL, board.SDA)
    drivers = [motor_setup(i2c, number) for number in range(count)]
    modes = [12] * count          # every driver starts in speed mode
    selected = 0
    print(MOTOR_HELP)
    while True:
        typed = typed_line()
        if typed:
            try:
                selected = motor_command(drivers, modes, selected, typed[0], typed[1:].strip())
            except ValueError as e:
                print(f"Invalid value: {e}")
        time.sleep(0.001)
        for driver in drivers:
            driver.update_quick_data_readout()


TESTS = {"imu": imu, "pcb": pcb, "solenoid": solenoid, "motor": motor}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in TESTS:
        sys.exit(__doc__)
    try:
        TESTS[sys.argv[1]]()
    except KeyboardInterrupt:
        print("\nStopped.")

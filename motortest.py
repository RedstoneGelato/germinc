import time
import board
import busio
from steelbar_powerful_bldc_driver import PowerfulBLDCDriver

# Set up I2C
i2c = busio.I2C(board.SCL, board.SDA)

# Create motor driver at address 0x20 (change this to your real address)
motor = PowerfulBLDCDriver(i2c, 0x20)

# Set basic limits and modes
motor.set_speed_limit(2000000)  # max speed
motor.configure_operating_mode_and_sensor(3, 1)  # FOC + sin/cos encoder
motor.configure_command_mode(12)  # speed mode

print("Spinning motor...")
motor.set_speed(200000)  # positive = forward, negative = reverse

time.sleep(3)

print("Stopping")
motor.set_speed(0)

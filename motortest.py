import time
import board
import busio
from steelbar_powerful_bldc_driver import PowerfulBLDCDriver

# Set up I2C
i2c = busio.I2C(board.SCL, board.SDA)

# Create motor driver at address 0x20 (change this to your real address)
motor = PowerfulBLDCDriver(i2c, 26)
motor.set_current_limit_foc(65536)  # set current limit to 1 amp (only works in FOC mode)
motor.set_id_pid_constants(1500, 200)
motor.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
motor.set_position_pid_constants(275, 0, 0)
motor.set_position_region_boundary(250000)
motor.set_speed_limit(10000000)
motor.configure_operating_mode_and_sensor(3, 1)  # configure FOC mode and sin/cos encoder
motor.configure_command_mode(12)  # configure speed command mode
motormode = 12


print("Spinning motor...")
while True:
    motor.set_speed(2000000)  # positive = forward, negative = reverse

#!/usr/bin/env python3
"""
robot_stop.py — ExecStopPost safety shutdown.

Runs whenever the robot's systemd service stops, for ANY reason:
manual `systemctl stop`, `systemctl restart`, the main process
crashing, or systemd killing it after a timeout. By the time this
script runs, the main process's I2C connections are already closed
by the OS, so this script opens its own fresh connections and does
not depend on any state from the process that just died.

Every hardware step is wrapped individually so a single failure
(e.g. a board that's already lost power, or an I2C bus that's in a
bad state) can't stop the rest from being attempted. This script
never raises past main() and always exits 0, so it can't leave the
systemd unit hanging or in a failed state during a competition.
"""

import sys
import time
import logging
from pathlib import Path

LOG_PATH = Path("/home/germinc1/robot_stop.log")  # adjust to your actual home dir

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("robot_stop")

# Same I2C addresses as MotorThread in main.py — keep these two files
# in sync if the wiring/addresses ever change.
MOTOR_ADDRESSES = {
    "motor1": 26,
    "motor2": 32,
    "motor3": 28,
    "motor4": 27,
    "motor5": 25,  # dribbler
}

PCB_I2C_ADDR = 0x64


def stop_motors():
    """Zero every motor's speed and clear its fault flags."""
    try:
        import board
        import busio
        from steelbar_powerful_bldc_driver import PowerfulBLDCDriver
    except Exception as e:
        log.error(f"Could not import motor driver libraries: {e}")
        return

    try:
        i2c = busio.I2C(board.SCL, board.SDA)
    except Exception as e:
        log.error(f"Could not open I2C bus for motors: {e}")
        return

    for name, addr in MOTOR_ADDRESSES.items():
        try:
            motor = PowerfulBLDCDriver(i2c, addr)
            motor.set_speed(0)
            time.sleep(0.01)
            motor.set_speed(0)  # send twice, belt-and-suspenders
            motor.clear_faults()
            log.info(f"{name} (addr {addr}): stopped and faults cleared")
        except Exception as e:
            log.error(f"{name} (addr {addr}): failed to stop cleanly - {e}")


def stop_pcb_leds():
    """Best-effort: turn the underlight LEDs off via the STM32 PCB."""
    try:
        from smbus2 import SMBus
    except Exception as e:
        log.error(f"Could not import smbus2: {e}")
        return

    try:
        bus = SMBus(1)
        bus.write_i2c_block_data(PCB_I2C_ADDR, 0x03, [0, 0])  # brightness = 0
        bus.close()
        log.info("PCB LEDs set to 0 brightness")
    except Exception as e:
        log.error(f"Could not reset PCB brightness: {e}")


def main():
    log.info("=== ExecStopPost safety shutdown triggered ===")
    stop_motors()
    stop_pcb_leds()
    log.info("=== ExecStopPost safety shutdown complete ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Never let this script itself raise unhandled - systemd should
        # always be able to consider the stop sequence "done" so it
        # doesn't hang or block a restart mid-competition.
        log.error(f"Unexpected top-level error: {e}")
    sys.exit(0)
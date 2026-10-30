import logging
import sys
import time
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent / "robot_stop.log"
PCB_I2C_ADDR = 0x64
PCB_SET_BRIGHTNESS = 0x03

logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("robot_stop")


def motor_addresses(hw):
    addresses = {f"motor{i + 1}": motor[0] for i, motor in enumerate(hw.DRIVE_MOTORS)}
    if hw.DRIBBLER is not None:
        addresses["dribbler"] = hw.DRIBBLER[0]
    return addresses


def stop_motors(addresses):
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
    for name, address in addresses.items():
        try:
            motor = PowerfulBLDCDriver(i2c, address)
            motor.set_speed(0)
            time.sleep(0.01)
            motor.set_speed(0)
            motor.clear_faults()
            log.info(f"{name} (addr {address}): stopped and faults cleared")
        except Exception as e:
            log.error(f"{name} (addr {address}): failed to stop cleanly - {e}")


def stop_pcb_leds():
    try:
        from smbus2 import SMBus
    except Exception as e:
        log.error(f"Could not import smbus2: {e}")
        return
    try:
        bus = SMBus(1)
        bus.write_i2c_block_data(PCB_I2C_ADDR, PCB_SET_BRIGHTNESS, [0, 0])
        bus.close()
        log.info("PCB LEDs set to 0 brightness")
    except Exception as e:
        log.error(f"Could not reset PCB brightness: {e}")


def run(load_hardware):
    try:
        log.info("=== ExecStopPost safety shutdown triggered ===")
        try:
            addresses = motor_addresses(load_hardware())
        except Exception as e:
            log.error(f"Could not read motor addresses: {e}")
            addresses = {}
        stop_motors(addresses)
        stop_pcb_leds()
        log.info("=== ExecStopPost safety shutdown complete ===")
    except Exception as e:
        log.error(f"Unexpected top-level error: {e}")
    sys.exit(0)

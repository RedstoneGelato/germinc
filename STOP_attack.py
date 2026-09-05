import time
import board
import busio
from steelbar_powerful_bldc_driver import PowerfulBLDCDriver

# Must match the addresses AND calibration values in the target robot's MotorThread exactly -
# these are copied from the goalie file. If you run this against the attacker, swap in its values instead.
MOTOR_CONFIGS = [
    {"addr": 26, "offset": 1161314304, "centre": 1244},
    {"addr": 27, "offset": 1304942336, "centre": 1239},
    {"addr": 28, "offset": 1772804352, "centre": 1251},
    {"addr": 25, "offset": 1352689664, "centre": 1251}
]

def main():
    print("Connecting to motors...")
    i2c = busio.I2C(board.SCL, board.SDA)

    motors = []
    for cfg in MOTOR_CONFIGS:
        try:
            m = PowerfulBLDCDriver(i2c, cfg["addr"])
            m.set_ELECANGLEOFFSET(cfg["offset"])
            m.set_SINCOSCENTRE(cfg["centre"])
            m.configure_operating_mode_and_sensor(3, 1)
            m.configure_command_mode(12)
            motors.append(m)
        except Exception as e:
            print(f"Could not reach motor at address {cfg['addr']}: {e}")

    if not motors:
        print("No motors responded - nothing to stop.")
        return

    print(f"Stopping {len(motors)} motor(s)...")
    # Send zero repeatedly for a moment, not just once - if main.py is still
    # alive and its MotorThread is still running, a single zero here would
    # just get immediately overwritten by its next command 5ms later.
    end_time = time.time() + 1.0
    while time.time() < end_time:
        for m in motors:
            try:
                m.set_speed(0)
            except Exception as e:
                print(f"Error stopping motor: {e}")
        time.sleep(0.005)

    for m in motors:
        try:
            m.clear_faults()
        except Exception as e:
            print(f"Error clearing faults: {e}")

    print("Motors stopped.")

if __name__ == "__main__":
    main()
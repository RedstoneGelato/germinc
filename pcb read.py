import time
import sys
from smbus2 import SMBus, i2c_msg

I2C_BUS = 1
I2C_ADDR = 0x64
CMD_READ_COLOURS = 0x01
CMD_READ_IR = 0x02

COLOUR_SENSOR_COUNT = 32
COLOUR_PACKET_SIZE = COLOUR_SENSOR_COUNT * 2   # 64 bytes, uint16 LE each
IR_SENSOR_COUNT = 12
IR_PACKET_SIZE = IR_SENSOR_COUNT               # 12 bytes, uint8 each
CMD_TO_RESPONSE_DELAY_S = 0.05
READ_RETRIES = 3
RETRY_DELAY_S = 0.02

def _send_command(bus: SMBus, cmd: int) -> None:
    """Write a single command byte to the STM32 (separate transaction, STOP)."""
    bus.write_byte(I2C_ADDR, cmd)

def _read_raw(bus: SMBus, length: int) -> bytes:
    """Issue a plain read transaction (no register byte) for `length` bytes."""
    msg = i2c_msg.read(I2C_ADDR, length)
    bus.i2c_rdwr(msg)
    return bytes(msg)

def _read_packet(bus: SMBus, cmd: int, length: int) -> bytes:
    """Send a command then read the resulting packet, with basic retry."""
    last_err = None
    for attempt in range(READ_RETRIES):
        try:
            _send_command(bus, cmd)
            time.sleep(CMD_TO_RESPONSE_DELAY_S)
            data = _read_raw(bus, length)
            if len(data) == length:
                return data
        except OSError as e:
            last_err = e
            time.sleep(RETRY_DELAY_S)
    raise IOError(f"Failed to read packet for cmd 0x{cmd:02X} after "
                  f"{READ_RETRIES} attempts: {last_err}")

def read_ir(bus: SMBus) -> list[int]:
    """Returns a list of 12 ints (0/1), IR1..IR12 in order."""
    data = _read_packet(bus, CMD_READ_IR, IR_PACKET_SIZE)
    return list(data)

def read_colours(bus: SMBus) -> list[int]:
    """Returns a list of 32 ints (0-4095 ADC counts), COLOUR1..COLOUR32 in order."""
    data = _read_packet(bus, CMD_READ_COLOURS, COLOUR_PACKET_SIZE)
    values = []
    for i in range(COLOUR_SENSOR_COUNT):
        lo = data[i * 2]
        hi = data[i * 2 + 1]
        values.append(lo | (hi << 8))
    return values


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    bus = SMBus(I2C_BUS)

    try:
        while True:
            try:
                ir = read_ir(bus)
                colours = read_colours(bus)
            except IOError as e:
                print(f"I2C read error: {e}")
                time.sleep(0.2)
                continue

            ir_str = " ".join(f"IR{i+1}:{v}" for i, v in enumerate(ir))
            print(ir_str)

            colour_str = " ".join(f"C{i+1}:{v:4d}" for i, v in enumerate(colours))
            print(colour_str)
            print("-" * 60)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        bus.close()

main()
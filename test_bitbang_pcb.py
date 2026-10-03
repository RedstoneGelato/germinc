#!/usr/bin/env python3
import time
import sys
import select
import board
import adafruit_bitbangio as bbi

I2C_SCL = board.D6
I2C_SDA = board.D5
I2C_ADDR = 0x64
CMD_READ_COLOURS = 0x01
CMD_READ_IR = 0x02
CMD_SET_BRIGHTNESS = 0x03

COLOUR_SENSOR_COUNT = 32
COLOUR_PACKET_SIZE = COLOUR_SENSOR_COUNT * 2
IR_SENSOR_COUNT = 12
IR_PACKET_SIZE = 24

CMD_TO_RESPONSE_DELAY_S = 0.02
READ_RETRIES = 3
RETRY_DELAY_S = 0.02


def _write(bus, data: bytes) -> None:
    # bitbangio requires the bus to be locked around every transaction
    while not bus.try_lock():
        pass
    try:
        bus.writeto(I2C_ADDR, data)
    finally:
        bus.unlock()


def _send_command(bus, cmd: int) -> None:
    _write(bus, bytes([cmd]))


def _read_raw(bus, length: int) -> bytes:
    buf = bytearray(length)
    while not bus.try_lock():
        pass
    try:
        bus.readfrom_into(I2C_ADDR, buf)
    finally:
        bus.unlock()
    return bytes(buf)


def _read_packet(bus, cmd: int, length: int) -> bytes:
    last_err = None
    for attempt in range(READ_RETRIES):
        try:
            _send_command(bus, cmd)
            time.sleep(CMD_TO_RESPONSE_DELAY_S)
            data = _read_raw(bus, length)
            if len(data) == length:
                return data
        except (OSError, RuntimeError) as e:  # bitbangio can raise RuntimeError on bus timeouts
            last_err = e
            time.sleep(RETRY_DELAY_S)
    raise IOError(f"Failed to read packet for cmd 0x{cmd:02X} after "
                  f"{READ_RETRIES} attempts: {last_err}")


def read_colours(bus) -> list:
    data = _read_packet(bus, CMD_READ_COLOURS, COLOUR_PACKET_SIZE)
    return [data[i*2] | (data[i*2+1] << 8) for i in range(COLOUR_SENSOR_COUNT)]


def read_ir(bus) -> list:
    """
    Returns list of 12 dicts, each with:
      'detected': 1 or 0
      'distance': 0=none, 1=far, 2=medium, 3=close, 4=very close
    Same command 0x02 as before, now returns 24 bytes instead of 12.
    """
    data = _read_packet(bus, CMD_READ_IR, IR_PACKET_SIZE)
    return [
        {
            'detected': data[i * 2] if data[i * 2 + 1] >= 2 else 0,
            'distance': data[i * 2 + 1]
        }
        for i in range(12)
    ]

def set_brightness(bus, value: float):
    """
    Send brightness value to STM32.
    Valid range: 0.0 to 65535.0 (matches TIM3 period).
    Sends command 0x03 followed by 2 bytes (uint16, little-endian).
    This matches the STM32 SlaveRxCpltCallback which expects
    exactly 2 data bytes after the 0x03 command byte.
    """
    val = int(max(0.0, min(65535.0, value)))
    lo  = val & 0xFF
    hi  = (val >> 8) & 0xFF
    # one write: START, ADDR+W, 0x03, lo, hi, STOP
    # (same bytes smbus2's write_i2c_block_data(addr, 0x03, [lo, hi]) sent)
    _write(bus, bytes([CMD_SET_BRIGHTNESS, lo, hi]))

def read_input():
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip()
    return None

def main():
    try:
        bus = bbi.I2C(I2C_SCL, I2C_SDA, frequency=400000)
    except Exception as e:
        print(f"Could not set up bitbang I2C on SCL={I2C_SCL}, SDA={I2C_SDA}: {e}")
        sys.exit(1)

    print(f"Reading from STM32 at 0x{I2C_ADDR:02X} on bitbang I2C "
          f"(SCL={I2C_SCL}, SDA={I2C_SDA}). Ctrl+C to stop.\n")

    try:
        while True:
            raw_ir = read_ir(bus)
            colours = read_colours(bus)
            brightness = read_input()

            if brightness is not None:
                if brightness.isnumeric():
                    brightness = int(brightness)
                    set_brightness(bus, brightness)

            print(raw_ir)
            print(colours)
            print("------------------------------")
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        bus.deinit()


if __name__ == "__main__":
    main()
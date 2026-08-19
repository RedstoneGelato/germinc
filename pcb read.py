#!/usr/bin/env python3
"""
PCB Read.py

Reads IR1-12 and COLOUR1-32 sensor values from the STM32H503 sensor board
over I2C, where the STM32 is the I2C slave.

Protocol (matches the STM32 main.c I2C slave implementation):
    1. Master writes a single command byte (a separate transaction, ends in STOP):
         0x01 = I2C_CMD_READ_COLOURS
         0x02 = I2C_CMD_READ_IR
    2. Master waits briefly for the STM32 main loop to process the command
       and prepare the response buffer.
    3. Master issues a separate read transaction to pull the response bytes:
         - COLOURS response: 64 bytes = 32 x uint16, little-endian
         - IR response:       12 bytes = 12 x uint8 (1 = detected, 0 = clear)

Requires: smbus2  (pip install smbus2 --break-system-packages)
Enable I2C on the Pi first: sudo raspi-config -> Interface Options -> I2C
"""

import time
import sys
from smbus2 import SMBus, i2c_msg
import math

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

I2C_BUS = 1        # /dev/i2c-1 on Raspberry Pi 40-pin header

# NOTE: the STM32 firmware currently has a mismatch between the
# I2C_SLAVE_ADDRESS macro (0x42) and the actual configured HAL address
# (hi2c1.Init.OwnAddress1 = 200, which is 0x64 in real 7-bit terms).
# Set this to whichever one your board actually responds to.
I2C_ADDR = 0x64
# I2C_ADDR = 0x64   # <- use this instead if OwnAddress1 hasn't been fixed to 0x42<<1

CMD_READ_COLOURS = 0x01
CMD_READ_IR = 0x02
CMD_READ_BALL = 0x03

COLOUR_SENSOR_COUNT = 32
COLOUR_PACKET_SIZE = COLOUR_SENSOR_COUNT * 2   # 64 bytes, uint16 LE each
IR_SENSOR_COUNT = 12
IR_PACKET_SIZE = IR_SENSOR_COUNT               # 12 bytes, uint8 each
BALL_PACKET_SIZE = 2                           # {direction (1-12, 0=none), strength (0-10)}

# Time to let the STM32 main loop pick up the command and fill the buffer
# before we issue the read transaction. The STM32 loop also does a full
# colour sensor scan every iteration (32 channels x 1 ms mux settle delay
# = ~32 ms), so this needs to comfortably exceed one loop period.
CMD_TO_RESPONSE_DELAY_S = 0.05

READ_RETRIES = 3
RETRY_DELAY_S = 0.02


# ---------------------------------------------------------------------------
# Low-level I2C helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# High-level reads
# ---------------------------------------------------------------------------

def read_ir(bus: SMBus) -> list[int]:
    """Returns a list of 12 ints (0/1), IR1..IR12 in order."""
    data = _read_packet(bus, CMD_READ_IR, IR_PACKET_SIZE)
    return list(data)


def read_ball(bus: SMBus) -> tuple[int, int]:
    """Returns (direction, strength). direction is 1-12 (clock position) or
    0 if no ball is currently detected. strength is 0-10."""
    data = _read_packet(bus, CMD_READ_BALL, BALL_PACKET_SIZE)
    return data[0], data[1]


def read_colours(bus: SMBus) -> list[int]:
    """Returns a list of 32 ints (0-4095 ADC counts), COLOUR1..COLOUR32 in order."""
    data = _read_packet(bus, CMD_READ_COLOURS, COLOUR_PACKET_SIZE)
    values = []
    for i in range(COLOUR_SENSOR_COUNT):
        lo = data[i * 2]
        hi = data[i * 2 + 1]
        values.append(lo | (hi << 8))
    return values

def read_ir_activity(bus):
    data = _read_packet(bus, 0x04, 24)
    return [data[i*2] | (data[i*2+1] << 8) for i in range(12)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    try:
        bus = SMBus(I2C_BUS)
    except FileNotFoundError:
        print(f"Could not open /dev/i2c-{I2C_BUS}. Is I2C enabled "
              f"(sudo raspi-config -> Interface Options -> I2C)?")
        sys.exit(1)

    print(f"Reading from STM32 at 0x{I2C_ADDR:02X} on I2C bus {I2C_BUS}. "
          f"Ctrl+C to stop.\n")

    try:
        while True:
            ir = read_ir(bus)
            colours = read_colours(bus)
            strength = read_ir_activity(bus)
            irstrengthlist = []
            ballpos = []
            line_threshold = 3000
            irx = 0
            iry = 0
            linex = 0
            liney = 0

            for i, active in enumerate(ir):
                if active:
                    w = strength[i]
                    angle = i * math.pi / 6 + math.pi/2
                    irx += math.cos(angle) * w
                    iry += math.sin(angle) * w

            if irx != 0 or iry != 0:
                ir[0] = math.atan2(iry, irx)
                average = sum(strength) // len(strength)
                irstrengthlist.append(average)
                if len(irstrengthlist) > 10:
                    irstrengthlist.pop(0)
                strength = sum(irstrengthlist) // len(irstrengthlist)
                strength = max(0, min(strength, 10000))
                ir[1] = 100 - math.sqrt(strength)
            else:
                ir = None

            if ir is None:
                ballpos = [0,0]
            else:
                ballpos = [round(math.cos(ir[0]) * ir[1]), round(math.sin(ir[0]) * ir[1])] #relative position of ball to the bot: +x is right, +y is front

            print("Ballpos" + ballpos)

            for i, value in enumerate(colours):
                if value > line_threshold:
                    angle = i * (math.pi / 16) + math.pi / 2   # colour1 = front, spread anticlockwise
                    excess = value - line_threshold
                    linex += math.cos(angle) * excess
                    liney += math.sin(angle) * excess

            on_line = (linex != 0 or liney != 0)
            if on_line:
                mag = math.hypot(linex, liney)
                desired_pos = [-linex / mag * 200, -liney / mag * 200]  # straight away from the line
            else:
                desired_pos = [0,0]

            print("colour sensor" + desired_pos)


    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        bus.close()


if __name__ == "__main__":
    main()
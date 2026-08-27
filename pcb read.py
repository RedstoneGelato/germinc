#!/usr/bin/env python3
import time
import sys
from smbus2 import SMBus, i2c_msg
import math

I2C_BUS = 1
I2C_ADDR = 0x64
CMD_READ_COLOURS = 0x01
CMD_READ_IR = 0x02
CMD_SET_BRIGHTNESS = 0x03

COLOUR_SENSOR_COUNT = 32
COLOUR_PACKET_SIZE = COLOUR_SENSOR_COUNT * 2
IR_SENSOR_COUNT = 12
IR_PACKET_SIZE = 24

CMD_TO_RESPONSE_DELAY_S = 0.05
READ_RETRIES = 3
RETRY_DELAY_S = 0.02


def _send_command(bus: SMBus, cmd: int) -> None:
    bus.write_byte(I2C_ADDR, cmd)


def _read_raw(bus: SMBus, length: int) -> bytes:
    msg = i2c_msg.read(I2C_ADDR, length)
    bus.i2c_rdwr(msg)
    return bytes(msg)


def _read_packet(bus: SMBus, cmd: int, length: int) -> bytes:
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


def read_colours(bus: SMBus) -> list:
    data = _read_packet(bus, CMD_READ_COLOURS, COLOUR_PACKET_SIZE)
    return [data[i*2] | (data[i*2+1] << 8) for i in range(COLOUR_SENSOR_COUNT)]


def read_ir(bus: SMBus) -> list:
    """
    Returns list of 12 dicts, each with:
      'detected': 1 or 0
      'distance': 0=none, 1=far, 2=medium, 3=close, 4=very close
    Same command 0x02 as before, now returns 24 bytes instead of 12.
    """
    data = _read_packet(bus, CMD_READ_IR, IR_PACKET_SIZE)
    return [
        {
            'detected': data[i * 2],
            'distance': data[i * 2 + 1]
        }
        for i in range(12)
    ]

def main():
    try:
        bus = SMBus(I2C_BUS)
    except FileNotFoundError:
        print(f"Could not open /dev/i2c-{I2C_BUS}.")
        sys.exit(1)

    print(f"Reading from STM32 at 0x{I2C_ADDR:02X} on I2C bus {I2C_BUS}. "
          f"Ctrl+C to stop.\n")

    line_threshold  = 3000
    irstrengthlist  = []  # moved outside loop so history accumulates

    try:
        while True:
            ir_raw  = read_ir(bus)
            colours = read_colours(bus)

            irx = 0
            iry = 0

            for i, active in enumerate(ir_raw):
                if active:
                    angle = i * math.pi / 6 + math.pi / 2
                    irx += math.cos(angle)
                    iry += math.sin(angle)

            if irx != 0 or iry != 0:
                ball_angle = math.atan2(iry, irx)
                active_count = sum(ir_raw)
                irstrengthlist.append(active_count)
                if len(irstrengthlist) > 10:
                    irstrengthlist.pop(0)
                avg_strength = sum(irstrengthlist) / len(irstrengthlist)
                avg_strength = max(0, min(avg_strength, 10000))
                ball_dist = 100 - math.sqrt(avg_strength)
                ballpos = [round(math.cos(ball_angle) * ball_dist),
                           round(math.sin(ball_angle) * ball_dist)]
            else:
                ballpos = [0, 0]

            print("Ball position")
            print(ballpos)

            linex = 0
            liney = 0
            for i, value in enumerate(colours):
                if value > line_threshold:
                    angle = i * (math.pi / 16) + math.pi / 2
                    excess = value - line_threshold
                    linex += math.cos(angle) * excess
                    liney += math.sin(angle) * excess

            if linex != 0 or liney != 0:
                mag = math.hypot(linex, liney)
                desired_pos = [-linex / mag * 200, -liney / mag * 200]
            else:
                desired_pos = [0, 0]

            print("Colour sensors")
            print(desired_pos)
            print("------------------------------")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        bus.close()


if __name__ == "__main__":
    main()
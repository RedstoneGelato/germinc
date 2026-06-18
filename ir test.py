import smbus2
import time

I2C_BUS = 1
IR_ADDR = 0x14

REG_DIRECTION = 0x49
REG_STRENGTH = 0x54

bus = smbus2.SMBus(I2C_BUS)

while True:
    direction = bus.read_byte_data(IR_ADDR, REG_DIRECTION)

    strength_bytes = bus.read_i2c_block_data(
        IR_ADDR,
        REG_STRENGTH,
        2
    )

    strength = (strength_bytes[0] << 8) | strength_bytes[1]

    print(
        f"Direction: {direction:2d} | "
        f"Strength: {strength:5d}"
    )

    time.sleep(0.05)
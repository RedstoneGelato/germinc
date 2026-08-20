import smbus2
import struct

bus = smbus2.SMBus(1)
I2C_ADDR = 0x64
brightness_percent = 0
brightness_float = (brightness_percent / 100) * 65535.0  # Convert percentage to float value

def set_brightness(value: float):
    """
    Send brightness value (float) to STM32.
    Valid range matches your TIM3 period: 0.0 to 65535.0
    """
    value = max(0.0, min(65535.0, value))
    float_bytes = list(struct.pack('<f', value))  # little-endian float
    # Write command byte + 4 float bytes in one transaction
    bus.write_i2c_block_data(I2C_ADDR, 0x03, float_bytes)

set_brightness(brightness_float)  # Set initial brightness
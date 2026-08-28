import smbus2

bus = smbus2.SMBus(1)
I2C_ADDR = 0x64
brightness_percent = 0
brightness_int = int(max(0.0, min(65535.0, brightness_percent * 65535.0 / 100.0)))

def set_brightness(value: float):
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
    # write_i2c_block_data sends: START, ADDR+W, 0x03 (reg), lo, hi, STOP
    # STM32 receives 0x03 first (1 byte), then queues receive of 2 more bytes
    bus.write_i2c_block_data(I2C_ADDR, 0x03, [lo, hi])

set_brightness(brightness_int)       # off
# set_brightness(3277)  # ~5% (matches startup value in STM32)
# set_brightness(32767) # ~50%
# set_brightness(65535) # full
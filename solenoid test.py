from gpiozero import OutputDevice
import time

kick_pin = OutputDevice(17, active_high=True, initial_value=False)

print("Press ENTER to fire solenoid. Ctrl+C to quit.")

try:
    while True:
        input()  # wait for enter
        print("KICK")

        kick_pin.on()
        time.sleep(0.05)  # adjust this carefully
        kick_pin.off()

except KeyboardInterrupt:
    kick_pin.off()
    kick_pin.close()
    print("Stopped")
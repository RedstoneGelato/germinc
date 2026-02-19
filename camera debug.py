import cv2
from picamera2 import Picamera2

# --- Initial camera setup ---
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (320, 240), "format": "BGR888"})
picam2.configure(config)

# Disable automatic white balance
picam2.set_controls({"AwbEnable": False})

# Initial gains (blue, red)
blue_gain = 1.0
red_gain = 1.0
picam2.set_controls({"ColourGains": (blue_gain, red_gain)})

picam2.start()

print("Controls:")
print("  U = +Blue, I = -Blue")
print("  P = +Red, Y = -Red")
print("  Q = quit")

while True:
    frame = picam2.capture_array("main")

    # Show frame
    cv2.imshow("Camera Gain Test", frame)

    # Read key
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # Adjust red
    if key == ord('p'):
        red_gain += 0.1
    if key == ord('y'):
        red_gain -= 0.1

    # Adjust blue
    if key == ord('u'):
        blue_gain += 0.1
    if key == ord('i'):
        blue_gain -= 0.1

    # Clamp gains to positive values
    blue_gain = max(0, blue_gain)
    red_gain = max(0, red_gain)

    # Apply new gains
    picam2.set_controls({"ColourGains": (blue_gain, red_gain)})

    # Print for debugging
    print(f"Blue gain: {blue_gain:.2f}, Red gain: {red_gain:.2f}")

cv2.destroyAllWindows()
picam2.stop()
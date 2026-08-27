import picamera2, cv2, time

cap = picamera2.Picamera2()
config = cap.create_preview_configuration(lores={"size": (160, 120), "format": "RGB888"})
cap.configure(config)
cap.set_controls({"AwbEnable": False, "ColourGains": (2.1, 2.7)})
cap.start()

time.sleep(1)  # let it settle before grabbing
frame = cap.capture_array("lores")
print("Frame shape:", frame.shape)  # expect (120, 160, 3) - height, width, channels
cv2.imwrite("/home/pi/step1_raw.jpg", frame)
cap.stop()
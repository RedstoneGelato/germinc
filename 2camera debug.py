import picamera2, cv2, numpy as np, time

cap = picamera2.Picamera2()
config = cap.create_preview_configuration(lores={"size": (160, 120), "format": "RGB888"})
cap.configure(config)
cap.set_controls({"AwbEnable": False, "ColourGains": (2.1, 2.7)})
cap.start()
time.sleep(1)

frame = cap.capture_array("lores")
frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # use whichever direction step 2 confirmed

ignore_x1, ignore_x2, ignore_y1, ignore_y2 = 55, 105, 70, 120  # current values - edit these to test new ones
boxed = frame.copy()
cv2.rectangle(boxed, (ignore_x1, ignore_y1), (ignore_x2, ignore_y2), (0, 0, 255), 1)
cv2.imwrite("/home/pi/step3_ignorebox.jpg", boxed)
cap.stop()
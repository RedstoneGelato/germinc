from picamera2 import Picamera2
import time

picam = Picamera2()
picam.start()

time.sleep(2)

while True:
    frame = picam.capture_array()
    print(frame.shape)
import cv2
import numpy as np
import picamera2
import time

frame = None
hsv = np.zeros((120,160,3), dtype=np.uint8)
cap = picamera2.Picamera2()
cap.set_controls({"FrameRate": 60})
config = cap.create_preview_configuration(
    main={"size": (320, 240), "format": "RGB888"},
    lores={"size": (160, 120), "format": "YUV420"})
cap.configure(config)
cap.set_controls({
    "AwbEnable": False,
    "ColourGains": (2.1, 2.7)   # blue, red tweak when needed
})
cap.start()

blue = [0,0,0,0]
yellow = [0,0,0,0]
frame = None
ready = False

# HSV ranges
lower_blue = np.array([90, 200, 100])
upper_blue = np.array([110, 255, 255])
lower_yellow = np.array([0, 180, 180])
upper_yellow = np.array([40, 255, 255])

kernel = np.ones((3,3), np.uint8)

def merge_blobs(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_min = y_min = float('inf')
    x_max = y_max = 0

    for c in contours:
        if cv2.contourArea(c) < 100:
            continue
        x, y, w, h = cv2.boundingRect(c)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    if x_min < x_max and y_min < y_max:
        return [x_min, y_min, x_max - x_min, y_max - y_min] # coords of top left corner, width, height
    else:
        return [0,0,0,0]

while True:
    frame = cap.capture_array("lores")
    bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
    frame = bgr
    try:
        hsv[:] = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    except:
        continue

    if hsv is None or frame is None:
        time.sleep(0.005)
        continue

    frame = frame.copy()
    hsv = hsv.copy()

    # reset
    blue   = [0,0,0,0]
    yellow = [0,0,0,0]

    masks = {
        "blue":   cv2.morphologyEx(cv2.inRange(hsv, lower_blue, upper_blue), cv2.MORPH_OPEN, kernel),
        "yellow": cv2.morphologyEx(cv2.inRange(hsv, lower_yellow, upper_yellow), cv2.MORPH_OPEN, kernel),
    }

    blue = merge_blobs(masks["blue"])
    yellow = merge_blobs(masks["yellow"])

    if frame is not None:
        for color, bbox in zip(["blue","yellow"], [blue,yellow]):
            x, y, w, h = bbox
            if w > 0 and h > 0:
                if color=="blue":   cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                if color=="yellow": cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)

    cv2.imshow("Debug", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.stop()
cv2.destroyAllWindows()
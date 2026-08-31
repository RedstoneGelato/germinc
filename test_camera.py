import time
import cv2
import numpy as np
import picamera2

# ---- copied from main.py's FrameGrabber/DetectionThread - keep these in sync manually ----
ROTATION = cv2.ROTATE_90_CLOCKWISE

LOWER_BLUE = np.array([90, 200, 100])
UPPER_BLUE = np.array([110, 255, 255])
LOWER_YELLOW = np.array([0, 180, 180])
UPPER_YELLOW = np.array([40, 255, 255])

KERNEL = np.ones((3, 3), np.uint8)

IGNORE_X1, IGNORE_X2 = 20, 85
IGNORE_Y1, IGNORE_Y2 = 50, 110

MIN_CONTOUR_AREA = 100
# --------------------------------------------------------------------------------------------


def merge_blobs(mask):
    """Identical logic to DetectionThread._merge_blobs - keep in sync if that changes."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_min = y_min = float('inf')
    x_max = y_max = 0

    for c in contours:
        if cv2.contourArea(c) < MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    if x_min < x_max and y_min < y_max:
        return [x_min, y_min, x_max - x_min, y_max - y_min]
    else:
        return [0, 0, 0, 0]


def main():
    cap = picamera2.Picamera2()
    config = cap.create_preview_configuration(lores={"size": (160, 120), "format": "RGB888"})
    cap.configure(config)
    cap.set_controls({"AwbEnable": False, "ColourGains": (2.1, 2.7)})
    cap.start()
    print("Ctrl+C to stop.\n")

    display_available = True

    try:
        while True:
            frame = cap.capture_array("lores")
            frame = cv2.rotate(frame, ROTATION)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            blue_raw = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
            yellow_raw = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)

            blue_raw[IGNORE_Y1:IGNORE_Y2, IGNORE_X1:IGNORE_X2] = 0
            yellow_raw[IGNORE_Y1:IGNORE_Y2, IGNORE_X1:IGNORE_X2] = 0

            blue_mask = cv2.morphologyEx(blue_raw, cv2.MORPH_OPEN, KERNEL)
            yellow_mask = cv2.morphologyEx(yellow_raw, cv2.MORPH_OPEN, KERNEL)

            blue_box = merge_blobs(blue_mask)
            yellow_box = merge_blobs(yellow_mask)

            annotated = frame.copy()

            # ignore region drawn first, so real detections sit visually on top if they overlap
            cv2.rectangle(annotated, (IGNORE_X1, IGNORE_Y1), (IGNORE_X2, IGNORE_Y2), (0, 0, 255), 1)
            cv2.putText(annotated, "ignore", (IGNORE_X1, max(IGNORE_Y1 - 4, 10)),
                        cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1)

            if blue_box != [0, 0, 0, 0]:
                x, y, w, h = blue_box
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 1)
                cv2.putText(annotated, "blue", (x, max(y - 4, 10)),
                            cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 0), 1)

            if yellow_box != [0, 0, 0, 0]:
                x, y, w, h = yellow_box
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 255), 1)  # magenta - yellow itself barely shows against a bright frame
                cv2.putText(annotated, "yellow", (x, max(y - 4, 10)),
                            cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 255), 1)

            print(f"blue={blue_box}  yellow={yellow_box}")

            if display_available:
                try:
                    cv2.imshow("camera debug", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error:
                    print("No display available - continuing with file output only.")
                    display_available = False

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
import math
import cv2
import numpy as np
import picamera2
import sys
import select

ROTATION = cv2.ROTATE_90_CLOCKWISE

blue,red = 2.4,2.7

CAPTURE_SIZE = (320, 240)

LOWER_BLUE = np.array([90, 200, 100])
UPPER_BLUE = np.array([110, 255, 255])
LOWER_YELLOW = np.array([20, 150, 100])
UPPER_YELLOW = np.array([40, 255, 255])

KERNEL = np.ones((3, 3), np.uint8)

IGNORE_X1, IGNORE_X2 = 60, 160
IGNORE_Y1, IGNORE_Y2 = 90, 230

# --------------------------------------------------------------------------------------------


def merge_blobs(mask, min_size):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_min = y_min = float('inf')
    x_max = y_max = 0

    for c in contours:
        if cv2.contourArea(c) < min_size:
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
    config = cap.create_preview_configuration(main={"size": CAPTURE_SIZE, "format": "RGB888"})
    cap.configure(config)
    cap.set_controls({"AwbEnable": False, "ColourGains": (blue, red)})
    cap.start()
    print(f"Capturing at {CAPTURE_SIZE} from the main stream. Ctrl+C to stop.\n")

    display_available = True
    frame_cx = frame_cy = None

    try:
        while True:
            frame = cap.capture_array("main")
            frame = cv2.rotate(frame, ROTATION)

            if frame_cx is None:
                h, w = frame.shape[:2]
                frame_cx, frame_cy = w / 2, h / 2
                print(f"Rotated frame size: {w}x{h}, center=({frame_cx},{frame_cy})")

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            blue_raw = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
            yellow_raw = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)

            blue_raw[IGNORE_Y1:IGNORE_Y2, IGNORE_X1:IGNORE_X2] = 0
            yellow_raw[IGNORE_Y1:IGNORE_Y2, IGNORE_X1:IGNORE_X2] = 0

            blue_mask = cv2.morphologyEx(blue_raw, cv2.MORPH_OPEN, KERNEL)
            yellow_mask = cv2.morphologyEx(yellow_raw, cv2.MORPH_OPEN, KERNEL)

            blue_box = merge_blobs(blue_mask, 280)
            yellow_box = merge_blobs(yellow_mask, 280)

            annotated = frame.copy()

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
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 255), 1)
                cv2.putText(annotated, "yellow", (x, max(y - 4, 10)),
                            cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 255), 1)

            if yellow_box == [0, 0, 0, 0]:
                goalpos = [0, 200]
            else:
                goalx = yellow_box[0]
                goaly = yellow_box[1] + yellow_box[3] / 2
                dx = goalx - frame_cx
                dy = frame_cy - goaly
                goalpos = [dx, dy]
            if blue_box == [0, 0, 0, 0]:
                own_goalpos = [0, -200]
            else:
                own_goalx = blue_box[0] + blue_box[2]
                own_goaly = blue_box[1] + blue_box[3] / 2
                own_dx = own_goalx - frame_cx
                own_dy = frame_cy - own_goaly
                own_goalpos = [own_dx, own_dy]

            print(f"goalpos={goalpos}  own goal={own_goalpos}")

            if display_available:
                try:
                    cv2.imshow("camera debug", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error:
                    print("No display available - continuing with console output only.")
                    display_available = False

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
import math
import cv2
import numpy as np
import time
from multiprocessing import Process, Queue, Value
import picamera2

# --------------------------------------------------------
# Frame grabber process
# --------------------------------------------------------
def frame_grabber(frame_queue, running_flag):
    cap = picamera2.Picamera2()
    config = cap.create_preview_configuration(
        main={"size": (320, 240), "format": "RGB888"},
        lores={"size": (160, 120), "format": "YUV420"}
    )
    cap.configure(config)
    cap.set_controls({
        "AwbEnable": False,
        "ColourGains": (2.1, 2.7)
    })
    cap.start()

    while running_flag.value:
        frame = cap.capture_array("main")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # drop frame if queue full
        if not frame_queue.full():
            frame_queue.put((frame, hsv))

        time.sleep(0.005)

    cap.stop()

# --------------------------------------------------------
# Detection process
# --------------------------------------------------------
def detection_process(frame_queue, result_queue, running_flag):
    kernel = np.ones((3, 3), np.uint8)

    # HSV ranges
    HSV_RANGES = {
        "blue":   (np.array([110, 125, 100]), np.array([130, 180, 160])),
        "orange": (np.array([0, 180, 180]), np.array([20, 255, 255])),
        "yellow": (np.array([20, 180, 100]), np.array([40, 255, 160])),
        "green":  (np.array([60, 100, 75]), np.array([80, 255, 125]))
    }

    def merge_blobs(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x_min = y_min = float('inf')
        x_max = y_max = 0
        for c in contours:
            if cv2.contourArea(c) < 300:
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

    while running_flag.value:
        if frame_queue.empty():
            time.sleep(0.001)
            continue

        frame, hsv = frame_queue.get()

        result = {}
        for color, (lower, upper) in HSV_RANGES.items():
            mask = cv2.morphologyEx(cv2.inRange(hsv, lower, upper), cv2.MORPH_OPEN, kernel)
            result[color] = merge_blobs(mask)

        result_queue.put((frame, result))

# --------------------------------------------------------
# Motor process
# --------------------------------------------------------
def motor_process(motor_speeds, running_flag):
    # Placeholder: insert BLDC motor initialization here

    while running_flag.value:
        m1, m2, m3, m4 = motor_speeds[:]
        # call motor1.set_speed(m1), etc.
        time.sleep(0.005)


def SpeedUp(m1, m2, m3, m4, maxspeed):
    spdmax = max(abs(m1), abs(m2), abs(m3), abs(m4))
    if spdmax == 0:
        return 0, 0, 0, 0
    multiplier = maxspeed / spdmax
    return m1*multiplier, m2*multiplier, m3*multiplier, m4*multiplier

# --------------------------------------------------------
# Main program
# --------------------------------------------------------
def main():
    frame_queue = Queue(maxsize=2)
    result_queue = Queue(maxsize=2)
    running_flag = Value('b', True)
    motor_speeds = [0.0, 0.0, 0.0, 0.0]

    # Start processes
    grabber_p = Process(target=frame_grabber, args=(frame_queue, running_flag))
    detection_p = Process(target=detection_process, args=(frame_queue, result_queue, running_flag))
    motor_p = Process(target=motor_process, args=(motor_speeds, running_flag))

    grabber_p.start()
    detection_p.start()
    motor_p.start()

    speedlimit = 2000000
    bot_position = [160, 70]

    try:
        while True:
            if not result_queue.empty():
                frame, colors = result_queue.get()
                ball_bbox = colors["orange"]
                ball_position = [ball_bbox[0] + ball_bbox[2] // 2, ball_bbox[1] + ball_bbox[3]]  # x,y
                angle = math.atan2(ball_position[1] - bot_position[1], ball_position[0] - bot_position[0])

                # crude motor logic
                if bot_position[1] - ball_position[1] < 10 and abs(bot_position[0] - ball_position[0]) < 10:
                    # dribbler placeholder
                    motors = (0,0,0,0)

                elif ball_position[1] - bot_position[1] < 50:
                    motors = SpeedUp(
                        math.sin(angle - math.pi/4),
                        math.sin(angle - 3*math.pi/4),
                        math.sin(angle - 5*math.pi/4),
                        math.sin(angle - 7*math.pi/4),
                        speedlimit
                    )

                else:
                    motors = (-speedlimit, -speedlimit, -speedlimit, -speedlimit)

                for i in range(4):
                    motor_speeds[i] = motors[i]

                # show debug
                if frame is not None:
                    for color, bbox in colors.items():
                        x, y, w, h = bbox
                        if w > 0 and h > 0:
                            if color=="blue":   cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                            if color=="orange": cv2.rectangle(frame, (x,y), (x+w,y+h), (0,165,255), 2)
                            if color=="yellow": cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)
                            if color=="green":  cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                    cv2.imshow("Cam", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    finally:
        running_flag.value = False
        grabber_p.join()
        detection_p.join()
        motor_p.join()
        cv2.destroyAllWindows()

main()
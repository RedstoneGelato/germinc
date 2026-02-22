import threading
import math
import cv2
import picamera2
import numpy as np
import time
import board
import busio
from steelbar_powerful_bldc_driver import PowerfulBLDCDriver

# ---------------------------------------------
# FRAME GRABBER
# ---------------------------------------------
class FrameGrabber(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True
        self.frame = None
        self.hsv = None

        self.cap = picamera2.Picamera2()
        config = self.cap.create_preview_configuration(
            main={"size": (320, 240), "format": "RGB888"},
            lores={"size": (160, 120), "format": "YUV420"}
        )
        self.cap.configure(config)
        self.cap.set_controls({
            "AwbEnable": False,
            "ColourGains": (2.1, 2.7)
        })
        self.cap.start()

    def run(self):
        while self.running:
            frame = self.cap.capture_array("main")
            self.frame = frame
            self.hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            time.sleep(0.001)

# ---------------------------------------------
# COLOUR (NO BLACK)
# ---------------------------------------------
class DetectionThread(threading.Thread):
    def __init__(self, grabber):
        super().__init__()
        self.daemon = True
        self.running = True
        self.grabber = grabber

        self.blue   = [0,0,0,0]
        self.yellow = [0,0,0,0]
        self.green  = [0,0,0,0]
        self.frame  = None

        # HSV ranges (blue/yellow/green only)
        self.lower_blue   = np.array([110, 125, 100])
        self.upper_blue   = np.array([130, 180, 160])
        self.lower_yellow = np.array([20, 180, 100])
        self.upper_yellow = np.array([40, 255, 160])
        self.lower_green  = np.array([60, 100, 75])
        self.upper_green  = np.array([80, 255, 125])

        self.kernel = np.ones((3,3), np.uint8)

    def run(self):
        while self.running:
            if self.grabber.hsv is None or self.grabber.frame is None:
                continue

            hsv   = self.grabber.hsv.copy()
            frame = self.grabber.frame.copy()

            self.blue   = self._merge_blobs(cv2.morphologyEx(cv2.inRange(hsv, self.lower_blue,   self.upper_blue),   cv2.MORPH_OPEN, self.kernel))
            self.yellow = self._merge_blobs(cv2.morphologyEx(cv2.inRange(hsv, self.lower_yellow, self.upper_yellow), cv2.MORPH_OPEN, self.kernel))
            self.green  = self._merge_blobs(cv2.morphologyEx(cv2.inRange(hsv, self.lower_green,  self.upper_green),  cv2.MORPH_OPEN, self.kernel))

            # drawing
            for color, bbox in zip(["blue", "yellow", "green"], [self.blue, self.yellow, self.green]):
                x,y,w,h = bbox
                if w > 0 and h > 0:
                    if color == "blue":   cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                    if color == "yellow": cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)
                    if color == "green":  cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            self.frame = frame

    def _merge_blobs(self, mask):
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x_min=y_min=float('inf')
        x_max=y_max=0

        for c in contours:
            area = cv2.contourArea(c)
            if area < 300 or area > 10000:
                continue
            x,y,w,h = cv2.boundingRect(c)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x+w)
            y_max = max(y_max, y+h)

        if x_min < x_max and y_min < y_max:
            return [x_min, y_min, x_max-x_min, y_max-y_min]
        else:
            return [0,0,0,0]

# ---------------------------------------------
# BLACK DETECTION (BALL)
# ---------------------------------------------
class BlackDetectionThread(threading.Thread):
    def __init__(self, grabber):
        super().__init__()
        self.daemon = True
        self.running = True
        self.grabber = grabber

        self.frame = None
        self.black = [0,0,0,0]

        self.dark_threshold = 60   # threshold
        self.kernel = np.ones((5,5), np.uint8)

    def run(self):
        while self.running:
            if self.grabber.frame is None:
                continue

            frame = self.grabber.frame.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, self.dark_threshold, 255, cv2.THRESH_BINARY_INV)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

            self.black = self._merge_blobs(mask)

            x,y,w,h = self.black
            if w>0 and h>0:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,0), 2)

            self.frame = frame

    def _merge_blobs(self, mask):
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        x_min=y_min=float('inf')
        x_max=y_max=0

        for c in contours:
            area = cv2.contourArea(c)
            if area < 300 or area > 10000:
                continue

            x,y,w,h = cv2.boundingRect(c)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x+w)
            y_max = max(y_max, y+h)

        if x_min < x_max and y_min < y_max:
            return [x_min, y_min, x_max-x_min, y_max-y_min]
        else:
            return [0,0,0,0]

# ---------------------------------------------
# MOTOR
# ---------------------------------------------
class MotorThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon  = True
        self.running = True
        self.speedlimit = 2000000

        self.motorspeed1 = 0
        self.motorspeed2 = 0
        self.motorspeed3 = 0
        self.motorspeed4 = 0

        #self.i2c = busio.I2C(board.SCL, board.SDA) 
        #self.motor1 = PowerfulBLDCDriver(self.i2c, 0x20) 
        #self.motor1.set_speed_limit(self.speedlimit) 
        #self.motor1.configure_operating_mode_and_sensor(3, 1) 
        #self.motor1.configure_command_mode(12)

        #self.motor2 = PowerfulBLDCDriver(self.i2c, 0x20) 
        #self.motor2.set_speed_limit(self.speedlimit) 
        #self.motor2.configure_operating_mode_and_sensor(3, 1) 
        #self.motor2.configure_command_mode(12)

        #self.motor3 = PowerfulBLDCDriver(self.i2c, 0x20) 
        #self.motor3.set_speed_limit(self.speedlimit) 
        #self.motor3.configure_operating_mode_and_sensor(3, 1) 
        #self.motor3.configure_command_mode(12)

        #self.motor4 = PowerfulBLDCDriver(self.i2c, 0x20) 
        #self.motor4.set_speed_limit(self.speedlimit) 
        #self.motor4.configure_operating_mode_and_sensor(3, 1) 
        #self.motor4.configure_command_mode(12)

    def run(self):
        while self.running:
            #self.motor1.set_speed(self.motorspeed1) 
            #self.motor2.set_speed(self.motorspeed2) 
            #self.motor3.set_speed(self.motorspeed3) 
            #self.motor4.set_speed(self.motorspeed4)
            time.sleep(0.001)

# ---------------------------------------------
# SPEED NORMALISATION
# ---------------------------------------------
def SpeedUp(m1,m2,m3,m4,maxspeed):
    spdmax = max(abs(m1), abs(m2), abs(m3), abs(m4))
    if spdmax == 0:
        return 0,0,0,0
    mul = maxspeed/spdmax
    return m1*mul, m2*mul, m3*mul, m4*mul

# ---------------------------------------------
# MAIN LOOP
# ---------------------------------------------
def main():
    grabber = FrameGrabber()
    grabber.start()

    detector = DetectionThread(grabber)
    detector.start()

    black_detector = BlackDetectionThread(grabber)
    black_detector.start()

    motors = MotorThread()
    motors.start()

    while True:
        bot_position = [160, 70]

        bx,by,bw,bh = black_detector.black
        if bw == 0 or bh == 0:
            continue

        ball_position = [bx + bw//2, by + bh]  # centre bottom of black

        angle = math.atan2(ball_position[1] - bot_position[1], ball_position[0] - bot_position[0])

        # filler bot movement
        if bot_position[1] - ball_position[1] < 10 and abs(bot_position[0] - ball_position[0]) < 10:
            print("0")

        elif ball_position[1] - bot_position[1] < 50:
            print("1")
            motors.motorspeed1, motors.motorspeed2, motors.motorspeed3, motors.motorspeed4 = SpeedUp(
                math.sin(angle - math.pi/4),
                math.sin(angle - 3*math.pi/4),
                math.sin(angle - 5*math.pi/4),
                math.sin(angle - 7*math.pi/4),
                motors.speedlimit
            )

        else:
            print("2")
            motors.motorspeed1 = motors.speedlimit * -1
            motors.motorspeed2 = motors.speedlimit * -1
            motors.motorspeed3 = motors.speedlimit * -1
            motors.motorspeed4 = motors.speedlimit * -1

        if black_detector.frame is not None:
            cv2.imshow("Cam", black_detector.frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            grabber.running = False
            detector.running = False
            black_detector.running = False
            motors.running = False
            break

    grabber.cap.stop()
    grabber.join()
    detector.join()
    black_detector.join()
    motors.join()
    cv2.destroyAllWindows()

main()
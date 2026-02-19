import threading
import math
import cv2
import picamera2
import numpy as np
import time
import board
import busio
from steelbar_powerful_bldc_driver import PowerfulBLDCDriver

class FrameGrabber(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True
        self.frame = None
        self.hsv = None

        self.cap = picamera2.Picamera2()
        cfg = self.cap.create_video_configuration(main={"size": (640, 480), "format": "RGB888", "crop": (1000, 500, 2000, 1500)})
        self.cap.configure(cfg)
        self.cap.set_controls({"AwbEnable": False})
        self.cap.set_controls({"ColourGains": [32, 0]})
        self.cap.start()

    def run(self):
        while self.running:
            frame = self.cap.capture_array("main")
            self.frame = frame
            self.hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


class DetectionThread(threading.Thread):
    def __init__(self, grabber):
        super().__init__()
        self.daemon = True
        self.running = True
        self.grabber = grabber

        self.blue = [0,0,0,0]
        self.orange = [0,0,0,0]
        self.yellow = [0,0,0,0]
        self.green = [0,0,0,0]
        self.frame = None

        # HSV ranges
        self.lower_blue   = np.array([85, 100, 50])
        self.upper_blue   = np.array([140, 255, 255])
        self.lower_orange = np.array([0, 180, 180])
        self.upper_orange = np.array([20, 255, 255])
        self.lower_yellow = np.array([15, 90, 125])
        self.upper_yellow = np.array([30, 255, 255])
        self.lower_green  = np.array([35, 50, 50])
        self.upper_green  = np.array([85, 255, 255])

        self.kernel = np.ones((3,3), np.uint8)

    def run(self):
        while self.running:
            if self.grabber.hsv is None or self.grabber.frame is None:
                continue

            frame = self.grabber.frame.copy()
            hsv   = self.grabber.hsv.copy()

            # reset
            self.blue   = [0,0,0,0]
            self.orange = [0,0,0,0]
            self.yellow = [0,0,0,0]
            self.green  = [0,0,0,0]

            masks = {
                "blue":   cv2.morphologyEx(cv2.inRange(hsv, self.lower_blue, self.upper_blue), cv2.MORPH_OPEN, self.kernel),
                "orange": cv2.morphologyEx(cv2.inRange(hsv, self.lower_orange, self.upper_orange), cv2.MORPH_OPEN, self.kernel),
                "yellow": cv2.morphologyEx(cv2.inRange(hsv, self.lower_yellow, self.upper_yellow), cv2.MORPH_OPEN, self.kernel),
                "green":  cv2.morphologyEx(cv2.inRange(hsv, self.lower_green, self.upper_green), cv2.MORPH_OPEN, self.kernel)
            }

            self.blue   = self._merge_blobs(masks["blue"])
            self.orange = self._merge_blobs(masks["orange"])
            self.yellow = self._merge_blobs(masks["yellow"])
            self.green  = self._merge_blobs(masks["green"])

            if frame is not None:
                for color, bbox in zip(["blue","orange","yellow","green"], [self.blue,self.orange,self.yellow,self.green]):
                    x, y, w, h = bbox
                    if w > 0 and h > 0:
                        if color=="blue":   cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                        if color=="orange": cv2.rectangle(frame, (x,y), (x+w,y+h), (0,165,255), 2)
                        if color=="yellow": cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)
                        if color=="green":  cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            self.frame = frame

    def _merge_blobs(self, mask):
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
            return [0,0,0,0]

class MotorThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
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
            pass

def SpeedUp(motor1,motor2,motor3,motor4,maxspeed):
    spdmax = max(abs(motor1), abs(motor2), abs(motor3), abs(motor4))

    if spdmax == 0:
        return 0,0,0,0
    
    else:
        multiplier = maxspeed/spdmax
        new1 = motor1 * multiplier
        new2 = motor2 * multiplier
        new3 = motor3 * multiplier
        new4 = motor4 * multiplier
        return new1,new2,new3,new4

def main():
    grabber = FrameGrabber()
    grabber.start()

    camera = DetectionThread(grabber)
    camera.start()

    motors = MotorThread()
    motors.start()

    while True:
        bot_position = [160, 70]
        ball_position = [camera.orange[0] + (camera.orange[2] // 2),camera.orange[1] + camera.orange[3]] # x, y
        angle = math.atan2(ball_position[1] - bot_position[1], ball_position[0] - bot_position[0])

        if bot_position[1] - ball_position[1] < 10 and abs(bot_position[0] - ball_position[0]) < 10: #activate dribbler and try to score
            print("0")
            pass

        elif ball_position[1] - bot_position[1] < 50: # pathfind to the ball
            print("1")
            motors.motorspeed1, motors.motorspeed2, motors.motorspeed3, motors.motorspeed4 = SpeedUp(math.sin(angle - math.pi/4), math.sin(angle - 3*math.pi/4), math.sin(angle - 5*math.pi/4), math.sin(angle - 7*math.pi/4), motors.speedlimit)

        else: # back up
            print("2")
            motors.motorspeed1 = motors.speedlimit * -1
            motors.motorspeed2 = motors.speedlimit * -1
            motors.motorspeed3 = motors.speedlimit * -1
            motors.motorspeed4 = motors.speedlimit * -1

        if camera.frame is not None:
            cv2.imshow("Cam", camera.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                grabber.running = False
                camera.running = False
                motors.running = False
                break

    grabber.join()
    camera.join()
    motors.join()
    cv2.destroyAllWindows()

main()

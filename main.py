import threading
import cv2
import numpy as np
import time
import board
import busio
from steelbar_powerful_bldc_driver import PowerfulBLDCDriver

class CameraThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True  # thread closes when main program closes

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # shared data
        self.bluecx = None
        self.bluecy = None
        self.orangecx = None
        self.orangecy = None
        self.yellowcx = None
        self.yellowcy = None
        self.frame = None

        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

            # Reset values each frame
            self.bluecx = self.bluecy = None
            self.orangecx = self.orangecy = None
            self.yellowcx = self.yellowcy = None

            # --- BLUE ---
            lower_blue = np.array([85, 100, 50])
            upper_blue = np.array([140, 255, 255])
            maskblue = cv2.inRange(hsv, lower_blue, upper_blue)
            contours, _ = cv2.findContours(maskblue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = 0, 0

            for c in contours:
                if cv2.contourArea(c) > 500:
                    x, y, w, h = cv2.boundingRect(c)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x + w)
                    y_max = max(y_max, y + h)

            if x_min < x_max:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                self.bluecx = (x_min + x_max) // 2
                self.bluecy = (y_min + y_max) // 2

            # --- ORANGE ---
            lower_orange = np.array([0, 180, 180])
            upper_orange = np.array([20, 255, 255])
            maskorange = cv2.inRange(hsv, lower_orange, upper_orange)
            contours, _ = cv2.findContours(maskorange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = 0, 0

            for c in contours:
                if cv2.contourArea(c) > 300:
                    x, y, w, h = cv2.boundingRect(c)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x + w)
                    y_max = max(y_max, y + h)

            if x_min < x_max:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 165, 255), 2)
                self.orangecx = (x_min + x_max) // 2
                self.orangecy = (y_min + y_max) // 2

            # --- YELLOW ---
            lower_yellow = np.array([15, 90, 125])
            upper_yellow = np.array([30, 255, 255])
            maskyellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            contours, _ = cv2.findContours(maskyellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = 0, 0

            for c in contours:
                if cv2.contourArea(c) > 500:
                    x, y, w, h = cv2.boundingRect(c)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x + w)
                    y_max = max(y_max, y + h)

            if x_min < x_max:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                self.yellowcx = (x_min + x_max) // 2
                self.yellowcy = (y_min + y_max) // 2

            # save the latest frame
            self.frame = frame

            # display and quit
            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.running = False

        self.cap.release()
        cv2.destroyAllWindows()

class MotorThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True

        self.command = None

        #self.i2c = busio.I2C(board.SCL, board.SDA)
        #self.motor = PowerfulBLDCDriver(self.i2c, 0x20)
        #self.motor.set_speed_limit(2000000)
        #self.motor.configure_operating_mode_and_sensor(3, 1)
        #self.motor.configure_command_mode(12)

    def run(self):
        while self.running:
            if self.command is not None:
                print(self.command)

                self.command = None

def main():
    camera = CameraThread()
    camera.start()
    
    motors = MotorThread()
    motors.start()

    while True:
        motors.command = "hi testing"

        print("Blue:", camera.bluecx, camera.bluecy)
        print("Orange:", camera.orangecx, camera.orangecy)
        print("Yellow:", camera.yellowcx, camera.yellowcy)
        print()

        time.sleep(0.2)

        if not camera.running:
            break

main()

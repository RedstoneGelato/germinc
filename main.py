import threading
import math
import cv2
import picamera2
import numpy as np
import time
import board
import busio
from steelbar_powerful_bldc_driver import PowerfulBLDCDriver
import adafruit_bno08x
from adafruit_bno08x.i2c import BNO08X_I2C

class FrameGrabber(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True

        self.frame = None
        self.hsv = None
        self.cap = picamera2.Picamera2()
        self.cap.set_controls({"FrameRate": 60})
        config = self.cap.create_preview_configuration(
            main={"size": (320, 240), "format": "RGB888"},
	        lores={"size": (160, 120), "format": "YUV420"})
        self.cap.configure(config)
        self.cap.set_controls({
            "AwbEnable": False,
            "ColourGains": (2.1, 2.7)   # blue, red tweak when needed
        })
        self.cap.start()

        self.camera_matrix = np.array([
            [120, 0, 80],
            [0, 120, 60],
            [0, 0, 1]
        ], dtype=np.float32)

        self.dist_coeffs = np.array([
            -0.35, 0.12, 0, 0, -0.02
        ], dtype=np.float32)

        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            None,
            self.camera_matrix,
            (160,120),
            cv2.CV_16SC2
        )

        self.hsv = np.zeros((120,160,3), dtype=np.uint8)

    def run(self):
        while self.running:
            frame = self.cap.capture_array("lores")
            bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
            undistorted = cv2.remap(bgr, self.map1, self.map2, cv2.INTER_LINEAR)
            self.frame = undistorted
            self.hsv[:] = cv2.cvtColor(undistorted, cv2.COLOR_BGR2HSV)

class DetectionThread(threading.Thread):
    def __init__(self, grabber):
        super().__init__()
        self.daemon = True
        self.running = True
        self.grabber = grabber

        self.blue = [0,0,0,0]
        self.yellow = [0,0,0,0]
        self.enemies = []
        self.frame = None
        self.debug = False
        self.ready = False

        # HSV ranges
        self.lower_blue = np.array([90, 200, 100])
        self.upper_blue = np.array([110, 255, 255])
        self.lower_yellow = np.array([0, 180, 180])
        self.upper_yellow = np.array([40, 255, 255])
        self.lower_green = np.array([60, 100, 75])
        self.upper_green = np.array([90, 255, 125])

        self.kernel = np.ones((3,3), np.uint8)

    def run(self):
        while self.running:
            if self.grabber.hsv is None or self.grabber.frame is None:
                time.sleep(0.005)
                continue

            frame = self.grabber.frame
            hsv = self.grabber.hsv
            self.ready = True

            # reset
            self.blue   = [0,0,0,0]
            self.yellow = [0,0,0,0]
            self.enemies  = []

            masks = {
                "blue":   cv2.morphologyEx(cv2.inRange(hsv, self.lower_blue, self.upper_blue), cv2.MORPH_OPEN, self.kernel),
                "yellow": cv2.morphologyEx(cv2.inRange(hsv, self.lower_yellow, self.upper_yellow), cv2.MORPH_OPEN, self.kernel),
                "green":  cv2.morphologyEx(cv2.inRange(hsv, self.lower_green, self.upper_green), cv2.MORPH_OPEN, self.kernel)
            }

            self.blue = self._merge_blobs(masks["blue"])
            self.yellow = self._merge_blobs(masks["yellow"])

            inv_green = cv2.bitwise_not(masks["green"])
            inv_green = cv2.bitwise_and(inv_green, cv2.bitwise_not(masks["blue"]))
            inv_green = cv2.bitwise_and(inv_green, cv2.bitwise_not(masks["yellow"]))
            inv_green = cv2.morphologyEx(inv_green, cv2.MORPH_OPEN, self.kernel)
            inv_green = cv2.morphologyEx(inv_green, cv2.MORPH_CLOSE, self.kernel)
            self.enemies = self.get_enemy_blobs(inv_green)

            if frame is not None and self.debug == True:
                for color, bbox in zip(["blue","yellow"], [self.blue,self.yellow]):
                    x, y, w, h = bbox
                    if w > 0 and h > 0:
                        if color=="blue":   cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                        if color=="yellow": cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)

                for x, y, w, h in self.enemies:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # red for enemies

            self.frame = frame
            time.sleep(0.005)

    def _merge_blobs(self, mask):
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

    def get_enemy_blobs(self, mask, min_area=250):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if h > 0 and h < 60 and w < 60:
                if y < 10 or y + h > 110:   # too close to edges
                    continue
            blobs.append([x, y, w, h])

        return blobs
    
class IMUThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True

        self.i2c = busio.I2C(board.SCL, board.SDA, frequency = 800000)
        self.imu = BNO08X_I2C(self.i2c)
        self.imu.enable_feature(adafruit_bno08x.BNO_REPORT_ROTATION_VECTOR)

        self.heading = 0
        self.alpha = 0.2
        self.ready = False

    def run(self):
        while self.running:
            quat = self.imu.quaternion  # (x, y, z, w)
            if quat is not None:
                self.ready = True

                x, y, z, w = quat

                # convert quaternion → yaw (heading)
                heading = math.atan2(
                    2*(w*z + x*y),
                    1 - 2*(y*y + z*z)
                )

                # smooth
                self.heading = (self.heading * (1 - self.alpha)) + (heading * self.alpha)
            time.sleep(0.01)

class MotorThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True

        self.speedlimit = 10000000
        self.motorspeed1 = 0
        self.motorspeed2 = 0
        self.motorspeed3 = 0
        self.motorspeed4 = 0

        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.motor1 = PowerfulBLDCDriver(self.i2c, 25)
        self.motor1.set_current_limit_foc(65536)  # set current limit to 1 amp (only works in FOC mode)
        self.motor1.set_id_pid_constants(1500, 200)
        self.motor1.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
        self.motor1.set_position_pid_constants(275, 0, 0)
        self.motor1.set_position_region_boundary(250000)
        self.motor1.set_speed_limit(self.speedlimit)
        self.motor1.configure_operating_mode_and_sensor(3, 1)
        self.motor1.configure_command_mode(12)
        self.motor2 = PowerfulBLDCDriver(self.i2c, 27)
        self.motor2.set_current_limit_foc(65536)  # set current limit to 1 amp (only works in FOC mode)
        self.motor2.set_id_pid_constants(1500, 200)
        self.motor2.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
        self.motor2.set_position_pid_constants(275, 0, 0)
        self.motor2.set_position_region_boundary(250000)
        self.motor2.set_speed_limit(self.speedlimit)
        self.motor2.configure_operating_mode_and_sensor(3, 1)
        self.motor2.configure_command_mode(12)
        self.motor3 = PowerfulBLDCDriver(self.i2c, 26)
        self.motor3.set_current_limit_foc(65536)  # set current limit to 1 amp (only works in FOC mode)
        self.motor3.set_id_pid_constants(1500, 200)
        self.motor3.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
        self.motor3.set_position_pid_constants(275, 0, 0)
        self.motor3.set_position_region_boundary(250000)
        self.motor3.set_speed_limit(self.speedlimit)
        self.motor3.configure_operating_mode_and_sensor(3, 1)
        self.motor3.configure_command_mode(12)
        self.motor4 = PowerfulBLDCDriver(self.i2c, 28)
        self.motor4.set_current_limit_foc(65536)  # set current limit to 1 amp (only works in FOC mode)
        self.motor4.set_id_pid_constants(1500, 200)
        self.motor4.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
        self.motor4.set_position_pid_constants(275, 0, 0)
        self.motor4.set_position_region_boundary(250000)
        self.motor4.set_speed_limit(self.speedlimit)
        self.motor4.configure_operating_mode_and_sensor(3, 1)
        self.motor4.configure_command_mode(12)

    def run(self):
        while self.running:
            self.motor1.set_speed(int(self.motorspeed1))
            self.motor2.set_speed(int(self.motorspeed2))
            self.motor3.set_speed(int(self.motorspeed3))
            self.motor4.set_speed(int(self.motorspeed4))
            time.sleep(0.005)

def VelocityToMotor(xvel, yvel, rot, maxspd):
    mag = math.hypot(xvel, yvel)
    if mag > 1:
        xvel /= mag
        yvel /= mag

    motor1 = xvel*math.cos(math.pi/6) + yvel*math.sin(math.pi/6) + rot
    motor2 = xvel*math.cos(5*math.pi/6) + yvel*math.sin(5*math.pi/6) + rot
    motor3 = xvel*math.cos(4*math.pi/3) + yvel*math.sin(4*math.pi/3) + rot
    motor4 = xvel*math.cos(5*math.pi/3) + yvel*math.sin(5*math.pi/3) + rot

    scale = maxspd/max(abs(motor1), abs(motor2), abs(motor3), abs(motor4), 1)
    motor1 *= scale
    motor2 *= scale
    motor3 *= scale
    motor4 *= scale

    return motor1,motor2,motor3,motor4

def Ultrasonic(left, right, top, back, fieldsize, max_diff):
    width, height = fieldsize

    # estimates from each wall
    x_left  = left
    x_right = width - right

    y_top    = top
    y_bottom = height - back

    # difference between opposing sensors
    dx = abs(x_left - x_right)
    dy = abs(y_top - y_bottom)

    # X POSITION
    if dx < max_diff:
        # sensors agree → average
        x = (x_left + x_right) / 2
    else:
        # disagreement → choose smaller reading
        if left < right:
            x = x_left
        else:
            x = x_right

    # Y POSITION
    if dy < max_diff:
        y = (y_top + y_bottom) / 2
    else:
        if top < back:
            y = y_top
        else:
            y = y_bottom

    return x, y

def safe_shutdown(grabber, camera, motors, imu):
    print("Shutting down safely...")

    # stop motors first
    motors.motorspeed1 = 0
    motors.motorspeed2 = 0
    motors.motorspeed3 = 0
    motors.motorspeed4 = 0

    # allow motor thread to send stop command
    time.sleep(0.05)

    # stop threads
    grabber.running = False
    camera.running = False
    motors.running = False
    imu.running = False

    try:
        grabber.cap.stop()
    except:
        pass

    # wait for threads
    grabber.join()
    camera.join()
    motors.join()
    imu.join()

    cv2.destroyAllWindows()

    print("Robot stopped.")

def main():
    grabber = FrameGrabber()
    grabber.start()

    camera = DetectionThread(grabber)
    camera.start()

    motors = MotorThread()
    motors.start()

    imu = IMUThread()
    imu.start()

    print("Waiting for sensors...")
    while not (imu.ready and camera.ready):
        time.sleep(0.05)

    print("running")

    heading_offset = imu.heading
    goal_colour = 0 # 0 is yellow, 1 is blue
    HAS_BALL = False

    try:
        while True:
            maxspd = 2000000 # ideal max speed
            spin_weight = 0.05 # bigger number = bot spins more instead of moves more
            field_size = [1820,2430] # width, height
            desired_heading = 0

            USreadings = [910,910,1215,1215] # sub in for actual ultrasonic values, left, right, top, back
            bot_position = Ultrasonic(USreadings[0],USreadings[1],USreadings[2],USreadings[3],field_size,200)

            ir = [math.pi/2,100] # sub in for actual ir values direction, strength
            ballpos = [math.cos(ir[0]) * ir[1], math.sin(ir[0]) * ir[1]] #relative position of ball: x,y

            compass = imu.heading - heading_offset
            compass = (compass + math.pi) % (2*math.pi) - math.pi

            if ir[1] > 800 and ballpos[1] > 0 and abs(compass) < math.pi/4: # ball is in ball capture zone check: close enough, infront of bot, facing forwards
                HAS_BALL = True
            else:
                HAS_BALL = False

            if HAS_BALL:
                spin_weight = 0.1

                if goal_colour == 0:
                    if camera.yellow == [0,0,0,0]:
                        desired_heading = 0
                        desired_pos= [0,200]
                    else:
                        goalx = camera.yellow[0] + camera.yellow[2]/2
                        goaly = camera.yellow[1] + camera.yellow[3]/2

                        dx = goalx - 80
                        dy = 60 - goaly

                        desired_heading = math.atan2(dx, dy)
                        desired_pos = [goalx,goaly]
                else:
                    if camera.blue == [0,0,0,0]:
                        desired_heading = 0
                        desired_pos = [0,200]
                    else:
                        goalx = camera.blue[0] + camera.blue[2]/2
                        goaly = camera.blue[1] + camera.blue[3]/2

                        dx = goalx - 80
                        dy = 60 - goaly

                        desired_heading = math.atan2(dx, dy)
                        desired_pos = [goalx,goaly]

                if ir[1] > 1000 and abs(ballpos[0]) < 10: # kick check: very close, center
                    pass
            else:
                spin_weight = 0.05
                desired_heading = 0

                if ballpos[1] > 10:
                    desired_pos = [ballpos[0], ballpos[1] - 10] # directly behind the ball
                elif ballpos[0] > 0:
                    desired_pos = [ballpos[0] - 10, ballpos[1] - 10] # bot can go to bottom right corner of ball without colliding
                else:
                    desired_pos = [ballpos[0] + 10, ballpos[1] - 10] # bottom left

                if bot_position[0] < 50:
                    desired_pos[0] = max(desired_pos[0],0)
                elif bot_position[0] > 1770:
                    desired_pos[0] = min(desired_pos[0],0)

            xvel = desired_pos[0] - 80
            yvel = desired_pos[1] - 60
            heading_error = desired_heading - compass
            heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
            rot = spin_weight * heading_error
            rot = max(min(rot, 1), -1)
            motors.motorspeed1,motors.motorspeed2,motors.motorspeed3,motors.motorspeed4 = VelocityToMotor(xvel,yvel,rot,maxspd)
            print(imu.heading)
    
    except KeyboardInterrupt:
        safe_shutdown(grabber,camera,motors, imu)

main()
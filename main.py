import threading
import math
import cv2
import picamera2
import numpy as np
import time
import socket
import json
from smbus2 import SMBus, i2c_msg
import board
import busio
from steelbar_powerful_bldc_driver import PowerfulBLDCDriver
import adafruit_bno08x
from adafruit_bno08x.i2c import BNO08X_I2C
from gpiozero import DigitalInputDevice #imports

script_activate_pin = DigitalInputDevice(25, pull_up = True) #gpio pin for on/off switch

TEAM_ID = "GERM_INC"
ROBOT_ID = 2 #goalie bot
COMMS_PORT = 5555 #used by comms

class FrameGrabber(threading.Thread): #raw camera capture
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True

        self.frame = None
        self.hsv = np.zeros((320,240,3), dtype=np.uint8)
        self.cap = picamera2.Picamera2()
        config = self.cap.create_preview_configuration(main={"size": (320,240), "format": "RGB888"})
        self.cap.configure(config)
        self.cap.set_controls({
            "AwbEnable": False,
            "ColourGains": (2.8, 2.2), #red, blue
            "FrameDurationLimits": (16666, 16666) #60fps
        })
        self.cap.start() #starts camera capture

    def run(self):
        while self.running:
            frame = self.cap.capture_array("main")
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) #rotate camera feed due to rotated camera physically
            self.frame = frame
            try:
                self.hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #convert rgb to hsv
            except:
                continue
            time.sleep(0.01)

class DetectionThread(threading.Thread): #analyse camera capture, split into 2 threads because lack of processing power
    def __init__(self, grabber):
        super().__init__()
        self.daemon = True
        self.running = True
        self.grabber = grabber

        self.blue = [0,0,0,0]
        self.yellow = [0,0,0,0]
        self.frame = None
        self.ready = False

        # HSV ranges for detecting colour blobs
        self.lower_blue = np.array([100, 220, 100])
        self.upper_blue = np.array([120, 255, 180])
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([40, 255, 255])

        self.kernel = np.ones((3,3), np.uint8) #smoothing mask

        # pixel region to ignore (center, ignore bot)
        self.ignore_x1 = 60
        self.ignore_x2 = 160
        self.ignore_y1 = 90
        self.ignore_y2 = 230

    def run(self):
        while self.running:
            if self.grabber.hsv is None or self.grabber.frame is None:
                time.sleep(0.005)
                continue

            hsv = self.grabber.hsv.copy()
            self.ready = True

            # reset
            self.blue = [0,0,0,0]
            self.yellow = [0,0,0,0]

            blue_raw = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
            yellow_raw = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow) #find colour blobs

            blue_raw[self.ignore_y1:self.ignore_y2, self.ignore_x1:self.ignore_x2] = 0
            yellow_raw[self.ignore_y1:self.ignore_y2, self.ignore_x1:self.ignore_x2] = 0 #set ignore region to 0, make all blobs disappear

            masks = {
                "blue":   cv2.morphologyEx(blue_raw, cv2.MORPH_OPEN, self.kernel),
                "yellow": cv2.morphologyEx(yellow_raw, cv2.MORPH_OPEN, self.kernel),
            } #apply smoothing

            self.yellow = self._merge_blobs(masks["yellow"], 200)
            self.blue = self._merge_blobs(masks["blue"], 200) #merge all small blobs into 1 big blob
            time.sleep(0.005)

    def _merge_blobs(self, mask, min_area): #merge multiple rect of same colour into 1
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x_min = y_min = float('inf')
        x_max = y_max = 0

        for c in contours:
            if cv2.contourArea(c) < min_area:
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
    
class IMUThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True

        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.imu = BNO08X_I2C(self.i2c)
        self.imu.enable_feature(adafruit_bno08x.BNO_REPORT_GAME_ROTATION_VECTOR)

        self.heading = 0
        self.ready = False

    def run(self):
        while self.running:
            quat = self.imu.game_quaternion  # (x, y, z, w)
            if quat is not None:
                self.ready = True

                x, y, z, w = quat

                # convert quaternion -> yaw (heading)
                self.heading = math.atan2(
                    2*(w*z + x*y),
                    1 - 2*(y*y + z*z)
                )
            time.sleep(0.01)

class PCBThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True

        self.ready = False
        self.I2C_BUS = 1
        self.I2C_ADDR = 0x64
        self.CMD_READ_COLOURS = 0x01
        self.CMD_READ_IR = 0x02
        self.COLOUR_SENSOR_COUNT = 32
        self.COLOUR_PACKET_SIZE = self.COLOUR_SENSOR_COUNT * 2
        self.IR_SENSOR_COUNT = 12
        self.IR_PACKET_SIZE = self.IR_SENSOR_COUNT * 2
        self.CMD_TO_RESPONSE_DELAY = 0.02
        self.READ_RETRIES = 3
        self.RETRY_DELAY = 0.02
        self.bus = SMBus(self.I2C_BUS)
        self.lock = threading.Lock()
        self.bus_lock = threading.Lock()

        self.ir = [
            {'detected': 0, 'distance': 0}
            for _ in range(self.IR_SENSOR_COUNT)
                    ]

        self.colours = [0] * self.COLOUR_SENSOR_COUNT
        self.brightness_target = 0
        self._brightness_sent = None

    def _send_command(self, cmd): #low level helper
        self.bus.write_byte(self.I2C_ADDR, cmd)

    def _read_raw(self, length): #low level helper
        msg = i2c_msg.read(self.I2C_ADDR, length)
        self.bus.i2c_rdwr(msg)
        return bytes(msg)

    def _read_packet(self, cmd, length): #low level helper
        last_err = None

        for _ in range(self.READ_RETRIES):
            try:
                with self.bus_lock:
                    self._send_command(cmd)
                    time.sleep(self.CMD_TO_RESPONSE_DELAY)
                    data = self._read_raw(length)
                if len(data) == length:
                    return data

            except OSError as e:
                last_err = e
                time.sleep(self.RETRY_DELAY)

        raise IOError(f"Failed to read packet: {last_err}")

    def _read_ir(self): #read ir sensors
        data = self._read_packet(self.CMD_READ_IR, self.IR_PACKET_SIZE)

        return [
            {
                'detected': data[i * 2] if data[i * 2 + 1] >= 2 else 0,
                'distance': data[i * 2 + 1]
            }
            for i in range(12)
        ]

    def _read_colours(self): #read colour sensors
        data = self._read_packet(self.CMD_READ_COLOURS, self.COLOUR_PACKET_SIZE)

        values = []
        for i in range(self.COLOUR_SENSOR_COUNT):
            lo = data[2*i]
            hi = data[2*i + 1]
            values.append(lo | (hi << 8))

        return values

    def set_brightness(self, value: float):  # called from main: just records the request
        self.brightness_target = int(max(0.0, min(65535.0, value)))

    def _write_brightness(self, val):  # called only from the PCB thread
        lo = val & 0xFF
        hi = (val >> 8) & 0xFF
        for _ in range(self.READ_RETRIES):
            try:
                with self.bus_lock:
                    self.bus.write_i2c_block_data(self.I2C_ADDR, 0x03, [lo, hi])
                self._brightness_sent = val
                return
            except OSError:
                time.sleep(self.RETRY_DELAY)

    def run(self):
        while self.running:
            try:
                target = self.brightness_target
                if target != self._brightness_sent:
                    self._write_brightness(target)
                new_ir = self._read_ir()
                new_colours = self._read_colours()
                with self.lock:
                    self.ir = new_ir
                    self.colours = new_colours
                self.ready = True
            except IOError as e:
                print(f"PCB I2C error: {e}")
                self.ready = False
            time.sleep(0.05)

        self.bus.close()

class MotorThread(threading.Thread): #setup motors with motor drivers
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True

        self.speedlimit = 546133333 #max spd
        self.motorspeed1 = 0
        self.motorspeed2 = 0
        self.motorspeed3 = 0
        self.motorspeed4 = 0

        self.i2c = busio.I2C(board.SCL, board.SDA)

        self.motor1 = PowerfulBLDCDriver(self.i2c, 26)
        self.motor1.set_current_limit_foc(262144) # max 8 amps is 524288
        self.motor1.set_id_pid_constants(1500, 200)
        self.motor1.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
        self.motor1.set_position_pid_constants(275, 0, 0)
        self.motor1.set_position_region_boundary(250000)
        self.motor1.set_ELECANGLEOFFSET(1168928512)
        self.motor1.set_SINCOSCENTRE(1240)
        self.motor1.set_speed_limit(self.speedlimit)
        self.motor1.configure_operating_mode_and_sensor(3, 1)
        self.motor1.configure_command_mode(12)

        self.motor2 = PowerfulBLDCDriver(self.i2c, 32)
        self.motor2.set_current_limit_foc(262144) # 4 amps
        self.motor2.set_id_pid_constants(1500, 200)
        self.motor2.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
        self.motor2.set_position_pid_constants(275, 0, 0)
        self.motor2.set_position_region_boundary(250000)
        self.motor2.set_ELECANGLEOFFSET(1226029824)
        self.motor2.set_SINCOSCENTRE(1257)
        self.motor2.set_speed_limit(self.speedlimit)
        self.motor2.configure_operating_mode_and_sensor(3, 1)
        self.motor2.configure_command_mode(12)

        self.motor3 = PowerfulBLDCDriver(self.i2c, 28)
        self.motor3.set_current_limit_foc(262144)
        self.motor3.set_id_pid_constants(1500, 200)
        self.motor3.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
        self.motor3.set_position_pid_constants(275, 0, 0)
        self.motor3.set_position_region_boundary(250000)
        self.motor3.set_ELECANGLEOFFSET(1317619456)
        self.motor3.set_SINCOSCENTRE(1236)
        self.motor3.set_speed_limit(self.speedlimit)
        self.motor3.configure_operating_mode_and_sensor(3, 1)
        self.motor3.configure_command_mode(12)

        self.motor4 = PowerfulBLDCDriver(self.i2c, 25)
        self.motor4.set_current_limit_foc(262144)
        self.motor4.set_id_pid_constants(1500, 200)
        self.motor4.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
        self.motor4.set_position_pid_constants(275, 0, 0)
        self.motor4.set_position_region_boundary(250000)
        self.motor4.set_ELECANGLEOFFSET(1392997120)
        self.motor4.set_SINCOSCENTRE(1225)
        self.motor4.set_speed_limit(self.speedlimit)
        self.motor4.configure_operating_mode_and_sensor(3, 1)
        self.motor4.configure_command_mode(12)

    def run(self):
        while self.running:
            self.motor1.set_speed(int(-self.motorspeed1))
            self.motor2.set_speed(int(-self.motorspeed2))
            self.motor3.set_speed(int(-self.motorspeed3))
            self.motor4.set_speed(int(-self.motorspeed4))
            time.sleep(0.005)

class TeammateLinkThread(threading.Thread): #comms between bots
    def __init__(self, send_interval=0.05):
        super().__init__()
        self.daemon = True
        self.running = True
        self.enabled = True  # set False to satisfy rule 4.2.6 (referee-requested disable)

        self.send_interval = send_interval

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.bind(("", COMMS_PORT))
        self.sock.settimeout(0.02)

        self.teammate_state = {} #most recent info from teammate
        self.teammate_last_seen = 0
        self.my_state = {"bot active": 0} #what robot wants to tell its teammate
    def run(self):
        last_send = 0
        while self.running:
            now = time.monotonic()

            if self.enabled and now - last_send >= self.send_interval:
                try:
                    msg = {"team": TEAM_ID, "robot": ROBOT_ID, **self.my_state}
                    self.sock.sendto(json.dumps(msg).encode("utf-8"), ("255.255.255.255", COMMS_PORT)) #send message
                except OSError as e:
                    print(f"Comms send error: {e}")
                last_send = now

            try:
                data, _ = self.sock.recvfrom(1024)
                msg = json.loads(data.decode("utf-8"))
                if (self.enabled and isinstance(msg, dict)
                        and msg.get("team") == TEAM_ID #message from bot of the same team
                        and msg.get("robot") != ROBOT_ID): #ignore a possible echo of own broadcast
                    self.teammate_state = msg #receive message
                    self.teammate_last_seen = now
            except socket.timeout:
                pass
            except (OSError, json.JSONDecodeError):
                pass

    def stop(self):
        self.running = False
        self.sock.close()

class Hysteresis: #used for decision smoothing and ignore flickers, instant enter to instantly switch
    def __init__(self, hold_time, instant_enter=None):
        self.hold_time = hold_time
        self.instant_enter = instant_enter
        self.current = None
        self._pending = None
        self._pending_since = None

    def update(self, raw_value):
        now = time.monotonic()

        if self.current is None: #first call - nothing to debounce yet
            self.current = raw_value
            return self.current

        if raw_value == self.current: #still agrees - clear any pending change
            self._pending = None
            return self.current

        if self.instant_enter and self.instant_enter(raw_value): #instantly swap with no delay
            self.current = raw_value
            self._pending = None
            return self.current

        if raw_value != self._pending: #new candidate - start timing it
            self._pending = raw_value
            self._pending_since = now
            return self.current

        if now - self._pending_since >= self.hold_time: #held long enough - commit it
            self.current = raw_value
            self._pending = None

        return self.current

    def reset(self): #clear state so that unpause doesnt jitter
        self.current = None
        self._pending = None
        self._pending_since = None

class GoalTracker: #camera to goal position
    def __init__(self, history=10, tolerance=60, lost_limit=40):
        self.history, self.tolerance, self.lost_limit = history, tolerance, lost_limit
        self.goalx_list, self.goaly_list = [], []
        self.own_goalx_list, self.own_goaly_list = [], []
        self.lostgoalcount = self.lostowngoalcount = 0
        self.unc_gx = self.unc_gy = self.unc_ogx = self.unc_ogy = 0

    def _update_axis(self, value, lst, unc): #keep length of list (history) to 10
        if len(lst) > self.history:
            if abs(value - np.mean(lst)) < self.tolerance:
                lst.append(value); lst.pop(0); unc = 0
            else:
                unc += 1
                if unc > self.history:
                    lst.clear(); lst.append(value); unc = 0
        else:
            lst.append(value)
        return unc

    def update(self, goal_colour, yellow, blue):
        primary, secondary = (yellow, blue) if goal_colour == 0 else (blue, yellow) #primary is shooting goal, secondary is own goal

        if primary == [0,0,0,0]: #doesnt see goal
            self.lostgoalcount += 1
        else:
            self.lostgoalcount = 0
            gx = primary[0] + primary[2]/2 - 120
            gy = 160 - (primary[1] + primary[3]) #middle bottom of goal
            self.unc_gx = self._update_axis(gx, self.goalx_list, self.unc_gx)
            self.unc_gy = self._update_axis(gy, self.goaly_list, self.unc_gy)

        if secondary == [0,0,0,0]:
            self.lostowngoalcount += 1
        else:
            self.lostowngoalcount = 0
            ogx = secondary[0] + secondary[2]/2 - 120
            ogy = 160 - secondary[1]
            self.unc_ogx = self._update_axis(ogx, self.own_goalx_list, self.unc_ogx)
            self.unc_ogy = self._update_axis(ogy, self.own_goaly_list, self.unc_ogy)

        if self.lostgoalcount > self.lost_limit: #confirm lost goal
            self.goalx_list.clear(); self.goaly_list.clear()
        if self.lostowngoalcount > self.lost_limit:
            self.own_goalx_list.clear(); self.own_goaly_list.clear()

        goalpos = [int(np.mean(self.goalx_list)), int(np.mean(self.goaly_list))] if self.goalx_list else [0, 250] #average of list
        own_goalpos = [int(np.mean(self.own_goalx_list)), int(np.mean(self.own_goaly_list))] if self.own_goalx_list else [0, -250]
        return goalpos, own_goalpos

def VelocityToMotor(xvel, yvel, rot, maxspd): #convert variables into specific motor speed values
    motor1 = xvel*math.cos(math.pi/4) + yvel*math.sin(math.pi/4) - rot
    motor2 = xvel*math.cos(3*math.pi/4) + yvel*math.sin(3*math.pi/4) - rot
    motor3 = xvel*math.cos(5*math.pi/4) + yvel*math.sin(5*math.pi/4) - rot
    motor4 = xvel*math.cos(7*math.pi/4) + yvel*math.sin(7*math.pi/4) - rot

    scale = maxspd/max(abs(motor1), abs(motor2), abs(motor3), abs(motor4), 1)
    motor1 *= scale
    motor2 *= scale
    motor3 *= scale
    motor4 *= scale

    return int(motor1),int(motor2),int(motor3),int(motor4)

def circular_mean(angles):
    return math.atan2(sum(math.sin(a) for a in angles), sum(math.cos(a) for a in angles))

def angdiff(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))  # wraps correctly through +-pi

def safe_shutdown(grabber, camera, motors, imu, pcb, comms):
    print("Shutting down safely...")

    # stop motors first
    motors.motorspeed1 = 0
    motors.motorspeed2 = 0
    motors.motorspeed3 = 0
    motors.motorspeed4 = 0
    motors.motor1.clear_faults()
    motors.motor2.clear_faults()
    motors.motor3.clear_faults()
    motors.motor4.clear_faults()

    # allow motor thread to send stop command
    time.sleep(0.05)

    # stop threads
    grabber.running = False
    camera.running = False
    motors.running = False
    imu.running = False
    pcb.running = False
    comms.stop()

    try:
        grabber.cap.stop()
    except:
        pass

    # wait for threads
    grabber.join()
    camera.join()
    motors.join()
    imu.join()
    pcb.join()
    comms.join()

    print("Robot stopped.")

#==========================================================================================#
#                                                                                          #
#                                   START OF ACTUAL CODE                                   #
#                                                                                          #
#==========================================================================================#

def main():
    grabber = FrameGrabber()
    grabber.start()
    camera = DetectionThread(grabber)
    camera.start()
    motors = MotorThread()
    motors.start()
    imu = IMUThread()
    imu.start()
    pcb = PCBThread()
    pcb.start()
    comms = TeammateLinkThread()
    comms.start()
    CameraToGoal = GoalTracker() #start threads

    print("Waiting for sensors...")
    while not (imu.ready and camera.ready and pcb.ready):
        time.sleep(0.05)

    print("Waiting for signal")
    #initialise variables
    compass = 0
    xvel = 0
    yvel = 0
    heading_error = 0
    rot = 0
    heading_offset = imu.heading
    desired_heading = 0

    basespd = 80000000 # ideal speed
    new_maxspd = 0
    ingoalspd = 60000000
    dribblerspd = 0
    dribbler_on = False
    dribbler_list = []
    base_spin = 50 # bigger number = bot spins more instead of moves more
    line_escape_speed = basespd * 1.5

    ir = [math.pi/2,100] # direction, distance
    ballpos = [0,100] #cartesian plane coord relative of bot
    directionlist = []
    irdirection = 0
    unconcordantdirection = 0
    distancelist = []
    ball_last_seen = 0
    BALL_LOST_TIME = 0.3   # seconds to keep the last ball position after losing it

    desired_pos = [0,0]
    new_desired_pos = [0,0]
    goalpos = [0,250] # cartesian plane coord relative of bot
    own_goalpos = [0,-250] # cartesian plane coord relative of bot
    goal_colour = 0 # 0 shoot for yellow, 1 shoot for blue

    ball_distance = 0
    ball_distance_count = 0
    ball_distance_total = 0

    led_brightness = 40000  # pcb led brightness: 0 - 65535
    line_threshold = 1500
    colour_see_number = 0
    on_line = False
    was_on_line = False
    line_list = []
    line_spd_multi = 1
    pcb.set_brightness(led_brightness)

    botstate_hyst = Hysteresis(hold_time=0.11)
    substate1_hyst = Hysteresis(hold_time=0.11, instant_enter=lambda v: v == 1)
    substate2_hyst = Hysteresis(hold_time=0.11, instant_enter=lambda v: v == 1)
    botstate = 3
    substate1 = 4
    substate2 = 4

    CONTROL_PERIOD = 0.01 #robot runs at 100hz

    while script_activate_pin.is_active: #calibrate bot
        with pcb.lock:
            ir_snapshot = pcb.ir
            colours_snapshot = pcb.colours #pull variables
        if camera.yellow != [0,0,0,0]: #calibrate which goal to shoot
            goal_colour = 0 if 160 - camera.yellow[1] > 0 else 1
        elif camera.blue != [0,0,0,0]:
            goal_colour = 1 if 160 - camera.blue[1] > 0 else 0
        else:
            pass

        if max(colours_snapshot) - 3000 > line_threshold: # calibrate pcb leds
            led_brightness += 50
        elif max(colours_snapshot) < line_threshold:
            led_brightness -= 50
        led_brightness = max(min(led_brightness,65535),0)
        pcb.set_brightness(led_brightness)

        heading_offset = imu.heading #calibrate heading
        time.sleep(0.01)

    print("running")
    robot_active = True

    try:
        next_loop = time.monotonic()
        while True:
            irx = 0 #reset variables
            iry = 0
            ball_distance_total = 0
            ball_distance_count = 0
            linex = 0
            liney = 0
            colour_see_number = 0

            with pcb.lock: #pull variables from sensors
                ir_snapshot = pcb.ir
                colours_snapshot = pcb.colours
            yellow = camera.yellow[:]
            blue = camera.blue[:]

#----------------------------------------------------------------------
#            pause and unpause bot
#----------------------------------------------------------------------
            if script_activate_pin.is_active: #paused bot
                if robot_active:
                    print("Paused")
                    robot_active = False
                    botstate_hyst.reset()
                    substate1_hyst.reset()
                    substate2_hyst.reset()
                    x_robot = 0
                    y_robot = 0
                    rot = 0
                    directionlist = []
                    CameraToGoal.goalx_list = []
                    CameraToGoal.goaly_list = []
                    CameraToGoal.own_goalx_list = []
                    CameraToGoal.own_goaly_list = []
                    dribbler_list = []
                    distancelist = []
                    ball_last_seen = 0

                motors.motorspeed1 = 0
                motors.motorspeed2 = 0
                motors.motorspeed3 = 0
                motors.motorspeed4 = 0
                new_desired_pos = [0,0]
                dribbler_on = False
                new_maxspd = 0 #reset all variables and stop motors
                comms.my_state.update({"bot active": 0}) # comm say bot off

                with pcb.lock: #pull variables from threads
                    ir_snapshot = pcb.ir
                    colours_snapshot = pcb.colours
                yellow = camera.yellow[:]
                blue = camera.blue[:]

                if yellow != [0,0,0,0]: #calibrate goal colour
                    goal_colour = 0 if 160 - yellow[1] > 0 else 1
                elif blue != [0,0,0,0]:
                    goal_colour = 1 if 160 - blue[1] > 0 else 0
                else:
                    pass

                if max(colours_snapshot) - 3000 > line_threshold: # calibrate pcb leds
                    led_brightness += 50
                elif max(colours_snapshot) < line_threshold:
                    led_brightness -= 50
                led_brightness = max(min(led_brightness,65535),0)
                pcb.set_brightness(led_brightness)

                heading_offset = imu.heading #calibrate imu heading

                time.sleep(0.02)
                continue
            else:
                robot_active = True #running bot
                comms.my_state.update({"bot active": 1}) #comms say bot active

#----------------------------------------------------------------------
#            ir to ball pos, compass, camera to goal pos
#----------------------------------------------------------------------
            compass = imu.heading - heading_offset #bot heading
            compass = (compass + math.pi) % (2*math.pi) - math.pi

            for i, sensor in enumerate(ir_snapshot):
                if sensor["detected"] == 1 and sensor["distance"] != 0:
                    if sensor["distance"] >= 2:
                        angle = i * math.pi / 6 + math.pi/2
 
                        irx += math.cos(angle)
                        iry += math.sin(angle)
 
                    ball_distance_total += sensor["distance"]
                    ball_distance_count += 1

            if irx != 0 or iry != 0:
                ball_last_seen = time.monotonic()
                irdirection = math.atan2(iry, irx) # direction

                if len(directionlist) > 10: # smoothing
                    if unconcordantdirection > 10:
                        directionlist.clear()
                        directionlist.append(irdirection)
                        unconcordantdirection = 0
                    elif abs(angdiff(circular_mean(directionlist), irdirection)) > 1:
                        unconcordantdirection += 1
                    else:
                        directionlist.pop(0)
                        directionlist.append(irdirection)
                        unconcordantdirection = 0
                else:
                    directionlist.append(irdirection)
                    unconcordantdirection = 0

                raw_distance = (ball_distance_total * 25) / ball_distance_count
                raw_distance = max(min(raw_distance, 99), 1)
                raw_distance = ((100 - raw_distance) * 0.3) ** 2
                distancelist.append(raw_distance)  # smooth distance too
                if len(distancelist) > 10:
                    distancelist.pop(0)
                ball_distance = sum(distancelist) / len(distancelist)

                ir = [circular_mean(directionlist), ball_distance] #direction, distance
                ballpos = [round(math.cos(ir[0]) * ir[1]), round(math.sin(ir[0]) * ir[1])]

            elif directionlist and time.monotonic() - ball_last_seen < BALL_LOST_TIME:
                pass  # brief dropout: keep previous ballpos / ir / ball_distance

            else:  # confirmed lost
                directionlist.clear()
                distancelist.clear()
                ballpos = [0,0]
                ir = [0,0]
                ball_distance = 300

            goalpos, own_goalpos = CameraToGoal.update(goal_colour, yellow, blue) #goal position

#----------------------------------------------------------------------
#            line detection
#----------------------------------------------------------------------
            for i, value in enumerate(colours_snapshot): #sums line detected sensors direction
                if value < line_threshold:
                    angle = i * (math.pi / 16) + math.pi/2 + compass #colour1 = front, spread anticlockwise
                    linex += math.cos(angle)
                    liney += math.sin(angle)
                    colour_see_number += 1
            on_line = colour_see_number > 0
            if on_line and not was_on_line:
                line_list.append(time.monotonic()) #timestamps of when the bot was on line
            was_on_line = on_line
            while line_list and time.monotonic() - line_list[0] > 3: #3 second clear
                line_list.pop(0)
            line_spd_multi = {0: 1, 1: 0.6, 2: 0.5, 3: 0.3, 4: 0.1}.get(len(line_list), 0.1) #slow down bot if touching line repeatedly rapidly

#----------------------------------------------------------------------
#            comms from and to other bot
#----------------------------------------------------------------------
            teammate_fresh = (time.monotonic() - comms.teammate_last_seen) < 0.5 #checks if the bots are still connected
            if isinstance(comms.teammate_state, dict) and teammate_fresh:
                comms_command = comms.teammate_state.get("command") #1 for go get ball, 0 for chill in goals
                attack_bot_state = comms.teammate_state.get("bot active") #0 for bot off, 1 for bot on
            else:
                comms_command = None
                attack_bot_state = None

#----------------------------------------------------------------------
#            determine states
#----------------------------------------------------------------------
            if comms_command == 0: #signal from attack to chill
                raw_botstate = 3
            elif ballpos == [0,0] and ir == [0,0]: #doesnt see ball
                raw_botstate = 0
            elif attack_bot_state == 0 or attack_bot_state is None: #attack bot is off
                raw_botstate = 1
            elif comms_command == 1 or ir_snapshot[0].get('distance') == 3 or ir_snapshot[1].get('distance') == 3 or ir_snapshot[11].get('distance') == 3 or ir_snapshot[2].get('distance') == 3 or ir_snapshot[9].get('distance') == 3: #signal from other bot to go get ball
                raw_botstate = 2
            else: #chill in goals
                raw_botstate = 3

            botstate = botstate_hyst.update(raw_botstate) #smoothing

#----------------------------------------------------------------------
#            state machine
#----------------------------------------------------------------------
            if botstate == 0: #do not see ball
                desired_heading = 0
                if own_goalpos != [0,-250]: # align middle and go backwards
                    desired_pos = [own_goalpos[0], own_goalpos[1] + 100]
                    ingoalspd = int(basespd / 5)
                else:
                    desired_pos = [goalpos[0], -250]
                    ingoalspd = basespd
                dribbler_on = False

            elif botstate == 1: #go for ball then score
                if (substate1 == 1 and ir_snapshot[0].get("distance") == 3) or (ir_snapshot[0].get("distance") == 3 and ir_snapshot[1].get("distance") == 3 and ir_snapshot[11].get("distance") == 3 and ir_snapshot[2].get("distance") != 3 and ir_snapshot[10].get("distance") != 3):
                    raw_substate1 = 1  #ball in bcz
                elif ballpos[1] < (50 if substate1 in (1, 4) else 80): #2 far backup, 3 close backup
                    raw_substate1 = 2 if ballpos[1] < -150 else 3
                else:
                    raw_substate1 = 4  #pathfind to ball
                substate1 = substate1_hyst.update(raw_substate1)

                if substate1 == 1:
                    dribbler_on = True
                    desired_heading = math.atan2(goalpos[1],goalpos[0] * 1.6) - math.pi/2
                    desired_heading = (desired_heading + math.pi) % (2 * math.pi) - math.pi
                    desired_pos = [goalpos[0] * 1.5, goalpos[1]]
                elif substate1 == 2:
                    dribbler_on = False
                    desired_heading = 0
                    desired_pos = ballpos
                elif substate1 == 3:
                    dribbler_on = False
                    desired_heading = 0
                    if abs(ballpos[0]) < 110 and ballpos[1] < 0: #wrap around ball
                        if len(line_list) > 1: #touched line
                            desired_pos = [-200, 0] if ballpos[0] < 0 else [200, 0] #go other way
                        else:
                            desired_pos = [-200, 0] if ballpos[0] > 0 else [200, 0] #wrap around base on ball position
                    else:
                        desired_pos = [0, -200] if ballpos[1] > 20 else [ballpos[0], -200]
                elif substate1 == 4:
                    desired_heading = math.atan2(goalpos[1],goalpos[0] * 1.6) - math.pi/2
                    desired_heading = (desired_heading + math.pi) % (2 * math.pi) - math.pi
                    if ballpos[1] < 80 and abs(ballpos[0]) > 80:
                        desired_pos = [ballpos[0], -10]
                    else:
                        desired_pos = [ballpos[0], ballpos[1] - 60]

                    if abs(desired_pos[0]) + abs(desired_pos[1]) < 150:
                        dribbler_on = True
                    else:
                        dribbler_on = False

            elif botstate == 2: # go for ball then pass
                if (substate2 == 1 and ir_snapshot[0].get("distance") == 3) or (ir_snapshot[0].get("distance") == 3 and ir_snapshot[1].get("distance") == 3 and ir_snapshot[11].get("distance") == 3 and ir_snapshot[2].get("distance") != 3 and ir_snapshot[10].get("distance") != 3):
                    raw_substate2 = 1  # ball in bcz
                elif ballpos[1] < (50 if substate2 in (1, 4) else 80):
                    raw_substate2 = 2 if ballpos[1] < -150 else 3  # far vs near backup
                else:
                    raw_substate2 = 4  # pathfind to ball
                substate2 = substate2_hyst.update(raw_substate2)

                if substate2 == 1:
                    dribbler_on = True
                    desired_heading = math.atan2(goalpos[1],goalpos[0] * 1.6) - math.pi/2
                    desired_heading = (desired_heading + math.pi) % (2 * math.pi) - math.pi
                    desired_pos = goalpos
                elif substate2 == 2:
                    dribbler_on = False
                    desired_heading = 0
                    desired_pos = [0, -200]
                elif substate2 == 3:
                    dribbler_on = False
                    desired_heading = 0
                    if abs(ballpos[0]) < 110 and ballpos[1] < 0:
                        if len(line_list) > 1:
                            desired_pos = [-200, 0] if ballpos[0] < 0 else [200, 0]
                        else:
                            desired_pos = [-200, 0] if ballpos[0] > 0 else [200, 0]
                    else:
                        desired_pos = [0, -200]
                elif substate2 == 4:
                    desired_heading = 0
                    if ballpos[1] < 80 and abs(ballpos[0]) > 80:
                        desired_pos = [ballpos[0], -10]
                    else:
                        desired_pos = [ballpos[0], ballpos[1] - 60]

                    if abs(desired_pos[0]) + abs(desired_pos[1]) < 150:
                        dribbler_on = True
                    else:
                        dribbler_on = False

            elif botstate == 3: #chill in goals
                desired_heading = 0
                if own_goalpos != [0,-250]: # align middle and go backwards
                    desired_pos = [ballpos[0], own_goalpos[1] + 70]
                    ingoalspd = int(basespd / 5)
                else:
                    desired_pos = [goalpos[0], -250]
                    ingoalspd = basespd
                dribbler_on = False

            dribbler_list.append(dribbler_on)
            if len(dribbler_list) > 100:
                dribbler_list.pop(0)
            if dribbler_on:
                pass
            else:
                if dribbler_list.count(True) > 5:
                    pass
                else:
                    pass

            print(f"botstate={botstate} on line={on_line} goalpos={goalpos} ballpos={ballpos}")

#----------------------------------------------------------------------
#            translate all variables into motor movement
#----------------------------------------------------------------------
            heading_error = desired_heading - compass
            heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi #angle difference between desired and actual
            spin_weight = base_spin * max(0.1,min(abs(heading_error),2)) if heading_error != 0 else base_spin #scale spd based on how much angle difference
            if abs(heading_error) < 0.05: #dont spin if difference too small
                rot = 0
            else:
                rot = spin_weight * heading_error

            spd_scale_helper = max(min(abs(desired_pos[0]) + abs(desired_pos[1]),220),0)
            spd_multi = 0.00001 * (spd_scale_helper ** 2) + 0.002 * spd_scale_helper + 0.1 #quadratic scaling of speed, further the bot wants to go, faster itll go
            spd_multi = max(min(spd_multi,1),0.2)
            maxspd = round(basespd * (1 + (abs(rot) / 160)) * spd_multi)
            maxspd *= line_spd_multi
            new_maxspd = new_maxspd * 0.9 + maxspd * 0.1 #alpha beta smoothing

            new_desired_pos = [desired_pos[0] * 0.1 + new_desired_pos[0] * 0.9, desired_pos[1] * 0.1 + new_desired_pos[1] * 0.9] #alpha beta smoothing of desired position
            if on_line:
                mag = math.hypot(linex, liney)
                new_desired_pos = [linex / mag * 200, liney / mag * 200]  # straight away from the line
                new_maxspd = line_escape_speed

            xvel = new_desired_pos[0]
            yvel = new_desired_pos[1]
            x_field = -yvel
            y_field = xvel
            angle = -compass
            x_robot = x_field * math.cos(angle) - y_field * math.sin(angle)
            y_robot = x_field * math.sin(angle) + y_field * math.cos(angle)

            motors.motorspeed1,motors.motorspeed2,motors.motorspeed3,motors.motorspeed4 = VelocityToMotor(x_robot,y_robot,rot,new_maxspd) #convert variables into motor speed

            # Maintain a fixed 100 Hz loop
            next_loop += CONTROL_PERIOD
            sleep_time = next_loop - time.monotonic()

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_loop = time.monotonic()
    
    except KeyboardInterrupt:
        print("User stopped.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        safe_shutdown(grabber,camera,motors,imu,pcb,comms)

main()
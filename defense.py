import threading
import math
import cv2
import picamera2
import numpy as np
import time
import sys
import socket
import json
from smbus2 import SMBus, i2c_msg
import select
import board
import busio
from steelbar_powerful_bldc_driver import PowerfulBLDCDriver
import adafruit_bno08x
from adafruit_bno08x.i2c import BNO08X_I2C
from gpiozero import DigitalInputDevice

script_activate_pin = DigitalInputDevice(25, pull_up = True)

TEAM_ID = "GERM_INC"
ROBOT_ID = 2 #goalie bot
COMMS_PORT = 5555

class FrameGrabber(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True

        self.frame = None
        self.hsv = np.zeros((160,120,3), dtype=np.uint8)
        self.cap = picamera2.Picamera2()
        config = self.cap.create_preview_configuration(
	        lores={"size": (160, 120), "format": "RGB888"})
        self.cap.configure(config)
        self.cap.set_controls({
            "AwbEnable": False,
            "ColourGains": (2.1, 2.7)   # blue, red tweak when needed
        })
        self.cap.start()

    def run(self):
        while self.running:
            frame = self.cap.capture_array("lores")
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            self.frame = frame
            try:
                self.hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            except:
                continue
            time.sleep(0.01)

class DetectionThread(threading.Thread):
    def __init__(self, grabber):
        super().__init__()
        self.daemon = True
        self.running = True
        self.grabber = grabber

        self.blue = [0,0,0,0]
        self.yellow = [0,0,0,0]
        self.frame = None
        self.ready = False

        # HSV ranges
        self.lower_blue = np.array([90, 200, 100])
        self.upper_blue = np.array([110, 255, 255])
        self.lower_yellow = np.array([0, 180, 180])
        self.upper_yellow = np.array([40, 255, 255])

        self.kernel = np.ones((3,3), np.uint8)

        # pixel region to ignore (center, ignore bot)
        self.ignore_x1 = 20
        self.ignore_x2 = 85
        self.ignore_y1 = 50
        self.ignore_y2 = 110

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
            yellow_raw = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

            blue_raw[self.ignore_y1:self.ignore_y2, self.ignore_x1:self.ignore_x2] = 0
            yellow_raw[self.ignore_y1:self.ignore_y2, self.ignore_x1:self.ignore_x2] = 0

            masks = {
                "blue":   cv2.morphologyEx(blue_raw, cv2.MORPH_OPEN, self.kernel),
                "yellow": cv2.morphologyEx(yellow_raw, cv2.MORPH_OPEN, self.kernel),
            }

            self.blue = self._merge_blobs(masks["blue"])
            self.yellow = self._merge_blobs(masks["yellow"])
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
    
class IMUThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True

        self.i2c = busio.I2C(board.SCL, board.SDA, frequency = 400000)
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
        self.CMD_TO_RESPONSE_DELAY = 0.05
        self.READ_RETRIES = 3
        self.RETRY_DELAY = 0.02
        self.bus = SMBus(self.I2C_BUS)

    def _send_command(self, cmd):
        self.bus.write_byte(self.I2C_ADDR, cmd)

    def _read_raw(self, length):
        msg = i2c_msg.read(self.I2C_ADDR, length)
        self.bus.i2c_rdwr(msg)
        return bytes(msg)

    def _read_packet(self, cmd, length):
        last_err = None

        for _ in range(self.READ_RETRIES):
            try:
                self._send_command(cmd)
                time.sleep(self.CMD_TO_RESPONSE_DELAY)

                data = self._read_raw(length)
                if len(data) == length:
                    return data

            except OSError as e:
                last_err = e
                time.sleep(self.RETRY_DELAY)

        raise IOError(f"Failed to read packet: {last_err}")

    def _read_ir(self):
        data = self._read_packet(self.CMD_READ_IR, self.IR_PACKET_SIZE)

        return [
            {
                'detected': data[i * 2] if data[i * 2 + 1] >= 2 else 0,
                'distance': data[i * 2 + 1]
            }
            for i in range(12)
        ]

    def _read_colours(self):
        data = self._read_packet(self.CMD_READ_COLOURS, self.COLOUR_PACKET_SIZE)

        values = []
        for i in range(self.COLOUR_SENSOR_COUNT):
            lo = data[2*i]
            hi = data[2*i + 1]
            values.append(lo | (hi << 8))

        return values

    def set_brightness(self, value: float):
        """
        Send brightness value to STM32.
        Valid range: 0.0 to 65535.0 (matches TIM3 period).
        Sends command 0x03 followed by 2 bytes (uint16, little-endian).
        This matches the STM32 SlaveRxCpltCallback which expects
        exactly 2 data bytes after the 0x03 command byte.
        """
        val = int(max(0.0, min(65535.0, value)))
        lo  = val & 0xFF
        hi  = (val >> 8) & 0xFF
        # write_i2c_block_data sends: START, ADDR+W, 0x03 (reg), lo, hi, STOP
        # STM32 receives 0x03 first (1 byte), then queues receive of 2 more bytes
        self.bus.write_i2c_block_data(self.I2C_ADDR, 0x03, [lo, hi])

    def run(self):
        while self.running:
            try:
                self.ir = self._read_ir()
                self.colours = self._read_colours()
                self.ready = True
            except IOError as e:
                print(f"PCB I2C error: {e}")
                self.ready = False

            time.sleep(0.004)

        self.bus.close()

class MotorThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True

        self.speedlimit = 546133333
        self.motorspeed1 = 0
        self.motorspeed2 = 0
        self.motorspeed3 = 0
        self.motorspeed4 = 0
        self.motorspeed5 = 0

        self.i2c = busio.I2C(board.SCL, board.SDA)

        self.motor1 = PowerfulBLDCDriver(self.i2c, 26)
        self.motor1.set_current_limit_foc(262144)  # max 8 amps is 524288
        self.motor1.set_id_pid_constants(1500, 200)
        self.motor1.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
        self.motor1.set_position_pid_constants(275, 0, 0)
        self.motor1.set_position_region_boundary(250000)
        self.motor1.set_ELECANGLEOFFSET(1161314304)
        self.motor1.set_SINCOSCENTRE(1244)
        self.motor1.set_speed_limit(self.speedlimit)
        self.motor1.configure_operating_mode_and_sensor(3, 1)
        self.motor1.configure_command_mode(12)

        self.motor2 = PowerfulBLDCDriver(self.i2c, 32)
        self.motor2.set_current_limit_foc(262144)  # 4 amps
        self.motor2.set_id_pid_constants(1500, 200)
        self.motor2.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
        self.motor2.set_position_pid_constants(275, 0, 0)
        self.motor2.set_position_region_boundary(250000)
        self.motor2.set_ELECANGLEOFFSET(1304942336)
        self.motor2.set_SINCOSCENTRE(1239)
        self.motor2.set_speed_limit(self.speedlimit)
        self.motor2.configure_operating_mode_and_sensor(3, 1)
        self.motor2.configure_command_mode(12)

        self.motor3 = PowerfulBLDCDriver(self.i2c, 28)
        self.motor3.set_current_limit_foc(262144)
        self.motor3.set_id_pid_constants(1500, 200)
        self.motor3.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
        self.motor3.set_position_pid_constants(275, 0, 0)
        self.motor3.set_position_region_boundary(250000)
        self.motor3.set_ELECANGLEOFFSET(1772804352)
        self.motor3.set_SINCOSCENTRE(1251)
        self.motor3.set_speed_limit(self.speedlimit)
        self.motor3.configure_operating_mode_and_sensor(3, 1)
        self.motor3.configure_command_mode(12)

        self.motor4 = PowerfulBLDCDriver(self.i2c, 27)
        self.motor4.set_current_limit_foc(262144)
        self.motor4.set_id_pid_constants(1500, 200)
        self.motor4.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
        self.motor4.set_position_pid_constants(275, 0, 0)
        self.motor4.set_position_region_boundary(250000)
        self.motor4.set_ELECANGLEOFFSET(1352689664)
        self.motor4.set_SINCOSCENTRE(1251)
        self.motor4.set_speed_limit(self.speedlimit)
        self.motor4.configure_operating_mode_and_sensor(3, 1)
        self.motor4.configure_command_mode(12)

        self.motor5 = PowerfulBLDCDriver(self.i2c, 25) #dribbler motor
        self.motor5.set_current_limit_foc(262144)
        self.motor5.set_id_pid_constants(1500, 200)
        self.motor5.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
        self.motor5.set_position_pid_constants(275, 0, 0)
        self.motor5.set_position_region_boundary(250000)
        self.motor5.set_ELECANGLEOFFSET(1326110464)
        self.motor5.set_SINCOSCENTRE(1221)
        self.motor5.set_speed_limit(self.speedlimit)
        self.motor5.configure_operating_mode_and_sensor(3, 1)
        self.motor5.configure_command_mode(12)

    def run(self):
        while self.running:
            self.motor1.set_speed(int(-self.motorspeed1))
            self.motor2.set_speed(int(-self.motorspeed2))
            self.motor3.set_speed(int(-self.motorspeed3))
            self.motor4.set_speed(int(-self.motorspeed4))
            self.motor5.set_speed(int(self.motorspeed5))
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

        self.teammate_state = {}      # most recent info FROM the teammate
        self.teammate_last_seen = 0
        self.my_state = {}              # what THIS robot wants to tell its teammate
    def run(self):
        last_send = 0
        while self.running:
            now = time.time()

            if self.enabled and now - last_send >= self.send_interval:
                try:
                    msg = {"team": TEAM_ID, "robot": ROBOT_ID, **self.my_state}
                    self.sock.sendto(json.dumps(msg).encode("utf-8"), ("255.255.255.255", COMMS_PORT))
                except OSError as e:
                    print(f"Comms send error: {e}")
                last_send = now

            try:
                data, _ = self.sock.recvfrom(1024)
                msg = json.loads(data.decode("utf-8"))
                if (self.enabled and isinstance(msg, dict)
                        and msg.get("team") == TEAM_ID # message from bot of the same team
                        and msg.get("robot") != ROBOT_ID):   # ignore a possible echo of our own broadcast
                    self.teammate_state = msg
                    self.teammate_last_seen = now
            except socket.timeout:
                pass
            except (OSError, json.JSONDecodeError):
                pass

    def stop(self):
        self.running = False
        self.sock.close()

def VelocityToMotor(xvel, yvel, rot, maxspd):
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

class Hysteresis:
    """
    Holds a value steady across brief flickers around a sensor threshold.
    A new value only overwrites the current one once it's been the
    requested value continuously for `hold_time` seconds

    instant_enter (optional): a function taking the raw value, returning
    True if that value should commit with no delay. Use this for safety
    states you want to react to immediately, while still debouncing how
    quickly you're willing to leave that state again.
    """
    def __init__(self, hold_time, instant_enter=None):
        self.hold_time = hold_time
        self.instant_enter = instant_enter
        self.current = None
        self._pending = None
        self._pending_since = None

    def update(self, raw_value):
        now = time.time()

        if self.current is None:                          # first call - nothing to debounce yet
            self.current = raw_value
            return self.current

        if raw_value == self.current:                       # still agrees - clear any pending change
            self._pending = None
            return self.current

        if self.instant_enter and self.instant_enter(raw_value):
            self.current = raw_value                         # safety case - commit with no delay
            self._pending = None
            return self.current

        if raw_value != self._pending:                        # new candidate - start timing it
            self._pending = raw_value
            self._pending_since = now
            return self.current

        if now - self._pending_since >= self.hold_time:         # held long enough - commit it
            self.current = raw_value
            self._pending = None

        return self.current

def read_input():
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip()
    return None

def safe_shutdown(grabber, camera, motors, imu, pcb, comms):
    print("Shutting down safely...")

    # stop motors first
    motors.motorspeed1 = 0
    motors.motorspeed2 = 0
    motors.motorspeed3 = 0
    motors.motorspeed4 = 0
    motors.motorspeed5 = 0
    motors.motor1.clear_faults()
    motors.motor2.clear_faults()
    motors.motor3.clear_faults()
    motors.motor4.clear_faults()
    motors.motor5.clear_faults()

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

    print("Waiting for sensors...")
    while not (imu.ready and camera.ready and pcb.ready):
        time.sleep(0.05)
    
    print("Calibrating heading... keep robot still")
    time.sleep(1)  # let imu settle

    print("Waiting for signal")
    #initialise variables
    compass = 0
    xvel = 0
    yvel = 0
    heading_error = 0
    rot = 0
    basespd = 200000 # ideal speed
    dribblerspd = -5000000
    spin_weight = 50 # bigger number = bot spins more instead of moves more
    line_threshold = 4000 # tune for colour sensor readings
    line_escape_speed = 50000000
    desired_heading = 0
    ir = [math.pi/2,50] # direction, distance
    ballpos = [0,0] #cartesian plane coord relative of bot
    goalpos = [0,200] # cartesian plane coord relative of bot
    goal_colour = 0 # 0 shoot for yellow, 1 shoot for blue
    heading_offset = imu.heading
    comms_command = {}
    directionlist = []
    irdirection = 0
    unconcordantdirection = 0
    brightness_float = 10000  # pcb led brightness: 0 - 65535
    pcb.set_brightness(brightness_float)
    botstate_hyst = Hysteresis(hold_time=0.15)

    while script_activate_pin.is_active:
        if camera.yellow[1] > 60:
            goal_colour = 1
        else:
            goal_colour = 0

        if max(pcb.colours) > line_threshold: # calibrate pcb leds
            brightness_float -= 200
            pcb.set_brightness(brightness_float)
        elif max(pcb.colours) + 500 < line_threshold:
            brightness_float += 200
            pcb.set_brightness(brightness_float)

        heading_offset = imu.heading
        time.sleep(0.01)

    print("running")
    robot_active = True

    try:
        while True:
            irx = 0
            iry = 0
            linex = 0
            liney = 0

            user_input = read_input()
            # TESTING speed
            if user_input == "1": basespd = 0
            if user_input == "2": basespd = 500000
            if user_input == "3": basespd = 5000000
            if user_input == "4": basespd = 20000000
            if user_input == "5": basespd = 50000000
            if user_input == "6": basespd = 100000000
            if user_input == "7": basespd = 150000000
            if user_input == "8": basespd = 200000000
            if user_input == "9": basespd = 250000000
            if user_input == "0": basespd = 300000000
            #dribbler
            if user_input == "'": dribblerspd = 0
            if user_input == ",": dribblerspd = -5000000
            if user_input == ".": dribblerspd = -20000000
            if user_input == "p": dribblerspd = -100000000
            if user_input == "y": dribblerspd = -500000000

            if script_activate_pin.is_active: #paused bot
                if robot_active:
                    print("Paused")
                    robot_active = False
                motors.motorspeed1 = 0
                motors.motorspeed2 = 0
                motors.motorspeed3 = 0
                motors.motorspeed4 = 0
                motors.motorspeed5 = 0
                comms.my_state.update({"bot active": 0}) # bot off, likely called damage or 30sec penalty

                if camera.yellow[1] > 60:
                    goal_colour = 1
                else:
                    goal_colour = 0

                if max(pcb.colours) > line_threshold: # calibrate pcb leds
                    brightness_float -= 200
                    pcb.set_brightness(brightness_float)
                elif max(pcb.colours) + 500 < line_threshold:
                    brightness_float += 200
                    pcb.set_brightness(brightness_float)

                heading_offset = imu.heading

                time.sleep(0.02)
                continue
            else:
                robot_active = True #running bot
                comms.my_state.update({"bot active": 1})

#----------------------------------------------------------------------
#            comms from and to other bot
#----------------------------------------------------------------------
            teammate_fresh = (time.time() - comms.teammate_last_seen) < 0.5 # checks if the bots are still connected
            if isinstance(comms.teammate_state, dict) and teammate_fresh:
                comms_command = comms.teammate_state.get("command") # 1 for go get ball, 0 for chill
                attack_bot_state = comms.teammate_state.get("bot active") # 0 for bot off, 1 for bot on
            else:
                comms_command = {}
                attack_bot_state = {}

#----------------------------------------------------------------------
#            convert camera readings into goal position
#----------------------------------------------------------------------
            if goal_colour == 0: #shoot in yellow
                if camera.yellow == [0,0,0,0]:
                    goalpos= [0,200]
                else:
                    goalx = camera.yellow[0] + camera.yellow[2]/2
                    goaly = camera.yellow[1] + camera.yellow[3]/2
                    dx = goalx - 60
                    dy = 80 - goaly
                    goalpos =[dx,dy]
            else: #shoot in blue
                if camera.blue == [0,0,0,0]:
                    goalpos = [0,200]
                else:
                    goalx = camera.blue[0] + camera.blue[2]/2
                    goaly = camera.blue[1] + camera.blue[3]/2
                    dx = goalx - 60
                    dy = 80 - goaly
                    goalpos = [dx,dy]

#----------------------------------------------------------------------
#            read ir then convert into ball position, compass
#----------------------------------------------------------------------
            for i, sensor in enumerate(pcb.ir):
                if sensor["detected"]:
                    angle = i * math.pi / 6 + math.pi / 2

                    irx += math.cos(angle)
                    iry += math.sin(angle)

            if irx != 0 or iry != 0:
                irdirection = math.atan2(iry, irx) # direction
    
                if len(directionlist) > 10: # smoothing
                    if unconcordantdirection > 10: # 40ms
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

                ball_distance = max(sensor["distance"] for sensor in pcb.ir if sensor["detected"]) * 25
                ir = [circular_mean(directionlist), ball_distance]
            else:
                ir = None


            if ir is None:
                ballpos = [0,0]
            else:
                ballpos = [round(math.cos(ir[0]) * ir[1]), round(math.sin(ir[0]) * ir[1])]

            compass = imu.heading - heading_offset
            compass = (compass + math.pi) % (2*math.pi) - math.pi

#----------------------------------------------------------------------
#            determine states
#----------------------------------------------------------------------
            if ir is None: #doesnt see ball
                raw_botstate = 0
            elif ballpos[1] < 30: #ball behind bot
                raw_botstate = 1
            elif attack_bot_state == 0 or attack_bot_state == None: # attack bot is off
                raw_botstate = 2
            elif comms_command == 1: #signal from other bot to go get ball
                raw_botstate = 3
            else: #chill in goals
                raw_botstate = 4

            botstate = botstate_hyst.update(raw_botstate)

#----------------------------------------------------------------------
#            state machine
#----------------------------------------------------------------------
            if botstate == 0: # do not see ball
                desired_heading = 0
                desired_pos = [goalpos[0], goalpos[1] - 80] # align middle and go backwards #TUNE -80 to be infront of goals
                motors.motorspeed5 = 0

            elif botstate == 1: #go backwards
                desired_heading = 0
                if ir[1] < 75: #not close
                    desired_pos = [0, -200]
                else:
                    if abs(ballpos[0]) < 10: #right behind ball
                        if goalpos[0] < 0: #bot right of goals
                            desired_pos = [-200, -200]
                        else:
                            desired_pos = [200, -200]
                    else:
                        desired_pos = [0, -200]

            elif botstate == 2: # go for ball then score
                if ir[1] > 74: #ball in bcz
                    motors.motorspeed5 = dribblerspd
                    desired_heading = math.atan2(goalpos[1], goalpos[0])
                    desired_pos = goalpos
                else:
                    motors.motorspeed5 = 0
                    desired_heading = 0
                    desired_pos = ballpos

            elif botstate == 3: # go for ball then pass
                if ir[1] > 74: #ball in bcz
                    motors.motorspeed5 = dribblerspd
                    desired_heading = 0
                    desired_pos = [0,200]
                else:
                    motors.motorspeed5 = 0
                    desired_heading = 0
                    desired_pos = ballpos

            elif botstate == 4: #chill in goals
                desired_heading = 0
                desired_pos = [goalpos[0], goalpos[1] - 100] # align middle and go backwards #TUNE -100 to be inside goals
                motors.motorspeed5 = 0

#----------------------------------------------------------------------
#            line detection
#----------------------------------------------------------------------
            for i, value in enumerate(pcb.colours):
                if value > line_threshold:
                    angle = i * (math.pi / 16) + math.pi / 2   # colour1 = front, spread anticlockwise
                    excess = value - line_threshold
                    linex += math.cos(angle) * excess
                    liney += math.sin(angle) * excess

            on_line = (linex != 0 or liney != 0)
            if False: #DEBUG
                mag = math.hypot(linex, liney)
                desired_pos = [-linex / mag * 200, -liney / mag * 200]  # straight away from the line

            #DEBUG
            print(ballpos)
            print(goalpos)
            print(botstate)
            print("================")

#----------------------------------------------------------------------
#            translate all variables into motor movement
#----------------------------------------------------------------------
            heading_error = desired_heading - compass
            heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
            rot = spin_weight * heading_error

            maxspd = round(basespd * (1 + (abs(rot) / 160)) * (1 + (1 / (ir[1] + 1))))
            if on_line:
                maxspd = line_escape_speed

            xvel = desired_pos[0]
            yvel = desired_pos[1]
            x_field = -yvel
            y_field = xvel
            angle = -compass
            x_robot = x_field * math.cos(angle) - y_field * math.sin(angle)
            y_robot = x_field * math.sin(angle) + y_field * math.cos(angle)

            motors.motorspeed1,motors.motorspeed2,motors.motorspeed3,motors.motorspeed4 = VelocityToMotor(x_robot,y_robot,rot,maxspd)
    
    except KeyboardInterrupt:
        print("User stopped.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        safe_shutdown(grabber,camera,motors,imu,pcb,comms)

main()
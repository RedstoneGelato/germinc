import threading
import random
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
from gpiozero import DigitalInputDevice #imports

script_activate_pin = DigitalInputDevice(25, pull_up = True) #gpio pin for on/off switch

TEAM_ID = "GERM_INC"
ROBOT_ID = 1 #attack bot
COMMS_PORT = 5555 #used by comms

class FrameGrabber(threading.Thread): #raw camera capture
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True

        self.frame = None
        self.hsv = np.zeros((320,240,3), dtype=np.uint8)
        self.cap = picamera2.Picamera2()
        config = self.cap.create_preview_configuration(main={"size": (320,240), "format": "RGB888"}) #camera configs
        self.cap.configure(config)
        self.cap.set_controls({
            "AwbEnable": False,
            "ColourGains": (2.8, 2.2)   # blue, red tweak when needed
        })
        self.cap.start()

    def run(self):
        while self.running:
            frame = self.cap.capture_array("main")
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) #rotate capture due to physical rotated camera
            self.frame = frame
            try:
                self.hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #convert rgb to hsv
            except:
                continue
            time.sleep(0.01)

class DetectionThread(threading.Thread): #analyse camera feed, split into 2 threads because lack of processing power
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
            yellow_raw = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

            blue_raw[self.ignore_y1:self.ignore_y2, self.ignore_x1:self.ignore_x2] = 0
            yellow_raw[self.ignore_y1:self.ignore_y2, self.ignore_x1:self.ignore_x2] = 0

            masks = {
                "blue":   cv2.morphologyEx(blue_raw, cv2.MORPH_OPEN, self.kernel),
                "yellow": cv2.morphologyEx(yellow_raw, cv2.MORPH_OPEN, self.kernel),
            }

            self.yellow = self._merge_blobs(masks["yellow"], 200)
            self.blue = self._merge_blobs(masks["blue"], 200)
            time.sleep(0.005)

    def _merge_blobs(self, mask, min_area):
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
        val = int(max(0.0, min(65535.0, value)))
        lo  = val & 0xFF
        hi  = (val >> 8) & 0xFF
        with self.bus_lock:
            self.bus.write_i2c_block_data(self.I2C_ADDR, 0x03, [lo, hi])

    def run(self):
        while self.running:
            try:
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

        self.i2c = busio.I2C(board.SCL, board.SDA)

        self.motor1 = PowerfulBLDCDriver(self.i2c, 26)
        self.motor1.set_current_limit_foc(262144)  # max 8 amps is 524288
        self.motor1.set_id_pid_constants(1500, 200)
        self.motor1.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
        self.motor1.set_position_pid_constants(275, 0, 0)
        self.motor1.set_position_region_boundary(250000)
        self.motor1.set_ELECANGLEOFFSET(1559051008)
        self.motor1.set_SINCOSCENTRE(1258)
        self.motor1.set_speed_limit(self.speedlimit)
        self.motor1.configure_operating_mode_and_sensor(3, 1)
        self.motor1.configure_command_mode(12)

        self.motor2 = PowerfulBLDCDriver(self.i2c, 25)
        self.motor2.set_current_limit_foc(262144)  # 4 amps
        self.motor2.set_id_pid_constants(1500, 200)
        self.motor2.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
        self.motor2.set_position_pid_constants(275, 0, 0)
        self.motor2.set_position_region_boundary(250000)
        self.motor2.set_ELECANGLEOFFSET(1349926656)
        self.motor2.set_SINCOSCENTRE(1247)
        self.motor2.set_speed_limit(self.speedlimit)
        self.motor2.configure_operating_mode_and_sensor(3, 1)
        self.motor2.configure_command_mode(12)

        self.motor3 = PowerfulBLDCDriver(self.i2c, 27)
        self.motor3.set_current_limit_foc(262144)
        self.motor3.set_id_pid_constants(1500, 200)
        self.motor3.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
        self.motor3.set_position_pid_constants(275, 0, 0)
        self.motor3.set_position_region_boundary(250000)
        self.motor3.set_ELECANGLEOFFSET(1769756160)
        self.motor3.set_SINCOSCENTRE(1247)
        self.motor3.set_speed_limit(self.speedlimit)
        self.motor3.configure_operating_mode_and_sensor(3, 1)
        self.motor3.configure_command_mode(12)

        self.motor4 = PowerfulBLDCDriver(self.i2c, 28)
        self.motor4.set_current_limit_foc(262144)
        self.motor4.set_id_pid_constants(1500, 200)
        self.motor4.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
        self.motor4.set_position_pid_constants(275, 0, 0)
        self.motor4.set_position_region_boundary(250000)
        self.motor4.set_ELECANGLEOFFSET(1790054912)
        self.motor4.set_SINCOSCENTRE(1227)
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

        self.teammate_state = {} # most recent info FROM the teammate
        self.teammate_last_seen = 0
        self.my_state = {"bot active": 0, "command": 1} # what THIS robot wants to tell its teammate
    def run(self):
        last_send = 0
        while self.running:
            now = time.monotonic()

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
                        and msg.get("robot") != ROBOT_ID): # ignore a possible echo of our own broadcast
                    self.teammate_state = msg
                    self.teammate_last_seen = now
            except socket.timeout:
                pass
            except (OSError, json.JSONDecodeError):
                pass

    def stop(self):
        self.running = False
        self.sock.close()

class Hysteresis:
    def __init__(self, hold_time, instant_enter=None):
        self.hold_time = hold_time
        self.instant_enter = instant_enter
        self.current = None
        self._pending = None
        self._pending_since = None

    def update(self, raw_value):
        now = time.monotonic()

        if self.current is None: # first call - nothing to debounce yet
            self.current = raw_value
            return self.current

        if raw_value == self.current: # still agrees - clear any pending change
            self._pending = None
            return self.current

        if self.instant_enter and self.instant_enter(raw_value): # safety case - commit with no delay
            self.current = raw_value
            self._pending = None
            return self.current

        if raw_value != self._pending: # new candidate - start timing it
            self._pending = raw_value
            self._pending_since = now
            return self.current

        if now - self._pending_since >= self.hold_time: # held long enough - commit it
            self.current = raw_value
            self._pending = None

        return self.current

    def reset(self): #clear state so that unpause doesnt jitter
        self.current = None
        self._pending = None
        self._pending_since = None

class MotorSequence: #custom, preset, handwritten sequences of moves
    def __init__(self, steps, break_condition):
        self.steps = steps
        self.break_condition = break_condition
        self.active = False
        self.step_index = 0
        self.step_start = None #initialise variables

    def start(self):
        self.active = True
        self.step_index = 0
        self.step_start = time.monotonic()

    def stop(self):
        self.active = False
        self.step_index = 0
        self.step_start = None

    def tick(self):
        if self.break_condition():
            self.stop()
            return ("break", None)

        duration, xvel, yvel, rot, maxspd = self.steps[self.step_index]
        if time.monotonic() - self.step_start >= duration: #checks if last move is still continuing
            self.step_index += 1
            self.step_start = time.monotonic()
            if self.step_index >= len(self.steps):
                self.stop()
                return ("done", None)
            duration, xvel, yvel, rot, maxspd = self.steps[self.step_index] #do whatever the preset says

        m1, m2, m3, m4 = VelocityToMotor(xvel, yvel, rot, maxspd)
        return ("running", (m1, m2, m3, m4))

class SequenceRunner: #group all motor sequences
    def __init__(self, *sequences):
        self.sequences = sequences

    def busy(self): #checks if any sequences are currently running
        return any(s.active for s in self.sequences)

    def stop_all(self): #pause all sequences
        for s in self.sequences:
            s.stop()

    def tick(self): #advance sequence
        for s in self.sequences:
            if s.active:
                status, cmds = s.tick()
                if status == "running":
                    return cmds
                return None
        return None

class GoalTracker: #camera to goal position
    def __init__(self, history=10, tolerance=60, lost_limit=40):
        self.history, self.tolerance, self.lost_limit = history, tolerance, lost_limit
        self.goalx_list, self.goaly_list = [], []
        self.own_goalx_list, self.own_goaly_list = [], []
        self.lostgoalcount = self.lostowngoalcount = 0
        self.unc_gx = self.unc_gy = self.unc_ogx = self.unc_ogy = 0

    def _update_axis(self, value, lst, unc):
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
        primary, secondary = (yellow, blue) if goal_colour == 0 else (blue, yellow)

        if primary == [0,0,0,0]:
            self.lostgoalcount += 1
        else:
            self.lostgoalcount = 0
            gx = primary[0] + primary[2]/2 - 120
            gy = 160 - (primary[1] + primary[3]/2)
            self.unc_gx = self._update_axis(gx, self.goalx_list, self.unc_gx)
            self.unc_gy = self._update_axis(gy, self.goaly_list, self.unc_gy)

        if secondary == [0,0,0,0]:
            self.lostowngoalcount += 1
        else:
            self.lostowngoalcount = 0
            ogx = secondary[0] + secondary[2]/2 - 120
            ogy = 160 - (secondary[1] + secondary[3]/2)
            self.unc_ogx = self._update_axis(ogx, self.own_goalx_list, self.unc_ogx)
            self.unc_ogy = self._update_axis(ogy, self.own_goaly_list, self.unc_ogy)

        if self.lostgoalcount > self.lost_limit:
            self.goalx_list.clear(); self.goaly_list.clear()
        if self.lostowngoalcount > self.lost_limit:
            self.own_goalx_list.clear(); self.own_goaly_list.clear()

        goalpos = [int(np.mean(self.goalx_list)), int(np.mean(self.goaly_list))] if self.goalx_list else [0, 250]
        own_goalpos = [int(np.mean(self.own_goalx_list)), int(np.mean(self.own_goaly_list))] if self.own_goalx_list else [0, -250]
        return goalpos, own_goalpos

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
    CameraToGoal = GoalTracker()
    flick_sequence_left = MotorSequence(
        steps=[
        #   (duration, xvel, yvel, rot,    maxspd,    dribblerspd)  -- all TUNE
            (0.1,      0,    0,    -10000, 100000000), #turn around
            (0.06,     0,    0,    10000,  500000000), #fast in-place snap-rotate to whip the ball
        ],
        break_condition=lambda: (
            script_activate_pin.is_active #bot paused
            or ballpos == [0, 0] #lost the ball mid-sequence
            or on_line #crossing the line
        ),
    )
    flick_sequence_right = MotorSequence(
        steps=[
        #   (duration, xvel, yvel, rot,    maxspd,    dribblerspd)  -- all TUNE
            (0.1,        0,    0,    10000,  100000000), #turn around
            (0.06,       0,    0,    -10000, 500000000), #fast in-place snap-rotate to whip the ball
        ],
        break_condition=lambda: (
            script_activate_pin.is_active #bot paused
            or ballpos == [0, 0] #lost the ball mid-sequence
            or on_line #crossing the line
        ),
    )
    start_sequence_left = MotorSequence(
        steps=[
        #   (duration, xvel, yvel, rot,    maxspd,    dribblerspd)  -- all TUNE
            (0.7,        0,  100,    0,  500000000), #forwards and get the ball
        ],
        break_condition=lambda: (
            script_activate_pin.is_active #bot paused
            or not (ir_snapshot[0].get("distance") == 3 or ir_snapshot[1].get("distance") == 3 or ir_snapshot[11].get("distance") == 3) #lost the ball mid-sequence
            or on_line #crossing the line
        ),
    )
    start_sequence_right = MotorSequence(
        steps=[
        #   (duration, xvel, yvel, rot,    maxspd,    dribblerspd)  -- all TUNE
            (0.7,        0,  100,    0,  500000000), #forwards and get the ball
        ],
        break_condition=lambda: (
            script_activate_pin.is_active #bot paused
            or not (ir_snapshot[0].get("distance") == 3 or ir_snapshot[1].get("distance") == 3 or ir_snapshot[11].get("distance") == 3) #lost the ball mid-sequence
            or on_line #crossing the line
        ),
    )
    sequences = SequenceRunner(flick_sequence_left, flick_sequence_right,
                            start_sequence_left, start_sequence_right)

    print("Waiting for sensors...")
    while not (imu.ready and camera.ready and pcb.ready):
        time.sleep(0.05)

    print("Waiting for signal")
    #initialise variables
    compass = 0
    xvel = 0
    yvel = 0
    heading_error = 0
    rot = 0 #positive ccw, negative cw
    heading_offset = imu.heading
    desired_heading = 0

    basespd = 80000000 #ideal speed
    new_maxspd = 0
    ingoalspd = basespd // 3
    dribblerspd = 200000000
    dribbler_on = False
    dribbler_list = []
    base_spin = 50 #bigger number = bot spins more instead of moves more
    line_escape_speed = basespd * 1.5

    ir = [math.pi/2,100] #direction, distance
    ballpos = [0,100] #cartesian plane coord relative of bot
    directionlist = []
    irdirection = 0
    unconcordantdirection = 0

    desired_pos = [0,200]
    new_desired_pos = [0,0]
    goalpos = [0,250] #cartesian plane coord relative of bot
    own_goalpos = [0,-250] #cartesian plane coord relative of bot
    goal_colour = 0 #0 shoot for yellow, 1 shoot for blue

    ball_distance = 0
    ball_distance_count = 0
    ball_distance_total = 0

    led_brightness = 40000  #pcb led brightness: 0 - 65535
    line_threshold = 1500 #threshold for white line
    colour_see_number = 0
    on_line = False
    was_on_line = False
    line_list = []
    line_spd_multi = 1
    pcb.set_brightness(led_brightness)

    botstate_hyst = Hysteresis(hold_time=0.11, instant_enter=lambda v: v == 1)
    substate_hyst = Hysteresis(hold_time=0.11)
    botstate = 2
    substate = 4

    CONTROL_PERIOD = 0.01

    while script_activate_pin.is_active: #calibrate
        with pcb.lock:
            ir_snapshot = pcb.ir
            colours_snapshot = pcb.colours
        if camera.yellow != [0,0,0,0]: #calibrate goal colours
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

            with pcb.lock: #pull variables from threads
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
                    substate_hyst.reset()
                    sequences.stop_all()
                    x_robot = 0
                    y_robot = 0
                    rot = 0
                    directionlist = []
                    CameraToGoal.goalx_list = []
                    CameraToGoal.goaly_list = []
                    CameraToGoal.own_goalx_list = []
                    CameraToGoal.own_goaly_list = []
                    dribbler_list = []

                motors.motorspeed1 = 0
                motors.motorspeed2 = 0
                motors.motorspeed3 = 0
                motors.motorspeed4 = 0
                new_desired_pos = [0,0]
                new_maxspd = 0
                dribbler_on = False
                comms.my_state.update({"bot active": 0}) # bot off, likely called damage or 30sec penalty
                comms.my_state.update({"command": 1}) #tell goalie to get ball

                with pcb.lock:
                    ir_snapshot = pcb.ir
                    colours_snapshot = pcb.colours
                yellow = camera.yellow[:]
                blue = camera.blue[:]

                if yellow != [0,0,0,0]:
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

                heading_offset = imu.heading

                time.sleep(0.02)
                continue
            else:
                if robot_active == False: #first loop since turned bot back on
                    if not sequences.busy() and (ir_snapshot[0].get("distance") == 3 or ir_snapshot[1].get("distance") == 3 or ir_snapshot[11].get("distance") == 3): #checsk if bot in kickoff position
                        if random.randint(0,1): #randomly do start sequence left or start sequence right
                            #start_sequence_left.start()
                            pass
                        else:
                            #start_sequence_right.start()
                            pass
                robot_active = True #running bot
                comms.my_state.update({"bot active": 1})

#----------------------------------------------------------------------
#            ir to ball pos, compass, camera to goal pos
#----------------------------------------------------------------------
            compass = imu.heading - heading_offset
            compass = (compass + math.pi) % (2*math.pi) - math.pi

            ir_snapshot[10] = {'detected': 0, 'distance': 0} #broken, interpolate results below
            for i, sensor in enumerate(ir_snapshot): #sum angles and strength
                if sensor["detected"] == 1 and sensor["distance"] != 0:
                    if sensor["distance"] >= 2:
                        angle = i * math.pi / 6 + math.pi/2

                        irx += math.cos(angle)
                        iry += math.sin(angle)

                    ball_distance_total += sensor["distance"]
                    ball_distance_count += 1
            if ball_distance_count > 0:
                if ir_snapshot[11].get('distance') == 3 and ir_snapshot[9].get('distance') == 3: #surrounding both 3
                    irx += math.cos(math.pi/6)
                    iry += math.sin(math.pi/6)
                    ball_distance_total += 3
                    ball_distance_count += 1
                elif (ball_distance_total - 1) / ball_distance_count == 2 and (ir_snapshot[11].get('distance') == 3 or ir_snapshot[9].get('distance') == 3) and ball_distance_count < 4: #one neighbour is close, only one sees close, not enough ir sensors see
                    irx += math.cos(math.pi/6)
                    iry += math.sin(math.pi/6)
                    ball_distance_total += 3
                    ball_distance_count += 1
                elif ball_distance_count < 4 and (ir_snapshot[11].get('distance') != 0 or ir_snapshot[9].get('distance') != 0):
                    irx += math.cos(math.pi/6)
                    iry += math.sin(math.pi/6)
                    ball_distance_total += 2
                    ball_distance_count += 1
                elif ir_snapshot[11].get('distance') == 3 or ir_snapshot[9].get('distance') == 3:
                    irx += math.cos(math.pi/6)
                    iry += math.sin(math.pi/6)
                    ball_distance_total += 2
                    ball_distance_count += 1

            if irx != 0 or iry != 0:
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
                
                ball_distance = (ball_distance_total * 25) / ball_distance_count #average strength
                ball_distance = max(min(ball_distance, 99), 1)
                ball_distance = ((100 - ball_distance) * 0.3) ** 2 #strength to distance

                ir = [circular_mean(directionlist), ball_distance] #direction, distance
                ballpos = [round(math.cos(ir[0]) * ir[1]), round(math.sin(ir[0]) * ir[1])]
            else:
                ballpos = [0,0] #doesnt see ball
                ir = [0,0]
                ball_distance = 300

            goalpos, own_goalpos = CameraToGoal.update(goal_colour, yellow, blue) #middle bottom of goal

#----------------------------------------------------------------------
#            line detection
#----------------------------------------------------------------------
            for i, value in enumerate(colours_snapshot):
                if value < line_threshold:
                    angle = i * (math.pi / 16) + math.pi/2 + compass #colour1 = front, spread anticlockwise
                    linex += math.cos(angle)
                    liney += math.sin(angle)
                    colour_see_number += 1
            on_line = colour_see_number > 0
            if on_line and not was_on_line:
                line_list.append(time.monotonic())
            was_on_line = on_line
            while line_list and time.monotonic() - line_list[0] > 3:
                line_list.pop(0)
            line_spd_multi = {0: 1, 1: 0.6, 2: 0.5, 3: 0.3, 4: 0.1}.get(len(line_list), 0.1)

#----------------------------------------------------------------------
#            comms from and to other bot
#----------------------------------------------------------------------
            teammate_fresh = (time.monotonic() - comms.teammate_last_seen) < 0.5 # checks if the bots are still connected
            if isinstance(comms.teammate_state, dict) and teammate_fresh:
                goalie_bot_state = comms.teammate_state.get("bot active") # 0 for bot off, 1 for bot on
            else:
                goalie_bot_state = None

#----------------------------------------------------------------------
#            determine states
#----------------------------------------------------------------------
            if ballpos == [0,0] and ir == [0,0]: #doesnt see ball
                raw_botstate = 0 if goalie_bot_state == 1 else 3
            elif (botstate == 1 and ir_snapshot[0].get("distance") == 3) or (ir_snapshot[0].get("distance") == 3 and ir_snapshot[1].get("distance") == 3 and ir_snapshot[11].get("distance") == 3 and ir_snapshot[2].get("distance") != 3 and ir_snapshot[10].get("distance") != 3): # ball in ball capture zone
                raw_botstate = 1 #try to shoot
            else:
                raw_botstate = 2 #try to get possession of ball

            botstate = botstate_hyst.update(raw_botstate)

#----------------------------------------------------------------------
#            state machine
#----------------------------------------------------------------------
            if botstate == 0: # do not see ball
                comms.my_state.update({"command": 1})
                desired_heading = 0
                if goalpos != [0,250]:
                    desired_pos = [goalpos[0], goalpos[1] - 180] # go midfield
                    ingoalspd = int(basespd / 5)
                else:
                    desired_pos = [own_goalpos[0], 250]
                    ingoalspd = basespd
                dribbler_on = False

            elif botstate == 1: # shoot
                comms.my_state.update({"command": 0})
                desired_pos = [goalpos[0] * 1.5, goalpos[1]]
                desired_heading = math.atan2(goalpos[1], goalpos[0] * 1.6) - math.pi/2
                desired_heading = (desired_heading + math.pi) % (2 * math.pi) - math.pi

                if not sequences.busy() and abs(goalpos[0]) > 35 and goalpos[1] < 120: #position too far for just pointing at the goal and shooting
                    pass #flick the ball towards goal
                else:
                    dribbler_on = True

            elif botstate == 2: # go for ball
                if ballpos[1] < -220 and goalpos[1] < 200 and goalie_bot_state == 1: #tell goalie to get ball
                    raw_substate = 1
                elif ballpos[1] < (50 if substate in (1, 4) else 80):
                    raw_substate = 2 if ballpos[1] < -150 else 3  # far vs near backup
                else:
                    raw_substate = 4 # just go for ball
                substate = substate_hyst.update(raw_substate)

                if substate == 1:
                    comms.my_state.update({"command": 1}) #send goalie to get ball
                    desired_heading = 0
                    desired_pos = [goalpos[0], goalpos[1] - 180]
                    dribbler_on = False
                elif substate == 2:
                    comms.my_state.update({"command": 0})
                    dribbler_on = False
                    desired_heading = 0
                    desired_pos = ballpos
                elif substate == 3:
                    comms.my_state.update({"command": 0})
                    dribbler_on = False
                    desired_heading = 0
                    if abs(ballpos[0]) < 110 and ballpos[1] < 0:
                        if len(line_list) > 1:
                            desired_pos = [-200, 0] if goalpos[0] < 40 or own_goalpos[0] < 40 else [200, 0]
                        else:
                            desired_pos = [-200, 0] if ballpos[0] > 0 else [200, 0]
                    else:
                        desired_pos = [0, -200] if ballpos[1] > 20 else [ballpos[0], -200]
                elif substate == 4: # just go for ball
                    comms.my_state.update({"command": 0})
                    desired_heading = 0
                    if ballpos[1] < 80 and abs(ballpos[0]) > 80:
                        desired_pos = [ballpos[0], -10]
                    else:
                        desired_pos = [ballpos[0], ballpos[1] - 60]

                    if abs(desired_pos[0]) + abs(desired_pos[1]) < 150:
                        dribbler_on = True
                    else:
                        dribbler_on = False

            elif botstate == 3:
                comms.my_state.update({"command": 1})
                desired_heading = 0
                if own_goalpos != [0,-250]: #align middle and go backwards
                    desired_pos = [own_goalpos[0], own_goalpos[1] + 100] if own_goalpos[1] < -20 else [own_goalpos[0], 0]
                    ingoalspd = int(basespd / 5)
                else:
                    desired_pos = [goalpos[0], -200]
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

            print(f"ballpos={ballpos} botstate = {botstate} online = {on_line} goalpos = {goalpos}")

#----------------------------------------------------------------------
#            translate all variables into motor movement
#----------------------------------------------------------------------
            sequence_ran_this_tick = False #reset if sequence ran or not
            cmds = sequences.tick()
            sequence_ran_this_tick = cmds is not None #checks if a sequence is running
            if sequence_ran_this_tick:
                motors.motorspeed1, motors.motorspeed2, motors.motorspeed3, motors.motorspeed4 = cmds #run the set sequence

            if not sequence_ran_this_tick:
                heading_error = desired_heading - compass
                heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
                spin_weight = base_spin * max(0.1,min(abs(heading_error),2)) if heading_error != 0 else base_spin
                if abs(heading_error) < 0.05:
                    rot = 0
                else:
                    rot = spin_weight * heading_error

                spd_scale_helper = max(min(abs(desired_pos[0]) + abs(desired_pos[1]),220),0)
                spd_multi = 0.00001 * (spd_scale_helper ** 2) + 0.002 * spd_scale_helper + 0.1
                spd_multi = max(min(spd_multi,1),0.3)
                if botstate == 1 or (botstate == 2 and substate == 2):
                    spd_multi = 2
                maxspd = round(basespd * (1 + (abs(rot) / 160)) * spd_multi) if botstate == 1 or botstate == 2 else round(ingoalspd * (1 + (abs(rot) / 160)) * spd_multi)
                maxspd *= line_spd_multi
                new_maxspd = new_maxspd * 0.9 + maxspd * 0.1

                new_desired_pos = [desired_pos[0] * 0.1 + new_desired_pos[0] * 0.9, desired_pos[1] * 0.1 + new_desired_pos[1] * 0.9]
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

                motors.motorspeed1,motors.motorspeed2,motors.motorspeed3,motors.motorspeed4 = VelocityToMotor(x_robot,y_robot,rot,new_maxspd)

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
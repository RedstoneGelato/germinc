import threading
import math
import cv2
import picamera2
import numpy as np
import time
import socket
import json
import board
from steelbar_powerful_bldc_driver import PowerfulBLDCDriver
import adafruit_bno08x
from adafruit_bno08x.i2c import BNO08X_I2C
import adafruit_bitbangio as bbi
from gpiozero import DigitalInputDevice #imports

script_activate_pin = DigitalInputDevice(25, pull_up = True) #gpio pin for on/off switch

TEAM_ID = "GERM_INC"
ROBOT_ID = 2 #goalie bot
COMMS_PORT = 5555 #used by comms

class AutoLockI2C:
    def __init__(self, bus):
        self._bus = bus
        self._held = False
 
    def try_lock(self):
        if self._held:
            return False
        self._held = self._bus.try_lock()
        return self._held
 
    def unlock(self):
        if self._held:
            self._bus.unlock()
            self._held = False
 
    def _run(self, fn, *args, **kwargs):
        if self._held:  # caller already holds the lock
            return fn(*args, **kwargs)
        while not self._bus.try_lock():
            pass
        try:
            return fn(*args, **kwargs)
        finally:
            self._bus.unlock()
 
    def writeto(self, *a, **k):
        return self._run(self._bus.writeto, *a, **k)
 
    def readfrom_into(self, *a, **k):
        return self._run(self._bus.readfrom_into, *a, **k)
 
    def writeto_then_readfrom(self, *a, **k):
        return self._run(self._bus.writeto_then_readfrom, *a, **k)
 
    def scan(self):
        return self._run(self._bus.scan)
 
 
# ONE shared bus for IMU + PCB + motors (SCL = D6, SDA = D5)
i2c = AutoLockI2C(bbi.I2C(board.D6, board.D5, frequency=400000))

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
            "ColourGains": (2.8, 2.2)   # blue, red tweak when needed
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
        self.lower_blue = np.array([90, 200, 100])
        self.upper_blue = np.array([110, 255, 255])
        self.lower_yellow = np.array([0, 180, 180])
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

class IMU:
    def __init__(self, bus):
        self.sensor = BNO08X_I2C(bus)
        self.sensor.enable_feature(adafruit_bno08x.BNO_REPORT_GAME_ROTATION_VECTOR)
        self.heading = 0.0
        self.ready = False
        self.errors = 0
 
    def update(self):
        try:
            h = get_heading(self.sensor)
        except Exception:  # bitbang glitch / bad SHTP packet: keep the last heading
            self.errors += 1
            return
        if h is not None:
            self.heading = h
            self.ready = True

class PCBReader:
    ADDR = 0x64
    CMD_READ_COLOURS = 0x01
    CMD_READ_IR = 0x02
    CMD_SET_BRIGHTNESS = 0x03
    COLOUR_SENSOR_COUNT = 32
    IR_SENSOR_COUNT = 12
    RESPONSE_DELAY = 0.02  # PCB needs this long between command and read (was CMD_TO_RESPONSE_DELAY)
 
    def __init__(self, bus):
        self.bus = bus
        self.ir = [{'detected': 0, 'distance': 0} for _ in range(self.IR_SENSOR_COUNT)]
        self.colours = [0] * self.COLOUR_SENSOR_COUNT
        self.ready = False
        self.errors = 0
 
        self._have_ir = False
        self._have_colours = False
        self._schedule = [self.CMD_READ_COLOURS, self.CMD_READ_IR]  # order polled; add a repeat to poll one more often
        self._next = 0
        self._pending = None  # (command, time it was sent) while waiting for the PCB to prepare its reply
        self._brightness = None
        self._brightness_sent = None
 
    def set_brightness(self, value: float):  # 0-65535, higher = brighter
        # only stores the value; it is sent from update() when no read is in flight,
        # so it can never land between a command and its response
        self._brightness = int(max(0.0, min(65535.0, value)))
 
    def _read(self, length):
        buf = bytearray(length)
        self.bus.readfrom_into(self.ADDR, buf)
        return buf
 
    def update(self):
        try:
            if self._pending is None:
                if self._brightness is not None and self._brightness != self._brightness_sent:
                    b = self._brightness
                    self.bus.writeto(self.ADDR, bytes([self.CMD_SET_BRIGHTNESS, b & 0xFF, (b >> 8) & 0xFF]))
                    self._brightness_sent = b
                cmd = self._schedule[self._next]
                self.bus.writeto(self.ADDR, bytes([cmd]))
                self._pending = (cmd, time.monotonic())
                return
 
            cmd, sent_at = self._pending
            if time.monotonic() - sent_at < self.RESPONSE_DELAY:
                return  # reply not ready yet - go do other work
            self._pending = None
            self._next = (self._next + 1) % len(self._schedule)
 
            if cmd == self.CMD_READ_IR:
                data = self._read(self.IR_SENSOR_COUNT * 2)
                self.ir = [
                    {
                        'detected': data[i * 2] if data[i * 2 + 1] >= 2 else 0,
                        'distance': data[i * 2 + 1]
                    }
                    for i in range(self.IR_SENSOR_COUNT)
                ]
                self._have_ir = True
            else:
                data = self._read(self.COLOUR_SENSOR_COUNT * 2)
                self.colours = [data[2 * i] | (data[2 * i + 1] << 8) for i in range(self.COLOUR_SENSOR_COUNT)]
                self._have_colours = True
            self.ready = self._have_ir and self._have_colours
 
        except (OSError, RuntimeError) as e:
            self._pending = None  # drop the exchange, start clean next pass
            self.errors += 1
            print(f"PCB I2C error: {e}")

MOTOR_SPEED_LIMIT = 546133333  # max spd
MOTOR_CONFIG = [  # (i2c address, ELECANGLEOFFSET, SINCOSCENTRE) - same values as before
    (26, 1161314304, 1244),  # motor 1
    (32, 1304942336, 1239),  # motor 2
    (28, 1772804352, 1251),  # motor 3
    (25, 1352689664, 1251),  # motor 4
]
 
class Motors:
    def __init__(self, bus):
        self.motors = []
        for addr, elec_offset, sincos_centre in MOTOR_CONFIG:
            m = PowerfulBLDCDriver(bus, addr)
            m.set_current_limit_foc(262144)  # max 8 amps is 524288
            m.set_id_pid_constants(1500, 200)
            m.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
            m.set_position_pid_constants(275, 0, 0)
            m.set_position_region_boundary(250000)
            m.set_ELECANGLEOFFSET(elec_offset)
            m.set_SINCOSCENTRE(sincos_centre)
            m.set_speed_limit(MOTOR_SPEED_LIMIT)
            m.configure_operating_mode_and_sensor(3, 1)
            m.configure_command_mode(12)
            self.motors.append(m)
        self._last = None
        self._last_sent = 0.0
 
    def set_speeds(self, s1, s2, s3, s4):
        speeds = (int(-s1), int(-s2), int(-s3), int(-s4))
        now = time.monotonic()
        if speeds == self._last and now - self._last_sent < 0.1:
            return
        try:
            for m, s in zip(self.motors, speeds):
                m.set_speed(s)
            self._last = speeds
            self._last_sent = now
        except (OSError, RuntimeError) as e:
            self._last = None  # force a resend next pass
            print(f"Motor I2C error: {e}")
 
    def stop(self):  # forced stop, used at shutdown
        self._last = None
        self.set_speeds(0, 0, 0, 0)
 
    def clear_faults(self):
        for m in self.motors:
            try:
                m.clear_faults()
            except Exception:
                pass

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

def get_heading(imu):
    quat = imu.game_quaternion  # (x, y, z, w)
    if quat is not None:
        x, y, z, w = quat
        return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

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

def safe_shutdown(grabber, camera, motors, comms):
    print("Shutting down safely...")
 
    # stop motors first (repeat in case a bitbang write glitches)
    for _ in range(3):
        motors.stop()
        time.sleep(0.01)
    motors.clear_faults()
 
    # stop threads (only camera + comms are threads now)
    grabber.running = False
    camera.running = False
    comms.stop()
 
    try:
        grabber.cap.stop()
    except:
        pass
 
    # wait for threads
    grabber.join()
    camera.join()
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
    motors = Motors(i2c)
    imu = IMU(i2c)
    pcb = PCBReader(i2c)
    comms = TeammateLinkThread()
    comms.start()
    CameraToGoal = GoalTracker() #start threads

    print("Waiting for sensors...")
    while not (imu.ready and camera.ready and pcb.ready):
        imu.update()
        pcb.update()
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
        imu.update()
        pcb.update()
        ir_snapshot = list(pcb.ir)
        colours_snapshot = list(pcb.colours)
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

            imu.update()
            pcb.update()
            ir_snapshot = list(pcb.ir)
            colours_snapshot = list(pcb.colours)
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

                motors.set_speeds(0, 0, 0, 0)
                new_desired_pos = [0,0]
                dribbler_on = False
                new_maxspd = 0 #reset all variables and stop motors
                comms.my_state.update({"bot active": 0}) # comm say bot off

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

            ir_snapshot[3] = {'detected': 0, 'distance': 0} #broken, interpolate results below
            for i, sensor in enumerate(ir_snapshot):
                if sensor["detected"] == 1 and sensor["distance"] != 0:
                    if sensor["distance"] >= 2:
                        angle = i * math.pi / 6 + math.pi/2

                        irx += math.cos(angle)
                        iry += math.sin(angle)

                    ball_distance_total += sensor["distance"]
                    ball_distance_count += 1
            if ball_distance_count > 0:
                if ir_snapshot[2].get('distance') == 3 and ir_snapshot[4].get('distance') == 3: #surrounding both 3
                    irx += math.cos(math.pi)
                    iry += math.sin(math.pi)
                    ball_distance_total += 3
                    ball_distance_count += 1
                elif (ball_distance_total - 1) / ball_distance_count == 2 and (ir_snapshot[2].get('distance') == 3 or ir_snapshot[4].get('distance') == 3) and ball_distance_count < 4: #one neighbour is close, only one sees close, not enough ir sensors see
                    irx += math.cos(math.pi)
                    iry += math.sin(math.pi)
                    ball_distance_total += 3
                    ball_distance_count += 1
                elif ball_distance_count < 4 and (ir_snapshot[2].get('distance') != 0 or ir_snapshot[4].get('distance') != 0):
                    irx += math.cos(math.pi)
                    iry += math.sin(math.pi)
                    ball_distance_total += 2
                    ball_distance_count += 1
                elif ir_snapshot[2].get('distance') == 3 or ir_snapshot[4].get('distance') == 3:
                    irx += math.cos(math.pi)
                    iry += math.sin(math.pi)
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

                ball_distance = (ball_distance_total * 25) / ball_distance_count #average distance
                ball_distance = max(min(ball_distance, 99), 1)
                ball_distance = ((100 - ball_distance) * 0.3) ** 2

                ir = [circular_mean(directionlist), ball_distance] #direction, distance
                ballpos = [round(math.cos(ir[0]) * ir[1]), round(math.sin(ir[0]) * ir[1])]
            else:
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
                    desired_pos = [own_goalpos[0], own_goalpos[1] + 100] if own_goalpos[1] < -20 else [own_goalpos[0], 200]
                    ingoalspd = int(basespd / 5)
                else:
                    desired_pos = [goalpos[0], -250]
                    ingoalspd = basespd
                dribbler_on = False

            elif botstate == 1: #go for ball then score
                if (substate1 == 1 and ir_snapshot[0].get("distance") == 3) or (ir_snapshot[0].get("distance") == 3 and ir_snapshot[1].get("distance") == 3 and ir_snapshot[11].get("distance") == 3 and ir_snapshot[2].get("distance") != 3 and ir_snapshot[10].get("distance") != 3):
                    raw_substate1 = 1  #ball in bcz
                elif ballpos[1] < (60 if substate1 in (1, 4) else 80): #2 far backup, 3 close backup
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
                    if abs(ballpos[0]) < 120 and ballpos[1] < 0: #wrap around ball
                        if len(line_list) > 1: #touched line
                            desired_pos = [-200, 0] if ballpos[0] < 0 else [200, 0] #go other way
                        else:
                            desired_pos = [-200, 0] if ballpos[0] > 0 else [200, 0] #wrap around base on ball position
                    else:
                        desired_pos = [0, -200] if ballpos[1] > 20 else [ballpos[0], -200]
                elif substate1 == 4:
                    desired_heading = math.atan2(goalpos[1],goalpos[0] * 1.6) - math.pi/2
                    desired_heading = (desired_heading + math.pi) % (2 * math.pi) - math.pi
                    if ballpos[1] < 120 and abs(ballpos[0]) > 120:
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
                elif ballpos[1] < (60 if substate2 in (1, 4) else 80):
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
                    if abs(ballpos[0]) < 70 and ballpos[1] < 0:
                        if len(line_list) > 1:
                            desired_pos = [-200, 0] if ballpos[0] < 0 else [200, 0]
                        else:
                            desired_pos = [-200, 0] if ballpos[0] > 0 else [200, 0]
                    else:
                        desired_pos = [0, -200]
                elif substate2 == 4:
                    desired_heading = 0
                    if ballpos[1] < 120 and abs(ballpos[0]) > 120:
                        desired_pos = [ballpos[0], -10]
                    else:
                        desired_pos = [ballpos[0] * 1.5, ballpos[1] - 60]

                    if abs(desired_pos[0]) + abs(desired_pos[1]) < 150:
                        dribbler_on = True
                    else:
                        dribbler_on = False

            elif botstate == 3: #chill in goals
                desired_heading = 0
                if own_goalpos != [0,-250]: # align middle and go backwards
                    desired_pos = [own_goalpos[0], own_goalpos[1] + 70] if own_goalpos[1] < -20 else [own_goalpos[0], 0]
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

            motors.set_speeds(*VelocityToMotor(x_robot, y_robot, rot, new_maxspd))

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
        safe_shutdown(grabber,camera,motors,comms)

main()
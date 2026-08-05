import threading
import math
import cv2
import picamera2
import numpy as np
import time
import sys
from smbus2 import SMBus, i2c_msg
import select
import board
import busio
from steelbar_powerful_bldc_driver import PowerfulBLDCDriver
import adafruit_bno08x
from adafruit_bno08x.i2c import BNO08X_I2C
from gpiozero import OutputDevice, DigitalInputDevice

kick_pin = OutputDevice(17, active_high=True, initial_value=False)
script_activate_pin = DigitalInputDevice(25, pull_up = False)

class FrameGrabber(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True

        self.frame = None
        self.hsv = np.zeros((120,160,3), dtype=np.uint8)
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

            masks = {
                "blue":   cv2.morphologyEx(cv2.inRange(hsv, self.lower_blue, self.upper_blue), cv2.MORPH_OPEN, self.kernel),
                "yellow": cv2.morphologyEx(cv2.inRange(hsv, self.lower_yellow, self.upper_yellow), cv2.MORPH_OPEN, self.kernel),
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
        self.IR_PACKET_SIZE = self.IR_SENSOR_COUNT
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

    def read_ir_activity(self):
        data = self._read_packet(0x04, 24)
        return [data[i*2] | (data[i*2+1] << 8) for i in range(12)]

    def _read_ir(self):
        data = self._read_packet(self.CMD_READ_IR, self.IR_PACKET_SIZE)
        return list(data)

    def _read_colours(self):
        data = self._read_packet(self.CMD_READ_COLOURS, self.COLOUR_PACKET_SIZE)

        values = []
        for i in range(self.COLOUR_SENSOR_COUNT):
            lo = data[2*i]
            hi = data[2*i + 1]
            values.append(lo | (hi << 8))

        return values

    def run(self):
        while self.running:
            try:
                self.ir = self._read_ir()
                self.colours = self._read_colours()
                self.strength = self.read_ir_activity()
                self.ready = True

            except IOError as e:
                print(f"PCB I2C error: {e}")
                self.ready = False

            time.sleep(0.01)

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
        self.motor1.set_ELECANGLEOFFSET(1161314304)
        self.motor1.set_SINCOSCENTRE(1244)
        self.motor1.set_speed_limit(self.speedlimit)
        self.motor1.configure_operating_mode_and_sensor(3, 1)
        self.motor1.configure_command_mode(12)
        self.motor2 = PowerfulBLDCDriver(self.i2c, 28)
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
        self.motor3 = PowerfulBLDCDriver(self.i2c, 27)
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
        self.motor4 = PowerfulBLDCDriver(self.i2c, 25)
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

    def run(self):
        while self.running:
            self.motor1.set_speed(int(-self.motorspeed1))
            self.motor2.set_speed(int(-self.motorspeed2))
            self.motor3.set_speed(int(-self.motorspeed3))
            self.motor4.set_speed(int(-self.motorspeed4))
            time.sleep(0.005)

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

def kick(trigger=False):
    # persistent state (stored on the function itself)
    if not hasattr(kick, "last_kick"):
        kick.last_kick = 0
        kick.active = False
        kick.start = 0

    kick_duration = 0.05
    kick_cooldown = 0.5

    now = time.time()

    # trigger kick
    if trigger and not kick.active:
        if now - kick.last_kick >= kick_cooldown:
            kick_pin.on()
            kick.active = True
            kick.start = now
            kick.last_kick = now

    # update (turn off after duration)
    if kick.active and (now - kick.start >= kick_duration):
        kick_pin.off()
        kick.active = False

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

def safe_shutdown(grabber, camera, motors, imu, pcb):
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
    kick_pin.close()

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

    print("Waiting for sensors...")
    while not (imu.ready and camera.ready and pcb.ready):
        time.sleep(0.05)
    
    print("Calibrating heading... keep robot still")
    time.sleep(1)  # let imu settle
    heading_offset = imu.heading

    print("Waiting for signal")
    #initialise variables
    compass = 0
    xvel = 0
    yvel = 0
    heading_error = 0
    rot = 0
    basespd = 2000000 # ideal speed
    spin_weight = 50 # bigger number = bot spins more instead of moves more
    line_threshold = 3000 # tune for colour sensor readings
    line_escape_speed = 50000000
    desired_heading = 0
    ir = [math.pi/2,100]
    ballpos = [0,0]
    goalpos = [0,200]
    goal_colour = 0
    irstrengthlist = []
    botstate_hyst = Hysteresis(hold_time=0.15)

    while not script_activate_pin.value:
        if camera.yellow[1] > 60:
            goal_colour = 1
        else:
            goal_colour = 0
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
            #others
            if user_input == "l": kick(True)

            if not script_activate_pin.is_active:
                if robot_active:
                    print("Paused")
                    robot_active = False
                motors.motorspeed1 = 0
                motors.motorspeed2 = 0
                motors.motorspeed3 = 0
                motors.motorspeed4 = 0
                kick()  # let an in-progress kick still finish and release the solenoid
                time.sleep(0.02)
                continue
            elif not robot_active:
                print("Resumed")
                robot_active = True

#----------------------------------------------------------------------
#            convert camera readings into goal position
#----------------------------------------------------------------------
            if goal_colour == 0: #shoot in yellow
                if camera.yellow == [0,0,0,0]:
                    goalpos= [0,200]
                else:
                    goalx = camera.yellow[0] + camera.yellow[2]/2
                    goaly = camera.yellow[1] + camera.yellow[3]/2
                    dx = goalx - 80
                    dy = 60 - goaly
                    goalpos =[dx,dy]
            else: #shoot in blue
                if camera.blue == [0,0,0,0]:
                    goalpos = [0,200]
                else:
                    goalx = camera.blue[0] + camera.blue[2]/2
                    goaly = camera.blue[1] + camera.blue[3]/2
                    dx = goalx - 80
                    dy = 60 - goaly
                    goalpos = [dx,dy]

#----------------------------------------------------------------------
#            read ir then convert into ball position
#----------------------------------------------------------------------
            for i, active in enumerate(pcb.ir):
                if active:
                    w = pcb.strength[i]
                    angle = i * math.pi / 6 + math.pi/2
                    irx += math.cos(angle) * w
                    iry += math.sin(angle) * w

            if irx != 0 or iry != 0:
                ir[0] = math.atan2(iry, irx)
                average = sum(pcb.strength) // len(pcb.strength)
                irstrengthlist.append(average)
                if len(irstrengthlist) > 10:
                    irstrengthlist.pop(0)
                strength = sum(irstrengthlist) // len(irstrengthlist)
                strength = max(0, min(strength, 10000))
                ir[1] = 100 - math.sqrt(strength)
            else:
                ir = None

            if ir is None:
                ballpos = [0,0]
            else:
                ballpos = [round(math.cos(ir[0]) * ir[1]), round(math.sin(ir[0]) * ir[1])] #relative position of ball to the bot: +x is right,+y is front
            compass = imu.heading - heading_offset
            compass = (compass + math.pi) % (2*math.pi) - math.pi

#----------------------------------------------------------------------
#            determine states
#----------------------------------------------------------------------
            if ir is None or ir[1] > 99: #doesnt see ball
                raw_botstate = 0
            elif ir[1] < 25 and ballpos[1] > 0 : # ball is in ball capture zone check: close enough, infront of bot
                raw_botstate = 1 #try to shoot
            else:
                raw_botstate = 2 #try to get possession of ball

            botstate = botstate_hyst.update(raw_botstate)

#----------------------------------------------------------------------
#            state machine
#----------------------------------------------------------------------
            if botstate == 0: # do not see ball
                desired_heading = 0
                desired_pos = [goalpos[0], -200] # align middle and go backwards
            else:
                if botstate == 1:
                    desired_heading = math.atan2(goalpos[1],goalpos[0])
                    desired_pos = goalpos

                    if ir[1] < 10 and abs(heading_error) < 0.3: # kick check: very close, facing the right way #TUNE: ir10 distance
                        kick(True)

                elif botstate == 2: # go for ball
                    if ballpos[1] < 0: # ball behind bot
                        if abs(ballpos[0]) < 25 and ballpos[1] < 25: #TUNE: ir 25 distance, same as above, corner of the bot without colliding the bot
                            if goalpos[0] > 0:
                                desired_pos = [200, -200]
                            else:
                                desired_pos = [-200, -200]
                        else:
                            desired_pos = [0,-200] # go straight backwards
                        desired_heading = 0
                    else:
                        desired_heading = math.atan2(goalpos[1],goalpos[0])
                        angle_dif = math.atan2(ballpos[1] - goalpos[1], ballpos[0] - goalpos[0]) #vector from goal to ball
                        desired_pos = [ballpos[0] + math.cos(angle_dif) * 10, ballpos[1] + math.sin(angle_dif) * 10] # go to a spot behind the ball such that the bot the ball and the goal are in a line
                        #TUNE: *10 makes it behind the ball without colliding with the ball

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
            if on_line:
                mag = math.hypot(linex, liney)
                desired_pos = [-linex / mag * 200, -liney / mag * 200]  # straight away from the line

#----------------------------------------------------------------------
#            translate all variables into motor movement
#----------------------------------------------------------------------
            heading_error = desired_heading - compass
            heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
            rot = spin_weight * heading_error

            maxspd = round(basespd * (1 + (abs(rot) / 160)) * (1 + (abs(ballpos[1]) / 1000)))
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
            kick()
    
    except KeyboardInterrupt:
        print("User stopped.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        safe_shutdown(grabber,camera,motors,imu,pcb)

main()
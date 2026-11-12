"""Every setting of the robots in one place: shared settings first, then the two robots' own (motors, kicker) at the end.
Frames: robot = +x right, +y forward; field = +y towards the goal we attack; compass = CCW radians, 0 = facing it."""
import math
from collections import namedtuple

TEAM_ID = "GERM_INC"
COMMS_PORT = 5555

SWITCH_PIN = 25
REF_MODULE = False          # True = RCJ comms module (3.3 V = play), False = our own on/off switch (closed = paused)
ZERO_MAX_ERR = 0.17
GOAL_SWITCH_PIN = 26
GOAL_SWITCH_CLOSED = "yellow"   # swap to "blue" if the robots go for the wrong goal
GOAL_FROM_CAMERA = False

VISION_CONFIG = "robot_vision_config.json"
VISION_MAX_FPS = 100
VISION_CV_THREADS = 2

CONTROL_PERIOD = 0.01

ROBOT_RADIUS = 11.0
CAPTURE_ZONE = (-5.0, 5.0, ROBOT_RADIUS - 3.0, ROBOT_RADIUS + 4.0)   # TUNE in test_localisation.py
BALL_RADIUS = 2.1
BALL_BLIND_RING = 2.0

ENEMY_DETECTION = True     # TUNE before each match: False when their robots look like the carpet

LINE_TOUCH_SAFETY = 6.0
MAX_OUT = ROBOT_RADIUS - LINE_TOUCH_SAFETY
PRECISE_STD = 6.0

LDR_COUNT = 32
LDR_LINE_THRESHOLD = 1500  # TUNE on each new carpet: below this = white line
LDR_START_ANGLE = math.pi / 2
LED_BRIGHTNESS_START = 40000

IR_COUNT = 12
IR_START_ANGLE = math.pi / 2
IR_ON_PCB = True

BASE_SPEED = 80000000
LINE_ESCAPE_SPEED = BASE_SPEED * 1.5
MOTOR_MAX = BASE_SPEED * 1.5
TURN_MAX = 0.6
TURN_FULL_ERR = 1.0
DRIBBLE_TURN_MAX = 0.4
DRIBBLE_SPEED_MIN = 0.38
DRIBBLE_MAX_ACCEL = 10.0


# ==================================== the two robots ====================================
# What differs between the striker and the goalie. The classes are only used as named boxes of settings:
# the main files pass one of them on (as "hw"), and the code reads e.g. hw.ROBOT_ID or hw.KICKER_PIN.
# One Motor per motor driver: its I2C address and the two calibration numbers `python3 test_parts.py motor` prints.
Motor = namedtuple("Motor", "i2c_address elec_angle_offset sincos_centre")


class AttackRobot:               # the striker (main_attack.py)
    ROBOT_ID = 1                 # comms id: must differ between the two robots
    DRIVE_MOTORS = [
        Motor(26, 1559051008, 1258),
        Motor(25, 1349926656, 1247),
        Motor(27, 1769756160, 1247),
        Motor(28, 1790054912, 1227),
    ]
    DRIBBLER = Motor(32, 1431223552, 1245)   # None = no dribbler
    DRIBBLER_SPEED = 200000000   # TUNE: flip the sign if it spins the ball away
    KICKER_PIN = 17              # GPIO pin of the solenoid, None = no kicker
    KICK_PULSE = 0.05            # TUNE carefully: seconds the solenoid is powered
    KICK_COOLDOWN = 1.0          # seconds between kicks


class DefenseRobot:              # the goalie (main_defence.py)
    ROBOT_ID = 2
    DRIVE_MOTORS = [
        Motor(26, 1168928512, 1240),
        Motor(32, 1226029824, 1257),
        Motor(28, 1317619456, 1236),
        Motor(25, 1392997120, 1225),
    ]
    DRIBBLER = Motor(27, 1431223552, 1245)   # TUNE: placeholder copied from the striker - calibrate before use
    DRIBBLER_SPEED = 200000000
    KICKER_PIN = 17              # TUNE: check the pin on this robot (python3 test_parts.py solenoid)
    KICK_PULSE = 0.05
    KICK_COOLDOWN = 1.0

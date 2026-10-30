"""Shared settings for both robots. Per-robot ids and motor addresses: hardware_attack.py / hardware_defense.py.
Frames: robot = +x right, +y forward; field = +y towards the goal we attack; compass = CCW radians, 0 = facing it."""
import math

TEAM_ID = "GERM_INC"
COMMS_PORT = 5555

SWITCH_PIN = 25
REF_MODULE = True          # True = RCJ comms module (3.3 V = play), False = our own on/off switch (closed = paused)
ZERO_MAX_ERR = 0.17
GOAL_SWITCH_PIN = 26
GOAL_SWITCH_CLOSED = "yellow"   # swap to "blue" if the robots go for the wrong goal
GOAL_FROM_CAMERA = False

VISION_CONFIG = "robot_vision_config.json"
VISION_MAX_FPS = 50        # TUNE: None = no cap
VISION_CV_THREADS = 1

CONTROL_PERIOD = 0.01

ROBOT_RADIUS = 11.0                         # TUNE: centre to bumper, cm
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
IR_ON_PCB = False
IR_RAW_MAX = 1000          # TUNE: raw reading with the ball right next to a sensor

BASE_SPEED = 80000000
LINE_ESCAPE_SPEED = BASE_SPEED * 1.5
MOTOR_MAX = BASE_SPEED * 1.5
TURN_MAX = 0.6
TURN_FULL_ERR = 1.0
DRIBBLE_TURN_MAX = 0.4
DRIBBLE_SPEED_MIN = 0.38   # TUNE on the robot: fastest sideways / backwards speed that keeps the ball
DRIBBLE_MAX_ACCEL = 10.0

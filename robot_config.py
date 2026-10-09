"""
robot_config.py - numbers shared by BOTH robots that you are likely to change.
Per-robot things (robot id, motor I2C addresses) live in hardware_attack.py / hardware_defense.py.

Coordinate conventions used everywhere (see also field.py):
    robot frame   origin = robot centre, +x = robot's right, +y = robot's front, cm
    field frame   origin = field centre, +y = towards the goal we attack, +x = right when facing it, cm
    "relative"    a vector from the robot to something, in cm, with FIELD axes (robot frame rotated by compass)
    compass       robot heading in radians, counter-clockwise positive, 0 = facing the goal we attack
"""
import math

TEAM_ID = "GERM_INC"
COMMS_PORT = 5555

SWITCH_PIN = 25            # on/off switch, GPIO to ground

VISION_CONFIG = "robot_vision_config.json"   # made by test_camera.py -> "Generate config file"

CONTROL_PERIOD = 0.01      # main loop runs at 100 Hz

# ---- robot body (cm)
ROBOT_RADIUS = 11.0                         # TUNE: centre to bumper
# where the ball sits when it's in the dribbler, robot frame (x_min, x_max, y_min, y_max) - TUNE in test_localisation.py
CAPTURE_ZONE = (-4.0, 4.0, ROBOT_RADIUS - 3.0, ROBOT_RADIUS + 3.0)

# ---- LDR line ring (PCB)
LDR_COUNT = 32
LDR_LINE_THRESHOLD = 1500                  # reading below this = white line
LDR_START_ANGLE = math.pi / 2              # sensor 0 is at the front, numbering goes anticlockwise
LED_BRIGHTNESS_START = 40000               # 0 - 65535

# ---- motion
BASE_SPEED = 80000000
BASE_SPIN = 50                             # bigger number = bot spins more instead of moves more
LINE_ESCAPE_SPEED = BASE_SPEED * 1.5

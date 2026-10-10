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
BALL_RADIUS = 2.1                          # cm - CHECK the ball you use

# ---- out of bounds: a robot is only out once NO part of it touches the white line
LINE_TOUCH_SAFETY = 6.0                    # cm of robot that must still be over the line at the furthest out we go
MAX_OUT = ROBOT_RADIUS - LINE_TOUCH_SAFETY # so the robot centre may be up to this far past the line's outer edge
PRECISE_STD = 6.0                          # pose spread (cm) below which the position is trusted for that; otherwise
                                           # the LDRs decide: seeing the line = escape (safe, but stays further in)

# ---- LDR line ring (PCB)
LDR_COUNT = 32
LDR_LINE_THRESHOLD = 1500                  # reading below this = white line
LDR_START_ANGLE = math.pi / 2              # sensor 0 is at the front, numbering goes anticlockwise
LED_BRIGHTNESS_START = 40000               # 0 - 65535

# ---- motion
BASE_SPEED = 80000000
BASE_SPIN = 50                             # bigger number = bot spins more instead of moves more
LINE_ESCAPE_SPEED = BASE_SPEED * 1.5
# dribbler on: slow down the more the movement direction differs from where the robot faces (the ball stays in the
# dribbler best when driving forwards). Speed multiplier = 1 driving straight forwards, falling linearly with the
# angle to DRIBBLE_SPEED_MIN driving straight backwards (sideways = halfway between).
DRIBBLE_SPEED_MIN = 0.3

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
VISION_MAX_FPS = 50        # TUNE: camera frame rate cap. Without one picamera2 runs the camera as fast as the sensor
                           # mode goes (up to 120 fps on the HQ camera) and the vision thread never rests = heat.
                           # The main loop is 100 Hz; the simulator assumes 33 fps. None = no cap.
VISION_CV_THREADS = 1      # OpenCV worker threads (whole program). The vision pipeline is about as fast on 1 thread
                           # as on 4 at 640x480, and 1 leaves the other cores (and less heat) for everything else.

CONTROL_PERIOD = 0.01      # main loop runs at 100 Hz

# ---- robot body (cm)
ROBOT_RADIUS = 11.0                         # TUNE: centre to bumper
# where the camera SEES the ball when it's in the dribbler, robot frame (x_min, x_max, y_min, y_max) - TUNE in
# test_localisation.py. The ball is reported as its nearest VISIBLE point, and our own top deck hides the floor
# just past the bumper (the near half of a held ball), so a held ball shows up ~3 cm past the bumper, not at it
# (sim, deck 4 cm: a ball in the dribbler is seen at y 13.9-14.2, |x| < 2; one 3 cm out of the dribbler at 15.3)
CAPTURE_ZONE = (-5.0, 5.0, ROBOT_RADIUS - 3.0, ROBOT_RADIUS + 4.0)
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
LINE_ESCAPE_SPEED = BASE_SPEED * 1.5
MOTOR_MAX = BASE_SPEED * 1.5               # no wheel is ever asked for more than this (moving + turning together)
# turning has its own speed (it doesn't slow down when the robot moves slowly): proportional to how far off the
# wanted heading the robot is, full speed (TURN_MAX x BASE_SPEED on every wheel) from TURN_FULL_ERR radians off
TURN_MAX = 0.6
TURN_FULL_ERR = 1.0
DRIBBLE_TURN_MAX = 0.4                     # ...at most this with the dribbler on (turning too fast loses the ball)
# dribbler on: accelerating and braking don't lose the ball - moving SIDEWAYS or BACKWARDS too fast (relative to where
# the robot faces), or spinning fast (DRIBBLE_TURN_MAX), does. So the speed is capped so that neither the sideways
# nor the backwards part of the movement is faster than DRIBBLE_SPEED_MIN x the speed (forwards: no limit).
DRIBBLE_SPEED_MIN = 0.38           # TUNE on the robot: fastest sideways / backwards speed (x BASE_SPEED) keeping the ball
# dribbler on: each wheel's speed may change by at most this many BASE_SPEEDs per second (10 = standstill to full
# speed in 0.1 s) - braking / accelerating doesn't lose the ball, this only smooths out jerks
DRIBBLE_MAX_ACCEL = 10.0

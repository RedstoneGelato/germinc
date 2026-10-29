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

SWITCH_PIN = 25            # start/stop input: the referee's comms module (OUT1), or our own on/off switch (REF_MODULE)
REF_MODULE = True          # True = SWITCH_PIN is wired to the RCJ comms module OUT1: 3.3 V = PLAY, 0 V = STOP. Pulled
                           # down, so a loose wire / module off = stopped. False = on/off switch, GPIO to ground, closed = paused
ZERO_MAX_ERR = 0.17        # rad (10 deg) - while paused the heading is only re-zeroed if the robot already faces within
                           # this of forward (put down facing the goal again). See common.Robot.zero_heading.
GOAL_SWITCH_PIN = 26       # attack-goal switch, wired the same way (GPIO to ground)
GOAL_SWITCH_CLOSED = "yellow"   # goal we attack with the goal switch closed (pin to ground); open = the other goal.
                                # Swap to "blue" if the robots go for the wrong goal.
GOAL_FROM_CAMERA = False   # True = ignore the goal switch and work the goal out from the camera while paused

VISION_CONFIG = "robot_vision_config.json"   # made by test_camera.py -> "Generate config file"
VISION_MAX_FPS = 50        # TUNE: camera frame rate cap. None = no cap.
VISION_CV_THREADS = 1      # OpenCV worker threads (whole program)

CONTROL_PERIOD = 0.01      # main loop runs at 100 Hz

# ---- robot body (cm)
ROBOT_RADIUS = 11.0                         # TUNE: centre to bumper
# where the camera SEES the ball when it's in the dribbler, robot frame (x_min, x_max, y_min, y_max) - TUNE in
# test_localisation.py. The ball is reported as its nearest VISIBLE point (a held ball shows up ~3 cm past the
# bumper)
CAPTURE_ZONE = (-5.0, 5.0, ROBOT_RADIUS - 3.0, ROBOT_RADIUS + 4.0)
BALL_RADIUS = 2.1                          # cm - CHECK the ball you use
BALL_BLIND_RING = 2.0                      # cm: the camera can't see the ball while its near edge is within this of our
                                           # bumper, so it never sees one in the dribbler.
                                           # perception.py then relies on the IR ring, or (no IR) assumes a ball that
                                           # vanished out of the capture zone is still in the dribbler. 0 = it can see it

# ---- enemy robots: set before each match
ENEMY_DETECTION = True                    # False = the camera reports no other robots (obstacles). Turn off when their
                                           # robots can't be told apart from the carpet, e.g. dark robots on a grey carpet
                                           # or green robots on a green one

# ---- out of bounds: a robot is only out once NO part of it touches the white line
LINE_TOUCH_SAFETY = 6.0                    # cm of robot that must still be over the line at the furthest out we go
MAX_OUT = ROBOT_RADIUS - LINE_TOUCH_SAFETY # so the robot centre may be up to this far past the line's outer edge
PRECISE_STD = 6.0                          # pose spread (cm) below which the position is trusted for that; otherwise
                                           # the LDRs decide: seeing the line = escape

# ---- LDR line ring (PCB)
LDR_COUNT = 32
LDR_LINE_THRESHOLD = 1500                  # reading below this = white line (TUNE on each new carpet)
LDR_START_ANGLE = math.pi / 2              # sensor 0 is at the front, numbering goes anticlockwise
LED_BRIGHTNESS_START = 40000               # 0 - 65535

# ---- IR ball sensors (ring of TSSPs for the pulsed ball): bearing to the ball + a rough strength, also where the camera
# can't see it (the ~2 cm blind ring right round the bumper, the mirror mount bars' slices, robots in the way)
IR_COUNT = 12                              # CHECK: sensors evenly round the robot
IR_START_ANGLE = math.pi / 2               # sensor 0 at the front, numbering goes anticlockwise (like the LDRs)
IR_ON_PCB = False                          # True once the PCB sends the IR readings (hardware.PCBThread, command 0x02:
                                           # IR_COUNT x uint16, 0 = sensor not picking the ball up, IR_RAW_MAX = strongest).
                                           # False = no IR: the robot code keeps the last camera ball for a moment instead
IR_RAW_MAX = 1000                          # TUNE: raw reading of a sensor with the ball right next to it

# ---- motion
BASE_SPEED = 80000000
LINE_ESCAPE_SPEED = BASE_SPEED * 1.5
MOTOR_MAX = BASE_SPEED * 1.5               # no wheel is ever asked for more than this (moving + turning together)
# turning has its own speed (it doesn't slow down when the robot moves slowly): proportional to how far off the
# wanted heading the robot is, full speed (TURN_MAX x BASE_SPEED on every wheel) from TURN_FULL_ERR radians off
TURN_MAX = 0.6
TURN_FULL_ERR = 1.0
DRIBBLE_TURN_MAX = 0.4                     # ...at most this with the dribbler on
# dribbler on: the speed is capped so that neither the sideways nor the backwards part of the movement (relative to
# where the robot faces) is faster than DRIBBLE_SPEED_MIN x the speed (forwards: no limit).
DRIBBLE_SPEED_MIN = 0.38           # TUNE on the robot: fastest sideways / backwards speed (x BASE_SPEED) keeping the ball
# dribbler on: each wheel's speed may change by at most this many BASE_SPEEDs per second (10 = standstill to full
# speed in 0.1 s)
DRIBBLE_MAX_ACCEL = 10.0

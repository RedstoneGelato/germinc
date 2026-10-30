# Tuning cheat sheet

The settings you are most likely to change, grouped by when you change them. Line numbers are as of this version;
search for the name if they have moved. The full list is in [GUIDE.md](GUIDE.md) section 7.

> **Two strategy files.** `strategy.py` is the RCJA field, `strategy_int.py` the international one (`main_int.py`,
> `main_attack_int.py`, `sim_int.py`). They have their own copies of the strategy constants: change both if you want
> both fields to behave the same.

After changing behaviour, try it in the simulator before the robot: `python simulator.py` (window) or
`python simulator.py --headless 60` (summary). `python arena.py --games 20` plays many games to check a change helps.

## Before every match

| Setting | Where | Now | What it does |
| --- | --- | --- | --- |
| `ENEMY_DETECTION` | [robot_config.py:26](robot_config.py:26) | `True` | `False` = the camera reports no other robots. Turn off when theirs look like the carpet (dark robots on grey, green on green). |
| `GOAL_SWITCH_CLOSED` | [robot_config.py:12](robot_config.py:12) | `"yellow"` | Goal attacked with the goal switch closed. Swap to `"blue"` if the robots go for the wrong goal. |
| `GOAL_FROM_CAMERA` | [robot_config.py:13](robot_config.py:13) | `False` | `True` = ignore the switch and work out the goal from the camera while paused (robot must face the goal it attacks). |
| `REF_MODULE` | [robot_config.py:9](robot_config.py:9) | `True` | `True` = RCJ comms module on `SWITCH_PIN` (3.3 V = play). `False` = our own on/off switch (closed = paused). |

## At a new venue / new carpet

| Setting | Where | Now | What it does |
| --- | --- | --- | --- |
| Colours (HSV), robot centre, ignore box, distance calibration | `robot_vision_config.json`, made by `python3 test_camera.py` | per robot | Redo the HSV for all 5 colours under the venue lights, then **Generate config file**. One file per robot - never copy between robots. |
| `LDR_LINE_THRESHOLD` | [robot_config.py:33](robot_config.py:33) | `1500` | LDR reading below this = white line. Check with `test_pcb.py` on carpet and on the line. |
| `LED_BRIGHTNESS_START` | [robot_config.py:35](robot_config.py:35) | `40000` | Starting LED brightness (0-65535). The robot adjusts it itself while paused (`common.LedCalibrator`). |
| `CAPTURE_ZONE` | [robot_config.py:22](robot_config.py:22) | `(-5, 5, R-3, R+4)` cm | Where the camera sees a ball that is in the dribbler. Check the cyan box in `test_localisation.py`. |
| Field size | [field.py](field.py) (`PLAY_W`, `PLAY_L`, `GOAL_W`, `PENALTY_*`) / [field_int.py](field_int.py) | RCJA / international | Only if the field is not standard. |

## Per robot (hardware)

| Setting | Where | What it does |
| --- | --- | --- |
| `DRIVE_MOTORS` | [hardware_attack.py:7](hardware_attack.py:7), [hardware_defense.py:5](hardware_defense.py:5) | `Motor(i2c_address, elec_angle_offset, sincos_centre)` for motor1..4. Calibration belongs to that motor + driver. The STOP scripts read these too. |
| `DRIBBLER` | [hardware_attack.py:13](hardware_attack.py:13), [hardware_defense.py:11](hardware_defense.py:11) | Same format, `None` = no dribbler. **The goalie's values are placeholders copied from the striker - calibrate before use.** |
| `DRIBBLER_SPEED` | [hardware_attack.py:14](hardware_attack.py:14) | Speed while on. Flip the sign if it spins the ball away. |
| `KICKER_PIN`, `KICK_PULSE`, `KICK_COOLDOWN` | [hardware_attack.py:16](hardware_attack.py:16), [hardware_defense.py:14](hardware_defense.py:14) | Solenoid GPIO pin (`None` = no kicker), seconds powered (change carefully, test with `test_solenoid.py`), seconds between kicks. |
| `ROBOT_ID` | [hardware_attack.py:5](hardware_attack.py:5), [hardware_defense.py:3](hardware_defense.py:3) | Comms id: must differ between the two robots. |
| `ROBOT_RADIUS` | [robot_config.py:21](robot_config.py:21) | Centre to bumper (cm). Used almost everywhere. |
| `MOTOR_CURRENT_LIMIT`, `MOTOR_SPEED_LIMIT`, PIDs | [hardware.py:184](hardware.py:184), `MotorThread._make_driver` | Motor driver limits and PID constants, both robots. |

## How it drives

| Setting | Where | Now | What it does |
| --- | --- | --- | --- |
| `BASE_SPEED` | [robot_config.py:42](robot_config.py:42) | `80000000` | Top driving speed. `LINE_ESCAPE_SPEED` and `MOTOR_MAX` (lines 43-44) are multiples of it. |
| `TURN_MAX`, `TURN_FULL_ERR` | [robot_config.py:45](robot_config.py:45) | `0.6`, `1.0` rad | Turning speed (x `BASE_SPEED`) and how far off the heading it turns at full speed. |
| `DRIBBLE_SPEED_MIN` | [robot_config.py:48](robot_config.py:48) | `0.38` | Fastest sideways / backwards speed that keeps the ball. Tune on the robot. |
| `DRIBBLE_TURN_MAX`, `DRIBBLE_MAX_ACCEL` | [robot_config.py:47](robot_config.py:47) | `0.4`, `10.0` | Gentler turning / speeding up while holding the ball. |
| `BRAKE_SMOOTH`, `BRAKE_LEAD` | [motion.py:7](motion.py:7) | `0.5`, `12` cm | How quickly it changes direction, and how far it still travels before slowing takes effect. |
| `PARK_ENTER` / `PARK_RESUME` | [motion.py:10](motion.py:10) | `2` / `5` cm | Stops the wheels near the target to avoid jitter. |
| Speed vs distance curve | [motion.py:13](motion.py:13) `SPEED_CURVE_*` | | How much it slows as it gets close to a target. |

## Lines and out of bounds

| Setting | Where | Now | What it does |
| --- | --- | --- | --- |
| `LINE_TOUCH_SAFETY` | [robot_config.py:28](robot_config.py:28) | `6` cm | How much of the robot must still be over the line at the furthest out it goes (sets `MAX_OUT`). |
| `PRECISE_STD` | [robot_config.py:30](robot_config.py:30) | `6` cm | Below this position spread the robot trusts its position for out-of-bounds; above it the LDRs decide. |
| `SLOW_START`, `SLOW_STOP` | [lines.py:16](lines.py:16) | `20`, `2` cm | Starts slowing this far from the edge, stops this close. |
| `LDR_VECTOR_IS_ESCAPE` | [lines.py:10](lines.py:10) | `True` | Flip if the robot drives **into** the line instead of away from it. |
| `SPEED_MULTI` | [lines.py:21](lines.py:21) | `{0: 1, 1: 0.6, ...}` | Slows down after several line touches within 3 s. |

## Strategy (`strategy.py`, and the copy in `strategy_int.py`)

| Setting | Where | Now | What it does |
| --- | --- | --- | --- |
| `GUARD_RADIUS` | [strategy.py:31](strategy.py:31) | `20` cm | How far in front of its goal the goalie guards. |
| `DANGER_DIST` | [strategy.py:40](strategy.py:40) | `35` cm | Ball this close inside our box: goalie goes for it. |
| `GOALIE_MAX_OUT` | [strategy.py:73](strategy.py:73) | `12` cm | How far past its box the goalie may chase while the striker plays. |
| `GOALIE_ALONE_ATTACKS` | [strategy.py:71](strategy.py:71) | `True` | Goalie plays striker when its teammate is off. |
| `BEHIND_DIST`, `ORBIT_DIST` | [strategy.py:13](strategy.py:13) | `R+12`, `R+14` cm | How far behind the ball the striker lines up, and how wide it goes round it. |
| `CARRY_KICK_RANGE`, `CARRY_KICK_HOLD` | [strategy.py:810](strategy.py:810) | `70` cm, `0.2` s | Striker kicks when this close and facing the goal for this long. |
| `QUICK_SHOT_RANGE`, `OPEN_SHOT_RANGE` | [strategy.py:813](strategy.py:813) | `130`, `120` cm | Longest shots it takes right after winning the ball / when charged. |
| `FEINT`, `RESTART_RUSH` | [strategy.py:833](strategy.py:833) | `True`, `False` | Turn whole behaviours on or off. Other on/off switches sit next to them. |
| Not-localised behaviour | [strategy_fallback.py](strategy_fallback.py) | | What both robots do while they don't know where they are. |

## Camera and detection

| Setting | Where | Now | What it does |
| --- | --- | --- | --- |
| `VISION_MAX_FPS` | [robot_config.py:16](robot_config.py:16) | `50` | Camera frame rate cap (`None` = no cap). |
| `VISION_CV_THREADS` | [robot_config.py:17](robot_config.py:17) | `1` | OpenCV threads. Try `2` on the Pi if the camera fps is low. |
| `BALL_MIN_AREA`, `GOAL_MIN_AREA`, `FIELD_MIN_AREA` | [vision.py:390](vision.py:390) | `2`, `15`, `600` cm² | Smallest blobs that count as ball / goal / "can see the field". |
| `LINE_MAX_RANGE` | [vision.py:397](vision.py:397) | `120` cm | White line points further than this are ignored. |
| `OBSTACLE_MIN_SIZE`, `OBSTACLE_MAX_RANGE` | [vision.py:399](vision.py:399) | `12`, `82` cm | Smallest / furthest blob that counts as a robot. |
| `WHITE_BODY_KERNEL` | [vision.py:412](vision.py:412) | `12` cm | White blobs thicker than this are white robots, not lines. `0` = off. |
| `BALL_LOST_TIME` | [perception.py:12](perception.py:12) | `0.3` s | How long the ball is remembered after losing sight of it. |
| `CAMERA_BLIND` | [common.py:15](common.py:15) | `0.3` s | No new frame for this long = stand still until frames come back. |

## Wiring and comms (rarely)

| Setting | Where | Now |
| --- | --- | --- |
| `SWITCH_PIN`, `GOAL_SWITCH_PIN` | [robot_config.py:8](robot_config.py:8) | `25`, `26` |
| `TEAM_ID`, `COMMS_PORT` | [robot_config.py:5](robot_config.py:5) | `"GERM_INC"`, `5555` |
| `IR_ON_PCB`, `IR_RAW_MAX` | [robot_config.py:39](robot_config.py:39) | `False`, `1000` (set `IR_ON_PCB = True` once the PCB sends IR readings) |

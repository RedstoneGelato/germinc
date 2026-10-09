# How the robot code works

## 1. The big picture

Every robot runs one main file: `main.py` (goalie) or `main_attack.py` (striker). Each starts several threads that run
side by side, then loops 100 times a second:

```
 camera ──► vision.py ──► detection.py ──► Detections (ball, goals, line points, obstacles, in cm)
                │                               │
 IMU ───────────┴─ compass                      ├──► localisation.py ──► Pose (x, y on the field)
                                                │                            │
                                                ▼                            ▼
                                          perception.py  (World: smoothed ball, goals, obstacles)
 LDR ring (PCB) ────────────────────────► lines.py      (LineState: on line? which way out?)
 other robot ───► comms.py                      │
                                                ▼
                                          strategy.py   (decides: where to go, which way to face)
                                                │
                                                ▼
                                          motion.py ──► hardware.py ──► motors
```

| Thread | File | Runs | Produces |
| --- | --- | --- | --- |
| IMU | `hardware.py` `IMUThread` | 100 Hz | `imu.compass()` heading |
| PCB | `hardware.py` `PCBThread` | ~25 Hz | `pcb.snapshot()` 32 LDR readings |
| Motors | `hardware.py` `MotorThread` | 200 Hz | sends the 4 speeds to the drivers |
| Camera | `vision.py` `VisionThread` | as fast as the camera | `vision.latest()` Detections |
| Localisation | `localisation.py` `LocalisationThread` | every camera frame | `loc.get()` Pose |
| Comms | `comms.py` `TeammateLinkThread` | 20 Hz send | `comms.teammate()` other robot's message |

The main loop itself only reads the latest values, decides, and sets motor speeds. Nothing in it waits on hardware.
One pass of it is `GoalieBrain.tick()` in `main.py` / `StrikerBrain.tick()` in `main_attack.py`, and
`simulator.py` runs exactly those (section 10), so behaviour changes can be tried on a laptop first.

## 2. Coordinates (read this before editing anything)

Everything is in **cm**. There are three ways of describing a position:

| Name | Origin | Axes | Used by |
| --- | --- | --- | --- |
| **robot frame** | robot centre | +x = robot's right, +y = robot's front | raw detections (`det.ball`, `det.goals`) |
| **relative** | robot centre | field axes: +y = towards the goal we attack | `world.ball`, `world.attack_goal`, `desired_pos` in strategy |
| **field frame** | field centre | +y = goal we attack, +x = right when facing it | `pose.x, pose.y`, `world.ball_field`, `field.py` |

- `compass` = robot heading in radians, counter-clockwise positive, **0 = facing the goal we attack**.
  It's zeroed while the robot is paused (`imu.zero()`), so **always pause the robot facing the goal you attack**.
- robot frame → relative: `utils.rotate(v, compass)`. relative → field: add `pose.x, pose.y`.
- The strategy works in **relative** coordinates: `desired_pos = [0, 50]` means "go 50 cm towards the attack goal",
  whichever way the robot is facing.

## 3. What happens in one camera frame

1. **`vision.py`** grabs a frame and reads the compass at the same moment.
2. **`robot_vision.py`** (`RobotVision.process`) uses the calibration from `robot_vision_config.json`:
   fisheye + top-down remap → rotate for the camera mounting → 5 HSV colour masks → blanks out the ignore box (our own body).
3. **`detection.py`** turns the masks into objects in robot-frame cm:
   - **green** → convex hull = "the field". Everything outside it is ignored (people, walls, the crowd).
   - **orange** → biggest blob = ball.
   - **yellow / blue** → goals (all blobs of that colour together).
   - **white** → line points (up to 250 pixels, converted to cm).
   - **inside the field but none of the 5 colours** → obstacles (other robots).
   - Positions use the blob point **nearest the robot**. The top-down view assumes everything is flat, so tall things
     get stretched outwards. Their nearest point is where they touch the floor.
4. **`localisation.py`** rotates the line points and goals into field axes and updates the particle filter (section 4).
5. In the main loop, **`perception.py`** (`World.update`) smooths the ball and works out which goal is ours.
   If a goal is out of view, it fills it in from the pose. It also keeps obstacles only if they show up in 3+ frames in a row.

## 4. How localisation works

The IMU already gives the heading, so only x and y are unknown.

- There are 300 **particles**, each a guess of (x, y). At start-up / after a pause they're spread over the whole field.
- **Every frame:**
  1. **Predict:** every particle moves by the estimated velocity plus some random noise (the robot might have moved).
  2. **Lines:** for each particle, the white points seen are placed on the field as if the robot were there.
     `field.FieldMap` has a precomputed "distance to the nearest drawn line" for every cm of the field.
     Particles where the points land on the lines score well; points landing on green score badly.
  3. **Goals:** each particle predicts where the goal's near end should appear; the closer to what was seen, the better.
  4. **Resample:** good particles get copied, bad ones disappear. A few random particles are always re-added
     (2%), so it can recover if it was wrong. When very lost it also re-adds some around the position
     worked out from the goals alone (`position_from_goals`).
- The answer is the weighted average (`pose.x, pose.y`) and the spread (`pose.std`).
  `pose.confident` = spread under 15 cm. Code that relies on the pose checks `confident` first.

## 5. Line avoidance (`lines.py`)

| Source | Sees | Role |
| --- | --- | --- |
| LDR ring | right under the robot | the only thing that says "on the line **now**" |
| camera | lines around the robot, before reaching them | second escape direction |
| pose (if confident) | distance to the boundary | slows outward movement near the edge (`limit_outward`), third escape direction, out-of-bounds check |

When confident and well inside the field (15+ cm from the boundary), LDR hits are treated as **penalty-area lines and ignored**.
The escape direction is a blend of whichever sources are available.

## 6. "I want to change …" → where to edit

| I want to … | File | What |
| --- | --- | --- |
| **Camera / colours** | | |
| retune colours, exposure, white balance | `test_camera.py` web page | then **Generate config file** |
| change the ignore box (robot body) | `test_camera.py` web page, Heading panel sliders | cm from the robot centre |
| change where "robot centre" is | `test_camera.py`, "Set robot centre" | |
| change minimum ball / goal / obstacle size, line point count | `detection.py` top | areas are in cm² |
| **Field / localisation** | | |
| field dimensions | `field.py` top | **measure the real field** |
| the line inside the goals (gap to the goal walls) | `field.py`: `GOAL_LINE_INSET`, `boundary_polyline()` | |
| add / remove a field line | `field.py` `white_line_segments()` | |
| how much to trust lines vs goals | `localisation.py` top: `LINE_SIGMA`, `LINE_WEIGHT`, `GOAL_SIGMA` | smaller sigma = trusted more |
| it reacts too slowly / jumps around | `localisation.py`: `PREDICT_NOISE_SPEED` | higher = follows fast moves, noisier |
| when the pose counts as confident | `localisation.py`: `CONFIDENT_STD` | |
| **Lines** | | |
| LDR threshold, sensor angles | `robot_config.py`: `LDR_*` | |
| robot drives INTO the line | `lines.py`: `LDR_VECTOR_IS_ESCAPE = False` | |
| slow down earlier near the edge | `lines.py`: `SLOW_START`, `SLOW_STOP` | |
| stop ignoring penalty lines | `lines.py`: `IGNORE_INNER_LINES = False` | |
| **Ball** | | |
| where the ball sits in the dribbler | `robot_config.py`: `CAPTURE_ZONE`, `ROBOT_RADIUS` | check in `test_localisation.py` (cyan box) |
| how long the ball is remembered after losing it | `perception.py`: `BALL_LOST_TIME` | |
| **Behaviour** | | |
| goalie decisions / distances | `strategy.py` `GoalieStrategy` + constants above it | all marked TUNE |
| striker decisions / distances | `strategy.py` `StrikerStrategy` + `S_*` constants above it | |
| what the robots tell each other | `main.py` / `main_attack.py` `comms.my_state.update(...)` in `tick()` | |
| the order things happen in each loop | `main.py` `GoalieBrain.tick()` / `main_attack.py` `StrikerBrain.tick()` | the simulator runs these too |
| **Movement** | | |
| top speed, spin, line escape speed | `robot_config.py`: `BASE_SPEED`, `BASE_SPIN`, `LINE_ESCAPE_SPEED` | shared by both |
| speed vs distance curve | `motion.py` `Mover.step` | |
| a speed profile for one strategy only | that strategy's `self.speed` dict | `base_speed`, `spd_min`, `spd_multi` |
| **Hardware (per robot)** | | |
| motor I2C address / calibration (attack) | `hardware_attack.py` `DRIVE_MOTORS` **and** `STOP_attack.py` | |
| motor I2C address / calibration (goalie) | `hardware_defense.py` `DRIVE_MOTORS` **and** `STOP_defense.py` | |
| PID / current limit (both) | `hardware.py` `MotorThread` | |
| robot id for comms | `hardware_attack.py` / `hardware_defense.py`: `ROBOT_ID` | |

## 7. The two robots

| | Goalie | Striker |
| --- | --- | --- |
| run | `python3 main.py` | `python3 main_attack.py` |
| hardware file | `hardware_defense.py` | `hardware_attack.py` |
| drive motors (motor1..4) | 26, 32, 28, 25 | 26, 25, 27, 28 |
| dribbler (stop script only) | 27 (**check**) | 32 |
| stop script | `STOP_defense.py` | `STOP_attack.py` |
| strategy | `GoalieStrategy` | `StrikerStrategy` |

Everything else is shared. **`robot_vision_config.json` is per robot** (each camera has its own lens, mounting and centre),
so generate it with `test_camera.py` on each Pi and don't copy it between robots. The systemd service on each Pi
must point at the right main file and stop script.

## 8. Setup / tuning order (each robot)

1. `python3 test_camera.py` → lens calibration → top-down → robot centre → ignore box → HSV for all 5 colours → **Generate config file**.
2. `python3 test_localisation.py --pcb` (motors stay off):
   - camera view: ball circled, goals crossed, lines magenta, field hull green, no red obstacles on empty field;
   - ball in the dribbler sits inside the cyan capture box;
   - face the attack goal, **Zero heading**, turn the robot by hand: the magenta dots stay on the field lines;
   - carry it around: the green robot follows, ± stays small;
   - push it onto a line: the blue escape arrow points back into the field.
3. `field.py`: fix dimensions if the lines don't match up.
4. Try strategy changes in `simulator.py` first, then run the main file on the field and tune `strategy.py` distances.

## 9. Adding the ultrasonic ring later

1. `hardware.py`: add an `UltrasonicThread` with `distances` (8 values in cm, `None` = no echo)
   and `angles` (each sensor's direction in the robot frame, radians, 0 = robot's right, counter-clockwise).
2. `common.py` `Robot.__init__`: start it.
3. Main loop, each tick: `bot.loc.walls = (us.distances, [a + compass for a in us.angles])`.
4. `localisation.py` `correct_walls()` already does the rest; tune `WALL_SIGMA` / `WALL_OUTLIER`.
   It's robust to readings that hit a robot instead of the wall.

## 10. Simulator (`simulator.py`)

```bash
python simulator.py                  # window: field, both robots, two opponents
python simulator.py --headless 30    # no window: 30 simulated seconds, prints score / outs / localisation error
```

It runs the real `GoalieBrain` / `StrikerBrain`, so everything from camera detection through to motor speeds
is the robot's own code. Only the hardware is fake:

| Real | Simulated |
| --- | --- |
| camera + lens + top-down calibration | a perfect top-down picture of the sim field from the robot's position, 1 px/cm, `VIEW_RADIUS` cm around the robot, then the real `robot_vision.py` masks and `detection.py` |
| LDR ring | white line under a sensor = 800, green = 3200 |
| IMU | the true heading (+ slow drift with noise on) |
| comms | each robot's `my_state` handed straight to the other |
| switch | `p` key |
| motors | the 4 motor speeds turned back into movement (ideal omni wheels, `TOP_SPEED`, `TOP_SPIN`) |

Window: left-drag moves the ball / robots / opponents, right-drag turns them. On the field, × = where the robot
*thinks* it is (circle = spread), the arrow = where the strategy wants to go, a red ring = escaping a line,
grey dots = the selected robot's particles. The right panel is the selected robot's camera, with the same overlay
as `test_localisation.py` (`c` switches to the raw colour masks). The keys are listed on screen and at the top of
`simulator.py`. `r` picks up the selected robot and drops it somewhere random, to test relocalisation.

Settings to make it match the real robots (top of `simulator.py`): `TOP_SPEED` / `TOP_SPIN` (measure them),
`CAMERA_EVERY` (camera fps), `VIEW_RADIUS` (how far the real top-down view reaches), `BALL_RADIUS`,
`LDR_PHYSICAL_OFFSET` (where LDR 0 physically is, see the comment there).

Not simulated, so these still need the real robot: lens / top-down calibration errors, tall objects stretching,
robots hiding things behind them, real colours and lighting, motor wiring or sign mistakes, the dribbler.


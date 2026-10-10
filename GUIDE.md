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
                                          strategy.py   (decides: where to go, which way to face,
                                                         dribbler on/off, kick)
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

## 5. Strategy (`strategy.py`)

Both robots plan in **field coordinates** using the localisation. While a robot doesn't trust its position
(`pose.confident` false: just put down, or lost), it automatically falls back to the comp-style relative logic in
`strategy_fallback.py`. In the simulator / printout those states show as 10 + the fallback botstate.

| Goalie botstate | Striker botstate |
| --- | --- |
| 0 no ball known: guard the middle of the goal | 0 no ball known: if it went out, wait behind the neutral spot it comes back on (restart play); else look where it was last seen (`SEARCH_TIME`, 1 s), then wait just behind the centre spot (`WAIT_POS`) |
| 1 ball in dribbler: stay put, turn up-field (max `GOALIE_KICK_TURN`), kick it clear | 1 ball in dribbler: carry to the best aim point, kick when lined up (sidestep if about to be tackled) |
| 2 clearing: go for the ball | 2 ball known: get behind it (orbit, then line up, then push) |
| 3 guarding: on an arc between ball and goal | 3 no ball, goalie off: stand in front of our goal |
| 4 ball out: guard as if it were on the neutral spot it'll come back to | 4 goalie is clearing: wait up-field for a pass |
| | 5 ball out: wait just behind the neutral spot nearest the ball, facing the attack goal |

The parts it's built from (all in `strategy.py`):
- `approach()`: get behind the ball relative to an aim point. If the robot is on the wrong side, it goes round
  (`orbit`, on whichever side stays in the field), then lines up (`line up`), then drives through (`push`).
- `best_aim()`: tries `AIM_POINTS` points across the goal and picks the one with the most room between the
  ball's path and the obstacles.
- `Avoider`: if an obstacle is on the straight path, aim beside it instead. Once it picks a side it keeps it
  until it's past (otherwise it re-decides every loop and shuffles left-right in front of the obstacle).
- `ShotPlanner` (striker with the ball): scores a grid of spots it could shoot from. Each spot needs a clear line
  to part of the goal and to be within `KICK_RANGE`; it prefers short trips that don't cross obstacles. The striker
  strafes there with the ball while always facing the aim point (robot, ball and aim in a line), then kicks.
  While carrying it never drives backwards: spots and dodge waypoints behind it cost extra (`CARRY_BACK_COST`),
  and if the target is still behind it, it moves sideways instead (`CARRY_BACK_ALLOW`). It keeps its spot unless it
  becomes unusable or another is clearly better (`SHOT_SWITCH`, `SHOT_HOLD`), and it still shoots early if a clear
  shot opens up on the way.
- `choose_push_aim()`: if getting behind the ball for a goal shot would mean standing past the line (ball hugging
  the line), push it up-field or inwards instead.
- Obstacles are stored as their nearest point; `obstacle_centres()` estimates their centres (one robot radius
  further away). `perception.py` remembers obstacles for `OBSTACLE_MEMORY` s after they drop out of view, so
  plans don't flip-flop when an obstacle flickers at the edge of the camera.
- A ball an opponent is touching (`CONTEST_DIST`) is pushed at full speed, and the striker won't hand it to the
  goalie. A free ball is approached normally, so the dribbler can catch it.
- **Who goes for the ball:** the striker compares `ball_cost()` for both robots (distance, plus a penalty for being
  on the wrong side of the ball). If the ball is deep in our half and the goalie is clearly better placed, it sends
  `command = 1`. The goalie also clears by itself when the ball is in its penalty area and close.
- **Kick (striker):** only when the ball is in the dribbler and the robot's *actual heading* would put the ball
  between the posts: the heading line must cross the goal line at least `BALL_RADIUS + POST_MARGIN` inside a post,
  within `KICK_RANGE`, with nothing in the way (`KICK_CLEARANCE`) - `GoalieStrategy.can_shoot()`. (Facing the aim
  point within an angle isn't enough: from close and at a steep angle a few degrees off hits the post.) Aim points
  are on the goal line, `BALL_RADIUS + AIM_MARGIN` inside the posts. The goalie clears with the simpler angle check
  (`KICK_ANGLE`), from anywhere.
- **Moving ball:** both robots go for where the ball will be (`lead_ball()`, from `world.ball_vel`, up to
  `LEAD_TIME_MAX` ahead), so they get behind a rolling ball instead of chasing it.
- **Catching:** running onto a free ball slows down near it (`CATCH_MULTI_MIN`, `CATCH_MULTI_PER_CM`), so the
  dribbler catches it instead of knocking it away - unless an opponent could get there first (`RACE_MARGIN`), then
  it's full speed. A robot without a dribbler (`DRIBBLER = None`) never "carries": a ball at its front is just pushed
  (botstate 2), and it still stops at `CLEAR_RETURN_Y`.
- **Catch from the front (striker):** if it isn't already lined up behind the ball, the striker drives straight
  onto it facing it, catches it with the dribbler and turns with it (botstate 1), instead of going round it
  (`CATCH_FRONT_*`; not within `CATCH_FRONT_Y` of our goal line, where a missed catch could knock it in) - but only
  when that's quicker than orbiting (`catch_is_faster()`: turning to the ball while driving to it, then turning back
  with it, vs. the way round; `EST_*` are rough robot speeds - tune on the robot). With the ball behind it or off to
  the side, it orbits.
- **Opponent in the way:** an opponent between the striker and the ball (`blocker_between()`) isn't driven into (that
  shoves it onto the ball, maybe into our goal): the striker goes round the opponent and the ball together
  (`around_group()`, `BLOCKER_*`), then gets on with it.
- **Shoot from close:** planned shots only from within `KICK_RANGE` (35 cm) along the heading - against a goalie
  that guards well, longer shots were mostly saved. Under pressure up to `PRESSURE_RANGE`.
- **Under pressure:** with an opponent within `PRESSURE_DIST` the striker shoots as soon as its heading crosses
  the goal mouth and the ball's path misses every robot (no `KICK_CLEARANCE` margin) - better than losing the ball.
- **Defending (from the simulator: nearly every goal against came while the goalie was off its line):**
  - the goalie **never goes round the ball**. If it isn't already between ball and goal when clearing, it stays
    on its guard arc until it is (`GoalieStrategy.guard()`).
  - near our goal (`CLEAR_AWAY_Y`) the goalie pushes the ball **away from the goal centre**, half way to straight
    up-field (`clear_aim()`), so "behind the ball" is the goal side and the ball actually leaves our end.
  - the striker does the same when the ball is in our half (`STRIKER_CLEAR_Y`) and it's goal-side of the ball:
    clear it from there instead of going round it (which leaves the opponent a free push at our goal).
  - the striker only waits for the goalie's clearance (botstate 4) while no opponent is on the ball.
  - the goalie doesn't challenge a ball an opponent has (`OPP_ON_BALL`): it guards. It only goes for free balls.
  - **block:** a free ball rolling towards our goal line that will cross between the posts (`BLOCK_*`): the
    goalie goes to where it will cross the goalie's line, whatever it was doing (`block_point()`).
  - guarding moves use a proportional speed (`GUARD_GAIN`) and less smoothing (`GUARD_SMOOTH`, passed to
    `motion.Mover.step` as `smooth`): the default speed curve crawls over the short moves a goalie makes, and the
    default smoothing lags ~0.1 s, which is the difference between saving a close shot or not.
  - a ball already behind the goalie's line (on our goal line): the goalie may go round it to push it out sideways.
  - the goalie guards **facing straight up-field** (no turning: in `VelocityToMotor` turning and moving share the
    motors, so turning towards the ball made its sideways moves crawl) and against where the ball will be
    `GUARD_LEAD` s ahead.
  - **ball carrier:** the opponent whose centre is right behind the ball has it (`carrier()`, using both robots'
    cameras). The ball sits in front of that robot, so carrier -> ball is where a kick would go: if that line ends in
    our goal, the goalie moves its guard spot `CARRIER_WEIGHT` of the way towards it.
  - near our goal, a ball to the side (`CLEAR_OUT_X`) is pushed **out over the side line**: an out ball comes back
    on the halfway line, far from our goal.
  - **goalie with the ball** (it has the striker's dribbler + kicker): stays put, turns up-field and kicks it clear.
- **Restart play:** when the ball goes out (or was last seen at the line rolling out), the striker waits just
  behind the neutral spot it'll come back on, facing their goal (`RESTART_WAIT`, `NEUTRAL_WAIT`).
- **On its own** (the other robot is off): the goalie only goes for the ball in our half or when it's clearly the
  closest (`SOLO_CHASE_Y`, `SOLO_CLOSER`); the striker guards our goal like a goalie when an opponent will get to a
  ball in our half first (`solo_defend()`).
- **Dodge:** striker with the ball, no shot, an opponent within `DODGE_DIST` that is coming at it (`DODGE_CLOSING`,
  from the opponent velocities perception tracks) or already touching (`DODGE_NEAR`): sidestep (`DODGE_STEP`) away
  from it, still facing the aim, up-field side if possible. An opponent standing still further away is just gone
  round (dodging it made the striker shuffle left and right in front of it).
- `around_ball()`: when the robot isn't meant to touch the ball (guarding, going round it, lining up) and the
  straight path would hit it, go via a point beside it.
- `approach()` goes round (orbit) instead of straight to the line-up point when the straight path would clip the
  ball - clipping it was knocking it the wrong way, sometimes into our own goal.

**Opponents are whole robots:** `detection.py` estimates each opponent's centre from the whole blob (the middle of
its angular extent, one robot radius beyond its nearest point), not just the part facing us, and the strategy
treats every opponent as a robot-sized circle around that centre. The camera views draw them as circles.

**Opponent sharing:** each robot also sends the opponent robots its camera sees (field-frame centres) as `"opps"`.
`perception.py` merges them into `world.opponents` (ours + the teammate's that we don't see, without ourselves and
without anything fully outside the field); the goalie uses it to find the ball carrier.

**Ball sharing:** each robot sends the ball position **its own camera** sees (field frame) as `"ball"`, plus `"pos"`
and `"chasing"`. `perception.py` uses the teammate's ball when ours isn't visible (`world.ball_source` = `"mate"`).
That needs our own position to be confident, to turn the teammate's field position into "relative". The teammate is
also removed from our obstacle list using its `"pos"`.

## 6. Line avoidance (`lines.py`)

| Source | Sees | Role |
| --- | --- | --- |
| LDR ring | right under the robot | the only thing that says "on the line **now**" |
| camera | lines around the robot, before reaching them | second escape direction |
| pose (if confident) | distance to the boundary | slows outward movement near the edge (`limit_outward`), third escape direction, out-of-bounds check |

**Out of bounds = no part of the robot touching the white line.** So the robot may go partly past the line: when
the pose is precise (spread below `robot_config.PRECISE_STD`), the robot centre may go up to `robot_config.MAX_OUT`
past the line's outer edge (= `ROBOT_RADIUS - LINE_TOUCH_SAFETY`, keeping `LINE_TOUCH_SAFETY` cm of robot over the
line). Then the **pose decides** when to escape, and LDRs seeing the line on their own are fine. Without a precise
pose the old safe rule is used: any LDR on the line = escape. `strategy.py` uses the same limit (`OUT_MARGIN`) for
getting behind a ball near the line and going round obstacles. To be more careful, raise `LINE_TOUCH_SAFETY`.

Only the boundary is white. The penalty box line is **black**, so neither the LDRs nor the camera's white mask
react to it, and `detection.py`'s obstacle filter (`OBSTACLE_KERNEL`) removes anything as thin as that line.
`IGNORE_INNER_LINES` (off) would ignore LDR hits far from the boundary, which only makes sense if white lines are
ever added inside the field.
The escape direction is a blend of whichever sources are available.

## 7. "I want to change …" → where to edit

| I want to … | File | What |
| --- | --- | --- |
| **Camera / colours** | | |
| retune colours, exposure, white balance | `test_camera.py` web page | then **Generate config file** |
| change the ignore box (robot body) | `test_camera.py` web page, Heading panel sliders | cm from the robot centre |
| change where "robot centre" is | `test_camera.py`, "Set robot centre" | |
| change minimum ball / goal / obstacle size, line point count | `detection.py` top | areas are in cm² |
| obstacles where there's no robot (ghosts) | `detection.py`: `OBSTACLE_MIN_SIZE`, `OBSTACLE_MAX_RANGE`, `OBSTACLE_LINE_MARGIN(_FAR)`, `OBSTACLE_KERNEL`; `perception.py`: `OBSTACLE_MEMORY` | far away the lens blurs lines (white and the black penalty line) into robot-sized blobs; `test_localisation.py` shows each obstacle as a red circle |
| **Field / localisation** | | |
| field dimensions | `field.py` top | **measure the real field** |
| the line inside the goals (gap to the goal walls) | `field.py`: `GOAL_LINE_INSET`, `boundary_polyline()` | |
| add / remove a field line | `field.py` `white_line_segments()` (white, used by localisation) / `black_line_segments()` | |
| how much to trust lines vs goals | `localisation.py` top: `LINE_SIGMA`, `LINE_WEIGHT`, `GOAL_SIGMA` | smaller sigma = trusted more |
| it reacts too slowly / jumps around | `localisation.py`: `PREDICT_NOISE_SPEED` | higher = follows fast moves, noisier |
| when the pose counts as confident | `localisation.py`: `CONFIDENT_STD` | |
| **Lines** | | |
| LDR threshold, sensor angles | `robot_config.py`: `LDR_*` | |
| robot drives INTO the line | `lines.py`: `LDR_VECTOR_IS_ESCAPE = False` | |
| slow down earlier near the edge | `lines.py`: `SLOW_START`, `SLOW_STOP` | |
| how far past the line the robot may go | `robot_config.py`: `LINE_TOUCH_SAFETY` (→ `MAX_OUT`), `PRECISE_STD` | |
| black penalty line thickness | `field.py`: `PENALTY_LINE_W` (also sets the obstacle filter) | |
| **Ball** | | |
| where the ball sits in the dribbler | `robot_config.py`: `CAPTURE_ZONE`, `ROBOT_RADIUS` | check in `test_localisation.py` (cyan box) |
| how long the ball is remembered after losing it | `perception.py`: `BALL_LOST_TIME` | |
| **Behaviour** | | |
| goalie / striker behaviour (localised) | `strategy.py` `GoalieStrategy` / `StrikerStrategy` + constants at the top | all marked TUNE |
| how far behind the ball to line up, orbit distance | `strategy.py`: `BEHIND_DIST`, `ORBIT_DIST`, `ALIGN_TOL` | |
| when to kick | `strategy.py`: `KICK_RANGE`, `POST_MARGIN`, `KICK_CLEARANCE` (`KICK_ANGLE`: goalie clears) | |
| ball size | `robot_config.py`: `BALL_RADIUS` | used by aims, the kick check and the simulator |
| going for a moving ball, catching it gently | `strategy.py`: `LEAD_*`, `CATCH_MULTI_*`; `perception.py`: `BALL_VEL_*` | |
| what counts as "ball out", where to wait for the restart | `strategy.py`: `BALL_OUT_MARGIN`, `NEUTRAL_WAIT`; spots in `field.NEUTRAL_SPOTS` | |
| where the striker goes to shoot from | `strategy.py`: `SHOT_*`, `CARRY_MULTI` | |
| when the goalie leaves its goal | `strategy.py`: `DANGER_DIST`, `HANDOVER_Y`, `HANDOVER_MARGIN`, `CLEAR_RETURN_Y` | |
| which way the ball is cleared near our goal | `strategy.py`: `clear_aim()`, `CLEAR_AWAY_Y`, `STRIKER_CLEAR_Y` | |
| goalie: where it guards, how fast it reacts, blocking rolling balls | `strategy.py`: `GUARD_RADIUS`, `GUARD_GAIN`, `GUARD_SMOOTH`, `GUARD_LEAD`, `BLOCK_*`, `OPP_ON_BALL`, `CARRIER_*` | |
| when the striker shoots under pressure | `strategy.py`: `PRESSURE_DIST`, `PRESSURE_RANGE` | |
| when the striker catches the ball from the front instead of going round it | `strategy.py`: `CATCH_FRONT_*` | |
| behaviour while NOT localised | `strategy_fallback.py` (the comp logic) | |
| dribbler motor / speed, kicker pin / pulse / cooldown | `hardware_attack.py` / `hardware_defense.py`: `DRIBBLER`, `DRIBBLER_SPEED`, `KICKER_PIN`, `KICK_*` | `None` = robot doesn't have one |
| what the robots tell each other | `main.py` / `main_attack.py` `comms.my_state.update(...)` in `tick()` | |
| the order things happen in each loop | `main.py` `GoalieBrain.tick()` / `main_attack.py` `StrikerBrain.tick()` | the simulator runs these too |
| **Movement** | | |
| top speed, line escape speed, wheel limit | `robot_config.py`: `BASE_SPEED`, `LINE_ESCAPE_SPEED`, `MOTOR_MAX` | shared by both |
| how fast it turns | `robot_config.py`: `TURN_MAX` (x `BASE_SPEED` on every wheel), `TURN_FULL_ERR` (heading error for full turn speed), `DRIBBLE_TURN_MAX` | turning speed is proportional to the heading error and doesn't depend on how fast it moves |
| how quickly it brakes / changes direction | `motion.py`: `BRAKE_SMOOTH` (vs the strategy's `smooth`) | the velocity is smoothed as one vector, faster the more the direction changes |
| speed vs distance curve | `motion.py` `Mover.step` | |
| how much slower it drives sideways / backwards with the dribbler on | `robot_config.py`: `DRIBBLE_SPEED_MIN` | `motion.dribble_speed_multi()` |
| how gently it speeds up / turns with the dribbler on | `robot_config.py`: `DRIBBLE_MAX_ACCEL` | `Mover.step` limits wheel speed changes |
| a speed profile for one strategy only | that strategy's `self.speed` dict | `base_speed`, `spd_min`, `spd_multi` |
| **Hardware (per robot)** | | |
| motor I2C address / calibration (attack) | `hardware_attack.py` `DRIVE_MOTORS` **and** `STOP_attack.py` | |
| motor I2C address / calibration (goalie) | `hardware_defense.py` `DRIVE_MOTORS` **and** `STOP_defense.py` | |
| PID / current limit (both) | `hardware.py` `MotorThread` | |
| robot id for comms | `hardware_attack.py` / `hardware_defense.py`: `ROBOT_ID` | |

## 8. The two robots

| | Goalie | Striker |
| --- | --- | --- |
| run | `python3 main.py` | `python3 main_attack.py` |
| hardware file | `hardware_defense.py` | `hardware_attack.py` |
| drive motors (motor1..4) | 26, 32, 28, 25 | 26, 25, 27, 28 |
| dribbler | 27 - **placeholder address + calibration, calibrate before running** (`DRIBBLER = None` to run without) | 32 |
| kicker | GPIO 17 - **check the pin** | GPIO 17 |
| stop script | `STOP_defense.py` | `STOP_attack.py` |
| strategy | `GoalieStrategy` | `StrikerStrategy` |

Everything else is shared. **`robot_vision_config.json` is per robot** (each camera has its own lens, mounting and centre),
so generate it with `test_camera.py` on each Pi and don't copy it between robots. The systemd service on each Pi
must point at the right main file and stop script.

## 9. Setup / tuning order (each robot)

1. `python3 test_camera.py` → lens calibration → top-down → robot centre → ignore box → HSV for all 5 colours → **Generate config file**.
   Top-down px/cm: **2 or more**. At 1 px/cm a 4 cm ball is so small the noise filter removes it.
2. `python3 test_localisation.py --pcb` (motors stay off):
   - camera view: ball circled, goals crossed, lines magenta, field hull green, no red obstacles on empty field;
   - ball in the dribbler sits inside the cyan capture box;
   - face the attack goal, **Zero heading**, turn the robot by hand: the magenta dots stay on the field lines;
   - carry it around: the green robot follows, ± stays small;
   - push it onto a line: the blue escape arrow points back into the field.
3. `field.py`: fix dimensions if the lines don't match up.
4. Try strategy changes in `simulator.py` first, then run the main file on the field and tune `strategy.py` distances.

## 10. Adding the ultrasonic ring later

1. `hardware.py`: add an `UltrasonicThread` with `distances` (8 values in cm, `None` = no echo)
   and `angles` (each sensor's direction in the robot frame, radians, 0 = robot's right, counter-clockwise).
2. `common.py` `Robot.__init__`: start it.
3. Main loop, each tick: `bot.loc.walls = (us.distances, [a + compass for a in us.angles])`.
4. `localisation.py` `correct_walls()` already does the rest; tune `WALL_SIGMA` / `WALL_OUTLIER`.
   It's robust to readings that hit a robot instead of the wall.

## 11. Simulator (`simulator.py`)

```bash
python simulator.py                  # window: field, both robots, two opponents
python simulator.py --headless 30    # no window: 30 simulated seconds, prints score / outs / localisation error
python simulator.py --opp v26        # opponents run a past version of our code (past_versions/) instead of the AI
python simulator.py --fast           # no camera simulation (see Fast mode below)
```

It runs the real `GoalieBrain` / `StrikerBrain`, so everything from camera detection through to motor speeds
is the robot's own code. Only the hardware is fake:

| Real | Simulated |
| --- | --- |
| camera + lens + top-down calibration | the field through a `CAM_FOV_DEG` fisheye lens `CAM_HEIGHT` cm up at `CAM_RES` pixels (far things get blurry like the real one), turned into a `TOPDOWN_PPC` px/cm top-down view, then the real `robot_vision.py` masks and `detection.py` |
| dribbler | holds the ball in the capture notch while on; loses it when turning faster than `DRIBBLE_MAX_SPIN`, accelerating harder than `DRIBBLE_MAX_ACCEL`, or when an opponent touches it |
| kicker | ball leaves at `KICK_SPEED` (+ robot speed), cooldown from the hardware file |
| goals | solid side and back walls (`GOAL_WALL_T` thick): the ball bounces off them, robots can't drive through them, so goals only count through the front |
| referee | ball **fully** outside the playing area (goal mouths count as in) → off the field for `BALL_OUT_DELAY` (2 s, no ball), then on the free neutral spot nearest to where it went out; not moving for `NO_PROGRESS_TIME` → nearest free spot at once. **Multiple defence:** ball in a penalty area with both defending robots in it → the one further from the ball is moved just outside the area. A robot **pushed** out by an opponent (`PUSHED_TIME`, `PUSHED_OUT_SPEED`) is put straight back in. Neutral spots (`field.NEUTRAL_SPOTS`): the kickoff spot and `NEUTRAL_SIDE` (30 cm) left / right of it on the halfway line. A robot fully outside the playing area is taken off as damaged for `DAMAGED_TIME` (30 s) or until the next kickoff (switched off, its teammate sees it as inactive), then put back level with its own goal, on the side further from the ball |
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
`CAMERA_EVERY` (camera fps), `CAM_FOV_DEG` / `CAM_HEIGHT` / `CAM_RES` (lens), `BALL_RADIUS`, `KICK_SPEED`,
`LDR_PHYSICAL_OFFSET` (where LDR 0 physically is, see the comment there).

**Many games at once (`arena.py`):** one game proves little (a lucky bounce decides it), so to check a strategy
change, play many seeded games in parallel and compare the totals:

```bash
python arena.py --games 22                       # against the simple opponents (key a in the window)
python arena.py --games 22 --opp ../old_version  # against another copy of our own code (self-play)
python arena.py --games 44 --fast --opp ../old   # fast mode, ~10x quicker (see below)
python arena.py --games 22 --opp v35 --watch     # also watch all the running games live in one window
```

**Past versions:** `past_versions/` holds milestone snapshots of the robot code (see its README). Play against one
with `python simulator.py --opp v26` (window) or `python arena.py --opp v26` - any folder with the robot code works too.

**Fast mode (`--fast`, `Sim(fast=True)`):** no camera rendering, detection or localisation - each camera frame's
`Detections` are made straight from the true positions (+ noise) in the same format `detection.py` gives, and the
pose is the true position (+ noise). Perception, line fusion, strategy and motion are still the real code. Use it
to try ideas quickly, then confirm the good ones in the full simulator (it can't catch vision / localisation
problems). For a fair self-play comparison, play both ways round (each version as "us" once) and add the results
up: the two sides of the simulator aren't perfectly identical.

Each seed jitters the kickoff a few cm and turns sensor noise on, so the games differ. It prints each game, the
total goals and W/D/L, and event counts (catches, how the ball was lost, who put it out, kicked vs pushed goals).
For self-play the opponents run the other folder's `main.py` / `main_attack.py` (attacking blue): get an old
version with `git worktree add ../old_version <commit>`. The window takes the same idea:
`Sim(seed=..., opp_code=simulator.load_code(folder))`.

Not simulated, so these still need the real robot: lens / top-down calibration errors, tall objects stretching,
robots hiding things behind them, real colours and lighting, motor wiring or sign mistakes, how well the real
dribbler actually grips. A robot only gets a dribbler / kicker in the simulator if its hardware file has one.

## 12. Camera: how far it sees

The lens points straight down from 20 cm. For a circular fisheye, the floor it sees reaches `20 x tan(FOV / 2)`:

| FOV | sees around the robot | cm per pixel at 55 / 75 / 90 cm (240 px across) | with 480 px |
| --- | --- | --- | --- |
| 140° | **55 cm**: from the centre it can't see either goal | 1.7 / - / - | 0.9 / - / - |
| 160° | **113 cm**: both goal lines from midfield | 2.0 / 3.5 / 4.9 | 1.0 / 1.8 / 2.5 |

So use 160°. Lines and goals are fine to the edge. The ball (about 4 cm) stops being detected around 60-75 cm at
240 px; capturing at 480 px roughly doubles that, and shared ball positions cover the rest. If the lens spec's FOV is
the *diagonal*, it would only see about 20 cm, so make sure it's a circular fisheye whose FOV is the image circle.

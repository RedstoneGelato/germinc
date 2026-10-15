# strategy.py explained (plain English)

This is a guide to reading `strategy.py`. It goes through the file in order, from top to bottom. Line numbers like
`L740` match the file as it is now. If the file changes, they'll drift a little, so search for the function name instead.

You already wrote a lot of this kind of Python in `_hist_1attack.py` / `_hist_1defense.py`: classes with `__init__`
and `self`, `time.monotonic()`, `math.atan2`, your `Hysteresis` class (strategy.py still uses it!), `angdiff`,
`any(s.active for s in self.sequences)`, `x, y, z, w = quat`. Section 2 lists the extra bits of syntax this file uses
that your files didn't.

Contents
1. The big picture: what strategy.py is for
2. Python syntax used in this file
3. The maths you'll see over and over (vectors and angles)
4. The file, section by section
   - 4.1 Header and imports
   - 4.2 The TUNE numbers
   - 4.3 Helper functions
   - 4.4 GoalieStrategy
   - 4.5 The striker: chase / carry / shoot / restarts
5. Switched-off features
6. Where to change things

---------------------------------------------------------------------------------------------------------------------

## 1. The big picture

Each robot runs a loop, many times a second. In `main_attack.py` (striker) and `main.py` (goalie), each loop does this:

```
world = what the robot knows: where it is (pose), the ball, the other robots...   (perception / localisation)
desired_pos, desired_heading, dribbler_on = strategy.update(world, compass, mate, line.touches)
speeds = bot.mover.step(desired_pos, desired_heading, ..., **strategy.speed)       (motion.py turns it into motor speeds)
bot.actuate(speeds, dribbler_on, strategy.kick)                                    (motors, dribbler, kicker)
```

So **strategy.py never drives a motor itself**. Every loop, it answers four questions:

| Answer | What it is |
|---|---|
| `desired_pos` | **where to go**: an arrow (x, y in cm) from the robot to the target point |
| `desired_heading` | **which way to face**: a compass angle (radians) |
| `dribbler_on` | True or False |
| `.kick` | True = fire the kicker this loop |

It also sets a few things other parts read:
- `.speed`: how fast to go, as a dictionary like `{"spd_multi": 0.6}`, meaning "60% speed".
- `.botstate` and `.phase`: a number and a word, for printing and for the simulator window ("chase", "carry", "guard"...).
- `.chasing` and `.command`: sent to the teammate over comms.

There are two "brains" (classes) in the file:

| Class | Which robot | What it does |
|---|---|---|
| `GoalieStrategy` | goalie (main.py) | guards the goal, clears the ball, looks at the restart spot. When the striker is off, it plays `StrikerStrategy` itself |
| `StrikerStrategy` | striker (main_attack.py) | ball in play: chase it fast, carry it at the goal, shoot. Ball out or not seen: wait for the restart, search, wait in the middle |

Both robots also have a "fallback" (`strategy_fallback.py`: the same ideas in the robot's own relative frame - it
started as your comp-style logic). It's used whenever the robot doesn't know where it is on the field
(`pose.confident` is False).

---------------------------------------------------------------------------------------------------------------------

## 2. Python syntax used in this file

### Modules: `import x` and `x.NAME`
```python
import field
import robot_config as cfg          # "as cfg": a short nickname
from utils import Hysteresis        # take just this one thing out of utils.py
```
After that, `field.PLAY_W` means "the `PLAY_W` defined in field.py", and `cfg.ROBOT_RADIUS` means "the `ROBOT_RADIUS` in
robot_config.py".

### Constants in CAPITALS
`KICK_RANGE = 50.0` is just a variable. Writing it in capitals is a convention that means "a setting, don't change it
while running". All the tuning numbers are at the top so they're easy to find.

### Comments and docstrings
`# ...` is a comment. A `"""..."""` block straight under a `def` or `class` is a **docstring**, which is the
description of that function. Python ignores both, so they're just for people.

### Tuples and lists: points are pairs of numbers
```python
me = (pose.x, pose.y)        # tuple (round brackets): can't be changed afterwards
target = [0.0, -22.0]        # list (square brackets): can be changed, e.g. target[0] = 5
me[0], me[1]                 # x and y   (index 0 = first item)
```
Every point and every direction in this file is a pair: `p[0]` is x and `p[1]` is y.

### Unpacking, and returning several things
```python
target, heading, dribbler = self.attack(world, compass, me, ball)   # attack() returns 3 things
return target, heading, True                                        # ...like this
self.speed, self.phase = {"spd_multi": 1.0}, "carry"                # set two variables on one line
```

### Functions with default values and named arguments
```python
def clamp_in_field(p, margin=FIELD_MARGIN):     # margin is optional; if you leave it out it's FIELD_MARGIN
clamp_in_field(target, OUT_MARGIN)             # pass it anyway
self.score(self.spot, me, centres, min_clear=..., max_range=...)   # named arguments (the order doesn't matter)
```

### `None`
`None` means "nothing" or "not known". `ball = world.ball_field` is `None` when no ball is seen. Test for it with
`if ball is None:` or `if ball is not None:`. Many functions return `None` to mean "no such thing", for example
`head_on()` returns the opponent pressed against our front, or `None`.

### Conditions
```python
if a and b:   if a or b:   if not a:
0 < t < 1                       # chained comparison: 0 < t AND t < 1
x = 2 if fast else 1            # one-line if: x = 2 when fast, otherwise 1
```

### The `or` trick
`a or b` gives `a` if `a` is "something" (not 0, not None, not empty), otherwise `b`. So:
```python
math.copysign(1.0, ball[0] or 1.0)        # ball[0], but 1.0 if ball[0] is exactly 0 (pick a side when dead centre)
self.facing_since = self.facing_since or now   # keep the old time if there is one, else start it now
```

### Loops
```python
for c in centres:                  # each item in a list
for i in range(AIM_POINTS):        # i = 0, 1, 2, ... AIM_POINTS-1
for k, a in enumerate(candidates): # k = 0, 1, 2... along with each item a
for o, v in zip(opponents, velocities):   # walk two lists side by side: each opponent with its velocity
continue                           # skip to the next item
```

### List comprehensions: build a list in one line
```python
others = [q for q in field.NEUTRAL_SPOTS if dist(q, p) > 1]
```
This reads as "make a list of every spot q in NEUTRAL_SPOTS where q isn't p". It's the same as:
```python
others = []
for q in field.NEUTRAL_SPOTS:
    if dist(q, p) > 1:
        others.append(q)
```

### `any`, `all`, `min`, `max` over a loop
You used `any(s.active for s in self.sequences)` in your own code. It's the same idea:
```python
any(dist(o, me) < 2 * R + 20 for o in world.opponents)    # True if at least one opponent is that close
all(o[1] > STEP_UP_CLEAR_Y for o in world.opponents)      # True if every opponent is past that line
min(dist(c, ball) for c in world.opponents)               # the closest opponent's distance to the ball
min(..., default=1e9)                                     # ...and 1e9 (= 1000000000) if there are no opponents
```

### `key=lambda`: "the one with the smallest / biggest ..."
```python
min(field.NEUTRAL_SPOTS, key=lambda p: dist(p, ball))     # the neutral spot closest to the ball
```
`lambda p: dist(p, ball)` is a tiny nameless function: "given p, give back dist(p, ball)". `min(..., key=...)` gives
back the **item** that scores lowest (the spot), not the score itself.

### Dictionaries
```python
self.speed = {"spd_multi": 0.6, "smooth": 0.3}    # name -> value pairs
mate.get("pos")                                   # look up "pos"; gives None if it isn't there (no crash)
```
`mate` is the dictionary of what the teammate sent over comms: `"bot active"`, `"pos"`, `"command"`, `"chasing"`,
`"has ball"`. In main_attack.py, `**strategy.speed` unpacks the dictionary into named arguments, so
`step(..., spd_multi=0.6, smooth=0.3)`.

### Clamping a number between two limits
```python
min(max(x, lo), hi)        # x, but no lower than lo and no higher than hi
min(1.0, max(0.0, (90.0 - d) / 70.0))    # a 0..1 "how far along" value
```
This pattern is everywhere. `(90 - d) / 70` clamped to 0..1 means "0 when d is 90 or more, 1 when d is 20 or less,
and in between it slides smoothly".

### Classes (you know these)
```python
class Avoider:
    def __init__(self):      # runs once when it's created: Avoider()
        self.wp = None       # self.something = memory that lasts between loops
    def __call__(self, robot, target, centres, ...):   # special: lets you use the object like a function
```
`self.avoid = Avoider()` makes one. Because of `__call__`, `self.avoid(me, target, centres)` then runs its `__call__`.

`@staticmethod` above a `def` in a class means "this function doesn't use `self`". It just lives inside the class to
keep things tidy, and you call it as `GoalieStrategy.guard(me, ball)`.

### A function inside a function
`def soften(x, B=76.0):` inside `chase()` and `def clear(side):` inside `kick_out_side()` are small helpers that only
that one function uses.

### Long lines
A line ending in `\`, or an open bracket that isn't closed yet, carries on onto the next line. It's still one line
of code.

### Numbers
`1e-9` is 0.000000001 ("basically zero", used to avoid dividing by 0). `1e9` is "huge". `float("inf")` is infinity.
`x ** 2` is x squared. `a // b` divides and rounds down.

---------------------------------------------------------------------------------------------------------------------

## 3. The maths you'll see over and over

**Field coordinates** (from field.py). The origin (0, 0) is the field centre. **+y points at the goal we attack** and
+x is to the right when facing it. Everything is in cm. Our goal line is at `field.OWN_GOAL_Y` = -91.5 and theirs at
`field.ATTACK_GOAL_Y` = +91.5. The neutral spots are (0, 0), (-30, 0) and (30, 0).

**Headings** (compass). Measured in radians: 0 = facing their goal (up-field), and positive = turned
anticlockwise (to the left). `compass` is where the robot faces now; `heading` is where we want it to face.

| Function or code | What it means |
|---|---|
| `dist(a, b)` | distance between two points (Pythagoras: `math.hypot`) |
| `unit(v)` | the same direction as v, but length 1 (an arrow pointing that way) |
| `(b[0]-a[0], b[1]-a[1])` | the arrow from a to b |
| `p + u * 40` | the point 40 cm from p in direction u, written `[p[0] + u[0]*40, p[1] + u[1]*40]` |
| `heading_for(d)` | the compass heading that faces along arrow d (`atan2(-dx, dy)`) |
| `fwd = (-math.sin(compass), math.cos(compass))` | the arrow pointing out of the robot's front right now |
| `a[0]*b[0] + a[1]*b[1]` | **dot product**: how much of arrow a points along unit arrow b. Positive = same way, negative = opposite. E.g. "how fast is the opponent coming at me" = its velocity dotted with the unit arrow from it to me |
| `n = (-u[1], u[0])` | arrow u turned 90° left (sideways). Used to make points "to the side of" something |
| `angle_diff(x, y)` | the difference between two angles, wrapped to -180°..180° (`atan2(sin(x - y), cos(x - y))`). This is your `angdiff` |
| `seg_dist(p, a, b)` | how close point p comes to the straight line from a to b, and where along it (0 = at a, 1 = at b) |
| `clamp_in_field(p, margin)` | move p back inside the field, at least `margin` from the white line |
| `rel(p, pose)` | turn a field point into "the arrow from me to it". The final answer must be relative, so most `update`s end with `rel(target, pose)` |

`R` = `cfg.ROBOT_RADIUS` = 11 cm. So `2 * R` is the centre-to-centre distance of two robots touching.

**Obstacles vs opponents.** `world.obstacles_field` is every robot the camera sees, stored as the nearest edge.
`obstacle_centres()` turns those into centre points (`centres`). `world.opponents` and `world.opponent_vel` are the
opponents' centres and velocities, merged from both of our robots' cameras.

---------------------------------------------------------------------------------------------------------------------

## 4. The file, section by section

The file is laid out top to bottom like this:

| Lines | What |
|---|---|
| 1-27 | the description at the top (docstring) |
| 29-37 | imports |
| 39-128 | TUNE numbers (shared, goalie, striker waiting for restarts) |
| 132-547 | HELPERS: small tools everyone uses |
| 550-894 | `class GoalieStrategy` |
| 897-end | STRIKER: its TUNE numbers, `chase()`, `carry_dir()`, ..., `class StrikerStrategy` |

Inside each brain's `update()`, comments like `# ---- DECIDE ...` and `# ---- ACT` mark the steps.

### 4.1 Header and imports (L1-37)

- **L1-27:** the docstring, a summary of the file and of what `update()` returns (the same as section 1 above).
- **L29-35:** imports:
  - `math`
  - `time` (for `time.monotonic()`)
  - `field` (field size and spots)
  - `robot_config as cfg` (robot size, ball size)
  - the fallback brains
  - your `Hysteresis`
- **L37:** `R = cfg.ROBOT_RADIUS`, a short name because it's used hundreds of times.

### 4.2 The TUNE numbers (L39-128)

Each line is `NAME = value   # what it does`, and the comment on each line is its explanation. Grouped by who uses them:

| Lines | Used by | What they're about |
|---|---|---|
| 40-58 | everything | margins from the line, getting behind the ball, going round robots, aiming at the goal, slowing down to catch a free ball |
| 61-114 | goalie | guard radius, blocking rolling shots, clearing, passing, kicking out over the side line, shoving matches, the switched-off step-up and look-at-spot, playing alone |
| 116-128 | striker | waiting in the middle, searching, when a ball counts as out, **where to wait for a restart** |

The striker's ball-play numbers are further down, at the start of the STRIKER section (L905-942).

### 4.3 Helpers (L132-547)

These are small tools that the brains call. Each one takes points and numbers and gives back an answer; none of them
remember anything, except `Avoider`. They're grouped under `# ---- ...` headings.

**Basic geometry**
- `unit`: makes an arrow length 1. Gives (0, 0) instead of crashing if the arrow is zero length.
- `heading_for`: arrow → compass heading.
- `clamp_in_field`: keeps a point inside the field.
- `dist`: distance between two points.
- `seg_dist`: how close a point comes to a line (see section 3).

**Robots and the ball**
- `obstacle_centres`: the camera sees each robot's near edge, so this turns those into centre points.

**Shooting**
- `best_aim`: tries `AIM_POINTS` (7) points across their goal and picks the one with the most room around the ball's
  path (then the one nearest the middle).
- `clearance`: the gap between a path and the nearest robot (below 0 = blocked).
- `path_blocked`: the first robot in the way of driving from a to b, or None.

**`class Avoider` (L243): drives round robots in the way.** Use it as `target = self.avoid(me, target, centres)`.
1. If the straight path is clear, it goes straight.
2. Otherwise it tries a waypoint on each side of each robot, keeps those where both legs (me → waypoint → target) are
   clear, and takes the shortest. The route it's already on gets a 15 cm bonus (`STICK`), so it doesn't flip left and
   right every loop.
3. If nothing is clear, `fallback()` goes round the first robot on the path.

**Going for the ball** (the goalie's clearing)
- `choose_push_aim`: which way to push the ball. At the goal, unless standing behind it for that would be off the
  field.
- `approach`: get behind the ball and push. Three outcomes:
  - **"orbit"**: wrong side, so go round it
  - **"line up"**: behind it but off to the side, so go to the point behind it
  - **"push"**: lined up, so drive through it

  `along` is how far in front of or behind the ball we are, and `perp` is how far off to the side.
- `around_ball`: don't bump the ball on the way somewhere.
- `clear_aim`: where the goalie pushes a ball near our goal (out over the side line, or up-field away from the goal).
- `lead_ball`: where a rolling ball will be by the time we get there.
- `catch_speed`: slow right at the ball, faster further away (the goalie, and the striker's restart catch).
- `guard_speed`: the goalie's speed, in proportion to how far off its spot it is.
- `angle_diff`: the difference between two angles (your `angdiff`).
- `contested`: an opponent is right at the ball.

**Ball out**
- `ball_out`: past the line.
- `going_out`: out, or at the line and rolling out.
- `restart_spot`: the neutral spot nearest the ball.

**Shoving, possession, kicks**
- `shover`: an opponent pushing against us with the ball in between.
- `possession`: "us", "them" or None.
- `knock_out_aim`, `kick_out_side`, `pass_aim`: the goalie's kicks.

**Misc**
- `rel`: field point → arrow from the robot (`update()` must return arrows).
- `mate_active`: the teammate is on.

### 4.4 GoalieStrategy (L550-894)

**`__init__` (L562):** its memory. `botstate_hyst` is your Hysteresis: a new botstate has to hold 0.11 s, except 1
(ball in the dribbler), which switches at once.

**`update()` (L584) runs every loop, in this order:**

1. **"not sure where we are"**: hand over to `GoalieFallback`.
2. **"striker off"**: play `StrikerStrategy` instead (`self.alone`), copy its answers, and stop there.
3. **"remember a ball going out"**.
4. **"DECIDE the botstate"**: the decision tree. `raw` is this loop's answer before smoothing:
   - no ball → **0**
   - ball out → **4**
   - ball in our dribbler (and no opponent on it) → **1**
   - otherwise:
     - striker off → **2** if the ball is in our half or we're clearly closer, else **3**
     - an opponent has it close to our goal → **2** (go and knock it out)
     - an opponent has it → **3** (guard; coming out for it let them shoot past us)
     - the striker told us to take it, or it's in our box and close → **2**
     - we were already clearing, it's still in our end, and we haven't just kicked → **2**
     - otherwise → **3**
   - too far out of our area → **3**

   Then `Hysteresis` smooths it.
5. **"an opponent shoving the ball against the goalie"**: with the ball in our dribbler → turn to a side line and kick
   it out ("kick out"): the side with nobody in the way, else the side away from the robot shoving us
   (`kick_out_side`). Without it → shove straight back ("push back").
6. **"ACT"**:
   - **block**: a free ball rolling into our goal → stand where it'll cross (`block_point`), slowing down from 25 cm
     out (`BLOCK_GAIN`) and not moving sideways at all if the ball already comes within 5 cm of its centre
     (`BLOCK_STAY`) - full speed in, it slid past the ball and deflected it into our goal.
   - **look at spot**: **switched off** (`GOALIE_LOOK_AT_SPOT = False`). It came up to `LOOK_DIST` from the neutral
     spot the striker can't see, but with the ball back in about 0.5 s the goalie wasn't home in time.
   - **guard middle** (botstate 0).
   - **step up**: **switched off** (`GOALIE_STEP_UP = False`) for the same reason.
   - **guard** (3), or guard where it'll come back (4): `guard()` puts us on the arc between the goal and the ball.
   - botstate 1 → `clear_ball()`, botstate 2 → `take_ball()`.

**`clear_ball()` (L766):** the ball is in our dribbler:
- an opponent close → **"clear now"**: kick the moment we face up-field
- striker up-field → **"clear pass"**: kick it past the striker
- otherwise → **"clear kick"**: turn towards the open part of their goal and kick

**`take_ball()` (L797):** go for the ball with `go_for_ball()`. The aim is:
- to knock it out, if an opponent has it
- away from our goal, if it's deep in our half
- their goal, otherwise

It never goes round the ball, because that leaves the goal open.

**Other methods:**
- `note_kick` (main.py records kick times)
- `carrier` (the opponent with the ball)
- `shot_line_guard` (**switched off**: `CARRIER_WEIGHT = 0`)
- `block_point` (where a rolling ball crosses our goal line, worked out from its speed)
- `guard` (the spot on the arc)
- `go_for_ball`: `lead_ball` → `approach` → `around_ball` → speed → `Avoider`

### 4.5 The striker (L897-end)

This is the striker's whole brain. It used to hand the ball-out / ball-not-seen moments to an older striker
(`CarefulStriker`). That class is gone now: the parts of it that were still used are in `StrikerStrategy` itself.

**Its numbers (L905-954):**
- `CHASE_*`: the chase curve
- `CARRY_*`: when to kick while carrying, and the carry direction
- `QUICK_SHOT_*`: shoot right after winning the ball
- `CHARGED_*`: shoot when about to be tackled
- `HEAD_ON_*` / `SLIP_*`: step round an opponent pushing head-on
- `BOX_KEEP`: keep out of our own penalty area
- `RESTART_SCREEN` / `SCREEN_*`: the restart screen (below)
- `RESTART_CATCH` / `CATCH_RESTART_*`: the restart catch (below)
- `BEHIND_*`: orbit round a ball right behind us instead of backing into it (chase step 2b)
- `RESTART_RUSH` / `RUSH_*`: the restart rush (switched off)

**`share(value, start, end)` (L956):** "how far value has got from start to end, as 0..1". `chase()` uses it to slide
smoothly between behaviours as the distance changes.

**`chase(me, ball)` (L962): go for a loose ball (copied from team TS).** It has five numbered steps in the comments:
1. The closer we get (90 → 20 cm), the further round the ball we aim (up to 1.7× its angle), so we curl in behind it.
2. Close up (50 → 20 cm): if the ball is nearly in front, go straight at it; if it's beside or behind, keep going
   round (`soften()` does this).
   - 2b. Ball right behind us (more than 150° off up-field) and within 70 cm: **orbit**. Go to a point beside it on
     our side, facing up-field, then behind it. Going straight back reversed into the ball and knocked it towards our
     own goal, and turning round to face our own goal to get the ball is never done. (`BEHIND_ORBIT = False`: just
     head 110° off up-field on its side within 50 cm instead.)
3. Within 55 cm, with the ball within 60° of up-field: turn to face it.
4. Speed: full from 45 cm, 80% at 30 cm, then **×0.6 under 25 cm** (gentle, to catch it).
5. Dribbler on within 33 cm with the ball in front.

**`shot_open` (L1007):** facing their goal mouth right now, within range, nobody in the way.

**`head_on` (L1022):** an opponent pressed against our front.

**`charged` (L1036):** an opponent within 50 cm coming at us faster than 30 cm/s.

**`carry_dir` (L1046): which way to carry the ball, so an opponent never hits our capture zone.**
- It collects the angles to the opponents coming at us within 60 cm.
- With none, it carries straight at the goal.
- Otherwise it tries turning 0°, ±10°, ±20° ... up to ±120°, and takes the first direction that keeps them all more
  than 60° off our front. If none does, it takes the direction keeping them furthest round.

**`keep_out_of_box` (L1079):** moves a target out of our penalty area (two robots in it is a "multiple defence"
foul).

**`class StrikerStrategy` (L1087)**

Its memory (`__init__`):
- `last_ball`: where and when it last saw the ball, and whether it was going out. For an out ball it remembers the
  neutral spot it'll come back to.
- `out_at`: the last time the ball was out.
- `won_at` / `facing_since`: for the carry's kicks.
- `avoid`: an Avoider for going round robots.

`update()` (L1124), every loop:
1. Not localised → fallback.
2. The ball isn't seen but the camera saw it in our capture zone a moment ago (`world.ball_in_capture`) → it's right in
   front of us.
3. Remember the ball (`last_ball`).
4. **Ball gone out** (not seen, or fully past the line):
   - still visible but out → **"ball out"**: wait `NEUTRAL_WAIT` behind the spot it'll come back to.
   - not seen → `no_ball()`.

   A ball just past the line but not yet taken off is still played: giving up on it at once handed it to them.
5. **Ball in play** → `attack()`.
6. Goalie on → keep out of our penalty area.

`no_ball()` (L1187), when the ball isn't seen:
- goalie off → **"defend"** in front of our goal
- it went out a moment ago → **"restart"**: wait behind the spot it'll come back on. If their keeper is home, face the
  far corner of their goal from that spot, so it can shoot at once.
- otherwise (lost it) → **"wait"** behind the neutral spot nearest to where it was last seen (no searching)

`restart_ball()` (L1221): True while the ball has just come back on a neutral spot after going out.

`attack()` (L1229):
- **no ball in the dribbler:**
  - **"screen"**: for 3 s after the ball really went out, once it's back on a spot, if it's behind us (we're up-field
    of it) and their nearest robot is within 60 cm of it → face up-field and go round the ball on that robot's side
    (bodying it on the way) to get behind the ball, never touching the ball and never turning round. This goes on
    until their robot is further than 60 cm from the ball, or the ball is in front of us.
  - **"restart orbit"** (`RESTART_CATCH`): for 1.5 s after the ball really went out, once it's back on a spot, if
    we're up-field of it (the wrong side) and an opponent is within 70 cm of it → full speed round the ball on our
    side to get behind it, facing up-field (it used to turn round and catch it facing our own goal).
  - otherwise **"chase"** (or **"rush"**, if the restart rush were on)
- **ball in the dribbler, an opponent pressed against our front** → **"slip past"**: step 25 cm sideways (away from
  the side it's on) and 15 cm forwards. Pushing straight into it got us shoved back to our own goal at kickoffs.
- **ball in the dribbler** → **"carry"** along `carry_dir`, and kick:
  - just won it (less than 1 s ago) with the shot open from up to 130 cm (quick shot), or
  - an opponent charging at us with the shot open from up to 120 cm, or
  - facing their goal (within 10°) for 0.2 s, within 70 cm

The **restart rush** (`RESTART_RUSH`, `RUSH_*`) is switched off: with 0.5 s restarts it cost us. When on, it goes in at
full speed at a restart ball nobody else is near ("rush"), kicks straight away if facing their goal mouth ("flick"),
and slows the carry down while turning for 1.5 s after.

---------------------------------------------------------------------------------------------------------------------

## 5. Switched-off features

| Code | Status |
|---|---|
| `shot_line_guard` | off (`CARRIER_WEIGHT = 0.0`) |
| goalie "step up", "look at spot" | off (`GOALIE_STEP_UP`, `GOALIE_LOOK_AT_SPOT` = False) |
| restart rush / flick | off (`RESTART_RUSH = False`) |
| `RESTART_SIDE = 0.0` | the restart waiting spot is straight behind the neutral spot |

## 6. Where to change things

| You want to... | Change |
|---|---|
| striker waits closer to / further from the restart spot | `NEUTRAL_WAIT` (L128). Closer than about R + 10 makes the spot "occupied", and the referee then uses another spot |
| goalie guards further out / closer | `GUARD_RADIUS` (L61) |
| striker shoots from further / closer | `CARRY_KICK_RANGE`, `QUICK_SHOT_RANGE`, `OPEN_SHOT_RANGE` (L910-915) |
| striker approaches the ball faster or slower | `chase()` step 4 (the `0.8`, `0.2`, `0.6`) |
| goalie stops attacking when the striker is off | `GOALIE_ALONE_ATTACKS = False` (L92) |
| carry turns more or less to keep opponents off the ball | `CARRY_FRONT_CONE`, `CARRY_CONTACT_RANGE`, `CARRY_CLOSING` (L922-927) |
| restart catch on / off | `RESTART_CATCH` (L932) |
| restart screen on / off | `RESTART_SCREEN` (L928) |
| ball right behind us: orbit (True) or the simple sidestep (False) | `BEHIND_ORBIT` |

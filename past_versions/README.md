# Past versions (simulator opponents)

Snapshots of the robot code at milestones, so the simulator can play against them:

```bash
python simulator.py --opp v26              # window: our current code vs v26 (attacking the blue goal)
python simulator.py --opp v26 --fast       # same, without the camera simulation
python arena.py --games 22 --opp v26       # many games against it
```

`--opp` takes a name from this folder or a path to any folder with the robot code in it. Only the robot-code
files are kept here (strategy, perception, motion, ...): the hardware files and the simulator are always the
current ones, so every version plays with today's robots (goalie with a dribbler + kicker) and today's rules.

| Version | What it added (each beat the one before it in the simulator) |
| --- | --- |
| v0 | the commit "simulator and a bunch of changes" (start of the strategy work) |
| v5 | goalie never goes round the ball, clears away from our goal; pressure shots that miss every robot |
| v14 | goalie guards (doesn't challenge) when an opponent has the ball; proportional goalie speed, blocking rolling balls |
| v22 | striker catches the ball from the front instead of orbiting round it; shorter shot range |
| v26 | goalie guards facing up-field (turning was starving its moves); guards where the ball will be |
| v32 | line escape fixed past the line, clearing out over the side line, opponent sharing over comms |
| v35 | goalie dribbler + kicker, new motion (turning / braking), restart play, solo modes, dodging |
| v39 | quicker search / restart detection, waits behind (not on) the centre spot |

The current code (the repo itself) adds the dodge only reacting to opponents that are coming at us or touching,
whole-robot opponent circles, going round an opponent between us and the ball, choosing catch-from-the-front vs
orbit by time.

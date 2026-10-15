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
| v68 | goalie guards closer (20 cm) with less lead; pusher rule; push back in shoving matches (striker going for the ball, goalie), goalie kicks out over the open / nearer side line or charges a carrier within 35 cm; possession variable; faster orbit / line up; goalie passes past the striker; long shots at an open goal; realistic dribbler (brakes hard, only sideways / backwards / spinning lose the ball); sidesteps may use more of the field |
| v79 | shoots as soon as the path misses every robot; smaller (15 cm) sidesteps that go forwards past the opponent (towards the open side of the goal against their goalie) with room to pass a robot in the middle and no flicking between sidestep / going to the spot; full-speed orbit / line up (striker); waits for restarts halfway in from the side spot; smoothed opponent velocities; no detours round robots it's already level with; kickoff rule in the simulator |
| v81 | v79 + smoothed opponent velocities, no detours round robots already passed, goalie back in goal after clearing (the last version with our own careful striker) |
| v84 | striker rebuilt from other teams' ideas: TS-style chase, Bodensee-style carry and kick, slip past a head-on opponent, quick / charged shots if open; old striker (CarefulStriker) for restarts, waiting right behind the spot and facing the far corner when their goalie is home; striker keeps out of our penalty area; goalie clears at once under pressure, steps up when the ball is deep in their half, and comes up to look at the other neutral spot while the ball is out and carries with nearby charging opponents kept off its front (fast sim, 44 games: TS 4-0, Bodensee 233-9, lnxrobots 75-8, Technulgy 51-4) |
| v88 | tuned for 0.5 s restarts (simulator BALL_OUT_DELAY 0.5): perception drops only the one obstacle that is the teammate (an opponent tackling it was invisible); goalie doesn't step up or come out to look at the other spot; "restart catch" (straight at a restart ball from the wrong side, slowing down at it); restart rush off (fast sim, 88 games: TS 155-16, Bodensee 226-27, lnxrobots 139-19, Technulgy 184-17) |
| ts / ba / lx / tn | not versions: ports of other teams' strategies (TS 2025 Nationals, Bodensee Adler 2026 international Open vision, lnxrobots Open gen 3, Technulgy Open 2025 with our v79 goalie) - approximate re-writes in our coordinates, see the top of each strategy.py's port section |
| comp | not a version: the old COMPETITION logic (strategy_fallback.py's state machines) as an opponent, with today's
perception / motion / hardware. Pushes the ball in rather than kicking - a good test for the goalie. |

v68 also has everything between v39 and it (the dodge only reacting to opponents coming at us or touching,
whole-robot opponent circles, going round an opponent between us and the ball, catch-from-the-front vs orbit by
time). It keeps its own robot_config.py / motion.py, so it plays with v68's dribbler settings.

#!/usr/bin/env python3
"""
arena.py - play many simulated games in parallel and add up the results (to check a strategy change helps).

    python arena.py                              # 10 games x 60 s against the simple opponents (simulator key a)
    python arena.py --opp old_version_folder     # against another copy of the robot code (self-play)
    python arena.py --games 20 --seconds 90 --jobs 8
    python arena.py --games 44 --fast --opp old  # fast mode (no camera simulation): screen ideas, then confirm without --fast
    python arena.py --games 22 --opp v35 --watch # also show every running game live in one window (q closes it)

Self-play isn't perfectly symmetric (the two teams are processed in a fixed order in places), so to compare two
versions fairly play both ways round - each one as "us" once - and add the results up.

Each game uses a different seed (sensor noise on, kickoff positions jittered a few cm), so a change is only
better if it wins over many seeds, not just one lucky game. Our robots always attack yellow.

To keep an old version to play against:   git worktree add ../lightweight_v1 <commit>
(or copy the .py files into a folder). simulator.load_code() imports it next to the current code.
"""
import argparse
import os
import math
import multiprocessing as mp
import queue
import sys
import time

_WATCH = None         # in each game process: the queue the live window reads from (--watch), else None
WATCH_EVERY = 0.05    # s (real time, whatever the sim speed): send the positions this often for the live window
TILE_PPC = 1.2        # live window: px per cm of each small field


def _init_worker(q):
    global _WATCH
    _WATCH = q


def play(args):
    seed, seconds, opp, fast = args
    import cv2
    cv2.setNumThreads(1)          # one game per core already: OpenCV's own threads would only fight over the cores
    import simulator as S
    S._SIM_T[0] = 1000.0          # the sim clock is per process, and pool processes play several games
    code = S.load_code(opp) if opp != "ai" else None
    sim = S.Sim(noise=True, seed=seed, opp_code=code, fast=fast)
    sim.opp_ai = True
    held = {"us": 0, "them": 0}
    ball_y = 0.0
    n = 0
    phases = {}
    last_sent = 0.0
    while sim.t < seconds:
        sim.step()
        if _WATCH is not None and (time.perf_counter() - last_sent > WATCH_EVERY or sim.t >= seconds - S.DT):
            last_sent = time.perf_counter()
            try:
                _WATCH.put_nowait((seed, sim.t, sim.score["us"], sim.score["them"], tuple(sim.ball),
                                   [(b.x, b.y, b.h, b.team, "goalie" in b.name) for b in sim.on_field]))
            except queue.Full:
                pass
        if sim.ticks % 10 == 0:
            for r in sim.robots:
                st = r.brain.strategy
                key = f"{r.name} b{st.botstate} {st.phase}"
                phases[key] = phases.get(key, 0) + 0.1
        if sim.held_by is not None:
            held[sim.held_by.team] += 1
        ball_y += sim.ball[1]
        n += 1
    outs = {r.name: r.outs for r in sim.players}
    return {"seed": seed, "us": sim.score["us"], "them": sim.score["them"], "kicks": sim.kicks,
            "outs": outs, "held_us": held["us"] / n, "held_them": held["them"] / n, "ball_y": ball_y / n, "events": sim.events, "shots": sim.shots,
            "phases": phases}


def watch(q, pending, args):
    """Live window: every running game as a small field (our robots cyan / magenta, opponents grey, ball orange),
    finished games' scores along the top. Closing it (q / Esc) only stops the window, the games carry on."""
    import cv2
    import numpy as np
    import field

    k = TILE_PPC
    w, h = int(field.WALL_W * k), int(field.WALL_L * k)

    def px(x, y):
        return int(round((x + field.WALL_W / 2) * k)), int(round((field.WALL_L / 2 - y) * k))

    base = np.full((h, w, 3), (40, 140, 40), np.uint8)
    for sgn, col in ((1, (0, 220, 255)), (-1, (220, 120, 0))):
        cv2.rectangle(base, px(-field.GOAL_W / 2, sgn * field.PLAY_L / 2),
                      px(field.GOAL_W / 2, sgn * (field.PLAY_L / 2 + field.GOAL_DEPTH)), col, -1)
    for a, b in field.black_line_segments():
        cv2.line(base, px(*a), px(*b), (20, 20, 20), max(1, int(field.PENALTY_LINE_W * k)))
    for a, b in field.white_line_segments():
        cv2.line(base, px(*a), px(*b), (255, 255, 255), max(1, int(field.LINE_W * k)))
    latest, finished = {}, {}
    cols = min(6, max(1, min(args.jobs, args.games)))
    r_px = int(round(10.5 * k))
    shown = True
    import os
    snapshot = os.environ.get("ARENA_WATCH_SNAPSHOT")      # debugging: save the frames to this file, no window
    if not snapshot:
        cv2.namedWindow("arena (q closes this window, the games carry on)")
    while not pending.ready():
        try:
            for _ in range(500):      # (bounded: never just sit reading the queue without drawing)
                seed, t, us, them, ball, bodies = q.get_nowait()
                latest[seed] = (t, us, them, ball, bodies)
                if t >= args.seconds - 0.02:
                    finished[seed] = (us, them)
        except queue.Empty:
            pass
        if not shown:
            time.sleep(0.2)
            continue
        running = sorted(s for s in latest if s not in finished)
        rows = max(1, math.ceil(len(running) / cols))
        img = np.full((rows * (h + 22) + 26, cols * (w + 6), 3), 30, np.uint8)
        top = "done: " + "  ".join(f"#{s} {a}-{b}" for s, (a, b) in sorted(finished.items()))
        tot = (sum(a for a, _ in finished.values()), sum(b for _, b in finished.values()))
        cv2.putText(img, f"vs {os.path.basename(os.path.normpath(args.opp))}   finished {len(finished)}/{args.games}   total {tot[0]}-{tot[1]}   {top}"[:200],
                    (6, 18), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
        for i, s in enumerate(running):
            t, us, them, ball, bodies = latest[s]
            tile = base.copy()
            for x, y, hd, team, goalie in bodies:
                c = px(x, y)
                fill = ((255, 200, 0) if goalie else (255, 0, 255)) if team == "us" else (70, 70, 70)
                cv2.circle(tile, c, r_px, fill, -1)
                cv2.circle(tile, c, r_px, (0, 0, 200) if team != "us" else (255, 255, 255), 1)
                cv2.line(tile, c, px(x - 13 * math.sin(hd), y + 13 * math.cos(hd)), (255, 255, 255), 1)
            if abs(ball[0]) < field.WALL_W / 2:
                cv2.circle(tile, px(*ball), max(2, int(2.1 * k)), (0, 140, 255), -1)
            ox, oy = (i % cols) * (w + 6), 26 + (i // cols) * (h + 22)
            img[oy:oy + h, ox:ox + w] = tile
            cv2.putText(img, f"#{s} {us}-{them}  {t:4.1f}s", (ox + 2, oy + h + 15), cv2.FONT_HERSHEY_PLAIN, 1.0,
                        (255, 255, 255), 1)
        if snapshot:
            cv2.imwrite(snapshot, img)
            time.sleep(0.5)
            continue
        cv2.imshow("arena (q closes this window, the games carry on)", img)
        if cv2.waitKey(40) & 0xFF in (ord("q"), 27):
            cv2.destroyAllWindows()
            shown = False
    if shown and not snapshot:
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opp", default="ai", help="'ai' (simple opponents), a name from past_versions/ (e.g. v26) or a "
                                                "folder with another version of the code")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--jobs", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--first-seed", type=int, default=1)
    ap.add_argument("--fast", action="store_true", help="fast mode: no camera simulation (Sim(fast=True)), ~10x faster;"
                                                        " screen with it, confirm in the full simulator")
    ap.add_argument("--phases", action="store_true", help="also print how long our robots spent in each botstate/phase")
    ap.add_argument("--watch", action="store_true", help="show all the running games live in one window")
    ap.add_argument("--dump", help="also save every game's results (incl. each kick) to this JSON file")
    args = ap.parse_args()

    if args.opp != "ai":         # check the opponent exists here (in the game processes it would just hang the pool)
        from simulator import code_folder
        args.opp = code_folder(args.opp)
    t0 = time.perf_counter()
    jobs = [(args.first_seed + i, args.seconds, args.opp, args.fast) for i in range(args.games)]
    q = mp.Queue(maxsize=2000) if args.watch else None
    with mp.Pool(min(args.jobs, len(jobs)), maxtasksperchild=1, initializer=_init_worker, initargs=(q,)) as pool:
        if args.watch:
            pending = pool.map_async(play, jobs)
            watch(q, pending, args)
            results = pending.get()
        else:
            results = pool.map(play, jobs)
    if args.dump:
        import json
        with open(args.dump, "w") as f:
            json.dump(results, f)
    us = sum(r["us"] for r in results)
    them = sum(r["them"] for r in results)
    w = sum(r["us"] > r["them"] for r in results)
    d = sum(r["us"] == r["them"] for r in results)
    for r in results:
        print(f"seed {r['seed']:3d}  {r['us']}-{r['them']}  kicks {r['kicks']:2d}  ball held us {100 * r['held_us']:3.0f}% "
              f"them {100 * r['held_them']:3.0f}%  mean ball y {r['ball_y']:+5.1f}  outs {r['outs']}")
    ev = {}
    for r in results:
        for k, v in r["events"].items():
            ev[k] = ev.get(k, 0) + v
    for team in ("us", "them"):
        shots = [sh for r in results for sh in r["shots"] if sh["team"] == team]
        if shots:
            bands = [(0, 50), (50, 75), (75, 100), (100, 1e9)]
            txt = "  ".join(f"{a:.0f}-{b:.0f}cm {sum(sh['goal'] for sh in shots if a <= sh['dist'] < b)}/"
                            f"{sum(1 for sh in shots if a <= sh['dist'] < b)}" for a, b in bands)
            print(f"{team} kicks: {sum(sh['goal'] for sh in shots)} goals / {len(shots)} kicks   by distance: {txt}")
    if args.phases:
        ph = {}
        for r in results:
            for k, v in r["phases"].items():
                ph[k] = ph.get(k, 0) + v
        total = len(results) * args.seconds
        print("time per robot state (% of game):")
        for k, v in sorted(ph.items()):
            print(f"  {k:40s} {100 * v / total:5.1f}%")
    print("events:", ", ".join(f"{k} {v:.0f}" for k, v in sorted(ev.items())))
    print(f"\nvs {os.path.basename(os.path.normpath(args.opp))}: goals {us}-{them}  W/D/L {w}/{d}/{len(results) - w - d}  "
          f"({len(results)} games x {args.seconds:.0f} s, {time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())

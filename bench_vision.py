#!/usr/bin/env python3
"""
bench_vision.py - time the camera pipeline (robot_vision.py + detection.py) off the robot, and check it still
finds the same things as the old one.

    python bench_vision.py                      # 60 synthetic frames: radial mode, top-down at 2 and 3 px/cm
    python bench_vision.py --modes radial --frames 200
    python bench_vision.py --ref past_versions/v88   # what to compare against (default: vision_old = before the speed-up)

No camera needed: the frames are rendered from the simulator's field drawing (robots, ball, goals, lines) through a
fisheye lens model at 640x480, the size vision.py captures, and both pipelines get a calibration made from that
same lens model (camera 20 cm up, pointing down): the radial "Distance calibration" and/or a top-down one. Real-camera numbers on the Pi will differ, but the ratio
between old and new is what this is for. Run it on the Pi too: the times printed there are the real ones.
"""
import argparse
import importlib
import json
import math
import os
import sys
import tempfile
import time

import cv2
import numpy as np

import simulator as sim      # field drawing + colours (patches time.monotonic; this script uses perf_counter)
import field

HERE = os.path.dirname(os.path.abspath(__file__))
CAP_W, CAP_H = 640, 480
CAM_HEIGHT = 20.0                 # cm
HALF_FOV = math.radians(80.0)     # the image circle (480 px across) covers 160 deg
F = (CAP_H / 2) / HALF_FOV        # equidistant fisheye focal length, px
K = np.array([[F, 0, CAP_W / 2], [0, F, CAP_H / 2], [0, 0, 1]], np.float64)
D = np.array([0.02, -0.005, 0.0, 0.0], np.float64)
VIEW_CM = 115.0                   # top-down view: this far around the robot
EGO_PPC = 6.0                     # px/cm of the floor image the camera is rendered from


def make_radial_config():
    """test_camera.py's "Distance calibration" for the same lens: raw fisheye frame, pixel radius -> cm curve."""
    theta = np.linspace(0.0, math.radians(79.0), 200)
    k1, k2, k3, k4 = D
    r_px = F * theta * (1 + k1 * theta ** 2 + k2 * theta ** 4 + k3 * theta ** 6 + k4 * theta ** 8)
    cfg = sim.sim_vision_config()
    cfg["capture_size"] = [CAP_W, CAP_H]
    cfg["radial"] = {"table_px": r_px.tolist(), "table_cm": (CAM_HEIGHT * np.tan(theta)).tolist()}
    path = os.path.join(tempfile.mkdtemp(), "bench_vision_radial.json")
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path


def make_config(ppc):
    side = int(2 * VIEW_CM * ppc)
    c = side / 2.0
    s = 1.0 / (ppc * CAM_HEIGHT)
    # top-down pixel -> normalised camera ray (x right, y towards the robot's back = down in the image)
    Hn = [[s, 0, -c * s], [0, s, -c * s], [0, 0, 1]]
    cfg = sim.sim_vision_config()
    cfg["capture_size"] = [CAP_W, CAP_H]
    cfg["lens"] = {"K": K.tolist(), "D": D.tolist()}
    cfg["topdown"] = {"Hn": Hn, "out_size": [side, side], "px_per_cm": ppc}
    path = os.path.join(tempfile.mkdtemp(), f"bench_vision_{ppc}.json")
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path


class Renderer:
    """Floor around a robot -> raw 640x480 fisheye frame (BGR, like picamera2's "RGB888")."""
    SS = 2

    def __init__(self):
        self.fimg = sim.FieldImage(EGO_PPC)
        self.half = int(math.ceil(VIEW_CM * 1.3 * EGO_PPC))
        n_w, n_h = CAP_W * self.SS, CAP_H * self.SS
        u, v = np.meshgrid(np.arange(n_w) + 0.5, np.arange(n_h) + 0.5)
        pts = (np.stack([u.ravel(), v.ravel()], 1) / self.SS).reshape(-1, 1, 2)
        norm = cv2.fisheye.undistortPoints(pts, K, D).reshape(-1, 2)
        theta = np.hypot((pts[:, 0, 0] - K[0, 2]), (pts[:, 0, 1] - K[1, 2])) / F
        gx, gy = norm[:, 0] * CAM_HEIGHT, -norm[:, 1] * CAM_HEIGHT        # floor cm, +y = robot front
        ex = (self.half + gx * EGO_PPC).astype(np.float32).reshape(n_h, n_w)
        ey = (self.half - gy * EGO_PPC).astype(np.float32).reshape(n_h, n_w)
        out = (theta > HALF_FOV).reshape(n_h, n_w)
        ex[out], ey[out] = -10, -10
        self.maps = cv2.convertMaps(ex, ey, cv2.CV_16SC2)

    def scene(self, rng):
        """Random robot pose, ball and 3 other robots. Returns (field image, our (x, y, heading), ball robot-frame
        [x, y] cm or None)."""
        img = self.fimg.img.copy()
        hw, hl = field.PLAY_W / 2, field.PLAY_L / 2
        me = (rng.uniform(-hw + 5, hw - 5), rng.uniform(-hl + 5, hl - 5), rng.uniform(-math.pi, math.pi))
        others = []
        while len(others) < 3:
            p = (rng.uniform(-hw, hw), rng.uniform(-hl, hl))
            if all(math.hypot(p[0] - q[0], p[1] - q[1]) > 2.2 * sim.cfg.ROBOT_RADIUS for q in others + [me]):
                others.append(p)
        for p in others:
            cv2.circle(img, self.fimg.px(*p), int(sim.cfg.ROBOT_RADIUS * EGO_PPC), sim.ROBOT, -1)
        ball = None
        if rng.random() < 0.85:          # sometimes no ball in view
            a, r = rng.uniform(-math.pi, math.pi), rng.uniform(sim.cfg.ROBOT_RADIUS + 3, 90)
            b = (me[0] + r * math.cos(a), me[1] + r * math.sin(a))
            cv2.circle(img, self.fimg.px(*b), int(round(sim.BALL_RADIUS * EGO_PPC)), sim.ORANGE, -1)
            c, s = math.cos(-me[2]), math.sin(-me[2])
            ball = [r * math.cos(a) * c - r * math.sin(a) * s, r * math.cos(a) * s + r * math.sin(a) * c]
        cv2.circle(img, self.fimg.px(me[0], me[1]), int(sim.cfg.ROBOT_RADIUS * EGO_PPC), sim.ROBOT, -1)
        return img, me, ball

    def render(self, img, pose, rng):
        x, y, h = pose
        c, s = math.cos(h), math.sin(h)
        k, hp, f = EGO_PPC, self.half, self.fimg
        M = np.float32([[c, s, (f.W / 2 + x) * k - c * hp - s * hp],
                        [-s, c, (f.L / 2 - y) * k + s * hp - c * hp]])
        ego = cv2.warpAffine(img, M, (2 * hp, 2 * hp), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=sim.WALL)
        raw = cv2.remap(ego, *self.maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        raw = cv2.resize(raw, (CAP_W, CAP_H), interpolation=cv2.INTER_AREA)
        raw = cv2.GaussianBlur(raw, (3, 3), 0.8)
        return cv2.add(raw, rng.normal(0, 6, raw.shape).astype(np.int16), dtype=cv2.CV_8U)


def load_pipeline(folder):
    """(robot_vision, detection) modules from another folder, without mixing them with the current ones."""
    mods = ("field", "robot_config", "robot_vision", "detection")
    saved = {k: sys.modules.pop(k) for k in mods if k in sys.modules}
    sys.path.insert(0, os.path.abspath(folder))
    try:
        out = importlib.import_module("robot_vision"), importlib.import_module("detection")
    finally:
        sys.path.pop(0)
        for k in mods:
            sys.modules.pop(k, None)
        sys.modules.update(saved)
    return out


def timed(rv_mod, det_mod, cfg_path, frames, repeat):
    rv = rv_mod.RobotVision(cfg_path)
    dets = []
    t_proc = t_det = 0.0
    for _ in range(repeat):
        dets = []
        for raw in frames:
            t0 = time.perf_counter()
            res = rv.process(raw, align=False)
            t1 = time.perf_counter()
            dets.append(det_mod.detect(rv, res, 0.0, 0.0))
            t2 = time.perf_counter()
            t_proc += t1 - t0
            t_det += t2 - t1
    n = len(frames) * repeat
    return dets, 1000 * t_proc / n, 1000 * t_det / n, rv


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compare(old, new):
    """Per-frame differences between two lists of Detections."""
    s = {"frames": len(old), "ball_both": 0, "ball_mismatch": 0, "ball_err": [], "goal_mismatch": 0, "goal_err": [],
         "field_mismatch": 0, "obst_count_mismatch": 0, "obst_err": [], "line_err": [], "line_n": []}
    for a, b in zip(old, new):
        s["field_mismatch"] += a.field_visible != b.field_visible
        if (a.ball is None) != (b.ball is None):
            s["ball_mismatch"] += 1
        elif a.ball is not None:
            s["ball_both"] += 1
            s["ball_err"].append(_d(a.ball, b.ball))
        for g in ("yellow", "blue"):
            ga, gb = a.goals[g], b.goals[g]
            if (ga is None) != (gb is None):
                s["goal_mismatch"] += 1
            elif ga is not None:
                s["goal_err"].append(_d(ga.near, gb.near))
        if len(a.obstacles) != len(b.obstacles):
            s["obst_count_mismatch"] += 1
        for o in a.obstacles:
            if b.obstacles:
                s["obst_err"].append(min(_d(o.centre, q.centre) for q in b.obstacles))
        if len(a.line_pts) and len(b.line_pts):
            d = np.linalg.norm(a.line_pts[:, None, :] - b.line_pts[None, :, :], axis=2)
            s["line_err"].append(float(np.median(d.min(axis=1))))
        s["line_n"].append((len(a.line_pts), len(b.line_pts)))
    return s


BALL_RANGE = 80.0      # cm: balls closer than this should be found


def truth_ball(dets, balls):
    """Ball detection against where the ball really was (the nearest point of a 2.1 cm ball, so error ~0)."""
    t = {"n": 0, "found": 0, "false": 0, "err": []}
    for det, b in zip(dets, balls):
        r = math.hypot(*b) if b is not None else None
        if r is not None and r < BALL_RANGE:
            t["n"] += 1
            if det.ball is not None:
                t["found"] += 1
                t["err"].append(_d(det.ball, [b[0] * (1 - sim.BALL_RADIUS / r), b[1] * (1 - sim.BALL_RADIUS / r)]))
        if det.ball is not None and (b is None or _d(det.ball, b) > 10):
            t["false"] += 1
    return t


def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--modes", nargs="+", default=["radial", "2", "3"],
                    help="'radial' (raw fisheye + distance curve) and/or top-down px/cm values")
    ap.add_argument("--ref", default=os.path.join(HERE, "past_versions", "vision_old"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None, help="cv2.setNumThreads (default: OpenCV's choice)")
    a = ap.parse_args()
    if a.threads is not None:
        cv2.setNumThreads(a.threads)

    rng = np.random.default_rng(a.seed)
    ren = Renderer()
    frames, balls = [], []
    for _ in range(a.frames):
        img, pose, ball = ren.scene(rng)
        frames.append(ren.render(img, pose, rng))
        balls.append(ball)

    import detection
    import robot_vision
    ref_rv, ref_det = load_pipeline(a.ref)
    print(f"{a.frames} frames {CAP_W}x{CAP_H}, x{a.repeat}, OpenCV {cv2.__version__}, {cv2.getNumThreads()} threads")
    print(f"reference: {os.path.relpath(a.ref, HERE)}   new: ./robot_vision.py + ./detection.py\n")
    for mode in a.modes:
        path = make_radial_config() if mode == "radial" else make_config(float(mode))
        old, op, od, orv = timed(ref_rv, ref_det, path, frames, a.repeat)
        new, np_, nd, nrv = timed(robot_vision, detection, path, frames, a.repeat)
        h, w = orv.valid.shape
        print(f"--- {mode} ({w}x{h})" if mode == "radial" else f"--- top-down {float(mode):g} px/cm ({w}x{h})")
        print(f"  old: process {op:6.2f} ms  detect {od:6.2f} ms  total {op + od:6.2f} ms  ({1000 / (op + od):5.0f} fps)")
        print(f"  new: process {np_:6.2f} ms  detect {nd:6.2f} ms  total {np_ + nd:6.2f} ms  ({1000 / (np_ + nd):5.0f} fps)"
              f"   {(op + od) / (np_ + nd):.1f}x faster")
        s = compare(old, new)
        print(f"  field visible differs: {s['field_mismatch']}/{s['frames']} frames")
        print(f"  ball: seen by one only {s['ball_mismatch']}/{s['frames']}, both {s['ball_both']}; "
              f"position diff median {pct(s['ball_err'], 50):.2f} cm, max {pct(s['ball_err'], 100):.2f} cm")
        print(f"  goals: seen by one only {s['goal_mismatch']}; near point diff median {pct(s['goal_err'], 50):.2f} cm, "
              f"max {pct(s['goal_err'], 100):.2f} cm")
        print(f"  obstacles: count differs {s['obst_count_mismatch']}/{s['frames']}; centre diff median "
              f"{pct(s['obst_err'], 50):.2f} cm, max {pct(s['obst_err'], 100):.2f} cm")
        for name, dets in (("old", old), ("new", new)):
            t = truth_ball(dets, balls)
            print(f"  ball vs the truth ({name}): found {t['found']}/{t['n']} within {BALL_RANGE:g} cm, "
                  f"{t['false']} false; error median {pct(t['err'], 50):.2f} cm, max {pct(t['err'], 100):.2f} cm")
        print(f"  line points: median distance to the old ones {pct(s['line_err'], 50):.2f} cm "
              f"(worst frame {pct(s['line_err'], 100):.2f} cm)\n")


if __name__ == "__main__":
    main()

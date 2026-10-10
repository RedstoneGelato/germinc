#!/usr/bin/env python3
"""
test_localisation.py - check vision + localisation + line fusion on the real robot WITHOUT driving the motors.

    python3 test_localisation.py                    # camera + IMU, web page on port 8001
    python3 test_localisation.py --pcb              # also read the LDR ring and show the line fusion
    python3 test_localisation.py --no-imu           # heading fixed at 0 (robot must face the attack goal)
    python3 test_localisation.py --attack blue      # which goal we shoot at (default yellow)

Open http://<pi-ip>:8001. Left: what the camera sees (robot frame). Right: the field model with the particles
(grey), the estimated position (green) and everything seen, drawn where the robot thinks it is.

Checks, in order:
  1. Camera view: ball circled orange, goals crossed, white line points magenta, obstacles red, field hull green.
     The cyan box is CAPTURE_ZONE (robot_config.py): put the ball in the dribbler, it must be inside it.
  2. Face the attack goal, press "Zero heading". Turn the robot by hand: the dots on the field view must stay
     on the drawn lines (if they spin the wrong way, the compass sign is wrong).
  3. Carry the robot around: the green dot should follow within a few cm, the spread (±) should stay small.
  4. --pcb: push the robot onto a line, the escape arrow (blue) must point back into the field.

Uses the I2C bus and the camera, so don't run it next to main.py or test_camera.py.
"""
import argparse
import json
import math
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

import robot_config as cfg
from lines import LineFusion
from localisation import LocalisationThread
from perception import World
from vision import VisionThread

FIELD_SCALE = 2.0   # px per cm in the field view

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Localisation test</title>
<style>body{margin:0;padding:12px;background:#111;color:#eee;font:14px system-ui,sans-serif}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start}img{max-width:100%;background:#000}
#cam{width:min(480px,100%)}#fld{width:min(420px,100%)}
pre{font:12px monospace;background:#000;padding:8px;border:1px solid #333;min-width:300px}
button{padding:6px 10px;background:#335;color:#eee;border:1px solid #557;border-radius:4px;cursor:pointer}</style></head>
<body><div class="row"><img id="cam" src="/stream/cam"><img id="fld" src="/stream/field">
<div><button onclick="fetch('/api/zero',{method:'POST'})">Zero heading (facing attack goal)</button>
<button onclick="fetch('/api/relocalise',{method:'POST'})">Relocalise</button><pre id="st">...</pre></div></div>
<script>async function poll(){try{const s=await (await fetch('/api/status')).json();
document.getElementById('st').textContent=JSON.stringify(s,null,1)}catch(e){}setTimeout(poll,300)}poll()</script>
</body></html>"""


class Debug:
    def __init__(self, vision, loc, imu, pcb, attack):
        self.vision, self.loc, self.imu, self.pcb = vision, loc, imu, pcb
        self.attack = attack
        self.world = World()
        self.lines = LineFusion()
        self.line_state = None
        self.lock = threading.Lock()
        self.cam_img = self.field_img = None
        self.status = {}
        self.fid = 0
        self.fmap = loc.loc.map
        base = cv2.resize(self.fmap.lines, None, fx=FIELD_SCALE, fy=FIELD_SCALE, interpolation=cv2.INTER_NEAREST)
        self.field_base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        self.field_base[base <= 128] = (40, 110, 40)
        self.field_base[base > 128] = (255, 255, 255)

    def fpx(self, x, y):
        mx, my = self.fmap.to_map(x, y)
        return int(mx * FIELD_SCALE), int(my * FIELD_SCALE)

    def loop(self):
        last = -1
        while True:
            last, det = self.vision.wait_new(last, timeout=0.5)
            if det is None:
                continue
            with self.vision.cond:
                res = self.vision.last_res
            pose = self.loc.get()
            self.world.update(last, det, pose, self.attack)
            if self.pcb is not None:
                self.line_state = self.lines.update(self.pcb.snapshot(), det.compass, self.world.line_pts, pose)
            cam = self.draw_cam(res, det) if res is not None else None
            fld = self.draw_field(det, pose)
            st = self.make_status(det, pose)
            with self.lock:
                self.cam_img, self.field_img, self.status = cam, fld, st
                self.fid += 1

    def draw_cam(self, res, det):
        rv = self.vision.rv
        img = res.frame.copy()
        px = lambda p: tuple(int(round(v)) for v in rv.to_px(p[0], p[1]))
        x1, y1, x2, y2 = rv.ignore_box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cz = cfg.CAPTURE_ZONE
        cv2.rectangle(img, px((cz[0], cz[3])), px((cz[1], cz[2])), (255, 255, 0), 1)
        if det.hull_px is not None:
            cv2.polylines(img, [det.hull_px], True, (0, 255, 0), 1)
        for p in det.line_pts:
            cv2.circle(img, px(p), 1, (255, 0, 255), -1)
        for name, colour in (("yellow", (0, 255, 255)), ("blue", (255, 0, 0))):
            g = det.goals[name]
            if g is not None:
                cv2.drawMarker(img, px(g.near), colour, cv2.MARKER_CROSS, 10, 2)
        for o in det.obstacles:
            cv2.circle(img, px(o.near), 3, (0, 0, 255), -1)
            if getattr(o, "centre", None) is not None:   # the whole robot it belongs to
                cv2.circle(img, px(o.centre), int(round(cfg.ROBOT_RADIUS * rv.px_per_cm)), (0, 0, 255), 2)
        if det.ball is not None:
            cv2.circle(img, px(det.ball), 6, (0, 140, 255), 2)
        c = px((0, 0))
        cv2.arrowedLine(img, c, px((0, 15)), (0, 255, 0), 1, tipLength=0.3)
        return img

    def draw_field(self, det, pose):
        img = self.field_base.copy()
        for x, y in self.loc.particles()[::2]:
            cv2.circle(img, self.fpx(x, y), 1, (150, 150, 150), -1)
        if pose.std > 500:
            return img
        w = self.world
        for p in w.line_pts:
            cv2.circle(img, self.fpx(p[0] + pose.x, p[1] + pose.y), 1, (255, 0, 255), -1)
        for p, _ in w.obstacles:
            cv2.circle(img, self.fpx(p[0] + pose.x, p[1] + pose.y), 6, (0, 0, 255), 2)
        if w.ball is not None:
            cv2.circle(img, self.fpx(w.ball[0] + pose.x, w.ball[1] + pose.y), 5, (0, 140, 255), -1)
        centre = self.fpx(pose.x, pose.y)
        r = int(cfg.ROBOT_RADIUS * FIELD_SCALE)
        cv2.circle(img, centre, r, (0, 255, 0) if pose.confident else (0, 200, 255), 2)
        cv2.circle(img, centre, max(1, int(pose.std * FIELD_SCALE)), (0, 255, 0), 1)
        a = det.compass + math.pi / 2
        tip = self.fpx(pose.x + 20 * math.cos(a), pose.y + 20 * math.sin(a))
        cv2.arrowedLine(img, centre, tip, (0, 255, 0), 2, tipLength=0.3)
        ls = self.line_state
        if ls is not None and ls.escape is not None:
            tip = self.fpx(pose.x + 30 * ls.escape[0], pose.y + 30 * ls.escape[1])
            cv2.arrowedLine(img, centre, tip, (255, 120, 0), 2, tipLength=0.3)
        return img

    def make_status(self, det, pose):
        w = self.world
        r1 = lambda v: None if v is None else [round(float(a), 1) for a in v]
        st = {
            "fps": round(self.vision.fps, 1),
            "compass_deg": round(math.degrees(det.compass), 1),
            "pose": {"x": round(pose.x, 1), "y": round(pose.y, 1), "std": round(pose.std, 1), "confident": pose.confident},
            "field_visible": det.field_visible,
            "ball_robot_frame": r1(det.ball),
            "ball_relative": r1(w.ball),
            "ball_in_capture": w.ball_in_capture,
            "attack": self.attack,
            "attack_goal_near": r1(w.attack_goal.near) if w.attack_goal else None,
            "own_goal_near": r1(w.own_goal.near) if w.own_goal else None,
            "goal_source": w.goal_source,
            "line_points": int(len(det.line_pts)),
            "obstacles": [[r1(p), round(s)] for p, s in w.obstacles],
            "px_per_cm": self.vision.rv.px_per_cm,
        }
        ls = self.line_state
        if ls is not None:
            st["lines"] = {"on_line": ls.on_line, "ldr_count": ls.ldr_count, "cam_near": ls.cam_near,
                           "source": ls.source, "escape": r1(ls.escape)}
        return st


DBG = None


def jpeg(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send_bytes(self, data, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            if self.path == "/":
                self.send_bytes(PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/status":
                with DBG.lock:
                    st = DBG.status
                self.send_bytes(json.dumps(st).encode(), "application/json")
            elif self.path in ("/stream/cam", "/stream/field"):
                self.stream("cam_img" if self.path.endswith("cam") else "field_img")
            else:
                self.send_bytes(b"not found", "text/plain", 404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        if self.path == "/api/zero" and DBG.imu is not None:
            DBG.imu.zero()
            DBG.loc.relocalise()
        elif self.path == "/api/relocalise":
            DBG.loc.relocalise()
        self.send_bytes(b"{}", "application/json")

    def stream(self, attr):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        last = -1
        while True:
            with DBG.lock:
                fid, img = DBG.fid, getattr(DBG, attr)
            if img is None or fid == last:
                time.sleep(0.02)
                continue
            last = fid
            data = jpeg(img)
            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                             + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")


def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return socket.gethostname()


def main():
    global DBG
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--no-imu", action="store_true")
    ap.add_argument("--pcb", action="store_true", help="read the LDR ring and show the line fusion")
    ap.add_argument("--attack", choices=["yellow", "blue"], default="yellow")
    args = ap.parse_args()

    imu = pcb = None
    if not args.no_imu:
        from hardware import IMUThread
        imu = IMUThread()
        imu.start()
        while not imu.ready:
            time.sleep(0.05)
        imu.zero()
    if args.pcb:
        from hardware import PCBThread
        pcb = PCBThread()
        pcb.start()
        pcb.set_brightness(cfg.LED_BRIGHTNESS_START)

    vision = VisionThread(imu.compass if imu else (lambda: 0.0), keep_frames=True)
    vision.start()
    loc = LocalisationThread(vision, lambda: args.attack)
    loc.start()

    DBG = Debug(vision, loc, imu, pcb, args.attack)
    threading.Thread(target=DBG.loop, daemon=True).start()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    server.daemon_threads = True
    print(f"Open  http://{local_ip()}:{args.port}   Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
        vision.stop()


if __name__ == "__main__":
    main()

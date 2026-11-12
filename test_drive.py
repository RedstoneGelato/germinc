"""Drive the robot by hand from a web page, to tune speeds, turning, the dribbler and the kicker.

    python3 test_drive.py attack                  striker. Then open http://localhost:8000 on the Pi (VNC)
    python3 test_drive.py defense                 goalie
    python3 test_drive.py attack --host 0.0.0.0   also reachable from a laptop: http://<address of the Pi>:8000
    python3 test_drive.py --fake                  no robot needed, to try the page on a PC

Stop the normal robot program first (it uses the same motors). Put the robot on a stand for the first try.
The robot stops by itself when the page stops sending commands for STOP_AFTER seconds (page closed, Wi-Fi lost).
With --host 0.0.0.0 anyone on the same network can drive the robot, so only use it on a network you trust.
The camera view is the picture the robot's own vision works on. It needs robot_vision_config.json (test_camera.py).
Nothing is saved: copy the numbers you like into robot_config.py.
"""
import argparse
import json
import math
import threading
import time
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import robot_config as cfg
from motion import wheel_speeds

STOP_AFTER = 0.3          # seconds without a command from the page before the motors are stopped
MAX_SPEED = 1.5           # top of the speed slider, as a multiple of BASE_SPEED (same as MOTOR_MAX)
MAX_KICK_PULSE = 0.1      # seconds. Longer pulses can overheat the solenoid, so the page cannot ask for more
MAX_DRIBBLER = 546133333  # the motor drivers' own speed limit (MOTOR_SPEED_LIMIT in hardware.py)

motors = None             # set in main()
kicker = None
imu = None
camera = None             # vision.VisionThread, or None when the camera did not start
last_command = 0.0        # time.monotonic() of the newest command from the page


class FakeMotors:
    """Stands in for hardware.MotorThread with --fake."""
    has_dribbler = True
    dribbler_on_speed = 200000000
    drivers = ()
    speeds = (0, 0, 0, 0)
    dribbler_speed = 0

    def set(self, speeds):
        self.speeds = tuple(speeds)

    def set_dribbler(self, on):
        self.dribbler_speed = self.dribbler_on_speed if on else 0

    def stop(self):
        self.speeds = (0, 0, 0, 0)
        self.dribbler_speed = 0


class FakeKicker:
    """Stands in for hardware.Kicker with --fake."""
    pulse = 0.05

    def ready(self):
        return True

    def kick(self):
        print(f"KICK for {self.pulse} s")
        return True

    def close(self):
        pass


class FakeCamera:
    """Stands in for vision.VisionThread with --fake: one grey picture with an orange square."""

    def __init__(self):
        import numpy as np
        picture = np.full((240, 240, 3), 70, np.uint8)
        picture[100:140, 100:140] = (0, 140, 255)
        self.cond = threading.Lock()
        self.last_res = SimpleNamespace(frame=picture)
        self.fps = 0.0

    def stop(self):
        pass

    def close(self):
        pass


def camera_jpeg():
    """The newest camera picture as JPEG bytes, or None when there is none yet."""
    import cv2
    if camera is None:
        return None
    with camera.cond:
        result = camera.last_res
    if result is None:
        return None
    ok, data = cv2.imencode(".jpg", result.frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return data.tobytes() if ok else None


def number(data, key, low, high):
    """The number data[key], limited to low..high. The page is not trusted to send sensible values."""
    value = float(data.get(key, 0))
    if not math.isfinite(value):
        raise ValueError(f"{key} is not a number")
    return max(low, min(high, value))


def wheel_targets(cmd):
    """The four wheel speeds for one command from the page. Robot frame: x = right, y = forward, rot = turn left."""
    speed = number(cmd, "speed", 0.0, MAX_SPEED) * cfg.BASE_SPEED
    one_wheel = cmd.get("wheel")
    if one_wheel in (0, 1, 2, 3):       # spin a single wheel, to check the wiring order and direction
        speeds = [0, 0, 0, 0]
        speeds[one_wheel] = int(speed)
        return tuple(speeds)

    x, y = number(cmd, "x", -1.0, 1.0), number(cmd, "y", -1.0, 1.0)
    drive = wheel_speeds(x, y, speed * min(1.0, math.hypot(x, y)))
    rot = cfg.BASE_SPEED * number(cmd, "turn", 0.0, 1.0) * number(cmd, "rot", -1.0, 1.0)
    speeds = [w - rot for w in drive]   # same mixing and limit as motion.Mover.step
    biggest = max(abs(s) for s in speeds)
    if biggest > cfg.MOTOR_MAX:
        speeds = [s * cfg.MOTOR_MAX / biggest for s in speeds]
    return tuple(int(s) for s in speeds)


def apply_command(cmd):
    global last_command
    speeds = wheel_targets(cmd)
    dribbler_speed = number(cmd, "dribbler_speed", -MAX_DRIBBLER, MAX_DRIBBLER)
    motors.set(speeds)
    motors.dribbler_on_speed = int(dribbler_speed)
    motors.set_dribbler(bool(cmd.get("dribbler")) and motors.has_dribbler)
    last_command = time.monotonic()


def kick(cmd):
    if kicker is None:
        return {"kicked": False, "why": "this robot has no kicker (KICKER_PIN is None)"}
    kicker.pulse = number(cmd, "pulse", 0.0, MAX_KICK_PULSE)
    if kicker.kick():
        return {"kicked": True, "why": f"pulse {kicker.pulse} s"}
    return {"kicked": False, "why": "still cooling down"}


def state():
    return {
        "wheels": motors.speeds,
        "dribbler": motors.dribbler_speed,
        "compass": None if imu is None else round(math.degrees(imu.compass()), 1),
        "kick_ready": kicker is not None and kicker.ready(),
        "camera_fps": None if camera is None else round(camera.fps, 1),
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/camera.jpg"):
            picture = camera_jpeg()
            if picture is None:
                return self.reply(503, "text/plain", b"no camera picture yet")
            return self.reply(200, "image/jpeg", picture)
        self.reply(200, "text/html; charset=utf-8", PAGE.encode())

    def do_POST(self):
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            cmd = json.loads(body or b"{}")
            if self.path == "/cmd":
                apply_command(cmd)
                answer = state()
            elif self.path == "/kick":
                answer = kick(cmd)
            else:
                return self.reply(404, "text/plain", b"not found")
        except (ValueError, TypeError, AttributeError) as e:
            return self.reply(400, "text/plain", f"bad command: {e}".encode())
        self.reply(200, "application/json", json.dumps(answer).encode())

    def reply(self, code, kind, body):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):       # the page sends 10 commands a second: do not print each one
        pass


def start_hardware(role):
    global motors, kicker, imu, camera
    if role is None:
        motors, kicker, camera = FakeMotors(), FakeKicker(), FakeCamera()
        return {"dribbler_speed": FakeMotors.dribbler_on_speed, "kick_pulse": FakeKicker.pulse}

    from hardware import make_kicker, make_motors
    hw = cfg.AttackRobot if role == "attack" else cfg.DefenseRobot
    motors = make_motors(hw)
    motors.start()
    kicker = make_kicker(hw)
    try:
        from hardware import IMUThread
        imu = IMUThread()
        imu.start()
    except Exception as e:
        print(f"No compass on the page (IMU did not start: {e!r})")
    try:
        from vision import VisionThread
        compass = imu.compass if imu is not None else (lambda: 0.0)
        camera = VisionThread(compass, keep_frames=True)    # keep_frames: it keeps the newest picture for us
        camera.start()
    except Exception as e:
        print(f"No camera view on the page (camera did not start: {e!r})")
    return {"dribbler_speed": hw.DRIBBLER_SPEED if motors.has_dribbler else 0, "kick_pulse": hw.KICK_PULSE}


def stop_hardware():
    motors.stop()
    time.sleep(0.05)                    # lets the motor thread send the stop
    for driver in motors.drivers:
        driver.clear_faults()
    motors.running = False
    if imu is not None:
        imu.running = False
    if kicker is not None:
        kicker.close()
    if camera is not None:
        camera.stop()
        camera.close()


def main():
    global PAGE
    parser = argparse.ArgumentParser(description="Drive the robot by hand from a web page.")
    parser.add_argument("role", nargs="?", choices=["attack", "defense"], help="which robot this is")
    parser.add_argument("--fake", action="store_true", help="no robot: just try the page")
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = reachable from other computers")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.fake == (args.role is not None):
        parser.error("give attack or defense, or --fake")

    defaults = start_hardware(None if args.fake else args.role)
    defaults["camera"] = camera is not None
    defaults["base_speed"] = cfg.BASE_SPEED
    defaults["turn_max"] = cfg.TURN_MAX
    defaults["max_speed"] = MAX_SPEED
    defaults["max_kick_pulse"] = MAX_KICK_PULSE
    PAGE = PAGE.replace("__DEFAULTS__", json.dumps(defaults))

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Open http://localhost:{args.port}   (Ctrl+C here to quit)")
    try:
        while True:
            time.sleep(0.05)
            if time.monotonic() - last_command > STOP_AFTER:
                motors.stop()
    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        server.shutdown()
        stop_hardware()


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Robot manual drive</title>
<style>
  body { font-family: sans-serif; max-width: 720px; margin: 0 auto; padding: 16px; background: #14171c; color: #e8eaed; }
  h1 { font-size: 20px; }
  fieldset { border: 1px solid #3a4150; border-radius: 8px; margin: 0 0 12px; padding: 12px; }
  legend { padding: 0 6px; color: #9fb4d0; }
  label { display: block; margin: 8px 0; }
  input[type=range] { width: 100%; }
  input[type=number] { width: 150px; font-size: 16px; background: #222833; color: inherit; border: 1px solid #3a4150;
                       border-radius: 6px; padding: 6px; }
  button { font-size: 16px; padding: 14px 0; border-radius: 8px; border: 1px solid #3a4150; background: #222833;
           color: inherit; touch-action: none; user-select: none; cursor: pointer; }
  button.on, button:active { background: #2f6fdf; }
  #pad { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
  #wheels { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
  #stop { width: 100%; background: #b3261e; font-weight: bold; margin-bottom: 12px; }
  #kick { width: 100%; }
  #camera { display: block; width: 100%; max-width: 480px; margin: 0 auto; }
  #status { font-family: monospace; white-space: pre-wrap; }
  .hint { color: #9aa3b2; font-size: 14px; }
  .bad { color: #ff8a80; }
</style>
</head>
<body>
<h1>Robot manual drive</h1>
<button id="stop">STOP (Space)</button>

<fieldset id="cameraBox">
  <legend>Camera (what the robot's vision sees)</legend>
  <img id="camera" alt="camera view">
</fieldset>

<fieldset>
  <legend>Move and turn (hold a button or key)</legend>
  <div id="pad">
    <button data-key="q">Q<br>turn left</button>
    <button data-key="w">W<br>forward</button>
    <button data-key="e">E<br>turn right</button>
    <button data-key="a">A<br>left</button>
    <button data-key="s">S<br>back</button>
    <button data-key="d">D<br>right</button>
  </div>
  <label>Drive speed: <b id="speedText"></b>
    <input id="speed" type="range" min="0" step="1" value="30"></label>
  <label>Turn speed: <b id="turnText"></b>
    <input id="turn" type="range" min="0" max="100" step="1"></label>
</fieldset>

<fieldset>
  <legend>Single wheel (hold) - checks the motor order and direction</legend>
  <div id="wheels">
    <button data-key="1">Wheel 1</button>
    <button data-key="2">Wheel 2</button>
    <button data-key="3">Wheel 3</button>
    <button data-key="4">Wheel 4</button>
  </div>
  <p class="hint">Uses the drive speed above. Order = DRIVE_MOTORS in robot_config.py.</p>
</fieldset>

<fieldset>
  <legend>Dribbler</legend>
  <button id="dribbler" style="width:100%">Dribbler is OFF (R)</button>
  <label>DRIBBLER_SPEED (a minus sign spins it the other way)
    <input id="dribblerSpeed" type="number" step="10000000"></label>
</fieldset>

<fieldset>
  <legend>Kicker</legend>
  <button id="kick">KICK (K)</button>
  <label>KICK_PULSE in seconds (most allowed: <span id="maxPulse"></span>)
    <input id="pulse" type="number" step="0.005" min="0"></label>
  <div id="kickText" class="hint"></div>
</fieldset>

<fieldset>
  <legend>Robot</legend>
  <div id="status">connecting...</div>
</fieldset>

<script>
const D = __DEFAULTS__;
const $ = id => document.getElementById(id);
const held = new Set();          // keys and buttons held down right now
let dribblerOn = false;

$("speed").max = D.max_speed * 100;
$("turn").value = D.turn_max * 100;
$("dribblerSpeed").value = D.dribbler_speed;
$("pulse").value = D.kick_pulse;
$("pulse").max = D.max_kick_pulse;
$("maxPulse").textContent = D.max_kick_pulse;

function showSliders() {
  const speed = $("speed").value / 100, turn = $("turn").value / 100;
  $("speedText").textContent = speed.toFixed(2) + " x BASE_SPEED = " + Math.round(speed * D.base_speed);
  $("turnText").textContent = "TURN_MAX " + turn.toFixed(2);
}
$("speed").oninput = $("turn").oninput = showSliders;
showSliders();

function showHeld() {
  document.querySelectorAll("button[data-key]").forEach(b => b.classList.toggle("on", held.has(b.dataset.key)));
  $("dribbler").classList.toggle("on", dribblerOn);
  $("dribbler").textContent = "Dribbler is " + (dribblerOn ? "ON" : "OFF") + " (R)";
}
function stopAll() { held.clear(); dribblerOn = false; showHeld(); send(); }

document.querySelectorAll("button[data-key]").forEach(b => {
  b.onpointerdown = () => { held.add(b.dataset.key); showHeld(); };
  b.onpointerup = b.onpointerleave = b.onpointercancel = () => { held.delete(b.dataset.key); showHeld(); };
});
$("stop").onclick = stopAll;
$("dribbler").onclick = () => { dribblerOn = !dribblerOn; showHeld(); };
$("kick").onclick = kick;

document.addEventListener("keydown", e => {
  if (e.target.type === "number") return;          // typing a number is not driving
  const key = e.key.toLowerCase();
  if (key === " ") { e.preventDefault(); stopAll(); }
  else if (e.repeat) return;
  else if (key === "r") { dribblerOn = !dribblerOn; showHeld(); }
  else if (key === "k") kick();
  else if ("wasdqe1234".includes(key)) { held.add(key); showHeld(); }
});
document.addEventListener("keyup", e => { held.delete(e.key.toLowerCase()); showHeld(); });
window.addEventListener("blur", stopAll);           // another window in front = keys can get stuck, so stop

const axis = (plus, minus) => (held.has(plus) ? 1 : 0) - (held.has(minus) ? 1 : 0);

async function post(path, data) {
  const r = await fetch(path, {method: "POST", body: JSON.stringify(data)});
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function send() {
  const wheel = ["1", "2", "3", "4"].findIndex(k => held.has(k));
  try {
    const s = await post("/cmd", {
      x: axis("d", "a"), y: axis("w", "s"), rot: axis("q", "e"),
      speed: $("speed").value / 100, turn: $("turn").value / 100,
      wheel: wheel < 0 ? null : wheel,
      dribbler: dribblerOn, dribbler_speed: Number($("dribblerSpeed").value) || 0,
    });
    $("status").className = "";
    $("status").textContent = "wheel speeds sent: " + s.wheels.join(", ") +
      "\\ndribbler speed sent: " + s.dribbler +
      "\\ncompass: " + (s.compass === null ? "no IMU" : s.compass + " deg") +
      "\\nkicker: " + (s.kick_ready ? "ready" : "not ready") +
      "\\ncamera: " + (s.camera_fps === null ? "none" : s.camera_fps + " frames per second (vision speed)");
  } catch (err) {
    $("status").className = "bad";
    $("status").textContent = "NOT CONNECTED - the robot has stopped itself. " + err.message;
  }
}

async function kick() {
  try {
    const k = await post("/kick", {pulse: Number($("pulse").value) || 0});
    $("kickText").textContent = (k.kicked ? "Kicked: " : "No kick: ") + k.why;
  } catch (err) { $("kickText").textContent = "No kick: " + err.message; }
}

if (D.camera) {                  // load the next picture a moment after each one arrives
  const img = $("camera");
  const next = () => { img.src = "/camera.jpg?" + Date.now(); };
  img.onload = () => setTimeout(next, 100);
  img.onerror = () => setTimeout(next, 1000);
  next();
} else {
  $("cameraBox").hidden = true;
}

setInterval(send, 100);          // also the "I am still here" signal: the robot stops 0.3 s after the last one
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()

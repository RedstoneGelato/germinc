"""
vision.py - camera thread: capture -> robot_vision.py (calibration, masks) -> detection.py -> latest Detections.

Uses the calibration exported by test_camera.py (robot_vision_config.json). Stays in the robot frame
(align=False) and lets the consumers rotate the few detected points by the compass instead of rotating images.
"""
import threading
import time

from detection import detect
from robot_vision import RobotVision
import robot_config as cfg
from utils import rotate


def analyse_frame(rv, raw, t, compass):
    """One camera frame -> Detections. The robot (VisionThread) and simulator.py both use exactly this."""
    res = rv.process(raw, align=False)
    return detect(rv, res, t, compass), res


class VisionThread(threading.Thread):
    MAX_ERRORS = 10           # frames failing in a row -> give up (the thread ends and common.Robot restarts the camera)

    def __init__(self, compass, config_path=cfg.VISION_CONFIG, keep_frames=False, first_frame_id=0):
        """compass() -> current robot heading in radians (sampled right after each capture).
        keep_frames=True also keeps the last processed frame/masks (for test_localisation.py).
        first_frame_id: carries the numbering on when the camera is restarted."""
        super().__init__()
        import picamera2                      # imported here so the rest of the code can be tested off the Pi
        self.daemon = True
        self.running = True
        self.compass = compass
        self.keep_frames = keep_frames

        self.rv = RobotVision(config_path)
        if self.rv.mode != "topdown":
            print(f"[vision] WARNING: no top-down setup in {config_path} (mode={self.rv.mode}). "
                  f"Distances will be wrong - do the top-down setup in test_camera.py.")
        self.cap = picamera2.Picamera2()
        config = self.cap.create_preview_configuration(main={"size": self.rv.size, "format": "RGB888"})
        self.cap.configure(config)
        self.rv.apply_camera(self.cap)
        self.cap.start()

        self.cond = threading.Condition()
        self.frame_id = first_frame_id
        self.last_frame_t = time.monotonic() + 2.0   # last new frame (the watchdog allows 2 s extra to start up)
        self.detections = None
        self.last_res = None
        self.fps = 0.0
        self.ready = False

    def run(self):
        n, t0 = 0, time.monotonic()
        errors = 0
        while self.running:
            try:
                raw = self.cap.capture_array("main")
                t = time.monotonic()
                c = self.compass()
                det, res = analyse_frame(self.rv, raw, t, c)
            except Exception as e:      # one bad frame: skip it. Keeps failing: end the thread (gets restarted)
                if not self.running:
                    break
                errors += 1
                if errors == 1:
                    print(f"[vision] frame failed: {e!r}")
                if errors >= self.MAX_ERRORS:
                    print(f"[vision] {errors} frames failed in a row - stopping the camera thread")
                    break
                time.sleep(0.05)
                continue
            errors = 0
            with self.cond:
                self.detections = det
                self.last_res = res if self.keep_frames else None
                self.frame_id += 1
                self.cond.notify_all()
            self.last_frame_t = t
            self.ready = True
            n += 1
            if t - t0 >= 1.0:
                self.fps, n, t0 = n / (t - t0), 0, t

    def wait_new(self, last_id, timeout=None):
        """Block until a frame newer than last_id exists. Returns (frame_id, Detections or None on timeout)."""
        with self.cond:
            if not self.cond.wait_for(lambda: self.frame_id != last_id, timeout):
                return last_id, None
            return self.frame_id, self.detections

    def latest(self):
        with self.cond:
            return self.detections

    def stop(self):
        self.running = False
        try:
            self.cap.stop()
        except Exception:
            pass

    def close(self):
        """Release the camera (after stop()) so a new VisionThread can open it."""
        try:
            self.cap.close()
        except Exception:
            pass


def relative(det, v):
    """Robot-frame vector from a Detections -> relative (field axes), using the compass at capture time."""
    return None if v is None else rotate(v, det.compass)

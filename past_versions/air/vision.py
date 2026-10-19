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
    def __init__(self, compass, config_path=cfg.VISION_CONFIG, keep_frames=False):
        """compass() -> current robot heading in radians (sampled right after each capture).
        keep_frames=True also keeps the last processed frame/masks (for test_localisation.py)."""
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
        self.frame_id = 0
        self.detections = None
        self.last_res = None
        self.fps = 0.0
        self.ready = False

    def run(self):
        n, t0 = 0, time.monotonic()
        while self.running:
            raw = self.cap.capture_array("main")
            t = time.monotonic()
            c = self.compass()
            det, res = analyse_frame(self.rv, raw, t, c)
            with self.cond:
                self.detections = det
                self.last_res = res if self.keep_frames else None
                self.frame_id += 1
                self.cond.notify_all()
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


def relative(det, v):
    """Robot-frame vector from a Detections -> relative (field axes), using the compass at capture time."""
    return None if v is None else rotate(v, det.compass)

import cv2
import numpy as np
from picamera2 import Picamera2

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (320,240), "format":"RGB888"})
picam2.configure(config)

picam2.set_controls({"AwbEnable": False})

blue_gain = 1.0
red_gain = 1.0
picam2.set_controls({"ColourGains": (blue_gain, red_gain)})

picam2.start()

# --- initial camera parameters ---
fx = 160
fy = 160
cx = 160
cy = 120

k1 = -0.30
k2 = 0.10
k3 = -0.01

print("""
CONTROLS

White balance
U/I : blue +-
P/Y : red +-

Focal length
W/S : fx
E/D : fy

Distortion
1/2 : k1
3/4 : k2
5/6 : k3

O   : toggle undistort
K   : print calibration arrays
Q   : quit
""")

undistort_enabled = True

def compute_maps():

    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.array([k1, k2, 0, 0, k3], dtype=np.float32)

    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        None,
        camera_matrix,
        (320,240),
        cv2.CV_16SC2
    )

    return camera_matrix, dist_coeffs, map1, map2


camera_matrix, dist_coeffs, map1, map2 = compute_maps()

while True:

    frame = picam2.capture_array()

    if undistort_enabled:
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

    cv2.imshow("Live Fisheye Calibration", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # white balance
    if key == ord('u'): blue_gain += 0.1
    if key == ord('i'): blue_gain -= 0.1
    if key == ord('p'): red_gain += 0.1
    if key == ord('y'): red_gain -= 0.1

    # focal length
    if key == ord('w'): fx += 5
    if key == ord('s'): fx -= 5
    if key == ord('e'): fy += 5
    if key == ord('d'): fy -= 5

    # distortion
    if key == ord('1'): k1 += 0.01
    if key == ord('2'): k1 -= 0.01
    if key == ord('3'): k2 += 0.01
    if key == ord('4'): k2 -= 0.01
    if key == ord('5'): k3 += 0.01
    if key == ord('6'): k3 -= 0.01

    if key == ord('o'):
        undistort_enabled = not undistort_enabled

    if key == ord('k'):
        print("\nCALIBRATION VALUES\n")
        print("camera_matrix = np.array([")
        print(f"    [{fx}, 0, {cx}],")
        print(f"    [0, {fy}, {cy}],")
        print("    [0, 0, 1]")
        print("], dtype=np.float32)\n")

        print("dist_coeffs = np.array([")
        print(f"    {k1}, {k2}, 0, 0, {k3}")
        print("], dtype=np.float32)\n")

    blue_gain = max(0, blue_gain)
    red_gain = max(0, red_gain)

    picam2.set_controls({"ColourGains": (blue_gain, red_gain)})

    camera_matrix, dist_coeffs, map1, map2 = compute_maps()

    print(f"WB: blue={blue_gain:.2f} red={red_gain:.2f} | fx={fx} fy={fy} | k1={k1:.3f} k2={k2:.3f} k3={k3:.3f}")

cv2.destroyAllWindows()
picam2.stop()
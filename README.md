# Germ Inc
## General Information
| | |
| ----------- | ---- |
| Year | 2026 |
| Year Level | 11 |
| Team Name | Germ Inc |

Team Members (in first name alphabetical order):
- Allen
- Jerold
- Kieran
- Sebastian

# REQUIREMENTS
## Software
- Visual Studio Code (text editor of choice)
- Coding Language: Python 3.11 (Raspberry Pi) & C (PCB)
- STM32 Cube IDE (Write and flash custom PCB firmware)
- Github (source control)
- RealVNC Viewer (Used to remotely access and view Raspberry Pi GUI)

## Hardware
Per robot:
- 5x Robomaster M2006 P36 BLDC motors (ran on custom Steelbar Robotics Brushless Motor Drivers)
- 4x GTF Omniwheel (50mm diameter)
- 4S 850mAh Li-Po battery
- Raspberry Pi 5 (active cooler installed)
- Raspberry Pi HQ Camera
- Adafruit BNO085 IMU
- Custom PCB (LDR ring; IR ball ring for where the camera can't see the ball - robot_config.IR_*, IR_ON_PCB)
  - 1x STM32H503RBT6 Microcontroller
  - 32x KT-0603W LEDs
  - 32x ALS-PT19-315C/L177/TR8 LDRs
  - 2x CD74HC4067SM96 Analogue converters
- On-off switch (GPIO 25)

# SETUP & INSTALLATION
On a Raspberry Pi 5 with Raspberry Pi OS installed:
1. Install circuitpython: https://learn.adafruit.com/circuitpython-on-raspberrypi-linux/installing-circuitpython-on-raspberry-pi 
2.	Install the motor driver library, and other dependencies: pip install git+https://github.com/Aw3someAndrew/SteelBar_CircuitPython_powerful_bldc_driver.git
3.	Clone the repository into a directory of choice

# DEPLOYING & USAGE
During competitions, both pi's are setup to automatically run the code on startup. To manually run the code, follow the below:
1. Activate virtual environment
2. Change directory to the folder with the github repository
3. run either main_attack.py (striker robot), or main.py (goalie robot)
4. Flick the switch (wired from GPIO 25 to ground on the Raspberry Pi) to run the code, or otherwise in standby

Note: It is recommended to startup in standby as to calibrate the robot's bottom LED, compass, and goal colour

## Surviving crashes
Set up the autostart with `./deploy/install_service.sh attack` (striker Pi) or `./deploy/install_service.sh defense` (goalie Pi).
It installs `deploy/germinc.service` as `/etc/systemd/system/germinc.service` with `Restart=always`: whenever the program stops
it runs the STOP script (motors off) and starts the program again a second later. Stop it with `sudo systemctl stop germinc`
before running anything by hand (only one program can use the camera), and watch it with `journalctl -u germinc -f`.

Inside the program (`common.py`):
- an exception in `tick()` stops the motors for that loop, is printed once, and the next loop carries on (`TickGuard`)
- a camera or comms thread that dies or stops producing is restarted by a watchdog (once a second)
- no new camera frame for 0.3 s: the robot stands still until the camera is back (`CameraWatch`)
- the IMU heading ignores jumps (the BNO08x resetting), and the heading / attack goal are saved to `/dev/shm` every second,
  so a restarted program carries on facing the right way without pausing the robot

Test it in the simulator: `python arena.py --games 20 --faults 1` breaks our robots on purpose (camera freezes, comms drops,
IMU jumps, crashes, restarts: see `faults.py`); compare with the same games without `--faults`.

# CODE LAYOUT
See [GUIDE.md](GUIDE.md) for how it all works and which file to edit for what.

`main.py` (goalie) and `main_attack.py` (striker) start everything and run the 100 Hz loop; each part lives in its own file:

| File | What it does |
| ---- | ---- |
| `robot_config.py` | Numbers shared by both robots, coordinate conventions |
| `hardware.py` | IMU, PCB (LDR line ring + LEDs), motor thread |
| `hardware_attack.py` / `hardware_defense.py` | Per-robot motor addresses + calibration, robot id |
| `common.py` | Start-up, standby (paused) calibration, shutdown |
| `vision.py` | Everything the camera does: calibration + colour masks (`RobotVision`), masks -> ball (orange), goals (yellow/blue), field (green), lines (white), obstacles (anything else on the field) (`detect()`), and the camera thread (`VisionThread`) |
| `field.py` | Field dimensions + line distance map (check against the rules / real field!) |
| `localisation.py` | Particle filter: robot x, y on the field from lines + goals (ultrasonic ring ready for later) |
| `perception.py` | Smoothed ball / goals / obstacles in cm for the strategy |
| `lines.py` | Out-of-bounds from LDR ring + camera + localisation |
| `strategy.py` | Goalie and striker behaviour in field coordinates (needs localisation) |
| `strategy_fallback.py` | The comp-style relative state machines, used while a robot isn't localised |
| `motion.py` | Desired movement -> motor speeds |
| `comms.py` | Link to the other robot |
| `simulator.py` | Runs both robots' real logic on a simulated field (laptop, no Pi needed) |
| `faults.py` | Breaks our robots on purpose in the simulator (`--faults`) to check they survive crashes |
| `STOP_attack.py` / `STOP_defense.py` | systemd ExecStopPost: stop all motors + LEDs (`stop_robot.py`, addresses read from the hardware files) |

Camera workflow (on each robot, the config is per robot):
1. `python3 test_camera.py` -> lens calibration, top-down, robot centre, ignore box, HSV for all 5 colours -> "Generate config file" (writes `robot_vision_config.json`)
2. `python3 test_localisation.py --pcb` -> check detections, the capture zone, localisation and line escape without the motors running
3. `python3 main.py` (goalie) or `python3 main_attack.py` (striker)

Logic changes: try them in `python simulator.py` on a laptop first (needs `opencv-python` and `numpy`).

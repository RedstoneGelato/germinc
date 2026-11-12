# Germ Inc
## General Information
| | |
| ----------- | ---- |
| Year | 2027 |
| Year Level | 12 |
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
- 5x Robomaster M2006 P36 BLDC motors
- Steelbar Robotics Brushless DC Motor Driver V5
- 4x GTF Omniwheel (50mm diameter)
- 6S 850mAh Li-Po battery
- Raspberry Pi 5 (active cooler installed)
- Raspberry Pi Camera Module 3 Wide
- Custom hyperbolic mirror
- Adafruit BNO085 IMU
- Custom PCB:
  - 1x STM32H503RBT6 Microcontroller
  - 32x KT-0603W LEDs
  - 32x ALS-PT19-315C/L177/TR8 LDRs
  - 2x CD74HC4067SM96 Analogue converters

# SETUP & INSTALLATION
On a Raspberry Pi 5 with Raspberry Pi OS installed:
1.	Clone the repository into a directory of choice
2.	Run the setup file that automatically gets all dependencies: `bash setup.sh attack` (striker) or `bash setup.sh defense` (goalie).
	It installs the packages, turns on I2C (1 MHz, GPIO 2/3) and VNC, makes the virtual environment at `~/env` and sets the robot to start at boot.
	Leave out `attack` / `defense` to install without autostart.
	To keep the same IPv4 address on the current network, run it as `STATIC_IP=192.168.1.51/24 GATEWAY=192.168.1.1 bash setup.sh attack` (use your network's numbers). Reboot afterwards, then calibrate the camera with `test_camera.py`.

# DEPLOYING & USAGE
During competitions, both pi's are setup to automatically run the code on startup. To manually run the code, follow the below:
1. Activate virtual environment
2. Change directory to the folder with the github repository
3. run either `python3 main_attack.py` (striker robot), or `python3 main_defence.py` (goalie robot).
   Add `--int` for the international field, e.g. `python3 main_attack.py --int`
4. Flick the switch (wired from GPIO 25 to ground on the Raspberry Pi) to run the code

Note: It is recommended to startup in standby to calibrate the robot's bottom LED and compass

To start the international version at boot instead: `bash deploy/install_service.sh attack int` (or `defense int`).

# WHAT EACH FILE DOES
How the robot program fits together, 100 times a second:

```
camera  ->  vision.py  ->  localisation.py (where am I?)  \
line sensors, IR ring, compass  ->  hardware.py            ->  perception.py (World: ball, goals, robots, me)
teammate's messages  ->  comms.py                         /
                                                               |
                 strategy.py (what to do)  <-------------------+
                       |
                 motion.py (wheel speeds)  ->  lines.py (stay inside the field)  ->  hardware.py (motors, kicker)
```

`common.py` is the loop that runs these steps in order.

**Start and stop the robot**

| File | What it does |
| --- | --- |
| `main_attack.py` | Starts the striker. `--int` = international field. |
| `main_defence.py` | Starts the goalie. `--int` = international field. |
| `stop_robot.py` | Safety stop: all motors to 0, LEDs off. The boot service runs it whenever the program ends. |
| `setup.sh`, `deploy/` | Install everything on a fresh Pi and start the robot at boot. |

**Settings**

| File | What it does |
| --- | --- |
| `robot_config.py` | Every hardware and speed setting, and at the end the two robots' own motors, dribbler and kicker. |
| `field.py`, `field_int.py` | Sizes and lines of the RCJA field and of the international field. |
| `TUNING.md` | List of the settings worth changing, with what each one does. |

**The robot program** (used by both robots)

| File | What it does | Gets its input from |
| --- | --- | --- |
| `common.py` | The main loop: reads sensors, asks the strategy, drives the motors. Restarts parts that crash. | everything below |
| `hardware.py` | Talks to the electronics: compass, sensor board (line sensors, IR ring, LEDs), motors, kicker. | the Pi's pins |
| `vision.py` | Camera picture to ball, goals, white lines and other robots, in cm from the robot. | camera, `robot_vision_config.json` |
| `localisation.py` | Where the robot is on the field. | `vision.py`, `field.py` |
| `perception.py` | One picture of the game (`World`) from all sensors and the teammate. | `vision.py`, `localisation.py`, `hardware.py`, `comms.py` |
| `comms.py` | Messages between the two robots over Wi-Fi. | the other robot |
| `strategy.py` | What to do on the RCJA field: where to go, where to face, dribble, kick. | `perception.py` |
| `strategy_int.py` | The same for the international field. | `perception.py` |
| `strategy_fallback.py` | Simple play for when the robot does not know where it is. | `perception.py` |
| `motion.py` | Turns "go there, face that way" into four wheel speeds. | the strategy |
| `lines.py` | Is the robot on a white line, and which way is back in. | `hardware.py`, `vision.py`, `localisation.py` |
| `utils.py` | Small maths helpers (angles, filters). | - |

**Tools to run on the Pi** (stop the robot program first: `sudo systemctl stop germinc`)

| File | What it does |
| --- | --- |
| `test_camera.py` | Camera calibration web page. Makes `robot_vision_config.json`. |
| `test_localisation.py` | Web page: what the robot sees and where it thinks it is. |
| `test_drive.py` | Web page: drive the robot by hand, try speeds, dribbler and kicker, see the camera. |
| `test_parts.py` | Tests one part alone: `imu`, `pcb`, `solenoid` or `motor` (motor calibration). |

**Tools to run on a PC** (no robot needed)

| File | What it does |
| --- | --- |
| `simulator.py` | The game in a window, with the real robot code. |
| `sim_int.py` | The simulator on the international field with its rules. |
| `arena.py` | Many simulator games at once, prints the scores. For comparing strategy changes. |
| `faults.py` | Breaks sensors on purpose in the simulator. |
| `past_versions/` | Old copies of the robot code and other teams' strategies, used as opponents in the simulator. |

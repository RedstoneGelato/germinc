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
- Adafruit BNO085 IMU
- Custom PCB:
  - 1x STM32H503RBT6 Microcontroller
  - 32x KT-0603W LEDs
  - 32x ALS-PT19-315C/L177/TR8 LDRs
  - 2x CD74HC4067SM96 Analogue converters

# SETUP & INSTALLATION
On a Raspberry Pi 5 with Raspberry Pi OS installed:
1.	Clone the repository into a directory of choice
2.	Run the setup file that automatically gets all dependencies

# DEPLOYING & USAGE
During competitions, both pi's are setup to automatically run the code on startup. To manually run the code, follow the below:
1. Activate virtual environment
2. Change directory to the folder with the github repository
3. run either main_attack.py (striker robot), or main.py (goalie robot)
4. Flick the switch (wired from GPIO 25 to ground on the Raspberry Pi) to run the code

Note: It is recommended to startup in standby to calibrate the robot's bottom LED and compass

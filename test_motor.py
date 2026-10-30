import time
import sys
import board
import struct
import busio
import select
from steelbar_powerful_bldc_driver import PowerfulBLDCDriver

motor = [None] * 8
motormode = [0] * 8
motorcount = 0
setupmotorcount = 0

tempuint32 = 0
tempint32 = 0
tempfloat = 0.0

selectedmotor = 0
maxspeed = 0
postarget = 0.0
currentlimit = 0
pidconstant = 0.0
boundary = 0.0
pos = 0.0
clearfaults = False

def read_input():
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip()
    return None

i2c = busio.I2C(board.SCL, board.SDA)

print("Please enter the number of motor drivers you want to control:")
tempuint32 = int(input())
if tempuint32 == 0 or tempuint32 > 8:
    print("Error motor count out of range, please reboot microcontroller to try again.")
    quit()
motorcount = tempuint32

setupmotorcount = 0
while setupmotorcount < motorcount:
    print(f"Please enter the i2c address of motor driver number {setupmotorcount}:")
    tempuint32 = int(input())
    if tempuint32 <= 7 or tempuint32 >= 120:
        print("Error invalid i2c address, please reboot microcontroller to try again.")
        quit()
    motor[setupmotorcount] = PowerfulBLDCDriver(i2c, tempuint32)
    
    print(f"The firmware version of motor driver number {setupmotorcount} is: {motor[setupmotorcount].get_firmware_version()}")
    if motor[setupmotorcount].get_firmware_version() != 3:
        print("Error unsupported motor driver version, please check for updates, maybe check wiring and i2c configuration, reboot microcontroller to try again.")
        quit()

    setupmotorcount += 1

setupmotorcount = 0
while setupmotorcount < motorcount:
    motor[setupmotorcount].set_current_limit_foc(65536)
    motor[setupmotorcount].set_id_pid_constants(1500, 200)
    motor[setupmotorcount].set_iq_pid_constants(1500, 200)
    motor[setupmotorcount].set_speed_pid_constants(4e-2, 4e-4, 3e-2)
    motor[setupmotorcount].set_position_pid_constants(275, 0, 0)
    motor[setupmotorcount].set_position_region_boundary(250000)
    motor[setupmotorcount].set_speed_limit(10000000)
    
    motor[setupmotorcount].configure_operating_mode_and_sensor(15, 1)
    motor[setupmotorcount].configure_command_mode(15)
    motor[setupmotorcount].set_calibration_options(300, 2097152, 50000, 500000)
    
    motor[setupmotorcount].start_calibration()
    print(f"Starting calibration of motor {setupmotorcount}")
    while not motor[setupmotorcount].is_calibration_finished():
        print(".", end="")
        sys.stdout.flush()
        time.sleep(0.5)
    print()
    print(f"ELECANGLEOFFSET: {motor[setupmotorcount].get_calibration_ELECANGLEOFFSET()}")
    print(f"SINCOSCENTRE: {motor[setupmotorcount].get_calibration_SINCOSCENTRE()}")

    motor[setupmotorcount].configure_operating_mode_and_sensor(3, 1)
    motor[setupmotorcount].configure_command_mode(12)
    motormode[setupmotorcount] = 12
    
    setupmotorcount += 1

while True:
    userinput = read_input()
    if userinput:
        command = userinput[0]
        param = userinput[1:].strip()
        
        if command == 'n' and param:
            try:
                tempuint32 = int(param)
                if tempuint32 < motorcount:
                    selectedmotor = tempuint32
                    print(f"Selected motor number {selectedmotor}")
                else:
                    raise ValueError('Invalid motor number')
            except ValueError:
                print("Invalid motor number")
        if command == 'm' and param:
            try:
                tempuint32 = int(param)
                if tempuint32 == 2 or tempuint32 == 12 or tempuint32 == 13:
                    motor[selectedmotor].configure_command_mode(tempuint32)
                    motormode[selectedmotor] = tempuint32
                    print(f"Command Mode {tempuint32}")
                else:
                    raise ValueError('Invalid command mode')
            except ValueError:
                print("Invalid command mode")
        elif command == 's' and param:
            try:
                maxspeed = int(param)
                motor[selectedmotor].set_speed_limit(abs(maxspeed))
                if motormode[selectedmotor] == 12:
                    motor[selectedmotor].set_speed(maxspeed)
                print(f"Speed {maxspeed}")
            except ValueError:
                print("Invalid speed value")
        elif command == 'p' and param:
            try:
                postarget = float(param)
                posmsb = int(postarget)
                poslsb = int((postarget * 256) % 256)
                if motormode[selectedmotor] == 13:
                    motor[selectedmotor].set_position(posmsb, poslsb)
                else:
                    print("Motor is not in position mode")
                print(f"Position {postarget}")
            except ValueError:
                print("Invalid position value")
        elif command == 'c' and param:
            try:
                currentlimit = int(param)
                motor[selectedmotor].set_current_limit_foc(abs(currentlimit))
                if motormode[selectedmotor] == 2:
                    motor[selectedmotor].set_torque(currentlimit)
                print(f"Current (Torque) {currentlimit}")
            except ValueError:
                print("Invalid current value")
        elif command == 'k' and param:
            try:
                pidconstant = float(param)
                motor[selectedmotor].set_position_pid_constants(pidconstant, 0, 0)
            except ValueError:
                print("Invalid PID constant value")
        elif command == 'b' and param:
            try:
                boundary = float(param)
                motor[selectedmotor].set_position_region_boundary(boundary)
            except ValueError:
                print("Invalid boundary value")
        elif command == 'f':
            motor[selectedmotor].clear_faults()
            print("Clear Faults")
        elif command == 'd' and param:
            try:
                tempuint32 = int(param)
                print(f"Delay {tempuint32}")
                time.sleep(tempuint32 / 1000)
            except ValueError:
                print("Invalid delay value")
        else:
            print("Unknown command or missing parameter")

    time.sleep(0.001)
    for i in range(motorcount):
        motor[i].update_quick_data_readout()

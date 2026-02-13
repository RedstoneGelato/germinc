#include "Wire.h"
#include "PowerfulBLDCdriver.h"

PowerfulBLDCdriver motor1;

void setup() {
  Serial.begin(115200); // initialise serial
  Wire.begin(); // initialise i2c0, make sure to look up the i2c pins of your microcontroller.
  Wire.setClock(1000000); // set i2c speed to 1MHz
  motor1.begin(25, &Wire); // motor1 has i2c address 25 and is using i2c0.
  Serial.print("The firmware version of the motor driver is: ");
  Serial.println(motor1.getFirmwareVersion()); // get the firmware version and print it to the serial monitor.
  motor1.setCurrentLimitFOC(65536); // set current limit to 1 amp (only works in FOC mode)
  motor1.setIdPidConstants(1500, 200); 
  motor1.setIqPidConstants(1500, 200);
  motor1.setSpeedPidConstants(4e-2, 4e-4, 3e-2); // Constants valid for FOC and Robomaster M2006 P36 motor only, see tuning constants document for more details
  motor1.configureOperatingModeAndSensor(15, 1); // configure calibration mode and sin/cos encoder
  motor1.configureCommandMode(15); // configure calibration mode
  motor1.setCalibrationOptions(300, 2097152, 50000, 500000); // set calibration voltage to 300/3399*vcc volts, speed to 2097152/65536 elecangle/s, settling time to 50000/50000 seconds, calibration time to 500000/50000 seconds
  motor1.startCalibration(); // start the calibration
  while (motor1.isCalibrationFinished() == false) { // wait for the calibration to finish, do call any other motor driver functions while calibration is ongoing
    Serial.print(".");
    delay(500);
  }
  Serial.println(); // print out the calibration results
  Serial.print("ELECANGLEOFFSET:");
  Serial.println(motor1.getCalibrationELECANGLEOFFSET()); 
  Serial.print("SINCOSCENTRE:");
  Serial.println(motor1.getCalibrationSINCOSCENTRE());

  motor1.configureOperatingModeAndSensor(3, 1); // configure FOC mode and sin/cos encoder
  motor1.configureCommandMode(12); // configure speed command mode
  delay(500);
}

void loop() {
  motor1.setSpeed(5000000); // set the motor speed to 5000000/2^16 elec rev/s
  delay(2000);

  motor1.updateQuickDataReadout(); // update quick data readout
  Serial.print("pos:");
  Serial.print(motor1.getPositionQDR()); // get the motor position from quick data readout
  Serial.print(" spd:");
  Serial.print(motor1.getSpeedQDR()); // get the motor speed from quick data readout
  Serial.print(" err1:");
  Serial.print(motor1.getERROR1QDR()); // See documentation for details
  Serial.print(" err2:");
  Serial.println(motor1.getERROR2QDR()); // ERROR2 is not fully implemented in motor driver firmware. Ignore the value of this for now.

  motor1.setSpeed(-5000000); // negative values reverse the direction of the motor.
  delay(2000);

  motor1.updateQuickDataReadout();
  Serial.print("pos:");
  Serial.print(motor1.getPositionQDR());
  Serial.print(" spd:");
  Serial.print(motor1.getSpeedQDR());
  Serial.print(" err1:");
  Serial.print(motor1.getERROR1QDR());
  Serial.print(" err2:");
  Serial.println(motor1.getERROR2QDR());
}
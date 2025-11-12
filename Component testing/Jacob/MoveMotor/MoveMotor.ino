#include <Servo.h>
Servo esc;
const int escPin = 11;

Servo motor;  // Create a Servo object to control the motor
int motorPin = 9;  // PWM signal connected to the white wire

void setup() {
  delay(2000);
  esc.attach(escPin, 1000, 2000);
  delay(1000);
  esc.writeMicroseconds(1300); //ideal speed for test (I think)
  motor.attach(motorPin);  // Attach the motor signal pin

  motor.writeMicroseconds(2000);  // Full turn

  delay(5000);

  motor.writeMicroseconds(1000);

  delay(5000);

  motor.writeMicroseconds(1500);  // Stop
  
  delay(1000);

  esc.writeMicroseconds(1500);
}

void loop() { }
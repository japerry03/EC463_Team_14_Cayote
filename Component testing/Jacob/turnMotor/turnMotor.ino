#include <Servo.h>
Servo esc;
const int escPin = 11;

Servo motor;  // Create a Servo object to control the motor
int motorPin = 9;  // PWM signal connected to the white wire

void setup() {
  delay(2000);
  esc.attach(escPin, 1000, 2000);
  motor.attach(motorPin);  // Attach the motor signal pin
  motor.writeMicroseconds(1500);
  delay(1000);

  esc.writeMicroseconds(1400); //slow start

  delay(1000);

  esc.writeMicroseconds(1300);

  delay(1000);

  esc.writeMicroseconds(1200);
  
  delay(1000);

  esc.writeMicroseconds(1100);

  delay(1000);

  esc.writeMicroseconds(1500);

  delay(3000);

  esc.writeMicroseconds(1600);

  delay(1000);

  esc.writeMicroseconds(1700);

  delay(1000);

  esc.writeMicroseconds(1800);

  delay(1000);

  esc.writeMicroseconds(1900);

  delay(1000);

  esc.writeMicroseconds(1500);
}

void loop() { }
#include <Servo.h>
Servo esc;
const int escPin = 11;

Servo motor;  // Create a Servo object to control the motor
int motorPin = 9;  // PWM signal connected to the white wire

void setup() {
  motor.attach(motorPin);  // Attach the motor signal pin
  delay(2000);
  esc.attach(escPin, 1000, 2000);
  delay(1000);
}

void loop() {
  
}

void drive(){
  esc.writeMicroseconds(1400); //start motor

  delay(1000);

  esc.writeMicroseconds(1350); //midSpeed

  delay(1000);

  esc.writeMicroseconds(1300); //midSpeed
};

void stop(){

  esc.writeMicroseconds(1350);

  delay(1000);

  esc.writeMicroseconds(1400);

  delay(1000);

  esc.writeMicroseconds(1500); //Stop
}

void left(){
  motor.writeMicroseconds(2000);
}

void right(){
  motor.writeMicroseconds(1000);
}

void straight()){
  motor.writeMicroseconds(1500);
}

void emergencyStop(){
  esc.writeMicroseconds(1500);

  delay(1000);

  esc.writeMicroseconds(1600);

  delay(500);

  esc.writeMicroseconds(1500);
}
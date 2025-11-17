#include <Servo.h>
Servo esc;
const int escPin = 11;

Servo motor;  // Create a Servo object to control the motor
int motorPin = 9;  // PWM signal connected to the white wire

// Command input pins from Jetsons
const int b0 = 2;
const int b1 = 3;
const int b2 = 4;

void setup() {
  pinMode(b0, INPUT);
  pinMode(b1, INPUT);
  pinMode(b2, INPUT);

  Serial.begin(9600);
  Serial.println("connected");
  
  motor.attach(motorPin);  // Attach the motor signal pin
  delay(2000);
  esc.attach(escPin, 1000, 2000);
  delay(1000);
}

void loop() {
  int cmd = readCommand();

  Serial.println("looping");

  Serial.print("cmd: "); Serial.println(cmd);

  switch (cmd) {
    case 1: drive(); Serial.println("drive"); break;
    case 2: stop(); Serial.println("stop"); break;
    case 3: left(); Serial.println("left"); break;
    case 4: right(); Serial.println("right"); break;
    case 5: straight(); Serial.println("straight"); break;
    case 6: emergencyStop(); Serial.println("E-brake"); break;
    default: break;
  }

  delay(1000);
}

int readCommand() {
  int bit0 = digitalRead(b0);
  Serial.println(bit0);
  int bit1 = digitalRead(b1);
  Serial.println(bit1);
  int bit2 = digitalRead(b2);
  Serial.println(bit2);

  int cmd = (bit2 << 2) | (bit1 << 1) | bit0; // 3-bit value

  return cmd;
}

//motor commands
void drive(){
  esc.writeMicroseconds(1400); //start motor

  delay(1000);

  esc.writeMicroseconds(1350); //midSpeed

  delay(1000);

  esc.writeMicroseconds(1300); //midSpeed
}

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

void straight(){
  motor.writeMicroseconds(1500);
}

void emergencyStop(){
  esc.writeMicroseconds(1500);

  delay(1000);

  esc.writeMicroseconds(1600);

  delay(500);

  esc.writeMicroseconds(1500);
}
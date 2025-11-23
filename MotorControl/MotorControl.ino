#include <Servo.h>
Servo esc;
const int escPin = 3;

Servo motor;  // Create a Servo object to control the motor
int motorPin = 2;  // PWM signal connected to the white wire

// Command input pins from Jetsons
void setup() {

  Serial.begin(115200);
  
  motor.attach(motorPin);  // Attach the motor signal pin
  delay(2000);
  esc.attach(escPin, 1000, 2000);
  delay(1000);
}

void loop() {
  if (Serial.available() > 0){
    char cmd = Serial.read();

    switch(cmd){
      case 'd': //Drive
        drive(); 
        break;
      case 's': //Stop
        stop(); 
        break;
      case 'e': //Emergencystop
        emergencyStop(); 
        break;
      case 'r': //Right
        right(); 
        break;
      case 'l': //Left
        left(); 
        break;
      case 'f': //Forward
        straight(); 
        break;
      case 'v': //reVerse
        reverse(); 
        break;
      case 'o': //reversestOp
        rstop(); 
        break;
      case 'b': //backward
        rStraight();
        break;
    }
  }
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
  motor.writeMicroseconds(1460);
}

void rStraight(){
  motor.writeMicroseconds(1520);
}

void emergencyStop(){
  esc.writeMicroseconds(1500);

  delay(1000);

  esc.writeMicroseconds(1600);

  delay(500);

  esc.writeMicroseconds(1500);
}

void reverse(){
  esc.writeMicroseconds(1600); //start motor

  delay(1000);

  esc.writeMicroseconds(1650); //midSpeed

  delay(1000);

  esc.writeMicroseconds(1700); //midSpeed
}

void rstop(){
  esc.writeMicroseconds(1650); //start motors

  delay(1000);

  esc.writeMicroseconds(1600); //midSpeed

  delay(1000);

  esc.writeMicroseconds(1500); //stopped
}

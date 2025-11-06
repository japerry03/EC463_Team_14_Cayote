#include <Servo.h>
Servo esc;
const int escPin = 11;
const int throttleMin = 1000;
const int testThrottle = throttleMin + 300; // small steady throttle

void setup() {
  esc.attach(escPin, throttleMin, 2000);
  delay(500);
  esc.writeMicroseconds(throttleMin); // idle
  delay(2000);                        // arm
  esc.writeMicroseconds(testThrottle); // steady test throttle
  delay(5000);                        // measure during this 5s window
  esc.writeMicroseconds(throttleMin); // stop
}

void loop() { }
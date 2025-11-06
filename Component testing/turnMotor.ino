#include <Servo.h>

Servo motor;  // Create a Servo object to control the motor

int motorPin = 9;  // PWM signal connected to the white wire

void setup() {
  motor.attach(motorPin);  // Attach the motor signal pin
  Serial.begin(115200);
  Serial.println("Motor control ready!");
}

void loop() {
  // Simple test: sweep motor from min to max speed
  Serial.println("Motor full forward");
  motor.writeMicroseconds(2000);  // Full throttle
  delay(2000);                     // Run for 2 seconds

  Serial.println("Motor stop");
  motor.writeMicroseconds(1500);  // Stop / neutral
  delay(2000);

  Serial.println("Motor full reverse");
  motor.writeMicroseconds(1000);  // Full reverse (if ESC supports it)
  delay(2000);

  Serial.println("Motor stop");
  motor.writeMicroseconds(1500);  // Stop again
  delay(2000);
}
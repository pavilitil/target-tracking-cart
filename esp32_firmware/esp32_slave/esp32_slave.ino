/**
 * ============================================================================
 * Target Tracking Assistant Cart — ESP32 Low-Level Slave Firmware
 * ============================================================================
 * Hardware:
 *   - MCU: ESP32 DevKit V1 (30-pin)
 *   - Motor Drivers: 2x L298N (ZX-040), 1 driver per side (2 channels paired per board)
 *   - Motors: 4x JGY-370 12V 160 RPM Worm Gear Motors (4WD Differential Drive)
 *   - Safety Sensors: 2x US-025A (Front D1, D2) + 2x HC-SR04 (Side D3, D4)
 *   - Link to Master (Raspberry Pi 4): USB UART @ 115200 baud (/dev/ttyUSB0)
 *
 * Protocol:
 *   - RX from RPi: <CMD,L_PWM,R_PWM>\n   (L_PWM, R_PWM in range -255..255)
 *   - TX to RPi:   <STAT,D1,D2,D3,D4,STATE>\n
 *     where D1=Front-Left(cm), D2=Front-Right(cm), D3=Left(cm), D4=Right(cm),
 *     STATE = OK | OBSTACLE | WATCHDOG | STOP | SENSOR_ERR
 *
 * Key Features:
 *   - 2 kHz Hardware PWM (optimal switching frequency for bipolar L298N H-bridges)
 *   - Slew-Rate Limiter (smooth acceleration & braking for worm gearboxes)
 *   - Active Front Obstacle Interlock (< 35 cm instant forward block)
 *   - 400 ms Command Watchdog (auto-stops if RPi disconnects or hangs)
 *   - Compatible with both ESP32 Arduino Core v2.x and v3.x
 * ============================================================================
 */

#include <Arduino.h>
#include <WiFi.h>

// ============================================================================
// 1. SAFE PINOUT CONFIGURATION (ESP32 DevKit V1 30-Pin)
//    Avoids strapping pins (GPIO 0, 2, 5, 12, 15) & UART0 (GPIO 1, 3)
// ============================================================================

// Left Side Motor Driver (L298N #1: Front-Left on OUT1/2, Rear-Left on OUT3/4)
constexpr uint8_t PIN_L_RPWM = 25;  // Forward PWM -> IN1 + IN3 (jumpered together)
constexpr uint8_t PIN_L_LPWM = 26;  // Reverse PWM -> IN2 + IN4 (jumpered together)
constexpr uint8_t PIN_L_EN   = 27;  // Enable -> ENA + ENB (remove stock jumpers, tie together)

// Right Side Motor Driver (L298N #2: Front-Right on OUT1/2, Rear-Right on OUT3/4)
constexpr uint8_t PIN_R_RPWM = 32;  // Forward PWM -> IN1 + IN3 (jumpered together)
constexpr uint8_t PIN_R_LPWM = 33;  // Reverse PWM -> IN2 + IN4 (jumpered together)
constexpr uint8_t PIN_R_EN   = 13;  // Enable -> ENA + ENB (remove stock jumpers, tie together)

// 4x Ultrasonic Sensors: D1/D2 = US-025A (3.3V direct), D3/D4 = HC-SR04 (5V + 2kOhm series on ECHO)
constexpr uint8_t NUM_SENSORS = 4;

// Index: 0 = Front-Left (D1), 1 = Front-Right (D2), 2 = Left (D3), 3 = Right (D4)
constexpr uint8_t TRIG_PINS[NUM_SENSORS] = {16, 17, 18, 19};
constexpr uint8_t ECHO_PINS[NUM_SENSORS] = {34, 35, 36, 39}; // Input-only GPIOs

// ============================================================================
// 2. CONTROL, PWM & SAFETY CONSTANTS
// ============================================================================
constexpr uint32_t SERIAL_BAUD        = 115200;
constexpr uint32_t PWM_FREQ_HZ        = 2000;  // 2 kHz optimal PWM for bipolar L298N (prevents overheating)
constexpr uint8_t  PWM_RESOLUTION     = 8;     // 8-bit: 0..255
constexpr int16_t  MAX_PWM            = 255;
constexpr int16_t  MIN_STARTING_PWM   = 35;    // Deadzone threshold for 4x JGY-370 via L298N (~2V drop)

// Slew Rate Limiter (smooth ramp for 5-10 kg load + worm gearbox protection)
constexpr uint32_t CONTROL_LOOP_MS    = 10;    // 100 Hz internal motor update loop
constexpr float    SLEW_ACCEL_PER_STEP = 6.0f; // ~600 PWM/sec (~0.42s from 0 to 255)
constexpr float    SLEW_DECEL_PER_STEP = 10.0f;// ~1000 PWM/sec (~0.25s smooth stop)
constexpr float    SLEW_ESTOP_PER_STEP = 45.0f;// Fast ramp-down on obstacle/watchdog

// Safety & Timing Thresholds
constexpr uint16_t OBSTACLE_STOP_CM   = 35;    // Instant stop forward motion if D1 or D2 < 35 cm
constexpr uint16_t MAX_DISTANCE_CM    = 400;   // Sensor cap when no echo returned
constexpr uint32_t PULSE_TIMEOUT_US   = 18000; // ~300 cm max wait (18 ms) to keep loop fast
constexpr uint32_t WATCHDOG_TIMEOUT_MS = 400;  // Stop motors if no command for > 400 ms
constexpr uint32_t SENSOR_INTERVAL_MS = 25;    // Read one HC-SR04 every 25 ms
constexpr uint32_t TELEMETRY_PERIOD_MS = 50;   // Send <STAT,...> at 20 Hz
constexpr uint8_t  SENSOR_FAULT_MISSES = 8;    // Consecutive triggers with no ECHO rising edge -> fault

// Orthogonal trigger order: FL(0) -> L(2) -> FR(1) -> R(3) (50 ms between front sensors)
constexpr uint8_t  SENSOR_ORDER[NUM_SENSORS] = {0, 2, 1, 3};

// LEDC Channel definitions (for ESP32 Core v2.x compatibility)
constexpr uint8_t CH_L_RPWM = 0;
constexpr uint8_t CH_L_LPWM = 1;
constexpr uint8_t CH_R_RPWM = 2;
constexpr uint8_t CH_R_LPWM = 3;

// ============================================================================
// 3. GLOBAL STATE VARIABLES
// ============================================================================
int16_t  targetLeftPWM     = 0;
int16_t  targetRightPWM    = 0;
float    currentLeftPWM    = 0.0f;
float    currentRightPWM   = 0.0f;

uint16_t distancesCm[NUM_SENSORS] = {200, 200, 200, 200};
uint8_t  currentSensorIdx  = 0;
uint8_t  orderPos          = 0;
uint8_t  confirmHits[NUM_SENSORS]     = {0, 0, 0, 0};
uint8_t  missedRiseCount[NUM_SENSORS] = {0, 0, 0, 0};
bool     priorityRePing    = false;
bool     sensorFault       = false;

uint32_t lastCmdTimeMs     = 0;
uint32_t lastControlTimeMs = 0;
uint32_t lastSensorTimeMs  = 0;
uint32_t lastTelemTimeMs   = 0;

bool     watchdogTripped   = true;
bool     obstacleBlocked   = false;

char     rxBuf[48];
uint8_t  rxIdx             = 0;
bool     rxInFrame         = false;

// ============================================================================
// 4. HARDWARE PWM ABSTRACTION (Supports ESP32 Core 2.x and 3.x)
// ============================================================================
void setupPWM() {
#if ESP_ARDUINO_VERSION >= ESP_ARDUINO_VERSION_VAL(3, 0, 0)
  ledcAttach(PIN_L_RPWM, PWM_FREQ_HZ, PWM_RESOLUTION);
  ledcAttach(PIN_L_LPWM, PWM_FREQ_HZ, PWM_RESOLUTION);
  ledcAttach(PIN_R_RPWM, PWM_FREQ_HZ, PWM_RESOLUTION);
  ledcAttach(PIN_R_LPWM, PWM_FREQ_HZ, PWM_RESOLUTION);
#else
  ledcSetup(CH_L_RPWM, PWM_FREQ_HZ, PWM_RESOLUTION);
  ledcSetup(CH_L_LPWM, PWM_FREQ_HZ, PWM_RESOLUTION);
  ledcSetup(CH_R_RPWM, PWM_FREQ_HZ, PWM_RESOLUTION);
  ledcSetup(CH_R_LPWM, PWM_FREQ_HZ, PWM_RESOLUTION);

  ledcAttachPin(PIN_L_RPWM, CH_L_RPWM);
  ledcAttachPin(PIN_L_LPWM, CH_L_LPWM);
  ledcAttachPin(PIN_R_RPWM, CH_R_RPWM);
  ledcAttachPin(PIN_R_LPWM, CH_R_LPWM);
#endif
}

void writePWMChannel(uint8_t pin, uint8_t channel, uint8_t duty) {
#if ESP_ARDUINO_VERSION >= ESP_ARDUINO_VERSION_VAL(3, 0, 0)
  (void)channel;
  ledcWrite(pin, duty);
#else
  (void)pin;
  ledcWrite(channel, duty);
#endif
}

// ============================================================================
// 5. MOTOR DRIVER CONTROL & SLEW RATE LIMITER
// ============================================================================
void setDriversEnabled(bool enabled) {
  digitalWrite(PIN_L_EN, enabled ? HIGH : LOW);
  digitalWrite(PIN_R_EN, enabled ? HIGH : LOW);
}

void applySingleMotorPWM(int16_t pwm, uint8_t rpwmPin, uint8_t rpwmCh, uint8_t lpwmPin, uint8_t lpwmCh) {
  pwm = constrain(pwm, -MAX_PWM, MAX_PWM);

  // Deadband elimination to avoid humming on JGY-370 worm gear motors
  if (abs(pwm) < MIN_STARTING_PWM) {
    writePWMChannel(rpwmPin, rpwmCh, 0);
    writePWMChannel(lpwmPin, lpwmCh, 0);
    return;
  }

  if (pwm > 0) {
    writePWMChannel(lpwmPin, lpwmCh, 0);
    writePWMChannel(rpwmPin, rpwmCh, static_cast<uint8_t>(pwm));
  } else {
    writePWMChannel(rpwmPin, rpwmCh, 0);
    writePWMChannel(lpwmPin, lpwmCh, static_cast<uint8_t>(-pwm));
  }
}

float slewStep(float currentVal, float targetVal, bool emergencyBrake) {
  float diff = targetVal - currentVal;
  if (fabs(diff) < 1.0f) {
    return targetVal;
  }

  float maxStep = SLEW_ACCEL_PER_STEP;
  if (emergencyBrake) {
    maxStep = SLEW_ESTOP_PER_STEP;
  } else if (fabs(targetVal) < fabs(currentVal) || (targetVal * currentVal < 0.0f)) {
    // Decelerating or reversing direction -> use faster deceleration ramp
    maxStep = SLEW_DECEL_PER_STEP;
  }

  if (diff > maxStep) {
    return currentVal + maxStep;
  } else if (diff < -maxStep) {
    return currentVal - maxStep;
  }
  return targetVal;
}

void updateMotors() {
  uint32_t now = millis();

  // 1. Check Watchdog (400 ms timeout)
  if (now - lastCmdTimeMs > WATCHDOG_TIMEOUT_MS) {
    watchdogTripped = true;
    targetLeftPWM   = 0;
    targetRightPWM  = 0;
  } else {
    watchdogTripped = false;
  }

  // 2. Check Front Ultrasonic Sensors (D1 = index 0, D2 = index 1) & Front Sensor Health
  bool frontObstacle = (distancesCm[0] < OBSTACLE_STOP_CM) || (distancesCm[1] < OBSTACLE_STOP_CM);
  sensorFault = (missedRiseCount[0] >= SENSOR_FAULT_MISSES) || (missedRiseCount[1] >= SENSOR_FAULT_MISSES);

  int16_t effectiveLeftTarget  = targetLeftPWM;
  int16_t effectiveRightTarget = targetRightPWM;

  // Clamp any positive wheel target when front obstacle < 35 cm or front sensor disconnected
  // Allows safe reversing or spot-turn negative wheel motion away from the obstacle
  if (frontObstacle || sensorFault) {
    obstacleBlocked = frontObstacle;
    if (effectiveLeftTarget > 0)  effectiveLeftTarget  = 0;
    if (effectiveRightTarget > 0) effectiveRightTarget = 0;
  } else {
    obstacleBlocked = false;
  }

  bool emergencyRamp = watchdogTripped || obstacleBlocked || sensorFault;

  // 3. Apply Slew-Rate Limiting (uses SLEW_ESTOP_PER_STEP = 45.0f on emergency without mechanical shock)
  currentLeftPWM  = slewStep(currentLeftPWM,  static_cast<float>(effectiveLeftTarget),  emergencyRamp);
  currentRightPWM = slewStep(currentRightPWM, static_cast<float>(effectiveRightTarget), emergencyRamp);

  int16_t outL = static_cast<int16_t>(roundf(currentLeftPWM));
  int16_t outR = static_cast<int16_t>(roundf(currentRightPWM));

  // Disable BTS7960 H-bridges at watchdog standstill to save idle current & heat
  bool driversActive = !(watchdogTripped && abs(outL) < MIN_STARTING_PWM && abs(outR) < MIN_STARTING_PWM);
  setDriversEnabled(driversActive);
  applySingleMotorPWM(outL, PIN_L_RPWM, CH_L_RPWM, PIN_L_LPWM, CH_L_LPWM);
  applySingleMotorPWM(outR, PIN_R_RPWM, CH_R_RPWM, PIN_R_LPWM, CH_R_LPWM);
}

// ============================================================================
// 6. ULTRASONIC SENSORS (4x HC-SR04 Non-Blocking Interrupt Sequencer)
// ============================================================================
portMUX_TYPE      echoMux = portMUX_INITIALIZER_UNLOCKED;
volatile uint32_t echoRiseUs[NUM_SENSORS]     = {0, 0, 0, 0};
volatile uint32_t echoDurationUs[NUM_SENSORS] = {0, 0, 0, 0};
volatile bool     echoReady[NUM_SENSORS]      = {false, false, false, false};
volatile bool     echoRisingSeen[NUM_SENSORS] = {false, false, false, false};

void IRAM_ATTR onEchoChange(uint8_t idx) {
  portENTER_CRITICAL_ISR(&echoMux);
  if (digitalRead(ECHO_PINS[idx]) == HIGH) {
    echoRiseUs[idx] = micros();
    echoRisingSeen[idx] = true;
  } else {
    uint32_t dt = micros() - echoRiseUs[idx];
    if (dt < PULSE_TIMEOUT_US) {
      echoDurationUs[idx] = dt;
      echoReady[idx] = true;
    }
  }
  portEXIT_CRITICAL_ISR(&echoMux);
}

void IRAM_ATTR isrEcho0() { onEchoChange(0); }
void IRAM_ATTR isrEcho1() { onEchoChange(1); }
void IRAM_ATTR isrEcho2() { onEchoChange(2); }
void IRAM_ATTR isrEcho3() { onEchoChange(3); }

void updateNextSensor() {
  // 1. Collect result from the previous sensor trigger (0 us blocking delay!)
  bool gotReady = false;
  bool gotRise = false;
  uint32_t dur = 0;

  portENTER_CRITICAL(&echoMux);
  gotReady = echoReady[currentSensorIdx];
  gotRise  = echoRisingSeen[currentSensorIdx];
  dur      = echoDurationUs[currentSensorIdx];
  echoReady[currentSensorIdx] = false;
  echoRisingSeen[currentSensorIdx] = false;
  portEXIT_CRITICAL(&echoMux);

  if (!gotRise) {
    if (missedRiseCount[currentSensorIdx] < 250) {
      missedRiseCount[currentSensorIdx]++;
    }
  } else {
    missedRiseCount[currentSensorIdx] = 0;
  }

  uint16_t rawCm = MAX_DISTANCE_CM;
  if (gotReady) {
    uint16_t cm = static_cast<uint16_t>((dur * 343UL) / 20000UL);
    if (cm >= 2 && cm <= MAX_DISTANCE_CM) {
      rawCm = cm;
    }
  }

  // Anti-spike filter with immediate 25 ms priority re-ping on first < 35 cm hit
  if (gotRise && rawCm < OBSTACLE_STOP_CM) {
    if (confirmHits[currentSensorIdx] < 2) {
      confirmHits[currentSensorIdx]++;
    }
    if (confirmHits[currentSensorIdx] == 1) {
      // First reading < 35 cm: hold old distance and re-ping SAME sensor in next 25 ms slot
      priorityRePing = true;
    } else {
      // Confirmed in 2 consecutive readings (within 25 ms): latch obstacle immediately
      distancesCm[currentSensorIdx] = rawCm;
      priorityRePing = false;
    }
  } else {
    confirmHits[currentSensorIdx] = 0;
    priorityRePing = false;
    if (gotRise) {
      distancesCm[currentSensorIdx] = static_cast<uint16_t>(
        (rawCm * 7 + distancesCm[currentSensorIdx] * 3) / 10
      );
    }
  }

  // 2. Select next sensor (re-ping same sensor if priorityRePing, else orthogonal order)
  if (!priorityRePing) {
    orderPos = (orderPos + 1) % NUM_SENSORS;
    currentSensorIdx = SENSOR_ORDER[orderPos];
  }

  portENTER_CRITICAL(&echoMux);
  echoReady[currentSensorIdx] = false;
  echoRisingSeen[currentSensorIdx] = false;
  portEXIT_CRITICAL(&echoMux);

  digitalWrite(TRIG_PINS[currentSensorIdx], LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PINS[currentSensorIdx], HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PINS[currentSensorIdx], LOW);
}

// ============================================================================
// 7. ZERO-ALLOCATION UART PROTOCOL PARSER & TELEMETRY SENDER
//    Command:   <CMD,L_PWM,R_PWM>\n
//    Telemetry: <STAT,D1,D2,D3,D4,STATE>\n
// ============================================================================
void parseCommandFrame(const char* frame) {
  // Expected content inside '<' and '>': "CMD,L_PWM,R_PWM"
  if (strncmp(frame, "CMD,", 4) != 0) {
    return;
  }

  char* endPtr1 = nullptr;
  long lVal = strtol(frame + 4, &endPtr1, 10);
  if (endPtr1 == (frame + 4) || *endPtr1 != ',') {
    return;
  }

  char* endPtr2 = nullptr;
  long rVal = strtol(endPtr1 + 1, &endPtr2, 10);
  if (endPtr2 == (endPtr1 + 1) || *endPtr2 != '\0') {
    return;
  }

  targetLeftPWM  = constrain(static_cast<int16_t>(lVal), -MAX_PWM, MAX_PWM);
  targetRightPWM = constrain(static_cast<int16_t>(rVal), -MAX_PWM, MAX_PWM);
  lastCmdTimeMs  = millis();
}

void pollSerialInput() {
  while (Serial.available() > 0) {
    char c = static_cast<char>(Serial.read());

    if (c == '<') {
      rxIdx = 0;
      rxInFrame = true;
    } else if (c == '>') {
      if (rxInFrame && rxIdx > 0) {
        rxBuf[rxIdx] = '\0';
        parseCommandFrame(rxBuf);
      }
      rxIdx = 0;
      rxInFrame = false;
    } else if (rxInFrame && c != '\r' && c != '\n') {
      if (rxIdx < sizeof(rxBuf) - 1) {
        rxBuf[rxIdx++] = c;
      } else {
        rxIdx = 0; // Prevent buffer overflow on corrupted stream
        rxInFrame = false;
      }
    }
  }
}

void sendTelemetry() {
  const char* stateStr = "OK";
  if (sensorFault) {
    stateStr = "SENSOR_ERR";
  } else if (obstacleBlocked) {
    stateStr = "OBSTACLE";
  } else if (watchdogTripped) {
    stateStr = "WATCHDOG";
  } else if (targetLeftPWM == 0 && targetRightPWM == 0 &&
             abs(static_cast<int>(currentLeftPWM)) < MIN_STARTING_PWM &&
             abs(static_cast<int>(currentRightPWM)) < MIN_STARTING_PWM) {
    stateStr = "STOP";
  }

  Serial.printf("<STAT,%u,%u,%u,%u,%s>\n",
                distancesCm[0],
                distancesCm[1],
                distancesCm[2],
                distancesCm[3],
                stateStr);
}

// ============================================================================
// 8. ARDUINO SETUP & MAIN LOOP
// ============================================================================
void setup() {
  // Disable unused Wi-Fi & Bluetooth radios to save ~100 mA on Raspberry Pi 4's 1.1A USB rail
  WiFi.mode(WIFI_OFF);
  btStop();

  Serial.begin(SERIAL_BAUD);
  currentSensorIdx = SENSOR_ORDER[0];

  // Configure BTS7960 Enable pins (keep LOW during initialization)
  pinMode(PIN_L_EN, OUTPUT);
  pinMode(PIN_R_EN, OUTPUT);
  setDriversEnabled(false);

  // Configure 20 kHz PWM channels
  setupPWM();
  writePWMChannel(PIN_L_RPWM, CH_L_RPWM, 0);
  writePWMChannel(PIN_L_LPWM, CH_L_LPWM, 0);
  writePWMChannel(PIN_R_RPWM, CH_R_RPWM, 0);
  writePWMChannel(PIN_R_LPWM, CH_R_LPWM, 0);

  // Configure 4x HC-SR04 Ultrasonic Sensor pins with non-blocking interrupts
  for (uint8_t i = 0; i < NUM_SENSORS; i++) {
    pinMode(TRIG_PINS[i], OUTPUT);
    digitalWrite(TRIG_PINS[i], LOW);
    pinMode(ECHO_PINS[i], INPUT);
  }
  attachInterrupt(digitalPinToInterrupt(ECHO_PINS[0]), isrEcho0, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ECHO_PINS[1]), isrEcho1, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ECHO_PINS[2]), isrEcho2, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ECHO_PINS[3]), isrEcho3, CHANGE);

  setDriversEnabled(true);
  lastCmdTimeMs = millis();
}

void loop() {
  // 1. Read incoming UART commands immediately
  pollSerialInput();

  uint32_t now = millis();

  // 2. Round-robin HC-SR04 measurement (every 25 ms -> 10 Hz per sensor)
  if (now - lastSensorTimeMs >= SENSOR_INTERVAL_MS) {
    lastSensorTimeMs = now;
    updateNextSensor();
  }

  // 3. Motor Slew-Rate, Safety & Watchdog loop (100 Hz / every 10 ms)
  if (now - lastControlTimeMs >= CONTROL_LOOP_MS) {
    lastControlTimeMs = now;
    updateMotors();
  }

  // 4. Send Telemetry frame to Raspberry Pi (20 Hz / every 50 ms)
  if (now - lastTelemTimeMs >= TELEMETRY_PERIOD_MS) {
    lastTelemTimeMs = now;
    sendTelemetry();
  }
}


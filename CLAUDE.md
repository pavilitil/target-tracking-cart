# CLAUDE.md — Target Tracking Assistant Cart (RPi4 Master + ESP32 Slave)

## Project Overview
Two-loop autonomous target-tracking assistant cart (follows a person at 1.0–1.5 m, carries 5–10 kg, avoids obstacles < 35 cm).
- **Master:** Raspberry Pi 4 (Debian Bookworm 64-bit, Python 3.13, ARM Cortex-A72) + CSI camera Waveshare OV5647 (`60.6°` FOV, `640x480 @ 30 FPS`).
- **Slave:** ESP32 DevKit V1 (30-pin) connected via USB UART (`/dev/ttyUSB0 @ 115200 baud`), driving 2× BTS7960 (20 kHz PWM, JGY-370 12V 160 RPM worm gear motors) and reading 4× HC-SR04 ultrasonic sensors.

## Token-Saving Rules for Agent
- **NEVER read or search inside `venv/`, `models/`, or `__pycache__/`.**
- Run Python commands ONLY via `./venv/bin/python` (created with `--system-site-packages` for `picamera2`).
- Run verification tests after changes with: `./venv/bin/python test_system.py`

## File Map (Read only what you edit)
- `config.py` — `CartConfig` dataclass (camera, calibration `distance_calib_k=350.0`, Dual PID gains, UART, thresholds).
- `vision_tracker.py` — `CameraStream` (`Picamera2` -> `OpenCV` -> `DEMO_SIMULATOR`), `PersonTracker` (`YOLOv8n` / `MediaPipe`), `draw_hud_overlay()`.
- `pid_controller.py` — `PIDController` and `DualPIDCartController` (Angular steering PID + Linear distance PID + slew-rate limiter + ultrasonic safety layer).
- `uart_comm.py` — `UARTBridge` background thread (`<CMD,L_PWM,R_PWM>\n` TX @ 25 Hz, `<STAT,D1,D2,D3,D4,STATE>\n` RX, hot-plug & simulation fallback).
- `app.py` — `RobotCartSupervisor` (30 FPS control loop) + Flask server (`:5000`, `/video_feed`, `/api/telemetry`, `/api/mode`, `/api/manual`, `/api/pid`, `/api/demo`).
- `templates/index.html`, `static/app.js`, `static/style.css` — Mobile Web UI (MJPEG stream, mode switcher, virtual touch joystick, D-pad, live telemetry).
- `esp32_firmware/esp32_slave.ino` — ESP32 C++ firmware (20 kHz LEDC PWM, slew-rate limiter, `< 35 cm` front obstacle block, `400 ms` watchdog).
- `test_system.py` — Automated test suite for Dual PID, UART protocol, and Flask endpoints.


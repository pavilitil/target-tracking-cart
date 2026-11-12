"""
Centralized configuration for the Target Tracking Assistant Cart (Master — Raspberry Pi 4).
Hardware profile:
  - Master: Raspberry Pi 4 (Debian Bookworm 64-bit)
  - Camera: Waveshare OV5647 CSI (60.6° FOV, non-fisheye), 640x480 @ 30 FPS
  - Slave:  ESP32 DevKit V1 (30-pin) over USB UART (/dev/ttyUSB0 @ 115200 baud)
  - Drive:  Differential drive, 2x JGY-370 12V 160 RPM worm gear motors + 2x BTS7960 43A
  - Safety: 4x HC-SR04 ultrasonic sensors (2 front, 1 left, 1 right)
"""

from dataclasses import dataclass


@dataclass
class CartConfig:
    # --- Camera & Vision ---
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30
    camera_fov_deg: float = 60.6  # Waveshare OV5647 standard lens

    # AI Detector backend priority: "yolov8n" -> "mediapipe" -> "hog" -> "demo"
    preferred_detector: str = "yolov8n"
    yolo_model_path: str = "yolov8n.pt"
    yolo_conf_threshold: float = 0.42
    yolo_imgsz: int = 320  # Fast inference resolution on Raspberry Pi 4 CPU

    # --- Distance Estimation & Target Zone (1.0m - 1.5m) ---
    # Empirical optical constant K for OV5647 (60.6° FOV) at 640x480:
    # distance_m = distance_calib_k / box_height_px
    #   box_height = 350 px -> 1.00 m (minimum follow distance)
    #   box_height = 280 px -> 1.25 m (optimal target setpoint)
    #   box_height = 233 px -> 1.50 m (outer follow threshold)
    distance_calib_k: float = 350.0
    target_distance_min_m: float = 1.00
    target_distance_max_m: float = 1.50
    target_distance_setpoint_m: float = 1.25

    target_box_height_px: int = 280     # Setpoint at 1.25 m
    box_height_deadband_px: int = 22    # Comfort zone (~1.15m..1.36m) for smooth hold
    max_close_box_height_px: int = 355  # Closer than ~0.98m -> gentle back-off or full stop

    # --- Angular PID (Steering by horizontal error cx - frame_width/2) ---
    steer_kp: float = 0.42
    steer_ki: float = 0.015
    steer_kd: float = 0.085
    steer_deadband_px: int = 18         # Avoid worm-gear hunting when person is centered
    steer_max_pwm: int = 135            # Max differential steering effort

    # --- Linear PID (Speed by box_height error: target_box_height - box_height) ---
    speed_kp: float = 0.90
    speed_ki: float = 0.020
    speed_kd: float = 0.140
    speed_max_forward_pwm: int = 185    # Safe cruising speed with 5-10 kg load
    speed_max_reverse_pwm: int = 85     # Gentle reverse if person steps too close (< 0.95 m)

    # Smooth acceleration / deceleration limits on RPi side (PWM units per control tick @ 30Hz)
    soft_ramp_accel_step: float = 20.0
    soft_ramp_decel_step: float = 28.0
    target_lost_timeout_s: float = 0.65 # Start smooth ramp-down to 0 after losing person

    # --- Motor PWM Limits ---
    max_pwm: int = 255
    min_effective_pwm: int = 28         # Minimum PWM to overcome JGY-370 worm gear friction

    # --- UART Communication (RPi <-> ESP32) ---
    serial_port: str = "/dev/ttyUSB0"
    serial_baudrate: int = 115200
    serial_timeout_s: float = 0.05
    uart_tx_rate_hz: int = 25           # Send <CMD,L,R> every 40 ms (well inside 400ms watchdog)

    # --- Ultrasonic Safety Thresholds (HC-SR04) ---
    obstacle_stop_cm: int = 35          # Hard stop forward movement (< 35 cm)
    obstacle_slow_cm: int = 75          # Proportional speed reduction (35..75 cm)
    side_nudge_cm: int = 40             # Gentle steering assist away from side walls

    # --- Web UI Server ---
    flask_host: str = "0.0.0.0"
    flask_port: int = 5000
    mjpeg_quality: int = 78


CONFIG = CartConfig()


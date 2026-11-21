"""
Centralized configuration for the Target Tracking Assistant Cart (Master — Raspberry Pi 4).
Hardware profile:
  - Master: Raspberry Pi 4 (Debian Bookworm 64-bit)
  - Camera: Waveshare OV5647 CSI (60.6° FOV, non-fisheye), 640x480 @ 30 FPS
  - Slave:  ESP32 DevKit V1 (30-pin) over USB UART (/dev/ttyUSB0 @ 115200 baud)
  - Drive:  Differential drive, 2x JGY-370 12V 160 RPM worm gear motors + 2x BTS7960 43A
  - Safety: 4x HC-SR04 ultrasonic sensors (2 front, 1 left, 1 right)

Tunable PID gains / calibration are persisted to `settings.json` (see
`CartConfig.load_from_json()` / `CartConfig.save_to_json()`) and auto-loaded at start-up.
"""

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Mapping, Tuple

logger = logging.getLogger("CartConfig")

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = "settings.json"


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
    # Exported NCNN model directory inside models/ (preferred over .pt on ARM64 CPU):
    #   yolo export model=models/yolov8n.pt format=ncnn imgsz=320
    yolo_ncnn_model_dir: str = "yolov8n_ncnn_model"
    yolo_conf_threshold: float = 0.42
    yolo_imgsz: int = 320  # Fast inference resolution on Raspberry Pi 4 CPU
    torch_num_threads: int = 2  # Cap PyTorch CPU threads to prevent 4-core thermal runaway

    # --- Smart Thermal Guard & Eco Pacing (Raspberry Pi 4 without/with fan) ---
    thermal_warn_c: float = 70.0        # Enter THERMAL_ECO above 70°C
    thermal_protect_c: float = 78.0     # Enter THERMAL_PROTECT above 78°C
    ai_max_fps_normal: float = 12.0     # Max YOLO inference rate below 70°C (Alpha-Beta fills 30 FPS)
    ai_max_fps_eco: float = 6.0         # Max YOLO inference rate at 70..78°C
    ai_max_fps_protect: float = 3.5     # Max YOLO inference rate >= 78°C
    ai_max_fps_idle: float = 2.0        # Eco AI rate when mode is STOP or MANUAL

    # --- Target Alpha-Beta Filter & Occlusion Handling ---
    track_cx_alpha: float = 0.60        # Position gain for horizontal centre (cx)
    track_cx_beta: float = 0.20         # Velocity gain for cx
    track_bh_alpha: float = 0.50        # Position gain for bounding-box height
    track_bh_beta: float = 0.12         # Velocity gain for bounding-box height
    track_predict_horizon_s: float = 0.30   # Max extrapolation between AI results
    occlusion_decay_tau_s: float = 0.45     # Velocity / command decay while target is occluded
    detection_stale_timeout_s: float = 1.0  # No AI result for this long -> treat target as lost

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
    # "Hold still" band around the setpoint (box smaller by <= far / bigger by <= near):
    # with target 280 px this is 252..345 px (~1.39 m .. ~1.01 m)
    hold_band_far_px: int = 28
    hold_band_near_px: int = 65

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
    # Short occlusion window: keep the previous motion vector (decaying) while the person is
    # hidden for less than this; afterwards ramp smoothly down to 0.
    target_lost_timeout_s: float = 0.40

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

    # --- Persisted tunables (settings.json): field name -> (min, max) ---
    PERSISTED_SETTINGS: ClassVar[Dict[str, Tuple[float, float]]] = {
        "steer_kp": (0.0, 10.0),
        "steer_ki": (0.0, 10.0),
        "steer_kd": (0.0, 10.0),
        "speed_kp": (0.0, 10.0),
        "speed_ki": (0.0, 10.0),
        "speed_kd": (0.0, 10.0),
        "target_box_height_px": (120, 440),
        "distance_calib_k": (100.0, 1000.0),
    }

    @staticmethod
    def _settings_path(path: str) -> str:
        """Relative paths are resolved next to config.py (independent of the working dir)."""
        return path if os.path.isabs(path) else os.path.join(_BASE_DIR, path)

    def settings_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.PERSISTED_SETTINGS}

    @classmethod
    def validate_settings(cls, data: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Returns (clean_values, errors). Unknown keys and None values are ignored;
        non-numeric or out-of-range values are reported in `errors` and skipped.
        """
        clean: Dict[str, Any] = {}
        errors: List[str] = []
        normalized: Dict[str, Any] = dict(data)
        if "target_box_height_px" not in normalized and "target_box_height" in normalized:
            normalized["target_box_height_px"] = normalized["target_box_height"]

        for name, (lo, hi) in cls.PERSISTED_SETTINGS.items():
            raw = normalized.get(name)
            if raw is None:
                continue
            try:
                if isinstance(raw, bool):
                    raise ValueError("bool")
                value = float(raw)
            except (TypeError, ValueError):
                errors.append(f"{name}: {raw!r} is not a number")
                continue
            if not math.isfinite(value) or not (lo <= value <= hi):
                errors.append(f"{name}: {raw!r} is outside {lo:g}..{hi:g}")
                continue
            is_int_field = type(getattr(cls, name)) is int
            clean[name] = int(round(value)) if is_int_field else value
        return clean, errors

    def load_from_json(self, path: str = SETTINGS_FILE) -> bool:
        """Loads persisted tunables. Returns True if the file was read and applied."""
        full_path = self._settings_path(path)
        try:
            with open(full_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return False
        except (OSError, ValueError) as exc:
            logger.warning("Cannot read %s (%s) - keeping default settings", full_path, exc)
            return False
        if not isinstance(data, dict):
            logger.warning("%s is not a JSON object - keeping default settings", full_path)
            return False

        clean, errors = self.validate_settings(data)
        for err in errors:
            logger.warning("Ignoring invalid value in %s - %s", full_path, err)
        for name, value in clean.items():
            setattr(self, name, value)
        logger.info("Loaded %d setting(s) from %s", len(clean), full_path)
        return True

    def save_to_json(self, path: str = SETTINGS_FILE) -> bool:
        """Atomically writes persisted tunables (temp file + fsync + rename). Skips disk write if unchanged."""
        full_path = self._settings_path(path)
        current_data = self.settings_dict()
        try:
            if os.path.exists(full_path):
                with open(full_path, "r", encoding="utf-8") as existing_fh:
                    if json.load(existing_fh) == current_data:
                        return True
        except Exception:
            pass
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                prefix=".settings_", suffix=".tmp", dir=os.path.dirname(full_path)
            )
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(current_data, fh, indent=2)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp_path, 0o644)
            os.replace(tmp_path, full_path)
            return True
        except OSError as exc:
            logger.error("Cannot save settings to %s: %s", full_path, exc)
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            return False


CONFIG = CartConfig()
CONFIG.load_from_json(SETTINGS_FILE)

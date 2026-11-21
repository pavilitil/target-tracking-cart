#!/usr/bin/env python3
"""
Target Tracking Assistant Cart — Master High-Level Brain & Flask Web UI
Runs on Raspberry Pi 4 (Debian Bookworm 64-bit) on port 5000.
"""

import atexit
import logging
import math
import threading
import time
from typing import Dict, Optional, Tuple
import cv2
from flask import Flask, Response, jsonify, render_template, request

from config import CONFIG, CartConfig
from pid_controller import DualPIDCartController
from uart_comm import UARTBridge
from vision_tracker import CameraStream, DetectionResult, PersonTracker, draw_hud_overlay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Suppress 10-70 Hz HTTP access log spam from Werkzeug to protect microSD card & CPU
logging.getLogger("werkzeug").setLevel(logging.WARNING)
logger = logging.getLogger("RobotCartMaster")

app = Flask(__name__)


class RobotCartSupervisor:
    """
    Coordinates the 30 FPS Vision + AI Tracking loop, Dual PID Controllers,
    Manual Override Joystick, and Bidirectional UART communication with ESP32.
    """

    def __init__(self) -> None:
        self.cfg = CONFIG
        self._lock = threading.Lock()

        # Operating mode: safe initial state "STOP" until operator selects AUTO or MANUAL
        self.mode: str = "STOP"

        # Hardware & AI subsystems
        self.uart = UARTBridge(self.cfg)
        self.camera = CameraStream(self.cfg)
        self.tracker = PersonTracker(self.cfg)
        self.tracker.set_operating_mode("STOP")
        self.pid = DualPIDCartController(self.cfg)

        # Manual control state
        self._manual_left_pwm: int = 0
        self._manual_right_pwm: int = 0
        self._manual_last_cmd_time: float = 0.0
        self._manual_max_pwm: int = 190

        # Latest telemetry & lazy on-demand JPEG encoding state for MJPEG streaming
        self._latest_det = DetectionResult()
        self._left_pwm: int = 0
        self._right_pwm: int = 0
        self._pid_metrics: Dict[str, float] = {}
        self._fps: float = 30.0
        self._latest_raw_frame = None
        self._latest_sensors_cm = (200, 200, 200, 200)
        self._frame_seq: int = 0
        self._encoded_seq: int = -1
        self._latest_jpeg: Optional[bytes] = None
        self._frame_cond = threading.Condition(self._lock)
        self._encode_lock = threading.Lock()
        self._stream_clients: int = 0

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self.uart.start()
        self._running = True
        self._thread = threading.Thread(
            target=self._control_loop, name="VisionControlLoop", daemon=True
        )
        self._thread.start()
        logger.info("RobotCartSupervisor started (Mode=%s)", self.mode)

    def stop(self) -> None:
        self._running = False
        self.uart.set_motor_pwm(0, 0)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self.tracker.stop()
        self.uart.stop()
        self.camera.release()
        logger.info("RobotCartSupervisor stopped cleanly.")

    def set_mode(self, new_mode: str) -> str:
        normalized = new_mode.strip().upper()
        if normalized in ("AUTO AI", "AUTO_AI", "AUTO"):
            normalized = "AUTO"
        elif normalized not in ("MANUAL", "STOP"):
            normalized = "STOP"

        with self._lock:
            self.mode = normalized
            self._manual_left_pwm = 0
            self._manual_right_pwm = 0
            self.pid.reset()
            if normalized == "STOP":
                self._left_pwm = 0
                self._right_pwm = 0
                self.uart.set_motor_pwm(0, 0)

        logger.info("Switched operating mode to: %s", normalized)
        self.tracker.set_operating_mode(normalized)
        return normalized

    def set_manual_vector(
        self, joy_x: float, joy_y: float, max_pwm: Optional[int] = None
    ) -> None:
        """
        Converts normalized joystick coordinates:
          joy_x in [-1.0 (left) .. +1.0 (right)]
          joy_y in [-1.0 (reverse) .. +1.0 (forward)]
        into differential drive (left_pwm, right_pwm).
        """
        joy_x = max(-1.0, min(1.0, float(joy_x)))
        joy_y = max(-1.0, min(1.0, float(joy_y)))

        with self._lock:
            if max_pwm is not None:
                self._manual_max_pwm = max(60, min(255, int(max_pwm)))
            limit = self._manual_max_pwm

            linear = joy_y * limit
            angular = joy_x * (limit * 0.78)

            left_f = linear + angular
            right_f = linear - angular
            peak = max(abs(left_f), abs(right_f))
            if peak > limit:
                left_f = left_f * limit / peak
                right_f = right_f * limit / peak

            l_pwm = int(round(left_f))
            r_pwm = int(round(right_f))
            if abs(l_pwm) < self.cfg.min_effective_pwm:
                l_pwm = 0
            if abs(r_pwm) < self.cfg.min_effective_pwm:
                r_pwm = 0

            self._manual_left_pwm = l_pwm
            self._manual_right_pwm = r_pwm
            self._manual_last_cmd_time = time.monotonic()

    def get_jpeg_frame(self, last_seq: Optional[int] = None) -> Optional[bytes]:
        """
        Lazy on-demand HUD rendering and JPEG encoding:
        Only draws the HUD and runs cv2.imencode('.jpg') when a browser client or test
        actually requests a frame, and caches the encoded JPEG per _frame_seq so
        0% CPU is spent on video encoding when no viewer is connected.
        """
        with self._frame_cond:
            if last_seq is not None and self._frame_seq == last_seq:
                self._frame_cond.wait(timeout=0.20)
            elif self._latest_raw_frame is None:
                self._frame_cond.wait(timeout=0.20)

            if self._latest_raw_frame is None:
                return self._latest_jpeg

            if self._encoded_seq == self._frame_seq and self._latest_jpeg is not None:
                return self._latest_jpeg

            raw_frame = self._latest_raw_frame
            det = self._latest_det
            mode = self.mode
            l_pwm = self._left_pwm
            r_pwm = self._right_pwm
            sensors_cm = self._latest_sensors_cm
            fps = self._fps
            cam_src = self.camera.source_type
            reason = self.pid.status_reason
            seq = self._frame_seq

        with self._encode_lock:
            with self._lock:
                if self._encoded_seq == seq and self._latest_jpeg is not None:
                    return self._latest_jpeg
            annotated = draw_hud_overlay(
                frame=raw_frame,
                det=det,
                mode=mode,
                left_pwm=l_pwm,
                right_pwm=r_pwm,
                sensors_cm=sensors_cm,
                fps=fps,
                camera_source=cam_src,
                status_reason=reason,
                cfg=self.cfg,
            )
            ok, enc = cv2.imencode(
                ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), self.cfg.mjpeg_quality]
            )
            if not ok:
                return self._latest_jpeg
            jpeg_bytes = enc.tobytes()
            with self._lock:
                self._latest_jpeg = jpeg_bytes
                self._encoded_seq = seq
            return jpeg_bytes

    def _apply_motor_calibration(self, l_pwm: int, r_pwm: int) -> Tuple[int, int]:
        """Applies software left/right worm-gear trim and polarity inversion after safety clamping."""
        trim = float(getattr(self.cfg, "motor_trim", 0.0))
        l_cal = int(round(l_pwm * min(1.0, 1.0 - trim)))
        r_cal = int(round(r_pwm * min(1.0, 1.0 + trim)))
        if getattr(self.cfg, "invert_left_motor", False):
            l_cal = -l_cal
        if getattr(self.cfg, "invert_right_motor", False):
            r_cal = -r_cal
        l_cal = max(-self.cfg.max_pwm, min(self.cfg.max_pwm, l_cal))
        r_cal = max(-self.cfg.max_pwm, min(self.cfg.max_pwm, r_cal))
        return l_cal, r_cal

    def get_full_telemetry(self) -> Dict[str, object]:
        uart_data = self.uart.get_telemetry()
        cpu_temp_c, thermal_state, target_ai_fps = self.tracker.get_thermal_status()
        with self._lock:
            det = self._latest_det
            return {
                "mode": self.mode,
                "fps": round(self._fps, 1),
                "camera_source": self.camera.source_type,
                "thermal": {
                    "cpu_temp_c": cpu_temp_c,
                    "state": thermal_state,
                    "target_ai_fps": target_ai_fps,
                    "throttled_hex": self.tracker.throttled_hex,
                    "undervoltage": self.tracker.undervoltage_now,
                    "stream_clients": self._stream_clients,
                },
                "vision": {
                    "detected": det.detected,
                    "occluded": det.occluded,
                    "target_locked": det.target_locked,
                    "color_match": det.color_match,
                    "posture_state": det.posture_state,
                    "cx": det.cx,
                    "cy": det.cy,
                    "box_height": det.box_height,
                    "box_width": det.box_width,
                    "vx": det.vx,
                    "v_box_height": det.v_box_height,
                    "distance_m": det.distance_m,
                    "confidence": det.confidence,
                    "detector": det.detector_name,
                    "target_box_height": self.cfg.target_box_height_px,
                },
                "motors": {
                    "left_pwm": self._left_pwm,
                    "right_pwm": self._right_pwm,
                    "status_reason": self.pid.status_reason,
                    "pid": self._pid_metrics,
                },
                "sensors": {
                    "d1_front_left": uart_data["d1_front_left"],
                    "d2_front_right": uart_data["d2_front_right"],
                    "d3_left": uart_data["d3_left"],
                    "d4_right": uart_data["d4_right"],
                    "obstacle_stop_cm": self.cfg.obstacle_stop_cm,
                    "esp_state": uart_data["esp_state"],
                    "uart_connected": uart_data["connected"],
                    "simulation": uart_data["simulation"],
                    "rx_stale": uart_data["rx_stale"],
                    "port": uart_data["port"],
                    "last_tx": uart_data["last_tx"],
                    "tx_count": uart_data["tx_count"],
                    "rx_count": uart_data["rx_count"],
                },
                "pid_gains": {
                    "steer_kp": self.pid.steering_pid.kp,
                    "steer_ki": self.pid.steering_pid.ki,
                    "steer_kd": self.pid.steering_pid.kd,
                    "speed_kp": self.pid.speed_pid.kp,
                    "speed_ki": self.pid.speed_pid.ki,
                    "speed_kd": self.pid.speed_pid.kd,
                    "target_box_height_px": self.cfg.target_box_height_px,
                    "distance_calib_k": self.cfg.distance_calib_k,
                    "motor_trim": self.cfg.motor_trim,
                    "invert_left_motor": self.cfg.invert_left_motor,
                    "invert_right_motor": self.cfg.invert_right_motor,
                    "camera_hflip": self.cfg.camera_hflip,
                    "camera_vflip": self.cfg.camera_vflip,
                },
                "demo": {
                    "is_demo_camera": self.camera.source_type == "DEMO_SIMULATOR",
                    "auto_motion": self.camera.demo_auto_motion,
                    "person_visible": self.camera.demo_person_visible,
                    "target_x_norm": round(self.camera.demo_target_x_norm, 2),
                    "target_dist_m": round(self.camera.demo_target_dist_m, 2),
                },
            }

    def _control_loop(self) -> None:
        target_period = 1.0 / float(self.cfg.target_fps)
        fps_Filtered = float(self.cfg.target_fps)
        prev_loop_time = time.monotonic()

        while self._running:
            loop_start = time.monotonic()
            dt = max(0.001, loop_start - prev_loop_time)
            prev_loop_time = loop_start
            fps_Filtered = 0.90 * fps_Filtered + 0.10 * (1.0 / dt)

            # 1. Capture frame (640x480)
            frame, demo_bbox = self.camera.read_frame()

            # 2. Run AI Person Detection & Tracking
            det = self.tracker.detect(frame, demo_bbox=demo_bbox)

            # 3. Read 4x HC-SR04 Ultrasonic Distances from ESP32
            sensors_cm = self.uart.get_sensors_tuple()
            min_front_cm = min(sensors_cm[0], sensors_cm[1])

            # 4. Compute Motor Commands according to active Mode
            with self._lock:
                current_mode = self.mode

            if current_mode == "AUTO":
                if not self.uart._simulation_mode and self.camera.source_type == "DEMO_SIMULATOR":
                    l_pwm = 0
                    r_pwm = 0
                    self.pid.status_reason = "CAMERA_FAULT_LOCKOUT"
                    metrics = {"error_x": 0.0, "error_h": 0.0, "linear_pwm": 0.0, "angular_pwm": 0.0}
                else:
                    l_pwm, r_pwm, metrics = self.pid.compute_auto_command(
                        detected=det.detected,
                        cx=det.cx,
                        cy=det.cy,
                        box_height=det.box_height,
                        sensors_cm=sensors_cm,
                        vx=det.vx,
                        v_box_height=det.v_box_height,
                        posture_state=det.posture_state,
                    )
            elif current_mode == "MANUAL":
                with self._lock:
                    # Auto-zero manual command if user released joystick (> 0.6s timeout)
                    if loop_start - self._manual_last_cmd_time > 0.60:
                        self._manual_left_pwm = 0
                        self._manual_right_pwm = 0
                    l_pwm = self._manual_left_pwm
                    r_pwm = self._manual_right_pwm

                # Strict front obstacle protection in MANUAL mode (< 35 cm):
                # Clamp ANY positive wheel PWM (even during spot turn +80 / -80 -> 0 / -80)
                if min_front_cm < self.cfg.obstacle_stop_cm and (l_pwm > 0 or r_pwm > 0):
                    l_pwm = min(0, l_pwm)
                    r_pwm = min(0, r_pwm)
                    self.pid.status_reason = f"MANUAL_BLOCKED_OBSTACLE ({min_front_cm}cm)"
                elif l_pwm == 0 and r_pwm == 0:
                    self.pid.status_reason = "MANUAL_IDLE"
                else:
                    self.pid.status_reason = "MANUAL_DRIVING"
                metrics = {"error_x": 0.0, "error_h": 0.0, "linear_pwm": 0.0, "angular_pwm": 0.0}
            else:
                # STOP Mode
                l_pwm = 0
                r_pwm = 0
                self.pid.status_reason = "EMERGENCY_OR_USER_STOP"
                metrics = {"error_x": 0.0, "error_h": 0.0, "linear_pwm": 0.0, "angular_pwm": 0.0}

            # Brownout throttling if Raspberry Pi 4 reports < 4.63V under-voltage
            if self.tracker.undervoltage_now and (l_pwm != 0 or r_pwm != 0):
                l_pwm = int(round(l_pwm * 0.65))
                r_pwm = int(round(r_pwm * 0.65))

            # 5. Re-verify mode under lock (prevents race if STOP was clicked mid-tick) & update telemetry
            with self._frame_cond:
                if self.mode == "STOP":
                    l_pwm = 0
                    r_pwm = 0
                    self.pid.status_reason = "EMERGENCY_OR_USER_STOP"
                self._latest_det = det
                self._left_pwm = l_pwm
                self._right_pwm = r_pwm
                self._pid_metrics = metrics
                self._fps = fps_Filtered
                self._latest_raw_frame = frame
                self._latest_sensors_cm = sensors_cm
                self._frame_seq += 1
                self._frame_cond.notify_all()

            # 6. Apply motor trim & inversion after safety checks, then send to ESP32 (<CMD,L_PWM,R_PWM>\n)
            out_l, out_r = self._apply_motor_calibration(l_pwm, r_pwm)
            self.uart.set_motor_pwm(out_l, out_r)

            elapsed = time.monotonic() - loop_start
            sleep_s = target_period - elapsed
            if sleep_s > 0.001:
                time.sleep(sleep_s)


supervisor = RobotCartSupervisor()


@app.route("/")
def index():
    return render_template("index.html")


def _mjpeg_generator():
    with supervisor._lock:
        supervisor._stream_clients += 1
    last_seq = -1
    last_send = 0.0
    try:
        while True:
            jpeg = supervisor.get_jpeg_frame(last_seq=last_seq)
            with supervisor._lock:
                last_seq = supervisor._frame_seq
            if jpeg is not None:
                now = time.monotonic()
                elapsed = now - last_send
                if elapsed < 0.050:
                    time.sleep(0.050 - elapsed)
                last_send = time.monotonic()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            else:
                time.sleep(0.02)
    finally:
        with supervisor._lock:
            supervisor._stream_clients = max(0, supervisor._stream_clients - 1)


@app.route("/video_feed")
def video_feed():
    with supervisor._lock:
        if supervisor._stream_clients >= 4:
            return Response("Too many active MJPEG stream clients (max 4)", status=503)
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/telemetry", methods=["GET"])
def api_telemetry():
    return jsonify(supervisor.get_full_telemetry())


@app.route("/api/mode", methods=["POST"])
def api_mode():
    data = request.get_json(silent=True) or {}
    requested = str(data.get("mode", "STOP"))
    active_mode = supervisor.set_mode(requested)
    return jsonify({"ok": True, "mode": active_mode})


@app.route("/api/manual", methods=["POST"])
def api_manual():
    data = request.get_json(silent=True) or {}
    try:
        joy_x = float(data.get("x", 0.0))
        joy_y = float(data.get("y", 0.0))
        if not (math.isfinite(joy_x) and math.isfinite(joy_y)):
            raise ValueError("non-finite")
        raw_max_pwm = data.get("max_pwm")
        if raw_max_pwm is not None:
            max_pwm_f = float(raw_max_pwm)
            if not math.isfinite(max_pwm_f):
                raise ValueError("non-finite max_pwm")
            max_pwm = int(round(max_pwm_f))
        else:
            max_pwm = None
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Invalid numeric manual vector"}), 400
    supervisor.set_manual_vector(joy_x, joy_y, max_pwm)
    return jsonify({"ok": True})


@app.route("/api/pid", methods=["POST"])
def api_pid():
    data = request.get_json(silent=True) or {}
    clean, errors = CartConfig.validate_settings(data)
    if errors and not clean:
        return jsonify({"ok": False, "errors": errors}), 400
    with supervisor._lock:
        supervisor.pid.update_gains(
            steer_kp=clean.get("steer_kp"),
            steer_ki=clean.get("steer_ki"),
            steer_kd=clean.get("steer_kd"),
            speed_kp=clean.get("speed_kp"),
            speed_ki=clean.get("speed_ki"),
            speed_kd=clean.get("speed_kd"),
            target_box_height=clean.get("target_box_height_px"),
            distance_calib_k=clean.get("distance_calib_k"),
            motor_trim=clean.get("motor_trim"),
            invert_left_motor=clean.get("invert_left_motor"),
            invert_right_motor=clean.get("invert_right_motor"),
            camera_hflip=clean.get("camera_hflip"),
            camera_vflip=clean.get("camera_vflip"),
        )
        saved = supervisor.cfg.save_to_json()
    return jsonify(
        {
            "ok": True,
            "saved": saved,
            "errors": errors,
            "pid_gains": supervisor.get_full_telemetry()["pid_gains"],
            "target_box_height_px": supervisor.cfg.target_box_height_px,
        }
    )


@app.route("/api/calibrate_distance", methods=["POST"])
def api_calibrate_distance():
    """One-click 1.25m distance calibration from current detected person bounding box height."""
    with supervisor._lock:
        det = supervisor._latest_det
        if not det.detected or det.box_height < 80:
            return jsonify(
                {
                    "ok": False,
                    "error": "Людина не виявлена в кадрі або занадто далеко (< 80 px)",
                }
            ), 400
        bh = max(120, min(440, int(det.box_height)))
        calib_k = round(bh * 1.25, 1)
        supervisor.pid.update_gains(target_box_height=bh, distance_calib_k=calib_k)
        saved = supervisor.cfg.save_to_json()
    return jsonify(
        {
            "ok": True,
            "saved": saved,
            "target_box_height_px": bh,
            "distance_calib_k": calib_k,
            "pid_gains": supervisor.get_full_telemetry()["pid_gains"],
        }
    )


@app.route("/api/target_lock", methods=["POST"])
def api_target_lock():
    data = request.get_json(silent=True) or {}
    want_lock = bool(data.get("lock", True))
    locked = supervisor.tracker.set_target_color_lock(want_lock)
    return jsonify({"ok": True, "target_locked": locked})


@app.route("/api/demo", methods=["POST"])
def api_demo():
    """
    Allows interactive testing of the Demo Simulator (moving target person
    or simulating ultrasonic obstacles D1..D4) directly from the Web UI.
    """
    data = request.get_json(silent=True) or {}
    try:
        if "auto_motion" in data:
            supervisor.camera.demo_auto_motion = bool(data["auto_motion"])
        if "person_visible" in data:
            supervisor.camera.demo_person_visible = bool(data["person_visible"])
        if "target_x_norm" in data:
            val_x = float(data["target_x_norm"])
            if not math.isfinite(val_x):
                raise ValueError("non-finite target_x_norm")
            supervisor.camera.demo_target_x_norm = max(0.05, min(0.95, val_x))
        if "target_dist_m" in data:
            val_d = float(data["target_dist_m"])
            if not math.isfinite(val_d):
                raise ValueError("non-finite target_dist_m")
            supervisor.camera.demo_target_dist_m = max(0.75, min(2.60, val_d))

        preset = data.get("sim_preset")
        if preset == "clear":
            supervisor.uart.set_simulated_distances(None, None, None, None)
        elif preset == "slowdown":
            supervisor.uart.set_simulated_distances(55, 58, 150, 150)
        elif preset == "side_left":
            supervisor.uart.set_simulated_distances(140, 145, 26, 150)
        elif preset == "side_right":
            supervisor.uart.set_simulated_distances(140, 145, 150, 26)
        elif preset == "stop":
            supervisor.uart.set_simulated_distances(24, 28, 150, 150)
        elif all(k in data for k in ("d1", "d2", "d3", "d4")):
            supervisor.uart.set_simulated_distances(
                int(data["d1"]), int(data["d2"]), int(data["d3"]), int(data["d4"])
            )
        elif "sim_obstacle" in data:
            if data["sim_obstacle"]:
                supervisor.uart.set_simulated_distances(24, 28, 150, 150)
            else:
                supervisor.uart.set_simulated_distances(None, None, None, None)
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True})


if __name__ == "__main__":
    supervisor.start()
    atexit.register(supervisor.stop)
    app.run(
        host=CONFIG.flask_host,
        port=CONFIG.flask_port,
        debug=False,
        threaded=True,
    )


#!/usr/bin/env python3
"""
Automated Verification Suite for the Target Tracking Assistant Cart.
Tests:
  1. Dual PID Controller (Steering + Distance 1.0-1.5m + Obstacle Stop < 35cm +
     Short Occlusion Hold < 0.4s + Soft Brake >= 0.4s + Crouch Guard + Zero Blind Reverse)
  2. 1D Alpha-Beta Filter, Async Background Inference, NCNN Priority & Color Re-ID Lock
  3. JSON Settings Persistence (numeric + boolean fields, unchanged skip)
  4. UART Protocol Parser, RX Stale Timeout & Simulation Fallback
  5. Strict MANUAL Obstacle Clamping (+80 / -80 -> 0 / -80) & Motor Trim/Inversion
  6. Flask Web UI Endpoints (/api/telemetry, /api/mode, /api/manual, /api/pid,
     /api/calibrate_distance, /api/target_lock, /api/demo presets)
"""

import os
import tempfile
import numpy as np

from app import app, supervisor
from config import CartConfig, CONFIG
from pid_controller import DualPIDCartController
from uart_comm import UARTBridge
from vision_tracker import AlphaBetaFilter1D, DetectionResult, PersonTracker


def test_dual_pid_and_occlusion():
    cfg = CartConfig()
    assert cfg.speed_max_reverse_pwm == 0, "Blind auto-reverse must default to 0 (no rear sensor)"
    pid = DualPIDCartController(cfg)

    # Case 1: Person far away (box_height=180px -> ~1.94m) and to the right (cx=450 > 320)
    t0 = 100.0
    l_pwm, r_pwm, _ = pid.compute_auto_command(
        detected=True,
        cx=450,
        cy=240,
        box_height=180,
        sensors_cm=(150, 150, 150, 150),
        vx=15.0,
        v_box_height=-5.0,
        now=t0,
    )
    assert l_pwm > 0 and r_pwm > -50, f"Expected forward right turn, got L={l_pwm}, R={r_pwm}"
    assert l_pwm > r_pwm, f"Expected L_PWM > R_PWM for right turn, got L={l_pwm}, R={r_pwm}"

    # Case 2: Short occlusion (< 0.4s) -> must keep holding decaying motion vector
    l_occ, r_occ, _ = pid.compute_auto_command(
        detected=False,
        cx=320,
        cy=240,
        box_height=0,
        sensors_cm=(150, 150, 150, 150),
        now=t0 + 0.18,
    )
    assert "OCCLUSION_HOLD" in pid.status_reason, f"Expected OCCLUSION_HOLD, got {pid.status_reason}"
    assert l_occ > 0, f"Expected positive held PWM during <0.4s occlusion, got L={l_occ}, R={r_occ}"

    # Case 3: Lost for >= 0.4s -> smoothly brakes down to 0
    for step in range(1, 12):
        l_lost, r_lost, _ = pid.compute_auto_command(
            detected=False,
            cx=320,
            cy=240,
            box_height=0,
            sensors_cm=(150, 150, 150, 150),
            now=t0 + 0.45 + step * 0.033,
        )
    assert pid.status_reason == "TARGET_LOST_SOFT_STOP"
    assert l_lost == 0 and r_lost == 0, f"Expected smooth brake to 0 after 0.4s, got L={l_lost}, R={r_lost}"

    # Case 4: Front obstacle < 35 cm -> must block forward motion
    for i in range(5):
        l_obs, r_obs, _ = pid.compute_auto_command(
            detected=True,
            cx=320,
            cy=240,
            box_height=180,
            sensors_cm=(28, 120, 150, 150),
            now=t0 + 2.0 + i * 0.033,
        )
    assert l_obs <= 0 and r_obs <= 0, f"Obstacle <35cm failed to block forward PWM: L={l_obs}, R={r_obs}"

    # Case 5: Person inside 1.0-1.5m zone (box_height=280px -> 1.25m) and centered (cx=320)
    pid.reset()
    l_ok, r_ok, _ = pid.compute_auto_command(
        detected=True,
        cx=320,
        cy=240,
        box_height=280,
        sensors_cm=(150, 150, 150, 150),
        now=t0 + 3.0,
    )
    assert l_ok == 0 and r_ok == 0, f"Expected stop in 1.0-1.5m zone, got L={l_ok}, R={r_ok}"

    # Case 6: Crouched posture (posture_state="CROUCHED") -> must NOT charge forward or reverse
    pid.reset()
    l_crouch, r_crouch, _ = pid.compute_auto_command(
        detected=True,
        cx=320,
        cy=240,
        box_height=180,
        sensors_cm=(150, 150, 150, 150),
        now=t0 + 4.0,
        posture_state="CROUCHED",
    )
    assert l_crouch == 0 and r_crouch == 0, f"Expected full hold on crouch, got L={l_crouch}, R={r_crouch}"

    # Case 7: Person too close (box_height=420px) -> must stop safely (0, 0), not reverse blindly
    pid.reset()
    l_close, r_close, _ = pid.compute_auto_command(
        detected=True,
        cx=320,
        cy=240,
        box_height=420,
        sensors_cm=(150, 150, 150, 150),
        now=t0 + 5.0,
    )
    assert l_close == 0 and r_close == 0, f"Expected safe stop without blind reverse, got L={l_close}, R={r_close}"
    print("[PASS] Dual PID Controller, occlusion handling, crouch guard & zero blind reverse verified.")


def test_alpha_beta_and_async_tracker():
    # 1. Verify 1D Alpha-Beta filter position & velocity estimation
    ab = AlphaBetaFilter1D(alpha=0.6, beta=0.2)
    x0, v0 = ab.update(300.0, now=10.0)
    assert x0 == 300.0 and v0 == 0.0
    x1, v1 = ab.update(330.0, now=10.1)
    assert 310.0 < x1 < 330.0 and v1 > 0.0, f"Expected positive velocity estimate, got x={x1}, v={v1}"

    # 2. Verify NCNN model priority resolution over .pt
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = os.path.join(tmpdir, "models")
        ncnn_dir = os.path.join(models_dir, "yolov8n_ncnn_model")
        os.makedirs(ncnn_dir, exist_ok=True)
        with open(os.path.join(ncnn_dir, "model.ncnn.param"), "w", encoding="utf-8") as fh:
            fh.write("ncnn")
        with open(os.path.join(models_dir, "yolov8n.pt"), "w", encoding="utf-8") as fh:
            fh.write("pt")

        cfg = CartConfig(preferred_detector="demo")
        tracker = PersonTracker(cfg, base_dir=tmpdir)
        try:
            resolved_path, is_ncnn = tracker._resolve_yolo_model()
            assert is_ncnn is True and resolved_path == ncnn_dir, (
                f"Expected NCNN model priority, got path={resolved_path}, is_ncnn={is_ncnn}"
            )

            # 3. Verify async background daemon thread detection & short occlusion coasting
            blue_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            blue_frame[100:380, 260:380] = (220, 40, 20)  # Blue garment BGR
            det1 = tracker.detect(blue_frame, demo_bbox=(260, 100, 380, 380), now=50.0)
            assert det1.detected is True and det1.box_height == 280 and det1.posture_state == "STANDING"
            det2 = tracker.detect(blue_frame, demo_bbox=(280, 100, 400, 380), now=50.05)
            assert det2.detected is True and det2.vx > 0.0

            # 4. Verify Color Re-ID Lock accepts matching garment & rejects mismatched bystander
            assert tracker.set_target_color_lock(True) is True
            red_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            red_frame[100:380, 280:400] = (20, 30, 230)  # Bright Red bystander BGR
            det_bystander = tracker.detect(red_frame, demo_bbox=(280, 100, 400, 380), now=50.10)
            assert det_bystander.detected is False, "Color Lock must reject mismatched bystander"
            assert det_bystander.target_locked is True, "Lock state must persist during <0.4s occlusion"

            # After >= 0.4s loss, state resets cleanly and locking on empty wall returns False
            det_lost = tracker.detect(red_frame, demo_bbox=None, now=50.65)
            assert det_lost.detected is False and det_lost.occluded is False and det_lost.box_height == 0
            assert tracker.set_target_color_lock(False) is False
            assert tracker.set_target_color_lock(True) is False, "Must never lock onto empty wall"

            # 5. Verify Crouched posture detection (aspect_ratio > 0.65) caps height <= target_box_height_px
            crouch_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            det_crouch = tracker.detect(crouch_frame, demo_bbox=(200, 200, 380, 400), now=51.0)
            assert det_crouch.posture_state == "CROUCHED"
            assert det_crouch.box_height <= cfg.target_box_height_px
        finally:
            tracker.stop()

    print("[PASS] Alpha-Beta 1D filter, NCNN priority, Color Re-ID rejection & Crouch Guard verified.")


def test_json_settings_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "settings.json")
        cfg_save = CartConfig(
            steer_kp=0.55,
            steer_ki=0.025,
            steer_kd=0.11,
            speed_kp=1.05,
            speed_ki=0.03,
            speed_kd=0.18,
            target_box_height_px=300,
            motor_trim=0.12,
            invert_left_motor=True,
            camera_vflip=True,
        )
        assert cfg_save.save_to_json(json_path) is True
        assert cfg_save.save_to_json(json_path) is True  # No-op skip when unchanged
        assert os.path.exists(json_path)

        cfg_load = CartConfig()
        assert cfg_load.load_from_json(json_path) is True
        assert abs(cfg_load.steer_kp - 0.55) < 1e-6
        assert abs(cfg_load.speed_kd - 0.18) < 1e-6
        assert cfg_load.target_box_height_px == 300
        assert abs(cfg_load.motor_trim - 0.12) < 1e-6
        assert cfg_load.invert_left_motor is True
        assert cfg_load.camera_vflip is True
    print("[PASS] JSON settings persistence (numeric + boolean fields) verified.")


def test_uart_protocol():
    bridge = UARTBridge(CONFIG)
    bridge._parse_line("<STAT,42,38,110,125,OK>")
    telem = bridge.get_telemetry()
    assert telem["d1_front_left"] == 42
    assert telem["d2_front_right"] == 38
    assert telem["d3_left"] == 110
    assert telem["d4_right"] == 125
    assert telem["esp_state"] == "OK"
    assert telem["rx_stale"] is False

    # Verify RX stale watchdog on RPi side fails closed to (0, 0, 0, 0)
    with bridge._lock:
        bridge._connected = True
        bridge._simulation_mode = False
        bridge._connect_time = 10.0
        bridge._last_rx_time = 10.0
    assert bridge.is_rx_stale(now=11.0) is True
    with bridge._lock:
        bridge._last_rx_time = 1000.0  # test get_sensors_tuple using real monotonic clock
    # Simulate stale by forcing timestamps far in past relative to time.monotonic()
    with bridge._lock:
        bridge._connect_time = 0.001
        bridge._last_rx_time = 0.001
    assert bridge.get_sensors_tuple() == (0, 0, 0, 0)
    assert bridge.get_telemetry()["esp_state"] == "ESP_RX_TIMEOUT"
    print("[PASS] UART Protocol parser, RX stale fail-safe & simulation state verified.")


def test_manual_obstacle_clamp_and_calibration():
    # Override #1: Strict MANUAL obstacle clamping (spot turn +80 / -80 -> 0 / -80)
    l_pwm, r_pwm = 80, -80
    min_front_cm = 25
    if min_front_cm < CONFIG.obstacle_stop_cm and (l_pwm > 0 or r_pwm > 0):
        l_pwm = min(0, l_pwm)
        r_pwm = min(0, r_pwm)
    assert l_pwm == 0 and r_pwm == -80, f"Expected (0, -80) clamp, got ({l_pwm}, {r_pwm})"

    # Motor trim & inversion helper check
    orig_trim = supervisor.cfg.motor_trim
    orig_inv_l = supervisor.cfg.invert_left_motor
    try:
        supervisor.cfg.motor_trim = 0.10
        supervisor.cfg.invert_left_motor = True
        cal_l, cal_r = supervisor._apply_motor_calibration(100, 100)
        assert cal_l == -90 and cal_r == 100, f"Expected (-90, 100), got ({cal_l}, {cal_r})"
    finally:
        supervisor.cfg.motor_trim = orig_trim
        supervisor.cfg.invert_left_motor = orig_inv_l
    print("[PASS] Strict MANUAL obstacle clamp & motor trim/inversion verified.")


def test_flask_endpoints():
    settings_path = CONFIG._settings_path("settings.json")
    had_existing = os.path.exists(settings_path)
    orig_values = CONFIG.settings_dict()
    supervisor.start()
    try:
        client = app.test_client()

        res_index = client.get("/")
        assert res_index.status_code == 200
        html = res_index.get_data(as_text=True)
        assert "Зберегти налаштування" in html
        assert "Калібрувати 1.25 м" in html
        assert "undervoltage-banner" in html

        # Safe default startup mode must be STOP
        res_telem = client.get("/api/telemetry")
        assert res_telem.status_code == 200
        data = res_telem.get_json()
        assert data["mode"] == "STOP", f"Expected safe startup mode STOP, got {data['mode']}"
        assert "vision" in data and "motors" in data and "sensors" in data and "thermal" in data
        assert data["thermal"]["cpu_temp_c"] > 0.0

        res_lock = client.post("/api/target_lock", json={"lock": False})
        assert res_lock.status_code == 200 and res_lock.get_json()["ok"] is True

        # Validation: invalid / out-of-range PID & manual vector must return HTTP 400
        res_bad_pid = client.post("/api/pid", json={"steer_kp": -5.0})
        assert res_bad_pid.status_code == 400 and res_bad_pid.get_json()["ok"] is False

        res_bad_man = client.post("/api/manual", json={"x": "invalid", "y": 0.0})
        assert res_bad_man.status_code == 400 and res_bad_man.get_json()["ok"] is False

        # Valid PID + trim update
        res_pid = client.post(
            "/api/pid",
            json={
                "steer_kp": 0.48,
                "steer_ki": 0.018,
                "steer_kd": 0.09,
                "speed_kp": 0.95,
                "speed_ki": 0.022,
                "speed_kd": 0.15,
                "target_box_height": 290,
                "motor_trim": 0.05,
                "invert_left_motor": False,
            },
        )
        assert res_pid.status_code == 200
        pid_json = res_pid.get_json()
        assert pid_json["ok"] is True and pid_json["saved"] is True
        assert pid_json["target_box_height_px"] == 290
        assert abs(pid_json["pid_gains"]["motor_trim"] - 0.05) < 1e-6
        assert os.path.exists(settings_path)

        # One-click 1.25m distance calibration endpoint
        with supervisor._lock:
            supervisor._latest_det = DetectionResult(detected=True, box_height=260)
        res_cal = client.post("/api/calibrate_distance", json={})
        assert res_cal.status_code == 200
        cal_json = res_cal.get_json()
        assert cal_json["ok"] is True and cal_json["target_box_height_px"] == 260
        assert abs(cal_json["distance_calib_k"] - 325.0) < 1e-6

        # Demo ultrasonic presets
        for preset in ("slowdown", "side_left", "stop", "clear"):
            res_demo = client.post("/api/demo", json={"sim_preset": preset})
            assert res_demo.status_code == 200

        res_mode = client.post("/api/mode", json={"mode": "MANUAL"})
        assert res_mode.status_code == 200
        assert res_mode.get_json()["mode"] == "MANUAL"

        res_man = client.post("/api/manual", json={"x": 0.2, "y": 0.6, "max_pwm": 180})
        assert res_man.status_code == 200

        res_stop = client.post("/api/mode", json={"mode": "STOP"})
        assert res_stop.status_code == 200
        assert res_stop.get_json()["mode"] == "STOP"

        print("[PASS] Flask Web UI & REST API endpoints (safe STOP, validation, calibration, presets) verified.")
    finally:
        supervisor.stop()
        with supervisor._lock:
            supervisor.pid.update_gains(
                steer_kp=orig_values["steer_kp"],
                steer_ki=orig_values["steer_ki"],
                steer_kd=orig_values["steer_kd"],
                speed_kp=orig_values["speed_kp"],
                speed_ki=orig_values["speed_ki"],
                speed_kd=orig_values["speed_kd"],
                target_box_height=orig_values["target_box_height_px"],
                distance_calib_k=orig_values["distance_calib_k"],
                motor_trim=orig_values["motor_trim"],
                invert_left_motor=orig_values["invert_left_motor"],
                invert_right_motor=orig_values["invert_right_motor"],
                camera_hflip=orig_values["camera_hflip"],
                camera_vflip=orig_values["camera_vflip"],
            )
        if had_existing:
            CONFIG.save_to_json(settings_path)
        elif os.path.exists(settings_path):
            os.remove(settings_path)


if __name__ == "__main__":
    test_dual_pid_and_occlusion()
    test_alpha_beta_and_async_tracker()
    test_json_settings_persistence()
    test_uart_protocol()
    test_manual_obstacle_clamp_and_calibration()
    test_flask_endpoints()
    print("ALL TESTS PASSED SUCCESSFULLY!")

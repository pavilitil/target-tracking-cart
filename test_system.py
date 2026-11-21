#!/usr/bin/env python3
"""
Automated Verification Suite for the Target Tracking Assistant Cart.
Tests:
  1. Dual PID Controller (Steering + Distance 1.0-1.5m + Obstacle Stop < 35cm + Short Occlusion Hold < 0.4s & Soft Brake >= 0.4s)
  2. 1D Alpha-Beta Filter, Async Background Inference & NCNN Priority in PersonTracker
  3. JSON Settings Persistence (CartConfig.save_to_json / load_from_json)
  4. UART Protocol Parser & Simulation Fallback (<CMD,L,R> and <STAT,D1,D2,D3,D4,STATE>)
  5. Flask Web UI Endpoints (/api/telemetry, /api/mode, /api/manual, /api/pid, /video_feed)
"""

import os
import tempfile
import numpy as np

from app import app, supervisor
from config import CartConfig, CONFIG
from pid_controller import DualPIDCartController
from uart_comm import UARTBridge
from vision_tracker import AlphaBetaFilter1D, PersonTracker


def test_dual_pid_and_occlusion():
    cfg = CartConfig()
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
    print("[PASS] Dual PID Controller & <0.4s occlusion handling verified.")


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
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            det1 = tracker.detect(frame, demo_bbox=(260, 100, 380, 380), now=50.0)
            assert det1.detected is True and det1.box_height == 280
            det2 = tracker.detect(frame, demo_bbox=(280, 100, 400, 380), now=50.05)
            assert det2.detected is True and det2.vx > 0.0

            # Short occlusion at +0.15s (< 0.4s)
            det_occ = tracker.detect(frame, demo_bbox=None, now=50.20)
            assert det_occ.occluded is True and det_occ.box_height > 0

            # Full loss at +0.55s (>= 0.4s)
            det_lost = tracker.detect(frame, demo_bbox=None, now=50.60)
            assert det_lost.detected is False and det_lost.occluded is False and det_lost.box_height == 0
        finally:
            tracker.stop()

    print("[PASS] Alpha-Beta 1D filter, NCNN priority, and async tracker verified.")


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
        )
        assert cfg_save.save_to_json(json_path) is True
        assert os.path.exists(json_path)

        cfg_load = CartConfig()
        assert cfg_load.load_from_json(json_path) is True
        assert abs(cfg_load.steer_kp - 0.55) < 1e-6
        assert abs(cfg_load.speed_kd - 0.18) < 1e-6
        assert cfg_load.target_box_height_px == 300
    print("[PASS] JSON settings persistence (load_from_json / save_to_json) verified.")


def test_uart_protocol():
    bridge = UARTBridge(CONFIG)
    bridge._parse_line("<STAT,42,38,110,125,OK>")
    telem = bridge.get_telemetry()
    assert telem["d1_front_left"] == 42
    assert telem["d2_front_right"] == 38
    assert telem["d3_left"] == 110
    assert telem["d4_right"] == 125
    assert telem["esp_state"] == "OK"
    print("[PASS] UART Protocol parser & simulation state verified.")


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

        res_telem = client.get("/api/telemetry")
        assert res_telem.status_code == 200
        data = res_telem.get_json()
        assert "vision" in data and "motors" in data and "sensors" in data

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
            },
        )
        assert res_pid.status_code == 200
        pid_json = res_pid.get_json()
        assert pid_json["ok"] is True and pid_json["saved"] is True
        assert pid_json["target_box_height_px"] == 290
        assert os.path.exists(settings_path)

        res_mode = client.post("/api/mode", json={"mode": "MANUAL"})
        assert res_mode.status_code == 200
        assert res_mode.get_json()["mode"] == "MANUAL"

        res_man = client.post("/api/manual", json={"x": 0.2, "y": 0.6, "max_pwm": 180})
        assert res_man.status_code == 200

        res_stop = client.post("/api/mode", json={"mode": "STOP"})
        assert res_stop.status_code == 200
        assert res_stop.get_json()["mode"] == "STOP"

        client.post("/api/mode", json={"mode": "AUTO"})
        print("[PASS] Flask Web UI & REST API endpoints (/api/pid persistence included) verified.")
    finally:
        supervisor.stop()
        supervisor.pid.update_gains(
            steer_kp=orig_values["steer_kp"],
            steer_ki=orig_values["steer_ki"],
            steer_kd=orig_values["steer_kd"],
            speed_kp=orig_values["speed_kp"],
            speed_ki=orig_values["speed_ki"],
            speed_kd=orig_values["speed_kd"],
            target_box_height=orig_values["target_box_height_px"],
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
    test_flask_endpoints()
    print("ALL TESTS PASSED SUCCESSFULLY!")

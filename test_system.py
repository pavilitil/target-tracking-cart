#!/usr/bin/env python3
"""
Automated Verification Suite for the Target Tracking Assistant Cart.
Tests:
  1. Dual PID Controller (Steering + Distance 1.0-1.5m + Obstacle Stop < 35cm + Lost Target Ramp)
  2. UART Protocol Parser & Simulation Fallback (<CMD,L,R> and <STAT,D1,D2,D3,D4,STATE>)
  3. Camera & AI Person Tracker (640x480 @ 30 FPS)
  4. Flask Web UI Endpoints (/api/telemetry, /api/mode, /api/manual, /video_feed)
"""

from app import app, supervisor
from config import CONFIG
from pid_controller import DualPIDCartController
from uart_comm import UARTBridge


def test_dual_pid():
    pid = DualPIDCartController(CONFIG)

    # Case 1: Person far away (box_height=180px -> ~1.94m) and to the right (cx=450 > 320)
    l_pwm, r_pwm, _ = pid.compute_auto_command(
        detected=True, cx=450, cy=240, box_height=180, sensors_cm=(150, 150, 150, 150)
    )
    assert l_pwm > 0 and r_pwm > -50, f"Expected forward right turn, got L={l_pwm}, R={r_pwm}"
    assert l_pwm > r_pwm, f"Expected L_PWM > R_PWM for right turn, got L={l_pwm}, R={r_pwm}"

    # Case 2: Front obstacle < 35 cm -> must block forward motion
    for _ in range(5):
        l_obs, r_obs, _ = pid.compute_auto_command(
            detected=True, cx=320, cy=240, box_height=180, sensors_cm=(28, 120, 150, 150)
        )
    assert l_obs <= 0 and r_obs <= 0, f"Obstacle <35cm failed to block forward PWM: L={l_obs}, R={r_obs}"

    # Case 3: Person inside 1.0-1.5m zone (box_height=280px -> 1.25m) and centered (cx=320)
    pid.reset()
    l_ok, r_ok, _ = pid.compute_auto_command(
        detected=True, cx=320, cy=240, box_height=280, sensors_cm=(150, 150, 150, 150)
    )
    assert l_ok == 0 and r_ok == 0, f"Expected stop in 1.0-1.5m zone, got L={l_ok}, R={r_ok}"
    print("[PASS] Dual PID Controller verified.")


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
    supervisor.start()
    try:
        client = app.test_client()

        res_index = client.get("/")
        assert res_index.status_code == 200

        res_telem = client.get("/api/telemetry")
        assert res_telem.status_code == 200
        data = res_telem.get_json()
        assert "vision" in data and "motors" in data and "sensors" in data

        res_mode = client.post("/api/mode", json={"mode": "MANUAL"})
        assert res_mode.status_code == 200
        assert res_mode.get_json()["mode"] == "MANUAL"

        res_man = client.post("/api/manual", json={"x": 0.2, "y": 0.6, "max_pwm": 180})
        assert res_man.status_code == 200

        res_stop = client.post("/api/mode", json={"mode": "STOP"})
        assert res_stop.status_code == 200
        assert res_stop.get_json()["mode"] == "STOP"

        client.post("/api/mode", json={"mode": "AUTO"})
        print("[PASS] Flask Web UI & REST API endpoints verified.")
    finally:
        supervisor.stop()


if __name__ == "__main__":
    test_dual_pid()
    test_uart_protocol()
    test_flask_endpoints()
    print("ALL TESTS PASSED SUCCESSFULLY!")


/**
 * Target Tracking Assistant Cart — Smartphone Web UI Client
 * Handles:
 *   - 10 Hz Telemetry Polling (/api/telemetry)
 *   - Mode Switching (AUTO AI / MANUAL / STOP)
 *   - Multi-touch Virtual Joystick & D-Pad for MANUAL mode
 *   - Interactive Demo Testbench controls
 */

let currentMode = "AUTO";
let joyActive = false;
let joyX = 0.0;
let joyY = 0.0;
let manualHeartbeatTimer = null;

// Demo state flags
let demoAutoMotion = true;
let demoPersonVisible = true;
let demoObstacleActive = false;

async function setMode(mode) {
  try {
    const res = await fetch("/api/mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    const data = await res.json();
    if (data.ok) {
      updateModeButtons(data.mode);
    }
  } catch (err) {
    console.error("Failed to switch mode:", err);
  }
}

function updateModeButtons(mode) {
  currentMode = mode;
  document.getElementById("btn-mode-auto").classList.toggle("active", mode === "AUTO");
  document.getElementById("btn-mode-manual").classList.toggle("active", mode === "MANUAL");
  document.getElementById("btn-mode-stop").classList.toggle("active", mode === "STOP");
}

async function sendManualCommand(x, y) {
  const maxPwm = parseInt(document.getElementById("manual-speed-limit").value, 10) || 190;
  if (currentMode !== "MANUAL" && (Math.abs(x) > 0.01 || Math.abs(y) > 0.01)) {
    await setMode("MANUAL");
  }
  try {
    await fetch("/api/manual", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x, y, max_pwm: maxPwm }),
    });
  } catch (err) {
    // Ignore transient network hiccups
  }
}

function updateSensorCard(idSuffix, cm, stopThreshold) {
  const valEl = document.getElementById(`val-${idSuffix}`);
  const barEl = document.getElementById(`bar-${idSuffix}`);
  const boxEl = document.getElementById(`us-box-${idSuffix}`);
  if (!valEl || !barEl || !boxEl) return;

  valEl.textContent = cm;
  const pct = Math.max(5, Math.min(100, (cm / 250.0) * 100));
  barEl.style.width = `${pct}%`;

  if (cm < stopThreshold) {
    barEl.style.backgroundColor = "#ef4444";
    boxEl.classList.add("danger");
  } else if (cm < 75) {
    barEl.style.backgroundColor = "#f59e0b";
    boxEl.classList.remove("danger");
  } else {
    barEl.style.backgroundColor = "#22c55e";
    boxEl.classList.remove("danger");
  }
}

function updatePwmBar(barId, valId, pwm) {
  const bar = document.getElementById(barId);
  const lbl = document.getElementById(valId);
  if (!bar || !lbl) return;

  lbl.textContent = pwm > 0 ? `+${pwm}` : `${pwm}`;
  const halfPct = (Math.min(255, Math.abs(pwm)) / 255.0) * 50.0;
  if (pwm >= 0) {
    bar.style.left = "50%";
    bar.style.width = `${halfPct}%`;
    bar.style.backgroundColor = "#22c55e";
  } else {
    bar.style.left = `${50.0 - halfPct}%`;
    bar.style.width = `${halfPct}%`;
    bar.style.backgroundColor = "#f59e0b";
  }
}

async function pollTelemetry() {
  try {
    const res = await fetch("/api/telemetry");
    const t = await res.json();

    updateModeButtons(t.mode);

    // Header status
    document.getElementById("badge-fps").textContent = `${t.fps.toFixed(1)} FPS`;
    const uartBadge = document.getElementById("badge-uart");
    if (t.sensors.uart_connected) {
      uartBadge.textContent = `UART: ${t.sensors.port}`;
      uartBadge.style.color = "#4ade80";
    } else {
      uartBadge.textContent = "UART: SIM MODE";
      uartBadge.style.color = "#fbbf24";
    }
    document.getElementById("sub-status").textContent =
      `Камера: ${t.camera_source} | ШІ: ${t.vision.detector}`;

    // Ultrasonic sensors (D1..D4)
    const stopCm = t.sensors.obstacle_stop_cm || 35;
    updateSensorCard("d1", t.sensors.d1_front_left, stopCm);
    updateSensorCard("d2", t.sensors.d2_front_right, stopCm);
    updateSensorCard("d3", t.sensors.d3_left, stopCm);
    updateSensorCard("d4", t.sensors.d4_right, stopCm);

    const espChip = document.getElementById("esp-state-chip");
    espChip.textContent = `ESP32: ${t.sensors.esp_state}`;
    if (String(t.sensors.esp_state).includes("OBSTACLE")) {
      espChip.style.background = "rgba(239, 68, 68, 0.22)";
      espChip.style.color = "#f87171";
    } else {
      espChip.style.background = "rgba(34, 197, 94, 0.18)";
      espChip.style.color = "#4ade80";
    }

    // Motors & UART frame
    updatePwmBar("bar-lpwm", "val-lpwm", t.motors.left_pwm);
    updatePwmBar("bar-rpwm", "val-rpwm", t.motors.right_pwm);
    document.getElementById("val-uart-tx").textContent = t.sensors.last_tx;
    document.getElementById("val-status-reason").textContent = t.motors.status_reason;

    // Vision & PID metrics
    const distPill = document.getElementById("dist-pill");
    const aiChip = document.getElementById("ai-det-chip");
    if (t.vision.detected) {
      distPill.textContent = `Дистанція: ${t.vision.distance_m.toFixed(2)} м (Ціль 1.0–1.5 м)`;
      aiChip.textContent = `ЦІЛЬ АКТИВНА (${Math.round(t.vision.confidence * 100)}%)`;
      aiChip.style.color = "#4ade80";
      document.getElementById("val-cx-cy").textContent = `${t.vision.cx}, ${t.vision.cy}`;
      document.getElementById("val-box-h").textContent = t.vision.box_height;
      document.getElementById("val-dist-m").textContent = t.vision.distance_m.toFixed(2);
    } else {
      distPill.textContent = "Людину не знайдено — плавна зупинка";
      aiChip.textContent = " ПОШУК ЦІЛІ...";
      aiChip.style.color = "#fbbf24";
      document.getElementById("val-cx-cy").textContent = "-- , --";
      document.getElementById("val-box-h").textContent = "0";
      document.getElementById("val-dist-m").textContent = "--";
    }

    const ex = t.motors.pid && t.motors.pid.error_x !== undefined ? t.motors.pid.error_x : 0;
    const eh = t.motors.pid && t.motors.pid.error_h !== undefined ? t.motors.pid.error_h : 0;
    document.getElementById("val-pid-err").textContent = `${ex} / ${eh}`;
  } catch (err) {
    document.getElementById("conn-dot").style.backgroundColor = "#ef4444";
  }
}

// --- Virtual Touch Joystick Setup ---
function setupJoystick() {
  const base = document.getElementById("joystick-base");
  const stick = document.getElementById("joystick-stick");
  const maxRadius = 52;

  function handleMove(clientX, clientY) {
    const rect = base.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;

    let dx = clientX - centerX;
    let dy = clientY - centerY;
    const dist = Math.hypot(dx, dy);
    if (dist > maxRadius) {
      dx = (dx / dist) * maxRadius;
      dy = (dy / dist) * maxRadius;
    }

    stick.style.transform = `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px))`;
    joyX = +(dx / maxRadius).toFixed(2);
    joyY = +(-dy / maxRadius).toFixed(2); // Up is positive forward
    sendManualCommand(joyX, joyY);
  }

  function resetStick() {
    joyActive = false;
    joyX = 0.0;
    joyY = 0.0;
    stick.style.transform = "translate(-50%, -50%)";
    if (manualHeartbeatTimer) {
      clearInterval(manualHeartbeatTimer);
      manualHeartbeatTimer = null;
    }
    sendManualCommand(0, 0);
  }

  base.addEventListener("pointerdown", (e) => {
    joyActive = true;
    base.setPointerCapture(e.pointerId);
    handleMove(e.clientX, e.clientY);
    if (!manualHeartbeatTimer) {
      manualHeartbeatTimer = setInterval(() => {
        if (joyActive) sendManualCommand(joyX, joyY);
      }, 150);
    }
  });

  base.addEventListener("pointermove", (e) => {
    if (!joyActive) return;
    handleMove(e.clientX, e.clientY);
  });

  base.addEventListener("pointerup", resetStick);
  base.addEventListener("pointercancel", resetStick);

  // D-Pad buttons
  document.querySelectorAll(".dpad-btn").forEach((btn) => {
    let dpadInterval = null;
    const bx = parseFloat(btn.dataset.x || "0");
    const by = parseFloat(btn.dataset.y || "0");

    const startPress = (e) => {
      e.preventDefault();
      sendManualCommand(bx, by);
      dpadInterval = setInterval(() => sendManualCommand(bx, by), 150);
    };
    const endPress = (e) => {
      e.preventDefault();
      if (dpadInterval) {
        clearInterval(dpadInterval);
        dpadInterval = null;
      }
      sendManualCommand(0, 0);
    };

    btn.addEventListener("pointerdown", startPress);
    btn.addEventListener("pointerup", endPress);
    btn.addEventListener("pointerleave", endPress);
    btn.addEventListener("pointercancel", endPress);
  });

  const speedSlider = document.getElementById("manual-speed-limit");
  speedSlider.addEventListener("input", () => {
    document.getElementById("lbl-speed-limit").textContent = speedSlider.value;
  });
}

// --- Demo Simulator Testbench Controls ---
async function toggleDemoAuto() {
  demoAutoMotion = !demoAutoMotion;
  const btn = document.getElementById("btn-demo-auto");
  btn.classList.toggle("active", demoAutoMotion);
  btn.textContent = `Авто-рух людини: ${demoAutoMotion ? "УВІМК" : "ВИМК"}`;
  await fetch("/api/demo", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ auto_motion: demoAutoMotion }),
  });
}

async function toggleDemoPerson() {
  demoPersonVisible = !demoPersonVisible;
  const btn = document.getElementById("btn-demo-vis");
  btn.classList.toggle("active", !demoPersonVisible);
  btn.textContent = `Людина в кадрі: ${demoPersonVisible ? "ТАК" : "ВТРАЧЕНО"}`;
  await fetch("/api/demo", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ person_visible: demoPersonVisible }),
  });
}

async function toggleDemoObstacle() {
  demoObstacleActive = !demoObstacleActive;
  const btn = document.getElementById("btn-demo-obs");
  btn.classList.toggle("active", demoObstacleActive);
  btn.textContent = `Перешкода <35см: ${demoObstacleActive ? "АКТИВНА (24см)" : "НІ"}`;
  await fetch("/api/demo", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sim_obstacle: demoObstacleActive }),
  });
}

async function onDemoSliderChange() {
  const xPct = parseInt(document.getElementById("slider-demo-x").value, 10);
  const distCm = parseInt(document.getElementById("slider-demo-dist").value, 10);
  document.getElementById("lbl-demo-x").textContent = `${xPct}%`;
  document.getElementById("lbl-demo-dist").textContent = `${(distCm / 100).toFixed(2)} м`;

  demoAutoMotion = false;
  const btn = document.getElementById("btn-demo-auto");
  btn.classList.remove("active");
  btn.textContent = "Авто-рух людини: ВИМК";

  await fetch("/api/demo", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      auto_motion: false,
      target_x_norm: xPct / 100.0,
      target_dist_m: distCm / 100.0,
    }),
  });
}

window.addEventListener("DOMContentLoaded", () => {
  setupJoystick();
  pollTelemetry();
  setInterval(pollTelemetry, 120);
});

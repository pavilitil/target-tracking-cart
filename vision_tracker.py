"""
Camera Capture (Picamera2 OV5647 CSI / OpenCV V4L2 / Interactive Demo Mode)
and Asynchronous AI Person Detection & Tracking (YOLOv8n NCNN / PyTorch / MediaPipe / HOG Fallback)
with 1D Alpha-Beta filtering (cx, box_height) and short occlusion hold (< 0.4s).
Resolution: 640x480 @ 30 FPS.
"""

import logging
import math
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

# Cap OpenMP / OpenBLAS / MKL thread pools on 4-core ARM Cortex-A72 to prevent thermal runaway
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np
from config import CartConfig, CONFIG

try:
    cv2.setNumThreads(2)
except Exception:
    pass

logger = logging.getLogger("VisionTracker")

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class DetectionResult:
    detected: bool = False
    cx: int = 320
    cy: int = 240
    box_height: int = 0
    box_width: int = 0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x1, y1, x2, y2)
    confidence: float = 0.0
    distance_m: float = 0.0
    detector_name: str = "NONE"
    vx: float = 0.0             # Horizontal velocity (px/s) from 1D Alpha-Beta filter
    v_box_height: float = 0.0   # Bounding-box height rate (px/s) from 1D Alpha-Beta filter
    occluded: bool = False      # True while coasting through a short occlusion (< 0.4s)
    target_locked: bool = False # True when operator clothing color Re-ID lock is active
    color_match: float = 0.0    # Clothing HSV histogram correlation (0.0..1.0)
    posture_state: str = "STANDING"  # "STANDING" | "CROUCHED" | "CLIPPED_NEAR"
    raw_box_height: int = 0     # Uncompensated pixel height before posture compensation


def read_cpu_temperature_c() -> float:
    """Reads Raspberry Pi SoC temperature in Celsius from sysfs thermal_zone0."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r", encoding="utf-8") as fh:
            return round(int(fh.read().strip()) / 1000.0, 1)
    except Exception:
        return 45.0


def read_throttled_bitmask() -> int:
    """Reads Raspberry Pi 4 PMIC / firmware under-voltage and thermal throttle bitmask."""
    sysfs_path = "/sys/devices/platform/soc/soc:firmware/get_throttled"
    try:
        if os.path.exists(sysfs_path):
            with open(sysfs_path, "r", encoding="utf-8") as fh:
                return int(fh.read().strip(), 16)
        out = subprocess.check_output(["vcgencmd", "get_throttled"], timeout=0.4).decode("ascii")
        if "=" in out:
            return int(out.strip().split("=")[1], 16)
    except Exception:
        pass
    return 0


class AlphaBetaFilter1D:
    """
    Lightweight 1D Alpha-Beta kinematic filter estimating position (x) and velocity (v).
    Prevents noisy finite-difference D-term spikes in the Dual PID controller and enables
    smooth constant-velocity coasting between asynchronous AI frames or short occlusions.
    """

    def __init__(
        self,
        alpha: float = 0.60,
        beta: float = 0.20,
        max_velocity: float = 1200.0,
    ) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_velocity = float(max_velocity)
        self.x: Optional[float] = None
        self.v: float = 0.0
        self.last_time: Optional[float] = None

    def reset(self) -> None:
        self.x = None
        self.v = 0.0
        self.last_time = None

    def update(self, measurement: float, now: Optional[float] = None) -> Tuple[float, float]:
        if now is None:
            now = time.monotonic()
        m = float(measurement)
        if self.x is None or self.last_time is None:
            self.x = m
            self.v = 0.0
            self.last_time = now
            return self.x, self.v

        dt = max(0.005, min(0.50, now - self.last_time))
        x_pred = self.x + self.v * dt
        residual = m - x_pred

        self.x = x_pred + self.alpha * residual
        self.v = self.v + (self.beta / dt) * residual
        self.v = max(-self.max_velocity, min(self.max_velocity, self.v))
        self.last_time = now
        return self.x, self.v

    def predict(
        self,
        now: Optional[float] = None,
        decay_tau: float = 0.45,
        max_horizon: float = 0.40,
    ) -> Tuple[Optional[float], float]:
        """
        Extrapolates state to `now` with exponential velocity decay without mutating the
        last measurement anchor.
        """
        if self.x is None or self.last_time is None:
            return None, 0.0
        if now is None:
            now = time.monotonic()
        dt = max(0.0, min(max_horizon, now - self.last_time))
        tau = max(0.05, float(decay_tau))
        decay = math.exp(-dt / tau)
        v_eff = self.v * decay
        x_pred = self.x + v_eff * dt
        return x_pred, v_eff


class CameraStream:
    """
    Auto-detecting camera source:
      1. Picamera2 (Waveshare OV5647 CSI camera on Raspberry Pi 4)
      2. OpenCV VideoCapture (V4L2 / USB camera fallback)
      3. Synthetic Demo Mode (when no physical camera is connected yet)
    """

    def __init__(self, config: CartConfig = CONFIG) -> None:
        self.cfg = config
        self.source_type: str = "DEMO"
        self._picam2 = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._demo_start_time = time.monotonic()

        # Interactive demo target state (can follow auto-figure-8 or manual web override)
        self.demo_auto_motion: bool = True
        self.demo_target_x_norm: float = 0.5   # 0.0 (left) .. 1.0 (right)
        self.demo_target_dist_m: float = 1.65  # 0.8m .. 2.8m
        self.demo_person_visible: bool = True

        self._init_hardware()

    def _init_hardware(self) -> None:
        # 1. Try Picamera2 (Waveshare OV5647 CSI Camera)
        try:
            from picamera2 import Picamera2

            cam_info = Picamera2.global_camera_info()
            if cam_info and len(cam_info) > 0:
                picam = Picamera2()
                cam_cfg = picam.create_video_configuration(
                    main={"size": (self.cfg.frame_width, self.cfg.frame_height), "format": "RGB888"},
                    controls={"FrameDurationLimits": (33333, 33333)},  # 30 FPS
                )
                picam.configure(cam_cfg)
                picam.start()
                self._picam2 = picam
                model_name = cam_info[0].get("Model", "OV5647")
                self.source_type = f"PICAMERA2 ({model_name})"
                logger.info("Initialized Picamera2 CSI camera: %s", self.source_type)
                return
            else:
                logger.info("Picamera2 available, but no CSI camera detected in global_camera_info().")
        except Exception as exc:
            logger.debug("Picamera2 init skipped: %s", exc)

        # 2. Try OpenCV VideoCapture
        try:
            for dev_idx in (0, 1):
                if not os.path.exists(f"/dev/video{dev_idx}"):
                    continue
                cap = cv2.VideoCapture(dev_idx)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)
                    cap.set(cv2.CAP_PROP_FPS, self.cfg.target_fps)
                    ok, test_frame = cap.read()
                    if ok and test_frame is not None and test_frame.size > 0:
                        # Avoid empty V4L2 ISP metadata nodes on RPi
                        if test_frame.shape[1] >= 160:
                            self._cap = cap
                            self.source_type = f"OPENCV (/dev/video{dev_idx})"
                            logger.info("Initialized OpenCV camera on /dev/video%d", dev_idx)
                            return
                    cap.release()
        except Exception as exc:
            logger.debug("OpenCV VideoCapture init skipped: %s", exc)

        # 3. Fallback to Synthetic Demo Mode
        self.source_type = "DEMO_SIMULATOR"
        logger.info(
            "No physical camera connected — running in interactive DEMO_SIMULATOR mode (640x480 @ 30 FPS)."
        )

    def _apply_camera_flip(self, frame: np.ndarray) -> np.ndarray:
        hflip = bool(getattr(self.cfg, "camera_hflip", False))
        vflip = bool(getattr(self.cfg, "camera_vflip", False))
        if hflip and vflip:
            return cv2.flip(frame, -1)
        if hflip:
            return cv2.flip(frame, 1)
        if vflip:
            return cv2.flip(frame, 0)
        return frame

    def read_frame(self) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """
        Returns (bgr_frame, synthetic_bbox_or_None).
        """
        if self._picam2 is not None:
            try:
                frame = self._picam2.capture_array("main")
                if frame is not None:
                    if frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    return self._apply_camera_flip(frame), None
            except Exception as exc:
                logger.warning("Picamera2 read error (%s) — falling back to Demo mode", exc)
                self._picam2 = None
                self.source_type = "DEMO_SIMULATOR"

        if self._cap is not None:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                if (
                    frame.shape[1] != self.cfg.frame_width
                    or frame.shape[0] != self.cfg.frame_height
                ):
                    frame = cv2.resize(frame, (self.cfg.frame_width, self.cfg.frame_height))
                return self._apply_camera_flip(frame), None
            else:
                logger.warning("OpenCV camera lost — falling back to Demo mode")
                self._cap.release()
                self._cap = None
                self.source_type = "DEMO_SIMULATOR"

        return self._generate_demo_frame()

    def _generate_demo_frame(self) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        w, h = self.cfg.frame_width, self.cfg.frame_height
        t = time.monotonic() - self._demo_start_time

        # Render dark studio/corridor background with perspective floor grid
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[: h // 2, :] = (28, 24, 20)   # Upper wall
        frame[h // 2 :, :] = (42, 38, 34)   # Floor

        # Perspective lines on floor
        horizon_y = h // 2 - 10
        vanish_x = w // 2
        for offset in range(-600, 601, 120):
            cv2.line(
                frame,
                (vanish_x + offset // 8, horizon_y),
                (vanish_x + offset, h),
                (58, 54, 48),
                1,
                cv2.LINE_AA,
            )
        for row_y in range(horizon_y + 20, h, 36):
            cv2.line(frame, (0, row_y), (w, row_y), (54, 50, 44), 1)

        if not self.demo_person_visible:
            return frame, None

        if self.demo_auto_motion:
            # Smoothly oscillate target horizontally and in distance (0.95m .. 1.95m)
            x_norm = 0.5 + 0.28 * math.sin(t * 0.55)
            dist_m = 1.38 + 0.42 * math.sin(t * 0.32)
        else:
            x_norm = self.demo_target_x_norm
            dist_m = self.demo_target_dist_m

        dist_m = max(0.75, min(2.80, dist_m))
        box_h = int(round(self.cfg.distance_calib_k / dist_m))
        box_h = max(90, min(h - 20, box_h))
        box_w = int(round(box_h * 0.42))

        cx = int(round(x_norm * (w - 120) + 60))
        cy = int(round(horizon_y + (h - horizon_y) * 0.32))

        x1 = max(8, cx - box_w // 2)
        x2 = min(w - 8, cx + box_w // 2)
        y1 = max(8, cy - box_h // 2)
        y2 = min(h - 8, cy + box_h // 2)

        # Draw stylized human silhouette so real detectors or demo tracker can visualize clearly
        head_r = max(12, box_h // 9)
        head_center = (cx, y1 + head_r + 6)
        # Shadow on floor
        cv2.ellipse(
            frame,
            (cx, y2 - 4),
            (box_w // 2, max(6, box_w // 6)),
            0,
            0,
            360,
            (24, 22, 20),
            -1,
            cv2.LINE_AA,
        )
        # Torso
        torso_top = y1 + head_r * 2 + 10
        torso_bottom = y1 + int(box_h * 0.58)
        cv2.rectangle(
            frame,
            (cx - box_w // 3, torso_top),
            (cx + box_w // 3, torso_bottom),
            (195, 135, 70),
            -1,
        )
        # Head
        cv2.circle(frame, head_center, head_r, (185, 205, 235), -1, cv2.LINE_AA)
        # Legs
        leg_w = max(8, box_w // 6)
        cv2.rectangle(
            frame,
            (cx - box_w // 4, torso_bottom),
            (cx - box_w // 4 + leg_w, y2 - 6),
            (85, 95, 115),
            -1,
        )
        cv2.rectangle(
            frame,
            (cx + box_w // 4 - leg_w, torso_bottom),
            (cx + box_w // 4, y2 - 6),
            (85, 95, 115),
            -1,
        )

        return frame, (x1, y1, x2, y2)

    def release(self) -> None:
        if self._picam2 is not None:
            try:
                self._picam2.stop()
            except Exception:
                pass
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass


class PersonTracker:
    """
    Asynchronous AI Person Detector & Target Tracker for Raspberry Pi 4 (ARM Cortex-A72):
      - Priority 1: Exported YOLOv8n NCNN model (`models/yolov8n_ncnn_model`) for ARM64 CPU
      - Priority 2: PyTorch YOLOv8n (`models/yolov8n.pt`, class 0 = person)
      - Priority 3: Demo-AI (Synthetic target fallback when running headless/demo)
    Runs heavy inference in a background daemon thread with a single-frame latest buffer
    so the camera capture & MJPEG stream stay locked at 30 FPS.
    Applies 1D Alpha-Beta filtering on `cx` and `box_height` and holds decaying target
    state across short occlusions (< 0.4s).
    """

    def __init__(self, config: CartConfig = CONFIG, base_dir: Optional[str] = None) -> None:
        self.cfg = config
        self._base_dir = base_dir or _BASE_DIR
        self.active_backend: str = "INITIALIZING"
        self._yolo_model = None

        # 1D Alpha-Beta filters for horizontal center (cx) and bounding-box height (box_height)
        self.cx_filter = AlphaBetaFilter1D(
            alpha=self.cfg.track_cx_alpha,
            beta=self.cfg.track_cx_beta,
            max_velocity=1200.0,
        )
        self.bh_filter = AlphaBetaFilter1D(
            alpha=self.cfg.track_bh_alpha,
            beta=self.cfg.track_bh_beta,
            max_velocity=800.0,
        )

        # Smoothed auxiliary geometry & tracking state
        self._locked_cx: Optional[float] = None
        self._locked_cy: Optional[float] = None
        self._locked_bh: Optional[float] = None
        self._locked_bw: Optional[float] = None
        self._last_conf: float = 0.0
        self._last_det_label: str = "INITIALIZING"
        self._last_seen_time: Optional[float] = None
        self._raw_visible: bool = False

        # Operator clothing color Re-ID lock (HSV histogram of torso)
        self._target_color_hist: Optional[np.ndarray] = None
        self._last_torso_hist: Optional[np.ndarray] = None
        self._last_raw_bbox: Optional[Tuple[int, int, int, int]] = None
        self._last_color_match: float = 0.0
        self._last_posture_state: str = "STANDING"
        self._last_raw_bh: int = 0

        # Smart Thermal Guard & Eco Pacing state (starts safe in STOP)
        self._operating_mode: str = "STOP"
        self.cpu_temp_c: float = read_cpu_temperature_c()
        self.throttled_hex: str = "0x0"
        self.undervoltage_now: bool = False
        self.thermal_state: str = "NORMAL"
        self._last_temp_check: float = 0.0

        # Background daemon inference thread with 1-slot latest-frame buffer
        self._lock = threading.Lock()
        self._frame_event = threading.Event()
        self._result_event = threading.Event()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_demo_bbox: Optional[Tuple[int, int, int, int]] = None
        self._latest_frame_time: float = 0.0
        self._submitted_seq: int = 0
        self._completed_seq: int = 0
        self._running: bool = True

        self._init_detector()
        self._last_det_label = self.active_backend

        self._worker_thread = threading.Thread(
            target=self._inference_worker,
            name="PersonTrackerAsyncWorker",
            daemon=True,
        )
        self._worker_thread.start()

    def _resolve_yolo_model(self) -> Tuple[str, bool]:
        """
        Resolves the preferred YOLO model path.
        Checks for the exported NCNN directory (`models/yolov8n_ncnn_model`) first
        before falling back to `models/yolov8n.pt`.
        Returns (model_path, is_ncnn).
        """
        ncnn_dir_cfg = getattr(self.cfg, "yolo_ncnn_model_dir", "yolov8n_ncnn_model")
        ncnn_candidates = [
            os.path.join(self._base_dir, "models", ncnn_dir_cfg),
            os.path.join("models", ncnn_dir_cfg),
            ncnn_dir_cfg,
        ]
        for cand in ncnn_candidates:
            if os.path.isdir(cand):
                try:
                    if any(os.scandir(cand)):
                        return cand, True
                except OSError:
                    continue

        pt_candidates = [
            os.path.join(self._base_dir, "models", self.cfg.yolo_model_path),
            os.path.join("models", self.cfg.yolo_model_path),
            self.cfg.yolo_model_path,
        ]
        for cand in pt_candidates:
            if os.path.exists(cand):
                return cand, False
        return self.cfg.yolo_model_path, False

    def _init_detector(self) -> None:
        # 1. Try YOLOv8n (preferring exported NCNN model on ARM64 over .pt)
        if self.cfg.preferred_detector.lower() in ("yolov8n", "yolo", "ncnn", "auto"):
            try:
                try:
                    import torch
                    torch.set_num_threads(max(1, int(getattr(self.cfg, "torch_num_threads", 2))))
                except Exception:
                    pass
                from ultralytics import YOLO

                model_path, is_ncnn = self._resolve_yolo_model()
                if is_ncnn:
                    try:
                        self._yolo_model = YOLO(model_path, task="detect")
                        self.active_backend = "YOLOv8n-NCNN"
                        logger.info("Loaded YOLOv8n NCNN model from %s", model_path)
                        return
                    except Exception as ncnn_exc:
                        logger.warning(
                            "NCNN model at %s failed (%s) — falling back to PyTorch .pt",
                            model_path,
                            ncnn_exc,
                        )
                        pt_path = os.path.join(self._base_dir, "models", self.cfg.yolo_model_path)
                        if not os.path.exists(pt_path):
                            pt_path = self.cfg.yolo_model_path
                        self._yolo_model = YOLO(pt_path)
                        self.active_backend = "YOLOv8n"
                        logger.info("Loaded YOLOv8n PyTorch detector from %s", pt_path)
                        return
                else:
                    self._yolo_model = YOLO(model_path)
                    self.active_backend = "YOLOv8n"
                    logger.info("Loaded YOLOv8n person detector from %s", model_path)
                    return
            except Exception as exc:
                logger.info("YOLOv8n unavailable (%s) — enabling Demo-AI fallback.", exc)

        self.active_backend = "Demo-AI"

    def stop(self) -> None:
        self._running = False
        self._frame_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)

    def reset(self) -> None:
        with self._lock:
            self.cx_filter.reset()
            self.bh_filter.reset()
            self._locked_cx = None
            self._locked_cy = None
            self._locked_bh = None
            self._locked_bw = None
            self._last_seen_time = None
            self._raw_visible = False

    def set_operating_mode(self, mode: str) -> None:
        with self._lock:
            self._operating_mode = mode

    @staticmethod
    def _extract_torso_hist(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Computes normalized 2D histogram of the person's torso region.
        Applies an HSV chromatic mask so black/white/gray garments do not generate Hue noise;
        falls back to unmasked S-V channels [1, 2] if fewer than 20 chromatic pixels exist.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        if bw < 16 or bh < 32:
            return None
        tx1 = max(0, int(x1 + bw * 0.22))
        tx2 = min(w, int(x2 - bw * 0.22))
        ty1 = max(0, int(y1 + bh * 0.22))
        ty2 = min(h, int(y1 + bh * 0.58))
        if tx2 - tx1 < 8 or ty2 - ty1 < 8:
            return None
        roi = frame[ty1:ty2, tx1:tx2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array((0, 30, 30), dtype=np.uint8),
            np.array((180, 255, 250), dtype=np.uint8),
        )
        if cv2.countNonZero(mask) >= 20:
            hist = cv2.calcHist([hsv], [0, 1], mask, [16, 16], [0, 180, 0, 256])
        else:
            hist = cv2.calcHist([hsv], [1, 2], None, [16, 16], [0, 256, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def set_target_color_lock(self, enabled: bool) -> bool:
        """
        Locks or unlocks operator clothing color Re-ID profile on-demand.
        Only locks when a target person is actively visible in the current frame.
        """
        with self._lock:
            if not enabled:
                self._target_color_hist = None
                self._last_torso_hist = None
                self._last_color_match = 0.0
                return False
            if not self._raw_visible or self._latest_frame is None or self._last_raw_bbox is None:
                return False
            hist = self._extract_torso_hist(self._latest_frame, self._last_raw_bbox)
            if hist is not None:
                self._last_torso_hist = hist
                self._target_color_hist = hist.copy()
                self._last_color_match = 1.0
                return True
            return False

    def is_target_color_locked(self) -> bool:
        with self._lock:
            return self._target_color_hist is not None

    def get_thermal_status(self) -> Tuple[float, str, float]:
        """Returns (cpu_temp_c, thermal_state, target_ai_fps)."""
        now = time.monotonic()
        if now - self._last_temp_check >= 1.5:
            self._last_temp_check = now
            self.cpu_temp_c = read_cpu_temperature_c()
            mask = read_throttled_bitmask()
            self.throttled_hex = hex(mask)
            self.undervoltage_now = bool(mask & 0x1)
        temp = self.cpu_temp_c
        with self._lock:
            mode = self._operating_mode

        warn_c = getattr(self.cfg, "thermal_warn_c", 70.0)
        prot_c = getattr(self.cfg, "thermal_protect_c", 78.0)
        if mode != "AUTO":
            self.thermal_state = "ECO_IDLE"
            ai_fps = getattr(self.cfg, "ai_max_fps_idle", 2.0)
        elif temp >= prot_c:
            self.thermal_state = "THERMAL_PROTECT"
            ai_fps = getattr(self.cfg, "ai_max_fps_protect", 3.5)
        elif temp >= warn_c:
            self.thermal_state = "THERMAL_ECO"
            ai_fps = getattr(self.cfg, "ai_max_fps_eco", 6.0)
        else:
            self.thermal_state = "NORMAL"
            ai_fps = getattr(self.cfg, "ai_max_fps_normal", 12.0)
        return temp, self.thermal_state, float(ai_fps)

    def _run_raw_inference(
        self,
        frame: np.ndarray,
        demo_bbox: Optional[Tuple[int, int, int, int]],
        locked_cx: Optional[float],
        locked_cy: Optional[float],
        target_hist: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[Tuple[int, int, int, int]], float, str, Optional[np.ndarray], float]:
        h, w = frame.shape[:2]
        raw_bbox: Optional[Tuple[int, int, int, int]] = None
        conf: float = 0.0
        det_label: str = self.active_backend
        best_hist: Optional[np.ndarray] = None
        best_sim: float = 0.0
        min_sim = float(getattr(self.cfg, "min_color_similarity", 0.35))

        # 1. Run YOLOv8n (NCNN or PyTorch) on real camera frames (when demo_bbox is None)
        if self._yolo_model is not None and demo_bbox is None:
            try:
                results = self._yolo_model.predict(
                    frame,
                    classes=[0],  # class 0 = person
                    conf=self.cfg.yolo_conf_threshold,
                    imgsz=self.cfg.yolo_imgsz,
                    verbose=False,
                )
                best_box = None
                best_score = -1.0
                for r in results:
                    if r.boxes is None or len(r.boxes) == 0:
                        continue
                    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    for coords, c_val in zip(boxes_xyxy, confs):
                        x1, y1, x2, y2 = (
                            int(coords[0]),
                            int(coords[1]),
                            int(coords[2]),
                            int(coords[3]),
                        )
                        c = float(c_val)
                        bx = (x1 + x2) * 0.5
                        by = (y1 + y2) * 0.5
                        cand_box = (x1, y1, x2, y2)
                        if locked_cx is not None and locked_cy is not None:
                            dist_penalty = math.hypot(bx - locked_cx, by - locked_cy) / float(w)
                            score = c - 0.75 * dist_penalty
                        else:
                            score = c * ((y2 - y1) / float(h))
                        sim = 0.0
                        cand_hist = None
                        if target_hist is not None:
                            cand_hist = self._extract_torso_hist(frame, cand_box)
                            if cand_hist is not None:
                                sim = max(
                                    0.0,
                                    float(
                                        cv2.compareHist(
                                            target_hist, cand_hist, cv2.HISTCMP_CORREL
                                        )
                                    ),
                                )
                            if sim < min_sim:
                                continue  # Reject bystander whose clothing does not match locked operator
                            score = score * 0.45 + sim * 0.65
                        if score > best_score:
                            best_score = score
                            best_box = cand_box
                            conf = c
                            best_sim = sim
                            best_hist = cand_hist
                raw_bbox = best_box
            except Exception as exc:
                logger.debug("YOLOv8n inference error: %s", exc)

        # 2. Use synthetic Demo BBox when in Demo Simulator mode
        if raw_bbox is None and demo_bbox is not None:
            cand_hist = None
            sim = 0.0
            accepted = True
            if target_hist is not None:
                cand_hist = self._extract_torso_hist(frame, demo_bbox)
                if cand_hist is not None:
                    sim = max(
                        0.0,
                        float(cv2.compareHist(target_hist, cand_hist, cv2.HISTCMP_CORREL)),
                    )
                if sim < min_sim:
                    accepted = False
            if accepted:
                raw_bbox = demo_bbox
                conf = 0.94
                best_hist = cand_hist
                best_sim = sim
            det_label = f"{self.active_backend} (Demo)"

        return raw_bbox, conf, det_label, best_hist, best_sim

    def _apply_inference_output(
        self,
        raw_bbox: Optional[Tuple[int, int, int, int]],
        conf: float,
        det_label: str,
        sample_time: float,
        seq: int,
        best_hist: Optional[np.ndarray] = None,
        best_sim: float = 0.0,
    ) -> None:
        with self._lock:
            self._last_det_label = det_label
            if raw_bbox is not None:
                self._last_raw_bbox = raw_bbox
                if best_hist is not None:
                    self._last_torso_hist = best_hist
                    self._last_color_match = round(best_sim, 2)
                else:
                    self._last_color_match = 0.0

                x1, y1, x2, y2 = raw_bbox
                raw_cx = (x1 + x2) * 0.5
                raw_cy = (y1 + y2) * 0.5
                raw_bw = max(10.0, float(x2 - x1))
                raw_bh = max(20.0, float(y2 - y1))
                self._last_raw_bh = int(round(raw_bh))

                # Posture / Crouching & Vertical Border-Clipping Compensation:
                # Cap effective_bh <= target_box_height_px so posture hold stops forward rush without reversing
                aspect_ratio = raw_bw / raw_bh
                frame_h = self.cfg.frame_height
                clipped_near = (y1 <= 8 and y2 >= frame_h - 8) or (y1 <= 6 and raw_bw >= 115)
                target_bh = float(self.cfg.target_box_height_px)
                if aspect_ratio > 0.65:
                    self._last_posture_state = "CROUCHED"
                    bh_from_width = raw_bw / 0.42
                    effective_bh = min(target_bh, max(raw_bh, min(1.65 * raw_bh, bh_from_width)))
                elif clipped_near:
                    self._last_posture_state = "CLIPPED_NEAR"
                    bh_from_width = raw_bw / 0.42
                    effective_bh = min(target_bh, max(raw_bh, min(1.50 * raw_bh, bh_from_width)))
                else:
                    self._last_posture_state = "STANDING"
                    effective_bh = raw_bh

                # If target was completely lost (> target_lost_timeout_s), re-seed filters cleanly
                if (
                    self._last_seen_time is not None
                    and (sample_time - self._last_seen_time) > self.cfg.target_lost_timeout_s
                ):
                    self.cx_filter.reset()
                    self.bh_filter.reset()
                    self._locked_cy = None
                    self._locked_bw = None

                filt_cx, _ = self.cx_filter.update(raw_cx, sample_time)
                filt_bh, _ = self.bh_filter.update(effective_bh, sample_time)

                alpha_aux = 0.65
                if self._locked_cy is None or self._locked_bw is None:
                    self._locked_cy = raw_cy
                    self._locked_bw = raw_bw
                else:
                    self._locked_cy = alpha_aux * raw_cy + (1.0 - alpha_aux) * self._locked_cy
                    self._locked_bw = alpha_aux * raw_bw + (1.0 - alpha_aux) * self._locked_bw

                self._locked_cx = filt_cx
                self._locked_bh = filt_bh
                self._last_conf = conf
                self._last_seen_time = sample_time
                self._raw_visible = True
            else:
                self._raw_visible = False
                self._last_raw_bbox = None
                self._last_torso_hist = None
                self._last_color_match = 0.0

            self._completed_seq = seq
            self._result_event.set()

    def _inference_worker(self) -> None:
        """
        Daemon thread that always processes the most recent frame from the 1-slot buffer.
        Intermediate frames arriving while YOLOv8n is busy are overwritten so lag never builds up.
        """
        last_seq = 0
        while self._running:
            self._frame_event.wait(timeout=0.10)
            if not self._running:
                break

            with self._lock:
                if self._latest_frame is None or self._submitted_seq == last_seq:
                    self._frame_event.clear()
                    continue
                frame = self._latest_frame
                demo_bbox = self._latest_demo_bbox
                sample_time = self._latest_frame_time
                seq = self._submitted_seq
                locked_cx = self._locked_cx
                locked_cy = self._locked_cy
                target_hist = self._target_color_hist
                self._frame_event.clear()

            last_seq = seq
            t_infer_start = time.monotonic()
            raw_bbox, conf, det_label, best_hist, best_sim = self._run_raw_inference(
                frame, demo_bbox, locked_cx, locked_cy, target_hist=target_hist
            )
            self._apply_inference_output(
                raw_bbox,
                conf,
                det_label,
                sample_time,
                seq,
                best_hist=best_hist,
                best_sim=best_sim,
            )

            # Dynamic Thermal Guard & Eco Pacing only when real AI model ran
            if demo_bbox is None and self._yolo_model is not None:
                _, _, target_ai_fps = self.get_thermal_status()
                min_period = 1.0 / max(1.0, target_ai_fps)
                elapsed_infer = time.monotonic() - t_infer_start
                if elapsed_infer < min_period:
                    time.sleep(min_period - elapsed_infer)

    def _build_current_result(self, w: int, h: int, now: float) -> DetectionResult:
        with self._lock:
            det_label = self._last_det_label
            last_seen = self._last_seen_time
            raw_visible = self._raw_visible

            if last_seen is None:
                return DetectionResult(detected=False, detector_name=det_label)

            elapsed_since_seen = max(0.0, now - last_seen)

            # 1. Active detection (and not stale from a hung detector)
            if raw_visible and elapsed_since_seen <= self.cfg.detection_stale_timeout_s:
                pred_cx, vx = self.cx_filter.predict(
                    now,
                    decay_tau=self.cfg.occlusion_decay_tau_s,
                    max_horizon=self.cfg.track_predict_horizon_s,
                )
                pred_bh, v_bh = self.bh_filter.predict(
                    now,
                    decay_tau=self.cfg.occlusion_decay_tau_s,
                    max_horizon=self.cfg.track_predict_horizon_s,
                )
                cx_val = pred_cx if pred_cx is not None else (self._locked_cx or w * 0.5)
                bh_val = pred_bh if pred_bh is not None else (self._locked_bh or 200.0)
                cy_val = self._locked_cy if self._locked_cy is not None else h * 0.5
                bw_val = self._locked_bw if self._locked_bw is not None else bh_val * 0.42

                cx_int = int(max(0, min(w - 1, round(cx_val))))
                cy_int = int(max(0, min(h - 1, round(cy_val))))
                bh_int = int(max(20, min(h, round(bh_val))))
                bw_int = int(max(10, min(w, round(bw_val))))

                sx1 = max(0, cx_int - bw_int // 2)
                sy1 = max(0, cy_int - bh_int // 2)
                sx2 = min(w - 1, cx_int + bw_int // 2)
                sy2 = min(h - 1, cy_int + bh_int // 2)
                dist_m = round(self.cfg.distance_calib_k / max(30.0, float(bh_int)), 2)

                return DetectionResult(
                    detected=True,
                    cx=cx_int,
                    cy=cy_int,
                    box_height=bh_int,
                    box_width=bw_int,
                    bbox=(sx1, sy1, sx2, sy2),
                    confidence=round(self._last_conf, 2),
                    distance_m=dist_m,
                    detector_name=det_label,
                    vx=round(vx, 2),
                    v_box_height=round(v_bh, 2),
                    occluded=False,
                    target_locked=self._target_color_hist is not None,
                    color_match=self._last_color_match,
                    posture_state=self._last_posture_state,
                    raw_box_height=self._last_raw_bh or bh_int,
                )

            # 2. Short occlusion (< target_lost_timeout_s = 0.4s):
            # Hold and extrapolate previous target state with exponential velocity decay
            if elapsed_since_seen < self.cfg.target_lost_timeout_s:
                pred_cx, vx = self.cx_filter.predict(
                    now,
                    decay_tau=self.cfg.occlusion_decay_tau_s,
                    max_horizon=self.cfg.target_lost_timeout_s,
                )
                pred_bh, v_bh = self.bh_filter.predict(
                    now,
                    decay_tau=self.cfg.occlusion_decay_tau_s,
                    max_horizon=self.cfg.target_lost_timeout_s,
                )
                cx_val = pred_cx if pred_cx is not None else (self._locked_cx or w * 0.5)
                bh_val = pred_bh if pred_bh is not None else (self._locked_bh or 200.0)
                cy_val = self._locked_cy if self._locked_cy is not None else h * 0.5
                bw_val = self._locked_bw if self._locked_bw is not None else bh_val * 0.42

                cx_int = int(max(0, min(w - 1, round(cx_val))))
                cy_int = int(max(0, min(h - 1, round(cy_val))))
                bh_int = int(max(20, min(h, round(bh_val))))
                bw_int = int(max(10, min(w, round(bw_val))))

                sx1 = max(0, cx_int - bw_int // 2)
                sy1 = max(0, cy_int - bh_int // 2)
                sx2 = min(w - 1, cx_int + bw_int // 2)
                sy2 = min(h - 1, cy_int + bh_int // 2)
                dist_m = round(self.cfg.distance_calib_k / max(30.0, float(bh_int)), 2)
                decay = math.exp(-elapsed_since_seen / max(0.05, self.cfg.occlusion_decay_tau_s))

                return DetectionResult(
                    detected=False,
                    cx=cx_int,
                    cy=cy_int,
                    box_height=bh_int,
                    box_width=bw_int,
                    bbox=(sx1, sy1, sx2, sy2),
                    confidence=round(self._last_conf * decay, 2),
                    distance_m=dist_m,
                    detector_name=det_label,
                    vx=round(vx, 2),
                    v_box_height=round(v_bh, 2),
                    occluded=True,
                    target_locked=self._target_color_hist is not None,
                    color_match=self._last_color_match,
                    posture_state=self._last_posture_state,
                    raw_box_height=self._last_raw_bh or bh_int,
                )

            # 3. Target lost for >= 0.4s -> reset filters cleanly
            self.cx_filter.reset()
            self.bh_filter.reset()
            self._locked_cx = None
            self._locked_cy = None
            self._locked_bh = None
            self._locked_bw = None
            self._last_raw_bbox = None
            self._last_torso_hist = None
            self._last_color_match = 0.0
            self._last_posture_state = "STANDING"
            return DetectionResult(detected=False, detector_name=det_label, occluded=False)

    def detect(
        self,
        frame: np.ndarray,
        demo_bbox: Optional[Tuple[int, int, int, int]] = None,
        now: Optional[float] = None,
    ) -> DetectionResult:
        """
        Non-blocking 30 FPS entry point: pushes the latest frame into the background daemon
        worker's 1-slot buffer and immediately returns the Alpha-Beta filtered / extrapolated
        detection result.
        """
        if now is None:
            now = time.monotonic()
        h, w = frame.shape[:2]

        with self._lock:
            self._latest_frame = frame
            self._latest_demo_bbox = demo_bbox
            self._latest_frame_time = now
            self._submitted_seq += 1
            target_seq = self._submitted_seq
            self._result_event.clear()
        self._frame_event.set()

        # In lightweight Demo Simulator mode or on the very first frame, wait briefly
        # so synchronous tests observe deterministic results; during real camera YOLOv8n inference
        # (demo_bbox is None and first frame already processed), never block the 30 FPS loop.
        if demo_bbox is not None or self._completed_seq == 0 or self._yolo_model is None:
            deadline = time.monotonic() + 0.150
            while self._completed_seq < target_seq and time.monotonic() < deadline:
                self._result_event.wait(timeout=0.005)

        return self._build_current_result(w, h, now)


def draw_hud_overlay(
    frame: np.ndarray,
    det: DetectionResult,
    mode: str,
    left_pwm: int,
    right_pwm: int,
    sensors_cm: Tuple[int, int, int, int],
    fps: float,
    camera_source: str,
    status_reason: str,
    cfg: CartConfig = CONFIG,
) -> np.ndarray:
    """
    Draws real-time AI bounding box, Ghost Trajectory Arc, safety badges,
    vertical L/R thrust bars, center crosshair, and PWM/ultrasonic telemetry.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    center_x, center_y = w // 2, h // 2

    # 1. Center Reference Crosshair & Deadband Vertical Guides
    db = cfg.steer_deadband_px
    cv2.line(out, (center_x - db, 42), (center_x - db, h - 42), (70, 70, 70), 1, cv2.LINE_AA)
    cv2.line(out, (center_x + db, 42), (center_x + db, h - 42), (70, 70, 70), 1, cv2.LINE_AA)
    cv2.drawMarker(
        out, (center_x, center_y), (160, 160, 160), cv2.MARKER_CROSS, 18, 1, cv2.LINE_AA
    )

    # 1b. Ghost Trajectory Arc (Visualizes differential steering & forward drive)
    net_fwd = max(0.0, float(left_pwm + right_pwm) * 0.5)
    diff_steer = float(left_pwm - right_pwm) * 0.5
    if abs(left_pwm) > 0 or abs(right_pwm) > 0:
        arc_len_px = max(48, min(190, int(48 + (net_fwd / 185.0) * 140)))
        arc_pts = []
        steps = 10
        base_y = h - 42
        for i in range(steps + 1):
            t = i / float(steps)
            px = int(round(center_x + (diff_steer / 135.0) * 140.0 * (t * t)))
            py = int(round(base_y - arc_len_px * t))
            px = max(18, min(w - 18, px))
            arc_pts.append((px, py))
        arc_color = (80, 230, 100) if net_fwd > 10 else (255, 190, 50)
        pts_arr = np.array(arc_pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts_arr], False, arc_color, 2, cv2.LINE_AA)
        cv2.circle(out, arc_pts[-1], 4, arc_color, -1, cv2.LINE_AA)

    # 1c. Left & Right Edge Vertical Thrust Bars
    bar_h_max = 140
    bar_bottom = h - 48
    for bx, pwm_val in ((8, left_pwm), (w - 18, right_pwm)):
        cv2.rectangle(out, (bx, bar_bottom - bar_h_max), (bx + 10, bar_bottom), (35, 40, 50), -1)
        fill_h = int(round((min(255, abs(pwm_val)) / 255.0) * bar_h_max))
        if fill_h > 0:
            b_color = (80, 230, 100) if pwm_val >= 0 else (60, 140, 255)
            cv2.rectangle(out, (bx + 1, bar_bottom - fill_h), (bx + 9, bar_bottom), b_color, -1)

    # 2. Target Person Bounding Box, Distance Vector & Safety Badges
    if det.detected or (det.occluded and det.box_height > 20):
        x1, y1, x2, y2 = det.bbox
        if det.occluded and not det.detected:
            box_color = (0, 190, 255)  # Amber-gold during short occlusion hold (<0.4s)
        elif cfg.target_distance_min_m <= det.distance_m <= cfg.target_distance_max_m:
            box_color = (80, 230, 100)  # Green: inside 1.0 - 1.5 m sweet spot
        elif det.distance_m > cfg.target_distance_max_m:
            box_color = (245, 200, 60)  # Cyan-blue: following forward
        else:
            box_color = (60, 140, 255)  # Orange: closer than 1.0 m

        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

        # Corner accents
        clen = min(18, max(4, (x2 - x1) // 4), max(4, (y2 - y1) // 4))
        for px, py, dx, dy in [
            (x1, y1, 1, 1),
            (x2, y1, -1, 1),
            (x1, y2, 1, -1),
            (x2, y2, -1, -1),
        ]:
            cv2.line(out, (px, py), (px + dx * clen, py), box_color, 3, cv2.LINE_AA)
            cv2.line(out, (px, py), (px, py + dy * clen), box_color, 3, cv2.LINE_AA)

        # Line from frame center to person center (cx, cy)
        cv2.line(out, (center_x, center_y), (det.cx, det.cy), box_color, 2, cv2.LINE_AA)
        cv2.circle(out, (det.cx, det.cy), 6, (0, 255, 255), -1, cv2.LINE_AA)

        tag = "HOLD" if (det.occluded and not det.detected) else f"PERSON {det.confidence:.0%}"
        label = f"{tag} | Dist: {det.distance_m:.2f}m | H:{det.box_height}px"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        ly = max(y1 - 8, th + 48)
        cv2.rectangle(out, (x1, ly - th - 6), (x1 + tw + 10, ly + 4), (20, 24, 30), -1)
        cv2.putText(
            out,
            label,
            (x1 + 5, ly - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            box_color,
            1,
            cv2.LINE_AA,
        )

        # Active Safety Badges below main label
        badges = []
        if det.target_locked:
            badges.append(f"[LOCKED {int(round(det.color_match * 100))}%]")
        if det.posture_state == "CROUCHED":
            badges.append("[CROUCH HOLD]")
        elif det.posture_state == "CLIPPED_NEAR":
            badges.append("[CLIPPED]")
        if "TTC" in status_reason:
            badges.append("[TTC BRAKE]")
        if det.occluded and not det.detected:
            if det.cx < 140:
                badges.append("<<< EXITED LEFT")
            elif det.cx > w - 140:
                badges.append("EXITED RIGHT >>>")
        if badges:
            badge_str = " ".join(badges)
            by_pos = min(h - 46, y2 + 16)
            cv2.rectangle(out, (x1, by_pos - 14), (x1 + len(badge_str) * 8 + 10, by_pos + 4), (20, 24, 30), -1)
            cv2.putText(
                out,
                badge_str,
                (x1 + 4, by_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )

    # 3. Top Telemetry Banner
    cv2.rectangle(out, (0, 0), (w, 36), (18, 20, 26), -1)
    mode_color = (
        (80, 230, 100)
        if mode == "AUTO"
        else ((255, 190, 50) if mode == "MANUAL" else (70, 80, 245))
    )
    cv2.putText(
        out,
        f"MODE: {mode}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        mode_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f"AI: {det.detector_name} | CAM: {camera_source} | {fps:.1f} FPS",
        (145, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (210, 215, 225),
        1,
        cv2.LINE_AA,
    )

    # 4. Bottom Telemetry & Ultrasonic Banner
    d1, d2, d3, d4 = sensors_cm
    cv2.rectangle(out, (0, h - 38), (w, h), (18, 20, 26), -1)
    s_color = (70, 75, 250) if min(d1, d2) < cfg.obstacle_stop_cm else (190, 220, 240)
    bottom_text = (
        f"PWM L:{left_pwm:+4d} R:{right_pwm:+4d} | "
        f"US(cm) FL:{d1:3d} FR:{d2:3d} L:{d3:3d} R:{d4:3d} | {status_reason}"
    )
    cv2.putText(
        out,
        bottom_text,
        (10, h - 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        s_color,
        1,
        cv2.LINE_AA,
    )

    return out

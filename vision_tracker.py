"""
Camera Capture (Picamera2 OV5647 CSI / OpenCV V4L2 / Interactive Demo Mode)
and AI Person Detection & Tracking (YOLOv8n / MediaPipe / HOG Fallback).
Resolution: 640x480 @ 30 FPS.
"""

import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from config import CartConfig, CONFIG

logger = logging.getLogger("VisionTracker")


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
                    return frame, None
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
                return frame, None
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
    AI Person Detector & Target Tracker supporting:
      - YOLOv8n (`ultralytics`, class 0 = person)
      - MediaPipe Pose (`mediapipe.solutions.pose`)
      - OpenCV HOG People Detector + Synthetic Demo BBox fallback
    Includes temporal smoothing and centroid target-locking.
    """

    def __init__(self, config: CartConfig = CONFIG) -> None:
        self.cfg = config
        self.active_backend: str = "INITIALIZING"
        self._yolo_model = None
        self._mp_pose = None
        self._hog = None

        # Smoothed target tracking state
        self._locked_cx: Optional[float] = None
        self._locked_cy: Optional[float] = None
        self._locked_bh: Optional[float] = None
        self._locked_bw: Optional[float] = None

        self._init_detector()

    def _init_detector(self) -> None:
        # 1. Try YOLOv8n (Ultralytics)
        if self.cfg.preferred_detector.lower() in ("yolov8n", "yolo", "auto"):
            try:
                from ultralytics import YOLO

                model_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "models",
                    self.cfg.yolo_model_path,
                )
                if not os.path.exists(model_path):
                    model_path = self.cfg.yolo_model_path
                self._yolo_model = YOLO(model_path)
                self.active_backend = "YOLOv8n"
                logger.info("Loaded YOLOv8n person detector successfully.")
                return
            except Exception as exc:
                logger.info("YOLOv8n unavailable (%s) — checking MediaPipe...", exc)

        # 2. Try MediaPipe Pose
        try:
            import mediapipe as mp

            self._mp_pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.active_backend = "MediaPipe"
            logger.info("Loaded MediaPipe Pose person tracker successfully.")
            return
        except Exception as exc:
            logger.info("MediaPipe unavailable (%s) — enabling OpenCV HOG + Demo AI.", exc)

        # 3. Fallback to OpenCV HOG Descriptor (if present in OpenCV <5) or Demo-AI
        if hasattr(cv2, "HOGDescriptor"):
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.active_backend = "OpenCV-HOG / Demo-AI"
        else:
            self._hog = None
            self.active_backend = "Demo-AI"

    def detect(
        self, frame: np.ndarray, demo_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> DetectionResult:
        h, w = frame.shape[:2]
        raw_bbox: Optional[Tuple[int, int, int, int]] = None
        conf: float = 0.0
        det_label: str = self.active_backend

        # 1. Run YOLOv8n on real camera frames (or if no demo_bbox)
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
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        coords = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = (
                            int(coords[0]),
                            int(coords[1]),
                            int(coords[2]),
                            int(coords[3]),
                        )
                        c = float(box.conf[0].cpu().numpy())
                        bx = (x1 + x2) * 0.5
                        by = (y1 + y2) * 0.5
                        # Prefer the person closest to our previously locked centroid
                        if self._locked_cx is not None and self._locked_cy is not None:
                            dist_penalty = math.hypot(
                                bx - self._locked_cx, by - self._locked_cy
                            ) / float(w)
                            score = c - 0.35 * dist_penalty
                        else:
                            score = c * ((y2 - y1) / float(h))
                        if score > best_score:
                            best_score = score
                            best_box = (x1, y1, x2, y2)
                            conf = c
                raw_bbox = best_box
            except Exception as exc:
                logger.debug("YOLOv8n inference error: %s", exc)

        # 2. Run MediaPipe Pose if active and no bbox yet
        if raw_bbox is None and self._mp_pose is not None and demo_bbox is None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self._mp_pose.process(rgb)
                if res.pose_landmarks:
                    xs = [lm.x * w for lm in res.pose_landmarks.landmark if lm.visibility > 0.35]
                    ys = [lm.y * h for lm in res.pose_landmarks.landmark if lm.visibility > 0.35]
                    if len(xs) >= 4:
                        pad_x = 24
                        pad_y = 20
                        x1 = max(0, int(min(xs)) - pad_x)
                        x2 = min(w - 1, int(max(xs)) + pad_x)
                        y1 = max(0, int(min(ys)) - pad_y)
                        y2 = min(h - 1, int(max(ys)) + pad_y)
                        raw_bbox = (x1, y1, x2, y2)
                        conf = 0.88
            except Exception as exc:
                logger.debug("MediaPipe error: %s", exc)

        # 3. Use synthetic Demo BBox when in Demo Simulator mode (or HOG fallback)
        if raw_bbox is None and demo_bbox is not None:
            raw_bbox = demo_bbox
            conf = 0.94
            det_label = f"{self.active_backend} (Demo)"
        elif raw_bbox is None and self._hog is not None and demo_bbox is None:
            rects, weights = self._hog.detectMultiScale(
                frame, winStride=(8, 8), padding=(16, 16), scale=1.08
            )
            if len(rects) > 0:
                idx = int(np.argmax([rw * rh for (rx, ry, rw, rh) in rects]))
                rx, ry, rw, rh = rects[idx]
                raw_bbox = (int(rx), int(ry), int(rx + rw), int(ry + rh))
                conf = float(weights[idx]) if len(weights) > idx else 0.75

        if raw_bbox is None:
            self._locked_cx = None
            self._locked_cy = None
            self._locked_bh = None
            self._locked_bw = None
            return DetectionResult(detected=False, detector_name=det_label)

        x1, y1, x2, y2 = raw_bbox
        raw_cx = (x1 + x2) * 0.5
        raw_cy = (y1 + y2) * 0.5
        raw_bw = max(10.0, float(x2 - x1))
        raw_bh = max(20.0, float(y2 - y1))

        # Exponential smoothing (70% new, 30% previous) for jitter-free PID input
        alpha = 0.70
        if self._locked_cx is None:
            self._locked_cx = raw_cx
            self._locked_cy = raw_cy
            self._locked_bw = raw_bw
            self._locked_bh = raw_bh
        else:
            self._locked_cx = alpha * raw_cx + (1.0 - alpha) * self._locked_cx
            self._locked_cy = alpha * raw_cy + (1.0 - alpha) * self._locked_cy
            self._locked_bw = alpha * raw_bw + (1.0 - alpha) * self._locked_bw
            self._locked_bh = alpha * raw_bh + (1.0 - alpha) * self._locked_bh

        cx_int = int(round(self._locked_cx))
        cy_int = int(round(self._locked_cy))
        bh_int = int(round(self._locked_bh))
        bw_int = int(round(self._locked_bw))

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
            confidence=round(conf, 2),
            distance_m=dist_m,
            detector_name=det_label,
        )


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
    Draws real-time AI bounding box, center crosshair, distance bar,
    ultrasonic radar indicators, and motor PWM telemetry onto the frame.
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

    # 2. Target Person Bounding Box & Distance Vector
    if det.detected:
        x1, y1, x2, y2 = det.bbox
        # Color-code by target distance zone (1.0m - 1.5m = Green, >1.5m = Cyan, <1.0m = Orange)
        if cfg.target_distance_min_m <= det.distance_m <= cfg.target_distance_max_m:
            box_color = (80, 230, 100)  # Green: inside 1.0 - 1.5 m sweet spot
        elif det.distance_m > cfg.target_distance_max_m:
            box_color = (245, 200, 60)  # Cyan-blue: following forward
        else:
            box_color = (60, 140, 255)  # Orange: closer than 1.0 m

        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

        # Corner accents
        clen = min(18, (x2 - x1) // 4, (y2 - y1) // 4)
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

        # Label above bounding box
        label = f"PERSON {det.confidence:.0%} | Dist: {det.distance_m:.2f}m | H:{det.box_height}px"
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


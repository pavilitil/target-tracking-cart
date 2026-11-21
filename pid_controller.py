"""
Dual PID Controller (Angular Steering + Linear Distance/Speed) and Differential Drive Mixer
for the Target Tracking Assistant Cart.
"""

import math
import time
from typing import Dict, Optional, Tuple
from config import CartConfig, CONFIG


class PIDController:
    """
    Discrete-time PID controller with deadband, integral anti-windup,
    and derivative smoothing filter.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        min_out: float,
        max_out: float,
        deadband: float = 0.0,
        integral_limit: float = 100.0,
        d_filter_alpha: float = 0.35,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_out = min_out
        self.max_out = max_out
        self.deadband = deadband
        self.integral_limit = integral_limit
        self.d_filter_alpha = d_filter_alpha

        self._integral: float = 0.0
        self._prev_error: Optional[float] = None
        self._filtered_d: float = 0.0
        self._last_time: Optional[float] = None
        self.last_p: float = 0.0
        self.last_i: float = 0.0
        self.last_d: float = 0.0
        self.last_output: float = 0.0

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None
        self._filtered_d = 0.0
        self._last_time = None
        self.last_p = 0.0
        self.last_i = 0.0
        self.last_d = 0.0
        self.last_output = 0.0

    def update(
        self,
        error: float,
        now: Optional[float] = None,
        error_rate: Optional[float] = None,
    ) -> float:
        """
        error_rate (units/s): externally estimated d(error)/dt, e.g. the velocity from the
        vision Alpha-Beta filter. When given it replaces the finite difference of the error,
        so measurement steps / re-acquisition jumps do not spike the D term.
        """
        if now is None:
            now = time.monotonic()

        if self._last_time is None:
            dt = 1.0 / 30.0
        else:
            dt = max(0.005, min(0.25, now - self._last_time))
        self._last_time = now

        # Apply deadband around zero error to prevent motor hunting
        if abs(error) <= self.deadband:
            self._integral *= 0.80
            self._prev_error = 0.0
            self._filtered_d = 0.0
            self.last_p = 0.0
            self.last_i = 0.0
            self.last_d = 0.0
            self.last_output = 0.0
            return 0.0

        # Subtract deadband for continuous transition without step jump
        effective_error = (
            error - self.deadband if error > 0 else error + self.deadband
        )

        # Reset integral on zero-crossing to prevent overshoot
        if self._prev_error is not None and (effective_error * self._prev_error < 0):
            self._integral = 0.0

        # Proportional term
        self.last_p = self.kp * effective_error

        # Integral term with anti-windup clamp
        self._integral += effective_error * dt
        self._integral = max(
            -self.integral_limit, min(self.integral_limit, self._integral)
        )
        self.last_i = self.ki * self._integral

        # Derivative term with exponential moving average filter
        if error_rate is not None:
            raw_d = float(error_rate)
        elif self._prev_error is None:
            raw_d = 0.0
        else:
            raw_d = (effective_error - self._prev_error) / dt
        self._filtered_d = (
            self.d_filter_alpha * raw_d + (1.0 - self.d_filter_alpha) * self._filtered_d
        )
        self.last_d = self.kd * self._filtered_d
        self._prev_error = effective_error

        output = self.last_p + self.last_i + self.last_d
        output = max(self.min_out, min(self.max_out, output))
        self.last_output = output
        return output


class DualPIDCartController:
    """
    Combines:
      1. Angular PID (Steering by horizontal target offset cx - frame_center_x)
      2. Linear PID  (Speed by bounding box height for 1.0m - 1.5m target distance)
      3. Smooth Slew-Rate Ramp (soft stop when target reached or person lost)
      4. Ultrasonic Safety Layer (4x HC-SR04 obstacle avoidance)
      5. Occlusion hold: target hidden < target_lost_timeout_s -> keep the previous motion
         vector with exponential decay; afterwards ramp smoothly down to 0.
    """

    def __init__(self, config: CartConfig = CONFIG) -> None:
        self.cfg = config

        # 1. Angular PID (Steering)
        self.steering_pid = PIDController(
            kp=self.cfg.steer_kp,
            ki=self.cfg.steer_ki,
            kd=self.cfg.steer_kd,
            min_out=-self.cfg.steer_max_pwm,
            max_out=self.cfg.steer_max_pwm,
            deadband=self.cfg.steer_deadband_px,
            integral_limit=120.0,
        )

        # 2. Linear PID (Distance / Speed)
        self.speed_pid = PIDController(
            kp=self.cfg.speed_kp,
            ki=self.cfg.speed_ki,
            kd=self.cfg.speed_kd,
            min_out=-self.cfg.speed_max_reverse_pwm,
            max_out=self.cfg.speed_max_forward_pwm,
            deadband=self.cfg.box_height_deadband_px,
            integral_limit=100.0,
        )

        self._smooth_linear: float = 0.0
        self._smooth_angular: float = 0.0
        self._last_seen_time: Optional[float] = None
        self._occlusion_active: bool = False
        self._hold_linear: float = 0.0
        self._hold_angular: float = 0.0
        self.status_reason: str = "IDLE"

    def reset(self) -> None:
        self.steering_pid.reset()
        self.speed_pid.reset()
        self._smooth_linear = 0.0
        self._smooth_angular = 0.0
        self._last_seen_time = None
        self._occlusion_active = False
        self._hold_linear = 0.0
        self._hold_angular = 0.0
        self.status_reason = "STOPPED"

    def update_gains(
        self,
        steer_kp: Optional[float] = None,
        steer_ki: Optional[float] = None,
        steer_kd: Optional[float] = None,
        speed_kp: Optional[float] = None,
        speed_ki: Optional[float] = None,
        speed_kd: Optional[float] = None,
        target_box_height: Optional[int] = None,
    ) -> None:
        # Gains are mirrored into the config so CartConfig.save_to_json() persists them
        if steer_kp is not None:
            self.steering_pid.kp = self.cfg.steer_kp = float(steer_kp)
        if steer_ki is not None:
            self.steering_pid.ki = self.cfg.steer_ki = float(steer_ki)
        if steer_kd is not None:
            self.steering_pid.kd = self.cfg.steer_kd = float(steer_kd)
        if speed_kp is not None:
            self.speed_pid.kp = self.cfg.speed_kp = float(speed_kp)
        if speed_ki is not None:
            self.speed_pid.ki = self.cfg.speed_ki = float(speed_ki)
        if speed_kd is not None:
            self.speed_pid.kd = self.cfg.speed_kd = float(speed_kd)
        if target_box_height is not None:
            self.cfg.target_box_height_px = int(target_box_height)

    @staticmethod
    def _slew_limit(current: float, target: float, accel_step: float, decel_step: float) -> float:
        diff = target - current
        # Use decel_step when slowing down toward 0 or reversing sign
        step = decel_step if (abs(target) < abs(current) or target * current < 0) else accel_step
        if abs(diff) <= step:
            return target
        return current + step if diff > 0 else current - step

    def compute_auto_command(
        self,
        detected: bool,
        cx: int,
        cy: int,
        box_height: int,
        sensors_cm: Tuple[int, int, int, int],
        vx: Optional[float] = None,
        v_box_height: Optional[float] = None,
        now: Optional[float] = None,
    ) -> Tuple[int, int, Dict[str, float]]:
        """
        Calculates (left_pwm, right_pwm, debug_metrics) in AUTO AI mode.
        sensors_cm = (D1_front_left, D2_front_right, D3_left, D4_right)
        vx, v_box_height = velocities (px/s) estimated by the 1D Alpha-Beta filter.
        """
        if now is None:
            now = time.monotonic()
        d1_fl, d2_fr, d3_l, d4_r = sensors_cm
        min_front_cm = min(d1_fl, d2_fr)

        frame_center_x = self.cfg.frame_width // 2
        error_x = 0.0
        error_h = 0.0

        if detected and box_height > 20:
            self._last_seen_time = now
            self._occlusion_active = False

            # 1. Angular Error: positive when person is to the right of center
            # d(cx - center_x)/dt = vx from Alpha-Beta filter (avoids finite-difference D spikes)
            error_x = float(cx - frame_center_x)
            error_x_rate = float(vx) if vx is not None else None
            raw_angular = self.steering_pid.update(error_x, now, error_rate=error_x_rate)

            # 2. Linear Error: positive when person is farther than setpoint (box smaller)
            # d(target_bh - box_height)/dt = -v_box_height from Alpha-Beta filter
            error_h = float(self.cfg.target_box_height_px - box_height)
            error_h_rate = -float(v_box_height) if v_box_height is not None else None

            low_bh = self.cfg.target_box_height_px - self.cfg.hold_band_far_px
            high_bh = self.cfg.target_box_height_px + self.cfg.hold_band_near_px
            if low_bh <= box_height <= high_bh:
                self.speed_pid.reset()
                raw_linear = 0.0
                self.status_reason = "TARGET_DIST_OK (1.0-1.5m)"
            else:
                raw_linear = self.speed_pid.update(error_h, now, error_rate=error_h_rate)
                if raw_linear > 0:
                    self.status_reason = "FOLLOWING_FORWARD"
                elif raw_linear < 0:
                    self.status_reason = "TOO_CLOSE_BACKING_OFF"
                else:
                    self.status_reason = "HOLDING_DISTANCE"

            self._hold_linear = raw_linear
            self._hold_angular = raw_angular
        else:
            # Short occlusion (< 0.4s): keep previous vector with velocity decay;
            # after 0.4s: smoothly brake to 0.
            lost_elapsed = (
                (now - self._last_seen_time)
                if self._last_seen_time is not None
                else float("inf")
            )
            if lost_elapsed < self.cfg.target_lost_timeout_s:
                self._occlusion_active = True
                tau = max(0.05, float(self.cfg.occlusion_decay_tau_s))
                decay = math.exp(-lost_elapsed / tau)
                raw_linear = self._hold_linear * decay
                raw_angular = self._hold_angular * decay
                self.status_reason = f"OCCLUSION_HOLD ({lost_elapsed:.2f}s)"
            else:
                self._occlusion_active = False
                self._hold_linear = 0.0
                self._hold_angular = 0.0
                raw_linear = 0.0
                raw_angular = 0.0
                self.steering_pid.reset()
                self.speed_pid.reset()
                self.status_reason = "TARGET_LOST_SOFT_STOP"

        # 3. Ultrasonic Safety & Obstacle Avoidance Layer
        if min_front_cm < self.cfg.obstacle_stop_cm:
            # Hard safety interlock: block any forward velocity immediately
            if raw_linear > 0:
                raw_linear = 0.0
            if self._smooth_linear > 0:
                self._smooth_linear = 0.0
            self.status_reason = f"OBSTACLE_STOP ({min_front_cm}cm < {self.cfg.obstacle_stop_cm}cm)"
        elif min_front_cm < self.cfg.obstacle_slow_cm and raw_linear > 0:
            # Proportional slowdown between 75 cm and 35 cm
            scale = (min_front_cm - self.cfg.obstacle_stop_cm) / float(
                self.cfg.obstacle_slow_cm - self.cfg.obstacle_stop_cm
            )
            raw_linear *= max(0.0, min(1.0, scale))
            self.status_reason = f"OBSTACLE_SLOWDOWN ({min_front_cm}cm)"

        # Gentle side obstacle avoidance nudge when moving forward
        if raw_linear > 15.0:
            if d3_l < self.cfg.side_nudge_cm and d3_l < d4_r:
                # Obstacle on left -> nudge right
                raw_angular += 28.0 * (1.0 - d3_l / float(self.cfg.side_nudge_cm))
            elif d4_r < self.cfg.side_nudge_cm and d4_r < d3_l:
                # Obstacle on right -> nudge left
                raw_angular -= 28.0 * (1.0 - d4_r / float(self.cfg.side_nudge_cm))

        # 4. Smooth Slew-Rate Limiting (protects worm gears & 5-10 kg cargo)
        self._smooth_linear = self._slew_limit(
            self._smooth_linear,
            raw_linear,
            self.cfg.soft_ramp_accel_step,
            self.cfg.soft_ramp_decel_step,
        )
        self._smooth_angular = self._slew_limit(
            self._smooth_angular,
            raw_angular,
            self.cfg.soft_ramp_accel_step * 1.4,
            self.cfg.soft_ramp_decel_step * 1.5,
        )

        # 5. Differential Drive Mixing
        # Left wheel = Linear + Angular, Right wheel = Linear - Angular
        left_f = self._smooth_linear + self._smooth_angular
        right_f = self._smooth_linear - self._smooth_angular

        # Normalize if either wheel exceeds max_pwm to preserve steering curvature
        peak = max(abs(left_f), abs(right_f))
        if peak > self.cfg.max_pwm:
            scale = self.cfg.max_pwm / peak
            left_f *= scale
            right_f *= scale

        left_pwm = int(round(left_f))
        right_pwm = int(round(right_f))

        # Apply minimum effective PWM deadzone cleanup
        if abs(left_pwm) < self.cfg.min_effective_pwm:
            left_pwm = 0
        if abs(right_pwm) < self.cfg.min_effective_pwm:
            right_pwm = 0

        # Final safety guarantee: never send positive PWM when front obstacle < 35 cm
        if min_front_cm < self.cfg.obstacle_stop_cm:
            if left_pwm > 0:
                left_pwm = 0
            if right_pwm > 0:
                right_pwm = 0

        metrics = {
            "error_x": round(error_x, 1),
            "error_h": round(error_h, 1),
            "linear_pwm": round(self._smooth_linear, 1),
            "angular_pwm": round(self._smooth_angular, 1),
            "steer_p": round(self.steering_pid.last_p, 1),
            "steer_i": round(self.steering_pid.last_i, 1),
            "steer_d": round(self.steering_pid.last_d, 1),
            "speed_p": round(self.speed_pid.last_p, 1),
            "speed_i": round(self.speed_pid.last_i, 1),
            "speed_d": round(self.speed_pid.last_d, 1),
        }
        return left_pwm, right_pwm, metrics


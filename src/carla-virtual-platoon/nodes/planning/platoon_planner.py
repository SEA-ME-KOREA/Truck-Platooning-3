#!/usr/bin/env python3

import time


class PlatoonPlanner:
    def __init__(
        self,
        *,
        desired_distance: float,
        gain: float,
        min_speed: float,
        max_speed: float,
        time_gap: float = 0.8,
        safe_distance: float = 4.5,
        kp: float = 0.3,
        ki: float = 0.01,
        kd: float = 1.8,
    ) -> None:
        self.desired_distance = desired_distance
        self.gain = gain
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.time_gap = time_gap
        self.safe_distance = safe_distance
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = time.time()

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = time.time()

    def adjust_speed(
        self, base_speed: float, front_vehicle_distance: float, ego_speed: float = 0.0
    ) -> float:
        if front_vehicle_distance <= 0.0:
            return float(base_speed)

        dynamic_target_distance = max(
            float(self.desired_distance),
            float(self.desired_distance) + float(self.time_gap) * max(0.0, float(ego_speed)),
        )
        error = float(front_vehicle_distance) - dynamic_target_distance

        now = time.time()
        dt = max(1e-3, now - self._prev_time)
        self._prev_time = now

        self._integral += error * dt
        self._integral = max(-20.0, min(20.0, self._integral))
        derivative = (error - self._prev_error) / dt
        self._prev_error = error

        pid_correction = self.kp * error + self.ki * self._integral + self.kd * derivative
        target_speed = float(base_speed + self.gain * error + pid_correction)

        if front_vehicle_distance < self.safe_distance:
            return 0.0

        target_speed = max(float(self.min_speed), target_speed)
        target_speed = min(float(self.max_speed), target_speed)
        return max(0.0, target_speed)

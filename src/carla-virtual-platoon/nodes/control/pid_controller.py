#!/usr/bin/env python3


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, integral_limit: float = 1e9) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = abs(integral_limit)
        self._integral = 0.0
        self._prev_error = None

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None

    def step(self, error: float, dt: float) -> float:
        self._integral += error * dt
        self._integral = max(-self.integral_limit, min(self.integral_limit, self._integral))
        derivative = 0.0 if self._prev_error is None or dt <= 0.0 else (error - self._prev_error) / dt
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * derivative

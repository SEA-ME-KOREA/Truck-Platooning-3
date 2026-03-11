#!/usr/bin/env python3

from pid_controller import PIDController


class LateralController:
    def __init__(self, *, kp: float, ki: float, kd: float, max_steer_deg: float) -> None:
        self._pid = PIDController(kp, ki, kd, integral_limit=10.0)
        self.max_steer_deg = max_steer_deg

    def reset(self) -> None:
        self._pid.reset()

    def compute(self, target_steer_error: float, dt: float) -> float:
        steer = self._pid.step(target_steer_error, dt)
        return max(-self.max_steer_deg, min(self.max_steer_deg, steer))

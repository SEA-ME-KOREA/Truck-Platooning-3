#!/usr/bin/env python3

from pid_controller import PIDController


class LongitudinalController:
    def __init__(self, *, kp: float, ki: float, kd: float, max_throttle: float, max_brake: float) -> None:
        self._pid = PIDController(kp, ki, kd, integral_limit=20.0)
        self.max_throttle = max_throttle
        self.max_brake = max_brake

    def reset(self) -> None:
        self._pid.reset()

    def compute(self, target_speed: float, current_speed: float, dt: float) -> float:
        cmd = self._pid.step(target_speed - current_speed, dt)
        return max(-self.max_brake, min(self.max_throttle, cmd))

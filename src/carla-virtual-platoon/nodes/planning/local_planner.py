#!/usr/bin/env python3


class LocalPlanner:
    """Shapes lateral target error before it reaches the actuator PID."""

    def __init__(
        self,
        *,
        steer_gain: float = 0.7,
        steer_limit: float = 0.35,
        error_deadband: float = 0.02,
        error_alpha: float = 0.8,
    ) -> None:
        self._steer_gain = float(steer_gain)
        self._steer_limit = abs(float(steer_limit))
        self._error_deadband = abs(float(error_deadband))
        self._error_alpha = min(1.0, max(0.0, float(error_alpha)))
        self._filtered_error = 0.0

    def reset(self) -> None:
        self._filtered_error = 0.0

    def compute_target_steer_error(self, lane_error: float) -> float:
        error = float(lane_error)
        if abs(error) < self._error_deadband:
            error = 0.0

        self._filtered_error = (
            self._error_alpha * self._filtered_error
            + (1.0 - self._error_alpha) * error
        )
        # Push harder as the truck departs the lane center so correction is
        # visible on wide Town04_Opt lanes instead of saturating too late.
        shaped_error = self._filtered_error + 0.35 * self._filtered_error * abs(
            self._filtered_error
        )
        target = shaped_error * self._steer_gain
        return max(-self._steer_limit, min(self._steer_limit, target))

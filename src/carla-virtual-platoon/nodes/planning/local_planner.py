#!/usr/bin/env python3


class LocalPlanner:
    """Keeps planning output at target-level, not actuator-level."""

    def compute_target_steer_error(self, lane_error: float) -> float:
        return float(lane_error)

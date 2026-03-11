#!/usr/bin/env python3


class PlatoonPlanner:
    def __init__(self, *, desired_distance: float, gain: float) -> None:
        self.desired_distance = desired_distance
        self.gain = gain

    def adjust_speed(self, base_speed: float, front_vehicle_distance: float) -> float:
        if front_vehicle_distance <= 0.0:
            return float(base_speed)
        error = front_vehicle_distance - self.desired_distance
        return max(0.0, float(base_speed + self.gain * error))

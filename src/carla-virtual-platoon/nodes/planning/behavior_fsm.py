#!/usr/bin/env python3

from dataclasses import dataclass


@dataclass
class PlannerInputs:
    lane_confidence: float
    stop_line_detected: bool
    front_vehicle_distance: float
    traffic_light_state: str


@dataclass
class PlannerOutputs:
    state: str
    target_speed: float


class BehaviorFSM:
    def __init__(
        self,
        *,
        cruise_speed: float,
        slow_speed: float,
        stop_distance: float,
        caution_distance: float,
        lane_conf_threshold: float,
        stop_line_hold_sec: float,
    ) -> None:
        self.cruise_speed = cruise_speed
        self.slow_speed = slow_speed
        self.stop_distance = stop_distance
        self.caution_distance = caution_distance
        self.lane_conf_threshold = lane_conf_threshold
        self.stop_line_hold_sec = stop_line_hold_sec
        self.stop_line_until = None

    def tick(self, inputs: PlannerInputs, now_sec: float) -> PlannerOutputs:
        mode = "LANE_FOLLOW"
        target_speed = self.cruise_speed

        if inputs.stop_line_detected:
            self.stop_line_until = now_sec + max(0.0, self.stop_line_hold_sec)
        if self.stop_line_until is not None and now_sec < self.stop_line_until:
            mode = "STOP_FOR_STOPLINE"
            target_speed = 0.0
        elif inputs.front_vehicle_distance < self.stop_distance:
            mode = "STOP_FOR_OBSTACLE"
            target_speed = 0.0
        elif inputs.front_vehicle_distance < self.caution_distance:
            mode = "PLATOON_FOLLOW"
            target_speed = self.slow_speed
        elif inputs.lane_confidence < self.lane_conf_threshold:
            mode = "INTERSECTION_CROSS"
            target_speed = self.slow_speed
        elif inputs.traffic_light_state == "RED":
            mode = "STOP_FOR_RED"
            target_speed = 0.0
        elif self.stop_line_until is not None and now_sec >= self.stop_line_until:
            self.stop_line_until = None

        return PlannerOutputs(state=mode, target_speed=target_speed)

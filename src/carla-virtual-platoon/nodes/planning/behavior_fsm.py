#!/usr/bin/env python3

from dataclasses import dataclass


@dataclass
class PlannerInputs:
    lane_confidence: float
    front_vehicle_distance: float


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
        lane_search_speed: float,
        stop_distance: float,
        caution_distance: float,
        lane_conf_threshold: float,
        lane_conf_stop_threshold: float,
    ) -> None:
        self.cruise_speed = cruise_speed
        self.slow_speed = slow_speed
        self.lane_search_speed = lane_search_speed
        self.stop_distance = stop_distance
        self.caution_distance = caution_distance
        self.lane_conf_threshold = lane_conf_threshold
        self.lane_conf_stop_threshold = lane_conf_stop_threshold

    def tick(self, inputs: PlannerInputs, now_sec: float) -> PlannerOutputs:
        mode = "LANE_FOLLOW"
        target_speed = self.cruise_speed

        if inputs.front_vehicle_distance < self.stop_distance:
            mode = "STOP_FOR_OBSTACLE"
            target_speed = 0.0
        elif inputs.lane_confidence < self.lane_conf_stop_threshold:
            mode = "LANE_RECOVERY"
            target_speed = self.lane_search_speed
        elif inputs.front_vehicle_distance < self.caution_distance:
            mode = "PLATOON_FOLLOW"
            target_speed = self.slow_speed
        elif inputs.lane_confidence < self.lane_conf_threshold:
            mode = "LANE_SEARCH"
            target_speed = self.lane_search_speed

        return PlannerOutputs(state=mode, target_speed=target_speed)

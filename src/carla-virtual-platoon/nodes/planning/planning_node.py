#!/usr/bin/env python3

import os
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from behavior_fsm import BehaviorFSM, PlannerInputs
from local_planner import LocalPlanner
from platoon_planner import PlatoonPlanner


class PlanningNode(Node):
    def __init__(self) -> None:
        super().__init__("planning_node")
        self.declare_parameter("lane_error_topic", "perception/lane_error")
        self.declare_parameter("lane_conf_topic", "perception/lane_confidence")
        self.declare_parameter("front_vehicle_topic", "perception/front_vehicle_distance")
        self.declare_parameter("stop_line_topic", "perception/stop_line_detected")
        self.declare_parameter("traffic_light_topic", "perception/traffic_light_state")
        self.declare_parameter("state_topic", "planning/state")
        self.declare_parameter("target_speed_topic", "planning/target_speed")
        self.declare_parameter("target_steer_error_topic", "planning/target_steer_error")

        self.declare_parameter("cruise_speed", 6.0)
        self.declare_parameter("slow_speed", 2.5)
        self.declare_parameter("stop_distance", 6.0)
        self.declare_parameter("caution_distance", 15.0)
        self.declare_parameter("lane_conf_threshold", 0.2)
        self.declare_parameter("stop_line_hold_sec", 1.5)
        self.declare_parameter("platoon_desired_distance", 12.0)
        self.declare_parameter("platoon_speed_gain", 0.15)

        self._lane_error = 0.0
        self._lane_confidence = 0.0
        self._front_vehicle_distance = 999.0
        self._stop_line_detected = False
        self._traffic_light_state = "UNKNOWN"

        self._fsm = BehaviorFSM(
            cruise_speed=float(self.get_parameter("cruise_speed").value),
            slow_speed=float(self.get_parameter("slow_speed").value),
            stop_distance=float(self.get_parameter("stop_distance").value),
            caution_distance=float(self.get_parameter("caution_distance").value),
            lane_conf_threshold=float(self.get_parameter("lane_conf_threshold").value),
            stop_line_hold_sec=float(self.get_parameter("stop_line_hold_sec").value),
        )
        self._local_planner = LocalPlanner()
        self._platoon_planner = PlatoonPlanner(
            desired_distance=float(self.get_parameter("platoon_desired_distance").value),
            gain=float(self.get_parameter("platoon_speed_gain").value),
        )

        self._state_pub = self.create_publisher(String, self.get_parameter("state_topic").value, 10)
        self._speed_pub = self.create_publisher(
            Float32, self.get_parameter("target_speed_topic").value, 10
        )
        self._steer_error_pub = self.create_publisher(
            Float32, self.get_parameter("target_steer_error_topic").value, 10
        )

        self.create_subscription(
            Float32, self.get_parameter("lane_error_topic").value, self._on_lane_error, 10
        )
        self.create_subscription(
            Float32, self.get_parameter("lane_conf_topic").value, self._on_lane_confidence, 10
        )
        self.create_subscription(
            Float32, self.get_parameter("front_vehicle_topic").value, self._on_front_vehicle, 10
        )
        self.create_subscription(
            Bool, self.get_parameter("stop_line_topic").value, self._on_stop_line, 10
        )
        self.create_subscription(
            String, self.get_parameter("traffic_light_topic").value, self._on_traffic_light, 10
        )
        self.create_timer(0.1, self._tick)

    def _on_lane_error(self, msg: Float32) -> None:
        self._lane_error = float(msg.data)

    def _on_lane_confidence(self, msg: Float32) -> None:
        self._lane_confidence = float(msg.data)

    def _on_front_vehicle(self, msg: Float32) -> None:
        self._front_vehicle_distance = float(msg.data)

    def _on_stop_line(self, msg: Bool) -> None:
        self._stop_line_detected = bool(msg.data)

    def _on_traffic_light(self, msg: String) -> None:
        self._traffic_light_state = msg.data or "UNKNOWN"

    def _tick(self) -> None:
        outputs = self._fsm.tick(
            PlannerInputs(
                lane_confidence=self._lane_confidence,
                stop_line_detected=self._stop_line_detected,
                front_vehicle_distance=self._front_vehicle_distance,
                traffic_light_state=self._traffic_light_state,
            ),
            now_sec=self.get_clock().now().nanoseconds / 1e9,
        )
        target_speed = outputs.target_speed
        if outputs.state == "PLATOON_FOLLOW":
            target_speed = self._platoon_planner.adjust_speed(
                target_speed, self._front_vehicle_distance
            )

        target_steer_error = self._local_planner.compute_target_steer_error(self._lane_error)

        self._state_pub.publish(String(data=outputs.state))
        self._speed_pub.publish(Float32(data=float(target_speed)))
        self._steer_error_pub.publish(Float32(data=float(target_steer_error)))


def main() -> None:
    rclpy.init()
    node = PlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

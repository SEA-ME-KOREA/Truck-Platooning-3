#!/usr/bin/env python3

import os
import sys

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, String

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
        self.declare_parameter("velocity_topic", "velocity")
        self.declare_parameter("state_topic", "planning/state")
        self.declare_parameter("target_speed_topic", "planning/target_speed")
        self.declare_parameter("target_steer_error_topic", "planning/target_steer_error")
        self.declare_parameter("pose_topic", "pose_from_carla")

        self.declare_parameter("cruise_speed", 3.0)
        self.declare_parameter("slow_speed", 2.2)
        self.declare_parameter("lane_search_speed", 1.2)
        self.declare_parameter("stop_distance", 8.0)
        self.declare_parameter("caution_distance", 55.0)
        self.declare_parameter("lane_conf_threshold", 0.1)
        self.declare_parameter("lane_conf_stop_threshold", 0.08)
        self.declare_parameter("steer_enable_conf_threshold", 0.12)
        self.declare_parameter("platoon_desired_distance", 24.0)
        self.declare_parameter("platoon_speed_gain", 0.22)
        self.declare_parameter("platoon_min_speed", 1.0)
        self.declare_parameter("platoon_max_speed", 3.5)
        self.declare_parameter("platoon_time_gap", 1.2)
        self.declare_parameter("platoon_safe_distance", 7.0)
        self.declare_parameter("platoon_pid_kp", 0.22)
        self.declare_parameter("platoon_pid_ki", 0.01)
        self.declare_parameter("platoon_pid_kd", 1.2)
        self.declare_parameter("steer_target_gain", 1.08)
        self.declare_parameter("steer_target_limit", 0.36)
        self.declare_parameter("steer_error_deadband", 0.01)
        self.declare_parameter("steer_error_alpha", 0.90)

        self._lane_error = 0.0
        self._lane_confidence = 0.0
        self._front_vehicle_distance = 999.0
        self._current_speed = 0.0
        self._pose_x = 0.0
        self._pose_y = 0.0
        self._front_pose_x = 0.0
        self._front_pose_y = 0.0
        self._have_pose = False
        self._have_front_pose = False
        self._truck_index = self._resolve_truck_index()
        steer_gain = float(self.get_parameter("steer_target_gain").value)
        steer_limit = float(self.get_parameter("steer_target_limit").value)
        steer_error_alpha = float(self.get_parameter("steer_error_alpha").value)
        if self._truck_index >= 2:
            steer_gain *= 0.82
            steer_limit *= 0.85
            steer_error_alpha = min(0.95, steer_error_alpha + 0.04)

        self._fsm = BehaviorFSM(
            cruise_speed=float(self.get_parameter("cruise_speed").value),
            slow_speed=float(self.get_parameter("slow_speed").value),
            lane_search_speed=float(self.get_parameter("lane_search_speed").value),
            stop_distance=float(self.get_parameter("stop_distance").value),
            caution_distance=float(self.get_parameter("caution_distance").value),
            lane_conf_threshold=float(self.get_parameter("lane_conf_threshold").value),
            lane_conf_stop_threshold=float(
                self.get_parameter("lane_conf_stop_threshold").value
            ),
        )
        self._local_planner = LocalPlanner(
            steer_gain=steer_gain,
            steer_limit=steer_limit,
            error_deadband=float(self.get_parameter("steer_error_deadband").value),
            error_alpha=steer_error_alpha,
        )
        self._platoon_planner = PlatoonPlanner(
            desired_distance=float(self.get_parameter("platoon_desired_distance").value),
            gain=float(self.get_parameter("platoon_speed_gain").value),
            min_speed=float(self.get_parameter("platoon_min_speed").value),
            max_speed=float(self.get_parameter("platoon_max_speed").value),
            time_gap=float(self.get_parameter("platoon_time_gap").value),
            safe_distance=float(self.get_parameter("platoon_safe_distance").value),
            kp=float(self.get_parameter("platoon_pid_kp").value),
            ki=float(self.get_parameter("platoon_pid_ki").value),
            kd=float(self.get_parameter("platoon_pid_kd").value),
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
            Float32, self.get_parameter("velocity_topic").value, self._on_velocity, 10
        )
        self.create_subscription(
            PoseStamped, self.get_parameter("pose_topic").value, self._on_pose, 10
        )
        if self._truck_index > 0:
            self.create_subscription(
                PoseStamped,
                f"/truck{self._truck_index - 1}/pose_from_carla",
                self._on_front_pose,
                10,
            )
        self.create_timer(0.1, self._tick)

    def _on_lane_error(self, msg: Float32) -> None:
        self._lane_error = float(msg.data)

    def _on_lane_confidence(self, msg: Float32) -> None:
        self._lane_confidence = float(msg.data)

    def _on_front_vehicle(self, msg: Float32) -> None:
        self._front_vehicle_distance = float(msg.data)

    def _on_velocity(self, msg: Float32) -> None:
        self._current_speed = float(msg.data)

    def _on_pose(self, msg: PoseStamped) -> None:
        self._pose_x = float(msg.pose.position.x)
        self._pose_y = float(msg.pose.position.y)
        self._have_pose = True

    def _on_front_pose(self, msg: PoseStamped) -> None:
        self._front_pose_x = float(msg.pose.position.x)
        self._front_pose_y = float(msg.pose.position.y)
        self._have_front_pose = True

    def _resolve_truck_index(self) -> int:
        namespace = (self.get_namespace() or '').strip('/')
        if namespace.startswith('truck'):
            try:
                return int(namespace.replace('truck', ''))
            except ValueError:
                return 0
        return 0

    def _ground_truth_front_distance(self) -> float:
        if self._truck_index <= 0 or not (self._have_pose and self._have_front_pose):
            return -1.0
        dx = self._front_pose_x - self._pose_x
        dy = self._front_pose_y - self._pose_y
        return max(0.0, (dx * dx + dy * dy) ** 0.5)

    def _tick(self) -> None:
        if self._lane_confidence < 0.05:
            self._local_planner.reset()
            self._platoon_planner.reset()

        front_vehicle_distance = self._front_vehicle_distance
        gt_front_distance = self._ground_truth_front_distance()
        if gt_front_distance > 0.0:
            front_vehicle_distance = gt_front_distance

        outputs = self._fsm.tick(
            PlannerInputs(
                lane_confidence=self._lane_confidence,
                front_vehicle_distance=front_vehicle_distance,
            ),
            now_sec=self.get_clock().now().nanoseconds / 1e9,
        )
        target_speed = outputs.target_speed
        if outputs.state == "PLATOON_FOLLOW":
            target_speed = self._platoon_planner.adjust_speed(
                target_speed, front_vehicle_distance, self._current_speed
            )

        target_steer_error = self._local_planner.compute_target_steer_error(self._lane_error)
        if (
            outputs.state in ("LANE_RECOVERY", "LANE_SEARCH")
            or self._lane_confidence
            < float(self.get_parameter("steer_enable_conf_threshold").value)
        ):
            self._local_planner.reset()
            target_steer_error = 0.0

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

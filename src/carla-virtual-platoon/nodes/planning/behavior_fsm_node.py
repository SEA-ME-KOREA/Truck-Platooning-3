#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


class BehaviorFsmNode(Node):
    def __init__(self) -> None:
        super().__init__("behavior_fsm_node")
        self.declare_parameter("lane_conf_topic", "perception/lane_confidence")
        self.declare_parameter("obstacle_dist_topic", "perception/front_obstacle_distance")
        self.declare_parameter("traffic_light_topic", "perception/traffic_light_state")
        self.declare_parameter("stop_line_topic", "perception/stop_line_detected")
        self.declare_parameter("target_speed_topic", "control/target_speed")
        self.declare_parameter("mode_topic", "control/behavior_mode")
        self.declare_parameter("cruise_speed", 6.0)
        self.declare_parameter("slow_speed", 2.5)
        self.declare_parameter("stop_distance", 6.0)
        self.declare_parameter("caution_distance", 15.0)
        self.declare_parameter("lane_conf_threshold", 0.2)
        self.declare_parameter("stop_line_hold_sec", 1.5)

        self._lane_conf: float = 0.0
        self._obstacle_dist: float = 999.0
        self._traffic_light: str = "UNKNOWN"
        self._stop_line_detected: bool = False
        self._stop_line_until = None

        self._speed_pub = self.create_publisher(
            Float32, self.get_parameter("target_speed_topic").value, 10
        )
        self._mode_pub = self.create_publisher(
            String, self.get_parameter("mode_topic").value, 10
        )

        self.create_subscription(
            Float32, self.get_parameter("lane_conf_topic").value, self._on_lane_conf, 10
        )
        self.create_subscription(
            Float32,
            self.get_parameter("obstacle_dist_topic").value,
            self._on_obstacle_dist,
            10,
        )
        self.create_subscription(
            String,
            self.get_parameter("traffic_light_topic").value,
            self._on_traffic_light,
            10,
        )
        self.create_subscription(
            Bool,
            self.get_parameter("stop_line_topic").value,
            self._on_stop_line,
            10,
        )
        self.create_timer(0.1, self._tick)

    def _on_lane_conf(self, msg: Float32) -> None:
        self._lane_conf = float(msg.data)

    def _on_obstacle_dist(self, msg: Float32) -> None:
        self._obstacle_dist = float(msg.data)

    def _on_traffic_light(self, msg: String) -> None:
        self._traffic_light = msg.data or "UNKNOWN"

    def _on_stop_line(self, msg: Bool) -> None:
        detected = bool(msg.data)
        if detected and not self._stop_line_detected:
            hold = float(self.get_parameter("stop_line_hold_sec").value)
            self._stop_line_until = self.get_clock().now().nanoseconds / 1e9 + max(0.0, hold)
        self._stop_line_detected = detected

    def _tick(self) -> None:
        cruise = float(self.get_parameter("cruise_speed").value)
        slow = float(self.get_parameter("slow_speed").value)
        stop_d = float(self.get_parameter("stop_distance").value)
        caution_d = float(self.get_parameter("caution_distance").value)
        lane_conf_th = float(self.get_parameter("lane_conf_threshold").value)

        mode = "LANE_FOLLOW"
        target_speed = cruise

        now_sec = self.get_clock().now().nanoseconds / 1e9
        if self._stop_line_until is not None and now_sec < self._stop_line_until:
            mode = "STOP_FOR_STOPLINE"
            target_speed = 0.0
        elif self._stop_line_until is not None and now_sec >= self._stop_line_until:
            self._stop_line_until = None

        if self._obstacle_dist < stop_d:
            mode = "STOP_FOR_OBSTACLE"
            target_speed = 0.0
        elif mode != "STOP_FOR_STOPLINE" and self._obstacle_dist < caution_d:
            mode = "OBSTACLE_CAUTION"
            target_speed = slow
        elif mode != "STOP_FOR_STOPLINE" and self._lane_conf < lane_conf_th:
            mode = "INTERSECTION_CROSS"
            target_speed = slow
        elif mode != "STOP_FOR_STOPLINE" and self._traffic_light == "RED":
            mode = "STOP_FOR_RED"
            target_speed = 0.0

        speed_msg = Float32()
        speed_msg.data = target_speed
        mode_msg = String()
        mode_msg.data = mode
        self._speed_pub.publish(speed_msg)
        self._mode_pub.publish(mode_msg)


def main() -> None:
    rclpy.init()
    node = BehaviorFsmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

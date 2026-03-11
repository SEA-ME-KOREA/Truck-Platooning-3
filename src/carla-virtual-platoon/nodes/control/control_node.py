#!/usr/bin/env python3

import os
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from lateral_controller import LateralController
from longitudinal_controller import LongitudinalController


class ControlNode(Node):
    def __init__(self) -> None:
        super().__init__("control_node")
        self.declare_parameter("state_topic", "planning/state")
        self.declare_parameter("target_speed_topic", "planning/target_speed")
        self.declare_parameter("target_steer_error_topic", "planning/target_steer_error")
        self.declare_parameter("enabled_topic", "control/enabled")
        self.declare_parameter("velocity_topic", "velocity")
        self.declare_parameter("steer_topic", "steer_control")
        self.declare_parameter("throttle_topic", "throttle_control")
        self.declare_parameter("brake_topic", "brake_control")
        self.declare_parameter("vehicle_cmd_steer_topic", "vehicle_cmd/steer")
        self.declare_parameter("vehicle_cmd_throttle_topic", "vehicle_cmd/throttle")
        self.declare_parameter("vehicle_cmd_brake_topic", "vehicle_cmd/brake")

        self.declare_parameter("steer_kp", 15.0)
        self.declare_parameter("steer_ki", 0.0)
        self.declare_parameter("steer_kd", 0.5)
        self.declare_parameter("speed_kp", 0.25)
        self.declare_parameter("speed_ki", 0.02)
        self.declare_parameter("speed_kd", 0.0)
        self.declare_parameter("emergency_distance", 4.0)
        self.declare_parameter("front_vehicle_topic", "perception/front_vehicle_distance")
        self.declare_parameter("max_steer_deg", 30.0)
        self.declare_parameter("max_throttle", 1.0)
        self.declare_parameter("max_brake", 1.0)

        self._state = "IDLE"
        self._target_speed = 0.0
        self._target_steer_error = 0.0
        self._enabled = False
        self._current_speed = 0.0
        self._front_vehicle_distance = 999.0

        self._lateral = LateralController(
            kp=float(self.get_parameter("steer_kp").value),
            ki=float(self.get_parameter("steer_ki").value),
            kd=float(self.get_parameter("steer_kd").value),
            max_steer_deg=float(self.get_parameter("max_steer_deg").value),
        )
        self._longitudinal = LongitudinalController(
            kp=float(self.get_parameter("speed_kp").value),
            ki=float(self.get_parameter("speed_ki").value),
            kd=float(self.get_parameter("speed_kd").value),
            max_throttle=float(self.get_parameter("max_throttle").value),
            max_brake=float(self.get_parameter("max_brake").value),
        )

        self.create_subscription(String, self.get_parameter("state_topic").value, self._on_state, 10)
        self.create_subscription(
            Float32, self.get_parameter("target_speed_topic").value, self._on_target_speed, 10
        )
        self.create_subscription(
            Float32,
            self.get_parameter("target_steer_error_topic").value,
            self._on_target_steer_error,
            10,
        )
        self.create_subscription(
            Bool, self.get_parameter("enabled_topic").value, self._on_enabled, 10
        )
        self.create_subscription(
            Float32, self.get_parameter("velocity_topic").value, self._on_velocity, 10
        )
        self.create_subscription(
            Float32, self.get_parameter("front_vehicle_topic").value, self._on_front_vehicle, 10
        )

        self._steer_pub = self.create_publisher(Float32, self.get_parameter("steer_topic").value, 10)
        self._throttle_pub = self.create_publisher(
            Float32, self.get_parameter("throttle_topic").value, 10
        )
        self._brake_pub = self.create_publisher(Float32, self.get_parameter("brake_topic").value, 10)
        self._vehicle_cmd_steer_pub = self.create_publisher(
            Float32, self.get_parameter("vehicle_cmd_steer_topic").value, 10
        )
        self._vehicle_cmd_throttle_pub = self.create_publisher(
            Float32, self.get_parameter("vehicle_cmd_throttle_topic").value, 10
        )
        self._vehicle_cmd_brake_pub = self.create_publisher(
            Float32, self.get_parameter("vehicle_cmd_brake_topic").value, 10
        )

        self.create_timer(0.05, self._tick)

    def _on_state(self, msg: String) -> None:
        self._state = msg.data or "UNKNOWN"

    def _on_target_speed(self, msg: Float32) -> None:
        self._target_speed = float(msg.data)

    def _on_target_steer_error(self, msg: Float32) -> None:
        self._target_steer_error = float(msg.data)

    def _on_enabled(self, msg: Bool) -> None:
        self._enabled = bool(msg.data)
        if not self._enabled:
            self._lateral.reset()
            self._longitudinal.reset()

    def _on_velocity(self, msg: Float32) -> None:
        self._current_speed = float(msg.data)

    def _on_front_vehicle(self, msg: Float32) -> None:
        self._front_vehicle_distance = float(msg.data)

    def _tick(self) -> None:
        if not self._enabled:
            self._publish_commands(0.0, 0.0, 0.0)
            return

        steer = self._lateral.compute(self._target_steer_error, 0.05)
        signed_longitudinal = self._longitudinal.compute(self._target_speed, self._current_speed, 0.05)

        if self._front_vehicle_distance < float(self.get_parameter("emergency_distance").value):
            steer = 0.0
            signed_longitudinal = -float(self.get_parameter("max_brake").value)

        throttle = max(0.0, signed_longitudinal)
        brake = max(0.0, -signed_longitudinal)
        self._publish_commands(steer, throttle, brake)

        # Preserve compatibility with the existing CARLA actuator bridge:
        # positive => throttle, negative => brake.
        self._steer_pub.publish(Float32(data=float(steer)))
        self._throttle_pub.publish(Float32(data=float(throttle - brake)))

    def _publish_commands(self, steer: float, throttle: float, brake: float) -> None:
        self._brake_pub.publish(Float32(data=float(brake)))
        self._vehicle_cmd_steer_pub.publish(Float32(data=float(steer)))
        self._vehicle_cmd_throttle_pub.publish(Float32(data=float(throttle)))
        self._vehicle_cmd_brake_pub.publish(Float32(data=float(brake)))


def main() -> None:
    rclpy.init()
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32, String


class ControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("controller_node")
        self.declare_parameter("lane_offset_topic", "perception/lane_center_offset")
        self.declare_parameter("lane_conf_topic", "perception/lane_confidence")
        self.declare_parameter("obstacle_dist_topic", "perception/front_obstacle_distance")
        self.declare_parameter("target_speed_topic", "control/target_speed")
        self.declare_parameter("mode_topic", "control/behavior_mode")
        self.declare_parameter("enabled_topic", "control/enabled")
        self.declare_parameter("velocity_topic", "velocity")
        self.declare_parameter("steer_topic", "steer_control")
        self.declare_parameter("throttle_topic", "throttle_control")
        self.declare_parameter("k_lane", 15.0)  # deg / normalized offset
        self.declare_parameter("k_speed_p", 0.25)
        self.declare_parameter("emergency_distance", 4.0)
        self.declare_parameter("max_steer_deg", 30.0)
        self.declare_parameter("max_throttle", 1.0)
        self.declare_parameter("max_brake", 1.0)
        self.declare_parameter("startup_hold_sec", 1.0)
        self.declare_parameter("startup_ramp_sec", 2.0)
        self.declare_parameter("use_imu_heading_assist", False)
        self.declare_parameter("imu_topic", "imu")
        self.declare_parameter("yaw_target_rad", 0.0)
        self.declare_parameter("yaw_k", 0.25)
        self.declare_parameter("yaw_alpha", 0.10)
        self.declare_parameter("yaw_max_bias_deg", 6.0)

        self._lane_offset = 0.0
        self._lane_conf = 0.0
        self._obstacle_dist = 999.0
        self._target_speed = 0.0
        self._current_speed = 0.0
        self._mode = "IDLE"
        self._enabled = False
        self._enabled_since = None
        self._yaw = None

        self.create_subscription(
            Float32, self.get_parameter("lane_offset_topic").value, self._on_lane_offset, 10
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
            Float32, self.get_parameter("target_speed_topic").value, self._on_target_speed, 10
        )
        self.create_subscription(
            Bool, self.get_parameter("enabled_topic").value, self._on_enabled, 10
        )
        self.create_subscription(
            Float32, self.get_parameter("velocity_topic").value, self._on_current_speed, 10
        )
        self.create_subscription(String, self.get_parameter("mode_topic").value, self._on_mode, 10)
        self.create_subscription(Imu, self.get_parameter("imu_topic").value, self._on_imu, 10)

        self._steer_pub = self.create_publisher(Float32, self.get_parameter("steer_topic").value, 10)
        self._throttle_pub = self.create_publisher(
            Float32, self.get_parameter("throttle_topic").value, 10
        )
        self.create_timer(0.05, self._tick)

    def _on_lane_offset(self, msg: Float32) -> None:
        self._lane_offset = float(msg.data)

    def _on_lane_conf(self, msg: Float32) -> None:
        self._lane_conf = float(msg.data)

    def _on_obstacle_dist(self, msg: Float32) -> None:
        self._obstacle_dist = float(msg.data)

    def _on_target_speed(self, msg: Float32) -> None:
        self._target_speed = float(msg.data)

    def _on_current_speed(self, msg: Float32) -> None:
        self._current_speed = float(msg.data)

    def _on_enabled(self, msg: Bool) -> None:
        new_enabled = bool(msg.data)
        if new_enabled and not self._enabled:
            self._enabled_since = self.get_clock().now()
        if not new_enabled:
            self._enabled_since = None
        self._enabled = new_enabled

    def _on_mode(self, msg: String) -> None:
        self._mode = msg.data or "UNKNOWN"

    def _on_imu(self, msg: Imu) -> None:
        try:
            qx = float(msg.orientation.x)
            qy = float(msg.orientation.y)
            qz = float(msg.orientation.z)
            qw = float(msg.orientation.w)
        except Exception:
            return
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        if self._yaw is None:
            self._yaw = yaw
            return
        alpha = float(self.get_parameter("yaw_alpha").value)
        self._yaw = (1.0 - alpha) * self._yaw + alpha * yaw

    @staticmethod
    def _wrap_pi(v: float) -> float:
        return (v + math.pi) % (2.0 * math.pi) - math.pi

    def _tick(self) -> None:
        if not self._enabled:
            steer_msg = Float32()
            steer_msg.data = 0.0
            throttle_msg = Float32()
            throttle_msg.data = 0.0
            self._steer_pub.publish(steer_msg)
            self._throttle_pub.publish(throttle_msg)
            return

        max_steer = float(self.get_parameter("max_steer_deg").value)
        k_lane = float(self.get_parameter("k_lane").value)
        k_speed = float(self.get_parameter("k_speed_p").value)
        emergency_d = float(self.get_parameter("emergency_distance").value)
        max_throttle = float(self.get_parameter("max_throttle").value)
        max_brake = float(self.get_parameter("max_brake").value)

        # Steering fallback: reduce lane authority when confidence drops.
        lane_weight = max(0.1, min(1.0, self._lane_conf))
        steer_cmd = -k_lane * self._lane_offset * lane_weight

        if bool(self.get_parameter("use_imu_heading_assist").value) and self._yaw is not None:
            yaw_target = float(self.get_parameter("yaw_target_rad").value)
            yaw_k = float(self.get_parameter("yaw_k").value)
            max_bias = float(self.get_parameter("yaw_max_bias_deg").value)
            yaw_err = self._wrap_pi(self._yaw - yaw_target)
            steer_cmd += max(-max_bias, min(max_bias, yaw_k * yaw_err))

        steer_cmd = max(-max_steer, min(max_steer, steer_cmd))

        speed_error = self._target_speed - self._current_speed
        throttle_brake = k_speed * speed_error

        if self._obstacle_dist < emergency_d:
            throttle_brake = -max_brake
            steer_cmd = 0.0

        throttle_brake = max(-max_brake, min(max_throttle, throttle_brake))

        # Prevent immediate slip when control is enabled: keep the truck settled,
        # then gradually allow steering/throttle authority.
        hold_sec = float(self.get_parameter("startup_hold_sec").value)
        ramp_sec = float(self.get_parameter("startup_ramp_sec").value)
        if self._enabled_since is not None:
            elapsed = (self.get_clock().now() - self._enabled_since).nanoseconds / 1e9
            if elapsed < hold_sec:
                steer_cmd = 0.0
                throttle_brake = 0.0
            elif ramp_sec > 0.0 and elapsed < (hold_sec + ramp_sec):
                alpha = (elapsed - hold_sec) / ramp_sec
                alpha = max(0.0, min(1.0, alpha))
                steer_cmd *= alpha
                if throttle_brake > 0.0:
                    throttle_brake *= alpha

        steer_msg = Float32()
        steer_msg.data = float(steer_cmd)
        throttle_msg = Float32()
        throttle_msg.data = float(throttle_brake)

        self._steer_pub.publish(steer_msg)
        self._throttle_pub.publish(throttle_msg)


def main() -> None:
    rclpy.init()
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

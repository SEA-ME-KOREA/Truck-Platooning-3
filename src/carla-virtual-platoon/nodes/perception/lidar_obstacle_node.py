#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32

try:
    from sensor_msgs_py import point_cloud2
except Exception:  # pragma: no cover
    point_cloud2 = None


class LidarObstacleNode(Node):
    def __init__(self) -> None:
        super().__init__("lidar_obstacle_node")
        self.declare_parameter("lidar_topic", "front_lidar")
        self.declare_parameter("distance_topic", "perception/front_obstacle_distance")
        self.declare_parameter("max_distance", 50.0)
        self.declare_parameter("roi_y_abs", 2.0)
        self.declare_parameter("min_z", -2.0)
        self.declare_parameter("max_z", 3.0)

        lidar_topic = self.get_parameter("lidar_topic").get_parameter_value().string_value
        distance_topic = self.get_parameter("distance_topic").get_parameter_value().string_value

        self._pub = self.create_publisher(Float32, distance_topic, 10)
        self._sub = self.create_subscription(PointCloud2, lidar_topic, self._on_lidar, 10)

        self.get_logger().info(f"lidar_obstacle_node subscribed to '{lidar_topic}'")

    def _on_lidar(self, msg: PointCloud2) -> None:
        max_distance = self.get_parameter("max_distance").get_parameter_value().double_value
        min_distance = max_distance

        if point_cloud2 is not None:
            roi_y_abs = self.get_parameter("roi_y_abs").get_parameter_value().double_value
            min_z = self.get_parameter("min_z").get_parameter_value().double_value
            max_z = self.get_parameter("max_z").get_parameter_value().double_value

            try:
                for p in point_cloud2.read_points(
                    msg, field_names=("x", "y", "z"), skip_nans=True
                ):
                    x, y, z = float(p[0]), float(p[1]), float(p[2])
                    if x <= 0.0:
                        continue
                    if abs(y) > roi_y_abs or z < min_z or z > max_z:
                        continue
                    d = math.sqrt(x * x + y * y + z * z)
                    if d < min_distance:
                        min_distance = d
            except Exception as exc:
                self.get_logger().warn(f"PointCloud2 parse failed: {exc}")

        out = Float32()
        out.data = float(min_distance)
        self._pub.publish(out)


def main() -> None:
    rclpy.init()
    node = LidarObstacleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

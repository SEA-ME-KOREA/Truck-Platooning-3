#!/usr/bin/env python3

import os
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Bool, Float32

try:
    from cv_bridge import CvBridge
except Exception:  # pragma: no cover
    CvBridge = None

try:
    from sensor_msgs_py import point_cloud2
except Exception:  # pragma: no cover
    point_cloud2 = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from debug_visualizer import DebugImagePublisher
from front_vehicle_detector import FrontVehicleDetector
from lane_detector import SlidingWindowLaneDetector
from stop_line_detector import StopLineDetector


class PerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("perception_node")

        self.declare_parameter("camera_topic", "front_camera")
        self.declare_parameter("lidar_topic", "front_lidar")
        self.declare_parameter("lane_error_topic", "perception/lane_error")
        self.declare_parameter("lane_curvature_topic", "perception/lane_curvature")
        self.declare_parameter("lane_confidence_topic", "perception/lane_confidence")
        self.declare_parameter("stop_line_topic", "perception/stop_line_detected")
        self.declare_parameter("front_vehicle_distance_topic", "perception/front_vehicle_distance")
        self.declare_parameter("lane_offset_legacy_topic", "perception/lane_center_offset")
        self.declare_parameter("front_obstacle_legacy_topic", "perception/front_obstacle_distance")

        self.declare_parameter("use_debug_visualization", False)
        self.declare_parameter("show_opencv_windows", False)
        self.declare_parameter("use_sliding_window_visualization", True)
        self.declare_parameter("publish_debug_topics", True)
        self.declare_parameter("raw_image_topic", "perception/debug/raw_image")
        self.declare_parameter("threshold_image_topic", "perception/debug/threshold_image")
        self.declare_parameter("warp_image_topic", "perception/debug/warp_image")
        self.declare_parameter("histogram_image_topic", "perception/debug/histogram_image")
        self.declare_parameter("sliding_window_image_topic", "perception/debug/sliding_window_image")
        self.declare_parameter("lane_overlay_image_topic", "perception/debug/lane_overlay_image")
        self.declare_parameter("stop_line_debug_topic", "perception/debug/stop_line_image")

        self.declare_parameter("roi_y_ratio", 0.55)
        self.declare_parameter("lane_canny_low", 70)
        self.declare_parameter("lane_canny_high", 140)
        self.declare_parameter("lane_windows", 8)
        self.declare_parameter("lane_margin", 50)
        self.declare_parameter("lane_minpix", 20)

        self.declare_parameter("max_distance", 50.0)
        self.declare_parameter("roi_y_abs", 2.0)
        self.declare_parameter("min_z", -2.0)
        self.declare_parameter("max_z", 3.0)

        self.declare_parameter("white_lower_hsv", [0, 0, 192])
        self.declare_parameter("white_upper_hsv", [179, 64, 255])
        self.declare_parameter("pixel_threshold", 7000)
        self.declare_parameter("roi_x_min", 40)
        self.declare_parameter("roi_x_max", 600)
        self.declare_parameter("roi_y_min", 400)
        self.declare_parameter("roi_y_max", 470)
        self.declare_parameter("binary_threshold", 50)
        self.declare_parameter("consecutive_frames", 2)
        self.declare_parameter("min_trigger_gap_sec", 1.0)

        self._bridge = CvBridge() if CvBridge else None
        self._lane_detector = SlidingWindowLaneDetector()
        self._stop_line_detector = StopLineDetector()
        self._front_vehicle_detector = FrontVehicleDetector()

        self._consecutive_stopline = 0
        self._last_stopline_trigger = -1e9

        self._lane_error_pub = self.create_publisher(
            Float32, self.get_parameter("lane_error_topic").value, 10
        )
        self._lane_curvature_pub = self.create_publisher(
            Float32, self.get_parameter("lane_curvature_topic").value, 10
        )
        self._lane_conf_pub = self.create_publisher(
            Float32, self.get_parameter("lane_confidence_topic").value, 10
        )
        self._lane_offset_legacy_pub = self.create_publisher(
            Float32, self.get_parameter("lane_offset_legacy_topic").value, 10
        )
        self._stop_line_pub = self.create_publisher(
            Bool, self.get_parameter("stop_line_topic").value, 10
        )
        self._front_vehicle_pub = self.create_publisher(
            Float32, self.get_parameter("front_vehicle_distance_topic").value, 10
        )
        self._front_obstacle_legacy_pub = self.create_publisher(
            Float32, self.get_parameter("front_obstacle_legacy_topic").value, 10
        )

        self._debug = DebugImagePublisher(
            self,
            {
                "raw_image": self.get_parameter("raw_image_topic").value,
                "threshold_image": self.get_parameter("threshold_image_topic").value,
                "warp_image": self.get_parameter("warp_image_topic").value,
                "histogram_image": self.get_parameter("histogram_image_topic").value,
                "sliding_window_image": self.get_parameter("sliding_window_image_topic").value,
                "lane_overlay_image": self.get_parameter("lane_overlay_image_topic").value,
                "stop_line_binary_image": self.get_parameter("stop_line_debug_topic").value,
            },
        )

        self.create_subscription(
            Image,
            self.get_parameter("camera_topic").value,
            self._on_image,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2,
            self.get_parameter("lidar_topic").value,
            self._on_lidar,
            qos_profile_sensor_data,
        )
        self.get_logger().info("perception_node started")

    def _publish_debug_image(self, key: str, frame) -> None:
        if self._bridge is None or frame is None:
            return
        if not bool(self.get_parameter("publish_debug_topics").value):
            return
        try:
            self._debug.publish(key, self._bridge.cv2_to_imgmsg(frame, encoding="bgr8"))
        except Exception:
            return

    def _show_debug_window(self, name: str, frame) -> None:
        if not bool(self.get_parameter("show_opencv_windows").value):
            return
        try:
            import cv2

            cv2.imshow(name, frame)
            cv2.waitKey(1)
        except Exception:
            return

    def _on_image(self, msg: Image) -> None:
        if self._bridge is None:
            return

        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"cv_bridge conversion failed: {exc}")
            return

        use_debug = bool(self.get_parameter("use_debug_visualization").value)
        lane_result = self._lane_detector.detect(
            frame,
            roi_y_ratio=float(self.get_parameter("roi_y_ratio").value),
            canny_low=int(self.get_parameter("lane_canny_low").value),
            canny_high=int(self.get_parameter("lane_canny_high").value),
            n_windows=int(self.get_parameter("lane_windows").value),
            margin=int(self.get_parameter("lane_margin").value),
            minpix=int(self.get_parameter("lane_minpix").value),
            use_debug_visualization=use_debug,
            use_sliding_window_visualization=bool(
                self.get_parameter("use_sliding_window_visualization").value
            ),
        )
        stop_result = self._stop_line_detector.detect(
            frame,
            white_lower_hsv=self.get_parameter("white_lower_hsv").value,
            white_upper_hsv=self.get_parameter("white_upper_hsv").value,
            pixel_threshold=int(self.get_parameter("pixel_threshold").value),
            roi_x_min=int(self.get_parameter("roi_x_min").value),
            roi_x_max=int(self.get_parameter("roi_x_max").value),
            roi_y_min=int(self.get_parameter("roi_y_min").value),
            roi_y_max=int(self.get_parameter("roi_y_max").value),
            binary_threshold=int(self.get_parameter("binary_threshold").value),
            use_debug_visualization=use_debug,
        )

        detected = stop_result.detected
        if detected:
            self._consecutive_stopline += 1
        else:
            self._consecutive_stopline = 0

        stop_line_out = False
        needed = max(1, int(self.get_parameter("consecutive_frames").value))
        min_gap = float(self.get_parameter("min_trigger_gap_sec").value)
        now_sec = self.get_clock().now().nanoseconds / 1e9
        if self._consecutive_stopline >= needed and (now_sec - self._last_stopline_trigger) >= min_gap:
            stop_line_out = True
            self._last_stopline_trigger = now_sec

        self._lane_error_pub.publish(Float32(data=float(lane_result.lane_error)))
        self._lane_curvature_pub.publish(Float32(data=float(lane_result.lane_curvature)))
        self._lane_conf_pub.publish(Float32(data=float(lane_result.lane_confidence)))
        self._lane_offset_legacy_pub.publish(Float32(data=float(lane_result.lane_error)))
        self._stop_line_pub.publish(Bool(data=bool(stop_line_out)))

        self._publish_debug_image("raw_image", frame)
        for key, image in lane_result.debug_images.items():
            self._publish_debug_image(key, image)
            if key in ("sliding_window_image", "lane_overlay_image"):
                self._show_debug_window(key, image)
        for key, image in stop_result.debug_images.items():
            self._publish_debug_image(key, image)

    def _on_lidar(self, msg: PointCloud2) -> None:
        max_distance = float(self.get_parameter("max_distance").value)
        points = []
        if point_cloud2 is not None:
            try:
                points = list(
                    point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
                )
            except Exception as exc:
                self.get_logger().warn(f"PointCloud2 parse failed: {exc}")

        result = self._front_vehicle_detector.detect(
            points,
            max_distance=max_distance,
            roi_y_abs=float(self.get_parameter("roi_y_abs").value),
            min_z=float(self.get_parameter("min_z").value),
            max_z=float(self.get_parameter("max_z").value),
        )
        out = Float32(data=float(result.distance))
        self._front_vehicle_pub.publish(out)
        self._front_obstacle_legacy_pub.publish(out)


def main() -> None:
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import os
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float32

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


class PerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("perception_node")

        self.declare_parameter("camera_topic", "front_camera")
        self.declare_parameter("lidar_topic", "front_lidar")
        self.declare_parameter("lane_error_topic", "perception/lane_error")
        self.declare_parameter("lane_curvature_topic", "perception/lane_curvature")
        self.declare_parameter("lane_confidence_topic", "perception/lane_confidence")
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

        self.declare_parameter("roi_y_ratio", 0.55)
        self.declare_parameter("lane_white_lower_hsv", [0, 0, 180])
        self.declare_parameter("lane_white_upper_hsv", [179, 70, 255])
        self.declare_parameter("lane_white_close_kernel_height", 17)
        self.declare_parameter("lane_white_open_kernel_size", 5)
        self.declare_parameter("warped_gap_close_kernel_height", 55)
        self.declare_parameter("warped_gap_close_kernel_width", 7)
        self.declare_parameter("lane_error_alpha", 0.65)
        self.declare_parameter("lane_preview_y_ratio", 0.35)
        self.declare_parameter("lane_far_preview_y_ratio", 0.18)
        self.declare_parameter("lane_far_preview_weight", 0.4)
        self.declare_parameter("lane_half_width_ratio", 0.18)
        self.declare_parameter("lane_min_width_ratio", 0.22)
        self.declare_parameter("lane_max_width_ratio", 0.48)
        self.declare_parameter("lane_center_bias_px", -28.0)
        self.declare_parameter("lane_fit_conf_threshold", 0.03)
        self.declare_parameter("lane_fallback_decay", 0.97)
        self.declare_parameter("lane_canny_low", 70)
        self.declare_parameter("lane_canny_high", 140)
        self.declare_parameter("lane_windows", 8)
        self.declare_parameter("warp_top_y_ratio", 0.25)
        self.declare_parameter("warp_top_left_x_ratio", 0.28)
        self.declare_parameter("warp_top_right_x_ratio", 0.72)
        self.declare_parameter("warp_bottom_left_x_ratio", 0.0)
        self.declare_parameter("warp_bottom_right_x_ratio", 1.0)
        self.declare_parameter("warp_dst_left_x_ratio", 0.18)
        self.declare_parameter("warp_dst_right_x_ratio", 0.82)
        self.declare_parameter("lane_margin", 50)
        self.declare_parameter("lane_minpix", 20)

        self.declare_parameter("max_distance", 60.0)
        self.declare_parameter("roi_y_abs", 3.2)
        self.declare_parameter("roi_y_min", -1.6)
        self.declare_parameter("roi_y_max", 1.6)
        self.declare_parameter("front_vehicle_max_angle_deg", 35.0)
        self.declare_parameter("min_z", -2.0)
        self.declare_parameter("max_z", 3.0)

        self._bridge = CvBridge() if CvBridge else None
        self._lane_detector = SlidingWindowLaneDetector()
        self._front_vehicle_detector = FrontVehicleDetector()

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
            white_lower_hsv=self.get_parameter("lane_white_lower_hsv").value,
            white_upper_hsv=self.get_parameter("lane_white_upper_hsv").value,
            white_close_kernel_height=int(
                self.get_parameter("lane_white_close_kernel_height").value
            ),
            white_open_kernel_size=int(self.get_parameter("lane_white_open_kernel_size").value),
            warped_gap_close_kernel_height=int(self.get_parameter("warped_gap_close_kernel_height").value),
            warp_top_y_ratio=float(self.get_parameter("warp_top_y_ratio").value),
            warp_top_left_x_ratio=float(self.get_parameter("warp_top_left_x_ratio").value),
            warp_top_right_x_ratio=float(self.get_parameter("warp_top_right_x_ratio").value),
            warp_bottom_left_x_ratio=float(self.get_parameter("warp_bottom_left_x_ratio").value),
            warp_bottom_right_x_ratio=float(self.get_parameter("warp_bottom_right_x_ratio").value),
            warp_dst_left_x_ratio=float(self.get_parameter("warp_dst_left_x_ratio").value),
            warp_dst_right_x_ratio=float(self.get_parameter("warp_dst_right_x_ratio").value),
            warped_gap_close_kernel_width=int(self.get_parameter("warped_gap_close_kernel_width").value),
            lane_error_alpha=float(self.get_parameter("lane_error_alpha").value),
            preview_y_ratio=float(self.get_parameter("lane_preview_y_ratio").value),
            far_preview_y_ratio=float(self.get_parameter("lane_far_preview_y_ratio").value),
            far_preview_weight=float(self.get_parameter("lane_far_preview_weight").value),
            lane_half_width_ratio=float(self.get_parameter("lane_half_width_ratio").value),
            lane_min_width_ratio=float(self.get_parameter("lane_min_width_ratio").value),
            lane_max_width_ratio=float(self.get_parameter("lane_max_width_ratio").value),
            lane_center_bias_px=float(self.get_parameter("lane_center_bias_px").value),
            fit_conf_threshold=float(self.get_parameter("lane_fit_conf_threshold").value),
            fallback_decay=float(self.get_parameter("lane_fallback_decay").value),
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

        self._lane_error_pub.publish(Float32(data=float(lane_result.lane_error)))
        self._lane_curvature_pub.publish(Float32(data=float(lane_result.lane_curvature)))
        self._lane_conf_pub.publish(Float32(data=float(lane_result.lane_confidence)))
        self._lane_offset_legacy_pub.publish(Float32(data=float(lane_result.lane_error)))

        self._publish_debug_image("raw_image", frame)
        for key, image in lane_result.debug_images.items():
            self._publish_debug_image(key, image)
            if key in ("sliding_window_image", "lane_overlay_image"):
                self._show_debug_window(key, image)

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
            roi_y_min=float(self.get_parameter("roi_y_min").value),
            roi_y_max=float(self.get_parameter("roi_y_max").value),
            max_angle_deg=float(self.get_parameter("front_vehicle_max_angle_deg").value),
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

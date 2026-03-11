#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

try:
    from cv_bridge import CvBridge
except Exception:  # pragma: no cover
    CvBridge = None


class StopLineDetectionNode(Node):
    def __init__(self) -> None:
        super().__init__("stop_line_detection_node")
        self.declare_parameter("camera_topic", "front_camera")
        self.declare_parameter("stop_line_topic", "perception/stop_line_detected")
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
        camera_topic = self.get_parameter("camera_topic").value
        stop_line_topic = self.get_parameter("stop_line_topic").value

        self._pub = self.create_publisher(Bool, stop_line_topic, 10)
        self._consecutive_hit = 0
        self._last_trigger_sec = -1e9
        self.create_subscription(Image, camera_topic, self._on_image, 10)
        self.get_logger().info(f"stop_line_detection_node subscribed to '{camera_topic}'")

    def _on_image(self, msg: Image) -> None:
        detected_raw = False
        if self._bridge is None:
            self._publish(False)
            return

        try:
            raw_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"cv_bridge conversion failed: {exc}")
            self._publish(False)
            return

        try:
            y, x = raw_img.shape[0:2]
            hsv = cv2.cvtColor(raw_img, cv2.COLOR_BGR2HSV)

            white_lower = np.array(self.get_parameter("white_lower_hsv").value, dtype=np.uint8)
            white_upper = np.array(self.get_parameter("white_upper_hsv").value, dtype=np.uint8)
            white_mask = cv2.inRange(hsv, white_lower, white_upper)
            filtered = cv2.bitwise_and(raw_img, raw_img, mask=white_mask)

            src = np.float32([[0, 420], [275, 260], [x - 275, 260], [x, 420]])
            dst = np.float32([[x // 8, y], [x // 8, 0], [x // 8 * 7, 0], [x // 8 * 7, y]])
            mat = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(filtered, mat, (x, y))

            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            bin_img = np.zeros_like(gray, dtype=np.uint8)
            binary_threshold = int(self.get_parameter("binary_threshold").value)
            bin_img[gray > binary_threshold] = 1

            x_min = max(0, int(self.get_parameter("roi_x_min").value))
            x_max = min(x, int(self.get_parameter("roi_x_max").value))
            y_min = max(0, int(self.get_parameter("roi_y_min").value))
            y_max = min(y, int(self.get_parameter("roi_y_max").value))

            if x_min < x_max and y_min < y_max:
                roi = bin_img[y_min:y_max, x_min:x_max]
                white_pixels = int(np.count_nonzero(roi))
                detected_raw = white_pixels > int(self.get_parameter("pixel_threshold").value)
        except Exception as exc:
            self.get_logger().warn(f"stop-line detection failed: {exc}")
            detected_raw = False

        if detected_raw:
            self._consecutive_hit += 1
        else:
            self._consecutive_hit = 0

        detected = False
        needed = max(1, int(self.get_parameter("consecutive_frames").value))
        min_gap = float(self.get_parameter("min_trigger_gap_sec").value)
        now_sec = self.get_clock().now().nanoseconds / 1e9
        if self._consecutive_hit >= needed and (now_sec - self._last_trigger_sec) >= min_gap:
            detected = True
            self._last_trigger_sec = now_sec

        self._publish(detected)

    def _publish(self, detected: bool) -> None:
        msg = Bool()
        msg.data = bool(detected)
        self._pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = StopLineDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

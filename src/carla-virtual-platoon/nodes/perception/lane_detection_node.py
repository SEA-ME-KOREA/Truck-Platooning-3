#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

try:
    import cv2
    import numpy as np
    from cv_bridge import CvBridge
except Exception:  # pragma: no cover - runtime dependency optional during skeleton setup
    cv2 = None
    np = None
    CvBridge = None

try:
    from lane_follower_legacy import LaneFollower
except Exception:  # pragma: no cover
    LaneFollower = None


class LaneDetectionNode(Node):
    def __init__(self) -> None:
        super().__init__("lane_detection_node")
        self.declare_parameter("camera_topic", "front_camera")
        self.declare_parameter("lane_offset_topic", "perception/lane_center_offset")
        self.declare_parameter("lane_conf_topic", "perception/lane_confidence")
        self.declare_parameter("debug_image_topic", "perception/lane_sliding_window_debug")
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("use_legacy_lane_follower", True)

        self._bridge = CvBridge() if CvBridge else None
        self._last_offset = 0.0
        self._legacy = LaneFollower() if LaneFollower is not None else None

        camera_topic = self.get_parameter("camera_topic").get_parameter_value().string_value
        offset_topic = self.get_parameter("lane_offset_topic").get_parameter_value().string_value
        conf_topic = self.get_parameter("lane_conf_topic").get_parameter_value().string_value
        debug_image_topic = (
            self.get_parameter("debug_image_topic").get_parameter_value().string_value
        )

        self._offset_pub = self.create_publisher(Float32, offset_topic, 10)
        self._conf_pub = self.create_publisher(Float32, conf_topic, 10)
        self._debug_pub = self.create_publisher(Image, debug_image_topic, 10)
        self.create_subscription(Image, camera_topic, self._on_image, 10)

        self.get_logger().info(f"lane_detection_node subscribed to '{camera_topic}'")

    def _publish(self, offset: float, confidence: float) -> None:
        offset_msg = Float32()
        offset_msg.data = float(offset)
        conf_msg = Float32()
        conf_msg.data = float(max(0.0, min(1.0, confidence)))
        self._offset_pub.publish(offset_msg)
        self._conf_pub.publish(conf_msg)
        self._last_offset = offset

    def _on_image(self, msg: Image) -> None:
        if not self._bridge or cv2 is None or np is None:
            self._publish(self._last_offset, 0.0)
            return

        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"cv_bridge conversion failed: {exc}")
            self._publish(self._last_offset, 0.0)
            return

        if bool(self.get_parameter("use_legacy_lane_follower").value) and self._legacy is not None:
            steer, pos_norm, debug_bgr, confidence = self._legacy.process(frame)
            normalized_offset = float((pos_norm - 0.5) * 2.0)
            self._publish(max(-1.0, min(1.0, normalized_offset)), float(confidence))
            if debug_bgr is not None:
                self._publish_debug(frame, debug_bgr)
            return

        h, w = frame.shape[:2]
        if h < 2 or w < 2:
            self._publish(self._last_offset, 0.0)
            return

        # Sliding-window style baseline detector for visual debugging.
        roi = frame[int(h * 0.55):, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 70, 140)
        binary = (edges > 0).astype(np.uint8)

        histogram = np.sum(binary[binary.shape[0] // 2 :, :], axis=0)
        base_x = int(np.argmax(histogram)) if histogram.size else binary.shape[1] // 2

        n_windows = 8
        margin = 50
        minpix = 20
        window_h = max(1, binary.shape[0] // n_windows)
        nonzero_y, nonzero_x = binary.nonzero()
        current_x = base_x
        lane_inds = []

        debug_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        for w_idx in range(n_windows):
            win_y_low = binary.shape[0] - (w_idx + 1) * window_h
            win_y_high = binary.shape[0] - w_idx * window_h
            win_x_low = max(0, current_x - margin)
            win_x_high = min(binary.shape[1], current_x + margin)

            cv2.rectangle(
                debug_bgr, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2
            )

            good_inds = (
                (nonzero_y >= win_y_low)
                & (nonzero_y < win_y_high)
                & (nonzero_x >= win_x_low)
                & (nonzero_x < win_x_high)
            ).nonzero()[0]
            lane_inds.append(good_inds)
            if good_inds.size > minpix:
                current_x = int(np.mean(nonzero_x[good_inds]))

        xs = np.concatenate(lane_inds) if lane_inds else np.array([], dtype=np.int32)
        lane_x_pixels = nonzero_x[xs] if xs.size else np.array([], dtype=np.int32)

        if lane_x_pixels.size < 50:
            self._publish(self._last_offset, 0.05)
            self._publish_debug(roi, debug_bgr)
            return

        if xs.size:
            debug_bgr[nonzero_y[xs], nonzero_x[xs]] = (0, 0, 255)

        lane_center_px = float(np.mean(lane_x_pixels))
        cv2.line(
            debug_bgr,
            (int(lane_center_px), 0),
            (int(lane_center_px), debug_bgr.shape[0] - 1),
            (255, 0, 0),
            2,
        )
        cv2.line(
            debug_bgr,
            (debug_bgr.shape[1] // 2, 0),
            (debug_bgr.shape[1] // 2, debug_bgr.shape[0] - 1),
            (255, 255, 0),
            2,
        )

        self._publish_debug(roi, debug_bgr)

        if lane_x_pixels.size < 50:
            self._publish(self._last_offset, 0.05)
            return

        image_center_px = roi.shape[1] / 2.0
        normalized_offset = (lane_center_px - image_center_px) / max(1.0, image_center_px)
        confidence = min(1.0, lane_x_pixels.size / 4000.0)
        self._publish(max(-1.0, min(1.0, normalized_offset)), confidence)

    def _publish_debug(self, roi_bgr, debug_bgr) -> None:
        if not self.get_parameter("publish_debug_image").value:
            return
        if self._bridge is None:
            return
        try:
            # Side-by-side: original ROI | sliding window overlay
            vis = np.hstack((roi_bgr, debug_bgr))
            msg = self._bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            self._debug_pub.publish(msg)
        except Exception as exc:
            self.get_logger().warn(f"debug image publish failed: {exc}")


def main() -> None:
    rclpy.init()
    node = LaneDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

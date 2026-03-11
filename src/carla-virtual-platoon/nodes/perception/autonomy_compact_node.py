#!/usr/bin/env python3

import math

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Bool, Float32, String

try:
    from cv_bridge import CvBridge
except Exception:  # pragma: no cover
    CvBridge = None

try:
    from sensor_msgs_py import point_cloud2
except Exception:  # pragma: no cover
    point_cloud2 = None

try:
    from lane_follower_legacy import LaneFollower
except Exception:  # pragma: no cover
    LaneFollower = None


class AutonomyCompactNode(Node):
    """Single-node pipeline: camera/lidar perception + behavior + control."""

    def __init__(self) -> None:
        super().__init__("autonomy_compact_node")

        self.declare_parameter("camera_topic", "front_camera")
        self.declare_parameter("lidar_topic", "front_lidar")
        self.declare_parameter("velocity_topic", "velocity")
        self.declare_parameter("enabled_topic", "control/enabled")
        self.declare_parameter("traffic_light_topic", "perception/traffic_light_state")
        self.declare_parameter("steer_topic", "steer_control")
        self.declare_parameter("throttle_topic", "throttle_control")
        self.declare_parameter("mode_topic", "control/behavior_mode")
        self.declare_parameter("target_speed_topic", "control/target_speed")
        self.declare_parameter("lane_offset_topic", "perception/lane_center_offset")
        self.declare_parameter("lane_conf_topic", "perception/lane_confidence")
        self.declare_parameter("obstacle_dist_topic", "perception/front_obstacle_distance")
        self.declare_parameter("stop_line_topic", "perception/stop_line_detected")
        self.declare_parameter("publish_debug", False)
        self.declare_parameter("debug_topic", "perception/lane_debug")

        self.declare_parameter("cruise_speed", 6.0)
        self.declare_parameter("slow_speed", 2.5)
        self.declare_parameter("stop_distance", 6.0)
        self.declare_parameter("caution_distance", 15.0)
        self.declare_parameter("lane_conf_threshold", 0.2)
        self.declare_parameter("stop_line_hold_sec", 1.5)
        self.declare_parameter("emergency_distance", 4.0)
        self.declare_parameter("k_lane", 15.0)
        self.declare_parameter("k_speed_p", 0.25)
        self.declare_parameter("max_steer_deg", 30.0)
        self.declare_parameter("max_throttle", 1.0)
        self.declare_parameter("max_brake", 1.0)
        self.declare_parameter("startup_hold_sec", 1.0)
        self.declare_parameter("startup_ramp_sec", 2.0)
        self.declare_parameter("loop_hz", 20.0)

        self.declare_parameter("lidar_max_distance", 50.0)
        self.declare_parameter("lidar_roi_y_abs", 2.0)
        self.declare_parameter("lidar_min_z", -2.0)
        self.declare_parameter("lidar_max_z", 3.0)

        self.declare_parameter("white_lower_hsv", [0, 0, 192])
        self.declare_parameter("white_upper_hsv", [179, 64, 255])
        self.declare_parameter("stopline_pixel_threshold", 7000)
        self.declare_parameter("stopline_roi_x_min", 40)
        self.declare_parameter("stopline_roi_x_max", 600)
        self.declare_parameter("stopline_roi_y_min", 400)
        self.declare_parameter("stopline_roi_y_max", 470)
        self.declare_parameter("stopline_binary_threshold", 50)
        self.declare_parameter("stopline_consecutive_frames", 2)
        self.declare_parameter("stopline_min_trigger_gap_sec", 1.0)

        self._bridge = CvBridge() if CvBridge else None
        self._lane = LaneFollower() if LaneFollower is not None else None

        self._lane_offset = 0.0
        self._lane_conf = 0.0
        self._obstacle_dist = 999.0
        self._stop_line_detected = False
        self._traffic_light = "UNKNOWN"
        self._target_speed = 0.0
        self._current_speed = 0.0
        self._mode = "IDLE"
        self._enabled = False
        self._enabled_since = None
        self._stop_line_until = None
        self._last_debug = None
        self._consecutive_stopline = 0
        self._last_stopline_trigger = -1e9

        self.create_subscription(Image, self.get_parameter("camera_topic").value, self._on_image, 10)
        self.create_subscription(PointCloud2, self.get_parameter("lidar_topic").value, self._on_lidar, 10)
        self.create_subscription(Float32, self.get_parameter("velocity_topic").value, self._on_velocity, 10)
        self.create_subscription(Bool, self.get_parameter("enabled_topic").value, self._on_enabled, 10)
        self.create_subscription(
            String, self.get_parameter("traffic_light_topic").value, self._on_traffic_light, 10
        )

        self._steer_pub = self.create_publisher(Float32, self.get_parameter("steer_topic").value, 10)
        self._throttle_pub = self.create_publisher(Float32, self.get_parameter("throttle_topic").value, 10)
        self._mode_pub = self.create_publisher(String, self.get_parameter("mode_topic").value, 10)
        self._target_speed_pub = self.create_publisher(
            Float32, self.get_parameter("target_speed_topic").value, 10
        )
        self._lane_offset_pub = self.create_publisher(
            Float32, self.get_parameter("lane_offset_topic").value, 10
        )
        self._lane_conf_pub = self.create_publisher(Float32, self.get_parameter("lane_conf_topic").value, 10)
        self._obstacle_pub = self.create_publisher(
            Float32, self.get_parameter("obstacle_dist_topic").value, 10
        )
        self._stop_line_pub = self.create_publisher(Bool, self.get_parameter("stop_line_topic").value, 10)
        self._debug_pub = self.create_publisher(Image, self.get_parameter("debug_topic").value, 10)

        hz = max(1.0, float(self.get_parameter("loop_hz").value))
        self.create_timer(1.0 / hz, self._tick)
        self.get_logger().info("autonomy_compact_node started")

    def _on_image(self, msg: Image) -> None:
        if self._bridge is None:
            return
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return

        # Lane detection (legacy follower imported from competition codebase).
        if self._lane is not None:
            _, pos_norm, debug_bgr, conf = self._lane.process(frame)
            self._lane_offset = float(max(-1.0, min(1.0, (pos_norm - 0.5) * 2.0)))
            self._lane_conf = float(max(0.0, min(1.0, conf)))
            self._last_debug = debug_bgr
        else:
            self._lane_offset = 0.0
            self._lane_conf = 0.0
            self._last_debug = None

        self._stop_line_detected = self._detect_stop_line(frame)

        self._lane_offset_pub.publish(Float32(data=float(self._lane_offset)))
        self._lane_conf_pub.publish(Float32(data=float(self._lane_conf)))
        self._stop_line_pub.publish(Bool(data=bool(self._stop_line_detected)))

        if bool(self.get_parameter("publish_debug").value) and self._last_debug is not None:
            try:
                self._debug_pub.publish(self._bridge.cv2_to_imgmsg(self._last_debug, encoding="bgr8"))
            except Exception:
                pass

    def _on_lidar(self, msg: PointCloud2) -> None:
        max_distance = float(self.get_parameter("lidar_max_distance").value)
        min_distance = max_distance
        if point_cloud2 is not None:
            roi_y_abs = float(self.get_parameter("lidar_roi_y_abs").value)
            min_z = float(self.get_parameter("lidar_min_z").value)
            max_z = float(self.get_parameter("lidar_max_z").value)
            try:
                for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                    x, y, z = float(p[0]), float(p[1]), float(p[2])
                    if x <= 0.0:
                        continue
                    if abs(y) > roi_y_abs or z < min_z or z > max_z:
                        continue
                    d = math.sqrt(x * x + y * y + z * z)
                    if d < min_distance:
                        min_distance = d
            except Exception:
                pass
        self._obstacle_dist = float(min_distance)
        self._obstacle_pub.publish(Float32(data=float(self._obstacle_dist)))

    def _on_velocity(self, msg: Float32) -> None:
        self._current_speed = float(msg.data)

    def _on_enabled(self, msg: Bool) -> None:
        new_enabled = bool(msg.data)
        if new_enabled and not self._enabled:
            self._enabled_since = self.get_clock().now()
        if not new_enabled:
            self._enabled_since = None
        self._enabled = new_enabled

    def _on_traffic_light(self, msg: String) -> None:
        self._traffic_light = msg.data or "UNKNOWN"

    def _detect_stop_line(self, frame) -> bool:
        try:
            y, x = frame.shape[:2]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            w_low = np.array(self.get_parameter("white_lower_hsv").value, dtype=np.uint8)
            w_high = np.array(self.get_parameter("white_upper_hsv").value, dtype=np.uint8)
            white_mask = cv2.inRange(hsv, w_low, w_high)
            filtered = cv2.bitwise_and(frame, frame, mask=white_mask)

            src = np.float32([[0, 420], [275, 260], [x - 275, 260], [x, 420]])
            dst = np.float32([[x // 8, y], [x // 8, 0], [x // 8 * 7, 0], [x // 8 * 7, y]])
            mat = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(filtered, mat, (x, y))

            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            bin_img = np.zeros_like(gray, dtype=np.uint8)
            bin_thr = int(self.get_parameter("stopline_binary_threshold").value)
            bin_img[gray > bin_thr] = 1

            x_min = max(0, int(self.get_parameter("stopline_roi_x_min").value))
            x_max = min(x, int(self.get_parameter("stopline_roi_x_max").value))
            y_min = max(0, int(self.get_parameter("stopline_roi_y_min").value))
            y_max = min(y, int(self.get_parameter("stopline_roi_y_max").value))
            if x_min >= x_max or y_min >= y_max:
                return False

            white_pixels = int(np.count_nonzero(bin_img[y_min:y_max, x_min:x_max]))
            detected_raw = white_pixels > int(self.get_parameter("stopline_pixel_threshold").value)
        except Exception:
            detected_raw = False

        if detected_raw:
            self._consecutive_stopline += 1
        else:
            self._consecutive_stopline = 0

        needed = max(1, int(self.get_parameter("stopline_consecutive_frames").value))
        min_gap = float(self.get_parameter("stopline_min_trigger_gap_sec").value)
        now_sec = self.get_clock().now().nanoseconds / 1e9
        if self._consecutive_stopline >= needed and (now_sec - self._last_stopline_trigger) >= min_gap:
            self._last_stopline_trigger = now_sec
            return True
        return False

    def _tick(self) -> None:
        cruise = float(self.get_parameter("cruise_speed").value)
        slow = float(self.get_parameter("slow_speed").value)
        stop_d = float(self.get_parameter("stop_distance").value)
        caution_d = float(self.get_parameter("caution_distance").value)
        lane_conf_th = float(self.get_parameter("lane_conf_threshold").value)
        hold_stopline = float(self.get_parameter("stop_line_hold_sec").value)

        mode = "LANE_FOLLOW"
        target_speed = cruise
        now_sec = self.get_clock().now().nanoseconds / 1e9

        if self._stop_line_detected:
            self._stop_line_until = now_sec + max(0.0, hold_stopline)
        if self._stop_line_until is not None and now_sec < self._stop_line_until:
            mode = "STOP_FOR_STOPLINE"
            target_speed = 0.0
        elif self._obstacle_dist < stop_d:
            mode = "STOP_FOR_OBSTACLE"
            target_speed = 0.0
        elif self._obstacle_dist < caution_d:
            mode = "OBSTACLE_CAUTION"
            target_speed = slow
        elif self._lane_conf < lane_conf_th:
            mode = "INTERSECTION_CROSS"
            target_speed = slow
        elif self._traffic_light == "RED":
            mode = "STOP_FOR_RED"
            target_speed = 0.0

        self._mode = mode
        self._target_speed = float(target_speed)

        self._mode_pub.publish(String(data=self._mode))
        self._target_speed_pub.publish(Float32(data=float(self._target_speed)))

        if not self._enabled:
            self._steer_pub.publish(Float32(data=0.0))
            self._throttle_pub.publish(Float32(data=0.0))
            return

        max_steer = float(self.get_parameter("max_steer_deg").value)
        k_lane = float(self.get_parameter("k_lane").value)
        k_speed = float(self.get_parameter("k_speed_p").value)
        emergency_d = float(self.get_parameter("emergency_distance").value)
        max_throttle = float(self.get_parameter("max_throttle").value)
        max_brake = float(self.get_parameter("max_brake").value)

        lane_weight = max(0.1, min(1.0, self._lane_conf))
        steer_cmd = -k_lane * self._lane_offset * lane_weight
        steer_cmd = max(-max_steer, min(max_steer, steer_cmd))

        speed_error = self._target_speed - self._current_speed
        throttle_brake = k_speed * speed_error
        if self._obstacle_dist < emergency_d:
            throttle_brake = -max_brake
            steer_cmd = 0.0
        throttle_brake = max(-max_brake, min(max_throttle, throttle_brake))

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

        self._steer_pub.publish(Float32(data=float(steer_cmd)))
        self._throttle_pub.publish(Float32(data=float(throttle_brake)))


def main() -> None:
    rclpy.init()
    node = AutonomyCompactNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

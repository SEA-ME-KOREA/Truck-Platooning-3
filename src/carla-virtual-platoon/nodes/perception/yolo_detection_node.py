#!/usr/bin/env python3
import json
from typing import Any, Dict, List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

try:
    from cv_bridge import CvBridge
except Exception:  # pragma: no cover
    CvBridge = None

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None


class YoloDetectionNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_detection_node")
        self.declare_parameter("camera_topic", "front_camera")
        self.declare_parameter("detections_topic", "perception/detections_json")
        self.declare_parameter("traffic_light_topic", "perception/traffic_light_state")
        self.declare_parameter("model_path", "yolov8n.pt")
        self.declare_parameter("enabled", False)
        self.declare_parameter("confidence_threshold", 0.3)

        self._bridge = CvBridge() if CvBridge else None
        self._model = None

        enabled = self.get_parameter("enabled").get_parameter_value().bool_value
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        if enabled and YOLO is not None:
            try:
                self._model = YOLO(model_path)
                self.get_logger().info(f"Loaded YOLO model: {model_path}")
            except Exception as exc:
                self.get_logger().warn(f"YOLO load failed: {exc}")

        camera_topic = self.get_parameter("camera_topic").get_parameter_value().string_value
        detections_topic = self.get_parameter("detections_topic").get_parameter_value().string_value
        traffic_light_topic = self.get_parameter("traffic_light_topic").get_parameter_value().string_value

        self._det_pub = self.create_publisher(String, detections_topic, 10)
        self._tl_pub = self.create_publisher(String, traffic_light_topic, 10)
        self._sub = self.create_subscription(Image, camera_topic, self._on_image, 10)

        self.get_logger().info(
            f"yolo_detection_node subscribed to '{camera_topic}' (enabled={enabled})"
        )

    def _publish(self, detections: List[Dict[str, Any]], traffic_light_state: str) -> None:
        det_msg = String()
        det_msg.data = json.dumps({"detections": detections}, ensure_ascii=True)
        tl_msg = String()
        tl_msg.data = traffic_light_state
        self._det_pub.publish(det_msg)
        self._tl_pub.publish(tl_msg)

    def _on_image(self, msg: Image) -> None:
        if self._model is None or self._bridge is None:
            self._publish([], "UNKNOWN")
            return

        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            results = self._model.predict(frame, verbose=False)
        except Exception as exc:
            self.get_logger().warn(f"YOLO inference failed: {exc}")
            self._publish([], "UNKNOWN")
            return

        detections: List[Dict[str, Any]] = []
        traffic_light_state = "UNKNOWN"
        conf_th = self.get_parameter("confidence_threshold").get_parameter_value().double_value

        for result in results:
            names = getattr(result, "names", {})
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                try:
                    conf = float(box.conf[0])
                    cls_idx = int(box.cls[0])
                    if conf < conf_th:
                        continue
                    label = str(names.get(cls_idx, cls_idx))
                    xyxy = [float(v) for v in box.xyxy[0].tolist()]
                except Exception:
                    continue

                detections.append({"label": label, "conf": conf, "xyxy": xyxy})

                if "traffic light" in label.lower():
                    traffic_light_state = "DETECTED"

        self._publish(detections, traffic_light_state)


def main() -> None:
    rclpy.init()
    node = YoloDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

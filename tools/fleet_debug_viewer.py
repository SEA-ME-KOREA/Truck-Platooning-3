#!/usr/bin/env python3

import argparse
from typing import Dict, Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

try:
    from cv_bridge import CvBridge
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"cv_bridge is required: {exc}")


class FleetDebugViewer(Node):
    def __init__(self, namespaces):
        super().__init__("fleet_debug_viewer")
        self._bridge = CvBridge()
        self._frames: Dict[str, Optional[np.ndarray]] = {ns: None for ns in namespaces}

        for ns in namespaces:
            self.create_subscription(
                Image,
                f"/{ns}/perception/debug/lane_overlay_image",
                lambda msg, ns=ns: self._on_image(ns, msg),
                10,
            )

        self.create_timer(0.1, self._render)
        self.get_logger().info(f"Showing fleet debug viewer for {', '.join(namespaces)}")

    def _on_image(self, namespace: str, msg: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return
        self._frames[namespace] = frame

    @staticmethod
    def _panel(frame: Optional[np.ndarray], title: str, width: int, height: int) -> np.ndarray:
        if frame is None:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(
                canvas,
                f"{title}: waiting...",
                (20, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            return canvas

        out = cv2.resize(frame, (width, height))
        cv2.rectangle(out, (0, 0), (width - 1, height - 1), (0, 0, 255), 2)
        cv2.putText(
            out,
            title,
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            title,
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return out

    def _render(self) -> None:
        tile_w = 640
        tile_h = 360
        ordered = []
        for ns in ("truck0", "truck1", "truck2"):
            if ns in self._frames:
                ordered.append(self._panel(self._frames[ns], ns, tile_w, tile_h))

        if not ordered:
            return

        mosaic = np.hstack(ordered)
        cv2.imshow("fleet_debug_viewer", mosaic)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespaces", nargs="*", default=["truck0", "truck1", "truck2"])
    args = parser.parse_args()

    rclpy.init()
    node = FleetDebugViewer(args.namespaces)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

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


class LaneDebugViewer(Node):
    def __init__(self, namespace: str) -> None:
        super().__init__("lane_debug_viewer")
        self._bridge = CvBridge()
        self._frames: Dict[str, Optional[np.ndarray]] = {
            "raw": None,
            "warp": None,
            "sliding": None,
            "overlay": None,
        }

        topics = {
            "raw": f"/{namespace}/perception/debug/raw_image",
            "warp": f"/{namespace}/perception/debug/warp_image",
            "sliding": f"/{namespace}/perception/debug/sliding_window_image",
            "overlay": f"/{namespace}/perception/debug/lane_overlay_image",
        }

        for key, topic in topics.items():
            self.create_subscription(
                Image,
                topic,
                lambda msg, key=key: self._on_image(key, msg),
                10,
            )

        self.create_timer(0.1, self._render)
        self.get_logger().info(f"Showing lane debug mosaic for namespace={namespace}")

    def _on_image(self, key: str, msg: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return
        self._frames[key] = frame

    @staticmethod
    def _resize(frame: Optional[np.ndarray], width: int, height: int, title: str) -> np.ndarray:
        if frame is None:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(
                canvas,
                f"{title}: waiting...",
                (20, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            return canvas

        out = cv2.resize(frame, (width, height))
        cv2.putText(
            out,
            title,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return out

    def _render(self) -> None:
        tile_w = 640
        tile_h = 360

        top_left = self._resize(self._frames["raw"], tile_w, tile_h, "raw_image")
        top_right = self._resize(self._frames["warp"], tile_w, tile_h, "warp_image")
        bottom_left = self._resize(
            self._frames["sliding"], tile_w, tile_h, "sliding_window_image"
        )
        bottom_right = self._resize(
            self._frames["overlay"], tile_w, tile_h, "lane_overlay_image"
        )

        top = np.hstack([top_left, top_right])
        bottom = np.hstack([bottom_left, bottom_right])
        mosaic = np.vstack([top, bottom])

        cv2.imshow("lane_debug_viewer", mosaic)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            raise KeyboardInterrupt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default="truck0")
    args = parser.parse_args()

    rclpy.init()
    node = LaneDebugViewer(args.namespace)
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

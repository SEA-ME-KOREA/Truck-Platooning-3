#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np


@dataclass
class StopLineDetectionResult:
    detected: bool
    debug_images: Dict[str, np.ndarray]


class StopLineDetector:
    def detect(
        self,
        frame_bgr: np.ndarray,
        *,
        white_lower_hsv,
        white_upper_hsv,
        pixel_threshold: int,
        roi_x_min: int,
        roi_x_max: int,
        roi_y_min: int,
        roi_y_max: int,
        binary_threshold: int,
        use_debug_visualization: bool = False,
    ) -> StopLineDetectionResult:
        debug_images: Dict[str, np.ndarray] = {}
        if frame_bgr is None or frame_bgr.size == 0:
            return StopLineDetectionResult(False, debug_images)

        y, x = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array(white_lower_hsv, dtype=np.uint8)
        upper = np.array(white_upper_hsv, dtype=np.uint8)
        white_mask = cv2.inRange(hsv, lower, upper)
        filtered = cv2.bitwise_and(frame_bgr, frame_bgr, mask=white_mask)

        src = np.float32([[0, 420], [275, 260], [x - 275, 260], [x, 420]])
        dst = np.float32([[x // 8, y], [x // 8, 0], [x // 8 * 7, 0], [x // 8 * 7, y]])
        mat = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(filtered, mat, (x, y))

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        binary = np.zeros_like(gray, dtype=np.uint8)
        binary[gray > int(binary_threshold)] = 255

        x_min = max(0, int(roi_x_min))
        x_max = min(x, int(roi_x_max))
        y_min = max(0, int(roi_y_min))
        y_max = min(y, int(roi_y_max))

        detected = False
        debug_view = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        if x_min < x_max and y_min < y_max:
            roi = binary[y_min:y_max, x_min:x_max]
            white_pixels = int(np.count_nonzero(roi))
            detected = white_pixels > int(pixel_threshold)
            if use_debug_visualization:
                color = (0, 255, 0) if detected else (0, 0, 255)
                cv2.rectangle(debug_view, (x_min, y_min), (x_max, y_max), color, 2)

        if use_debug_visualization:
            debug_images["stop_line_binary_image"] = debug_view
            debug_images["stop_line_warp_image"] = warped

        return StopLineDetectionResult(detected, debug_images)

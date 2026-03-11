#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np


@dataclass
class LaneDetectionResult:
    lane_error: float
    lane_curvature: float
    lane_confidence: float
    debug_images: Dict[str, np.ndarray]


class SlidingWindowLaneDetector:
    """Competition-style lane detector with step-by-step debug outputs."""

    def __init__(self) -> None:
        self._prev_lane_error = 0.0

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    @staticmethod
    def _binary_to_bgr(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    def detect(
        self,
        frame_bgr: np.ndarray,
        *,
        roi_y_ratio: float = 0.55,
        canny_low: int = 70,
        canny_high: int = 140,
        n_windows: int = 8,
        margin: int = 50,
        minpix: int = 20,
        use_debug_visualization: bool = False,
        use_sliding_window_visualization: bool = True,
    ) -> LaneDetectionResult:
        debug_images: Dict[str, np.ndarray] = {}
        if frame_bgr is None or frame_bgr.size == 0:
            return LaneDetectionResult(self._prev_lane_error, 0.0, 0.0, debug_images)

        h, w = frame_bgr.shape[:2]
        roi_y = int(max(0, min(h - 1, h * roi_y_ratio)))
        roi = frame_bgr[roi_y:, :]
        if roi.size == 0:
            return LaneDetectionResult(self._prev_lane_error, 0.0, 0.0, debug_images)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, canny_low, canny_high)

        src = np.float32(
            [
                [0, roi.shape[0] - 1],
                [roi.shape[1] * 0.32, roi.shape[0] * 0.25],
                [roi.shape[1] * 0.68, roi.shape[0] * 0.25],
                [roi.shape[1] - 1, roi.shape[0] - 1],
            ]
        )
        dst = np.float32(
            [
                [roi.shape[1] * 0.15, roi.shape[0] - 1],
                [roi.shape[1] * 0.15, 0],
                [roi.shape[1] * 0.85, 0],
                [roi.shape[1] * 0.85, roi.shape[0] - 1],
            ]
        )
        mat = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(edges, mat, (roi.shape[1], roi.shape[0]))
        binary = (warped > 0).astype(np.uint8)

        histogram = np.sum(binary[binary.shape[0] // 2 :, :], axis=0)
        base_x = int(np.argmax(histogram)) if histogram.size else binary.shape[1] // 2

        nonzero_y, nonzero_x = binary.nonzero()
        current_x = base_x
        lane_inds = []
        window_h = max(1, binary.shape[0] // max(1, n_windows))
        sliding_vis = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

        for idx in range(n_windows):
            win_y_low = binary.shape[0] - (idx + 1) * window_h
            win_y_high = binary.shape[0] - idx * window_h
            win_x_low = max(0, current_x - margin)
            win_x_high = min(binary.shape[1], current_x + margin)

            good_inds = (
                (nonzero_y >= win_y_low)
                & (nonzero_y < win_y_high)
                & (nonzero_x >= win_x_low)
                & (nonzero_x < win_x_high)
            ).nonzero()[0]
            lane_inds.append(good_inds)

            if use_debug_visualization and use_sliding_window_visualization:
                cv2.rectangle(
                    sliding_vis,
                    (win_x_low, win_y_low),
                    (win_x_high, win_y_high),
                    (0, 255, 0),
                    2,
                )

            if good_inds.size > minpix:
                current_x = int(np.mean(nonzero_x[good_inds]))

        xs = np.concatenate(lane_inds) if lane_inds else np.array([], dtype=np.int32)
        lane_x_pixels = nonzero_x[xs] if xs.size else np.array([], dtype=np.int32)
        lane_y_pixels = nonzero_y[xs] if xs.size else np.array([], dtype=np.int32)

        confidence = min(1.0, lane_x_pixels.size / 4000.0)
        lane_curvature = 0.0
        lane_error = self._prev_lane_error

        if lane_x_pixels.size >= 50:
            lane_center_px = float(np.mean(lane_x_pixels))
            image_center_px = binary.shape[1] / 2.0
            lane_error = self._clamp(
                (lane_center_px - image_center_px) / max(1.0, image_center_px),
                -1.0,
                1.0,
            )
            self._prev_lane_error = lane_error

            if lane_x_pixels.size >= 100:
                fit = np.polyfit(lane_y_pixels.astype(np.float32), lane_x_pixels.astype(np.float32), 2)
                lane_curvature = float(fit[0])
                plot_y = np.linspace(0, binary.shape[0] - 1, binary.shape[0]).astype(np.int32)
                fit_x = (fit[0] * plot_y ** 2 + fit[1] * plot_y + fit[2]).astype(np.int32)
                fit_x = np.clip(fit_x, 0, binary.shape[1] - 1)
                sliding_vis[plot_y, fit_x] = (255, 0, 255)

            if use_debug_visualization:
                sliding_vis[nonzero_y[xs], nonzero_x[xs]] = (0, 0, 255)
                cv2.line(
                    sliding_vis,
                    (int(lane_center_px), 0),
                    (int(lane_center_px), sliding_vis.shape[0] - 1),
                    (255, 0, 0),
                    2,
                )
                cv2.line(
                    sliding_vis,
                    (sliding_vis.shape[1] // 2, 0),
                    (sliding_vis.shape[1] // 2, sliding_vis.shape[0] - 1),
                    (255, 255, 0),
                    2,
                )

        if use_debug_visualization:
            histogram_vis = np.zeros((256, binary.shape[1], 3), dtype=np.uint8)
            if histogram.size:
                hist_norm = histogram.astype(np.float32)
                hist_norm *= 255.0 / max(1.0, float(np.max(hist_norm)))
                for x_idx, value in enumerate(hist_norm.astype(np.int32)):
                    cv2.line(histogram_vis, (x_idx, 255), (x_idx, 255 - value), (0, 255, 255), 1)

            overlay = frame_bgr.copy()
            warped_bgr = self._binary_to_bgr(warped)
            lane_overlay = cv2.warpPerspective(
                sliding_vis,
                cv2.getPerspectiveTransform(dst, src),
                (roi.shape[1], roi.shape[0]),
            )
            overlay[roi_y:, :] = cv2.addWeighted(overlay[roi_y:, :], 0.6, lane_overlay, 0.8, 0.0)

            debug_images["threshold_image"] = self._binary_to_bgr(edges)
            debug_images["warp_image"] = warped_bgr
            debug_images["histogram_image"] = histogram_vis
            debug_images["sliding_window_image"] = sliding_vis
            debug_images["lane_overlay_image"] = overlay

        return LaneDetectionResult(lane_error, lane_curvature, confidence, debug_images)

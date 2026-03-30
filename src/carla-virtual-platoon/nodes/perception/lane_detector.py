#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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
        self._smoothed_lane_error = 0.0
        self._prev_left_fit: Optional[np.ndarray] = None
        self._prev_right_fit: Optional[np.ndarray] = None
        self._prev_confidence = 0.0

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    @staticmethod
    def _binary_to_bgr(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _poly_x(fit: np.ndarray, y_value: float) -> float:
        return float(fit[0] * y_value ** 2 + fit[1] * y_value + fit[2])

    @staticmethod
    def _select_strategic_lanes(peaks: np.ndarray, midpoint: int):
        left_lanes = sorted([int(p) for p in peaks if p < midpoint], reverse=True)
        right_lanes = sorted([int(p) for p in peaks if p > midpoint])
        center_left = left_lanes[0] if left_lanes else None
        center_right = right_lanes[0] if right_lanes else None
        adj_left = left_lanes[1] if len(left_lanes) > 1 else None
        adj_right = right_lanes[1] if len(right_lanes) > 1 else None
        return center_left, center_right, adj_left, adj_right

    @staticmethod
    def _get_fallback_seeds(binary: np.ndarray, midpoint: int, prefer_right_bias: float = 0.1):
        if binary.size == 0:
            return None, None
        roi = binary[int(binary.shape[0] * 0.6) :, :]
        if roi.size == 0:
            return None, None

        hist = np.sum(roi, axis=0).astype(np.float32)
        if hist.size == 0 or float(np.max(hist)) <= 10.0:
            return None, None

        left_hist = hist[:midpoint]
        right_hist = hist[midpoint:].copy()
        if prefer_right_bias > 0.0 and right_hist.size:
            right_hist *= 1.0 + float(prefer_right_bias)

        left_seed = int(np.argmax(left_hist)) if left_hist.size else None
        right_seed = int(np.argmax(right_hist)) + midpoint if right_hist.size else None
        return left_seed, right_seed

    @staticmethod
    def _find_histogram_peaks(histogram: np.ndarray, min_height: float) -> np.ndarray:
        if histogram.size < 3:
            return np.array([], dtype=np.int32)
        peaks = []
        for idx in range(1, histogram.size - 1):
            value = histogram[idx]
            if value < min_height:
                continue
            if value >= histogram[idx - 1] and value >= histogram[idx + 1]:
                peaks.append(idx)
        return np.array(peaks, dtype=np.int32)

    def _score_lane_pair(
        self,
        left_peak: int,
        right_peak: int,
        *,
        image_width: int,
        preview_y: int,
        lane_min_width_px: float,
        lane_max_width_px: float,
        expected_width_px: float,
        image_center_px: float,
    ) -> float:
        lane_width = float(right_peak - left_peak)
        if lane_width < lane_min_width_px or lane_width > lane_max_width_px:
            return -1e9

        lane_center = 0.5 * (float(left_peak) + float(right_peak))
        width_error = abs(lane_width - expected_width_px) / max(1.0, expected_width_px)
        center_error = abs(lane_center - image_center_px) / max(1.0, image_width * 0.5)

        continuity_error = center_error
        if self._prev_left_fit is not None and self._prev_right_fit is not None:
            prev_center = 0.5 * (
                self._poly_x(self._prev_left_fit, preview_y)
                + self._poly_x(self._prev_right_fit, preview_y)
            )
            continuity_error = abs(lane_center - prev_center) / max(1.0, image_width * 0.5)
        elif self._prev_left_fit is not None:
            prev_left = self._poly_x(self._prev_left_fit, preview_y)
            continuity_error = abs(float(left_peak) - prev_left) / max(1.0, image_width * 0.5)
        elif self._prev_right_fit is not None:
            prev_right = self._poly_x(self._prev_right_fit, preview_y)
            continuity_error = abs(float(right_peak) - prev_right) / max(1.0, image_width * 0.5)

        score = 3.0
        score -= 1.6 * width_error
        score -= 1.2 * center_error
        score -= 2.2 * continuity_error
        return score

    def _validate_lane_geometry(
        self,
        left_fit: Optional[np.ndarray],
        right_fit: Optional[np.ndarray],
        *,
        preview_y: int,
        lane_min_width_px: float,
        lane_max_width_px: float,
        image_width: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if left_fit is None or right_fit is None:
            return left_fit, right_fit

        left_x_preview = self._poly_x(left_fit, preview_y)
        right_x_preview = self._poly_x(right_fit, preview_y)
        lane_width = right_x_preview - left_x_preview

        if lane_width < lane_min_width_px or lane_width > lane_max_width_px:
            prev_left = self._prev_left_fit
            prev_right = self._prev_right_fit
            if prev_left is not None and prev_right is not None:
                prev_width = self._poly_x(prev_right, preview_y) - self._poly_x(prev_left, preview_y)
                if lane_min_width_px <= prev_width <= lane_max_width_px:
                    return prev_left, prev_right

            image_center = image_width * 0.5
            left_dist = abs(left_x_preview - (image_center - lane_width * 0.5))
            right_dist = abs(right_x_preview - (image_center + lane_width * 0.5))
            if left_dist > right_dist:
                return None, right_fit
            return left_fit, None

        return left_fit, right_fit

    def detect(
        self,
        frame_bgr: np.ndarray,
        *,
        roi_y_ratio: float = 0.55,
        white_lower_hsv=(0, 0, 180),
        white_upper_hsv=(179, 70, 255),
        white_close_kernel_height: int = 17,
        white_open_kernel_size: int = 5,
        lane_error_alpha: float = 0.85,
        preview_y_ratio: float = 0.35,
        far_preview_y_ratio: float = 0.18,
        far_preview_weight: float = 0.4,
        lane_half_width_ratio: float = 0.18,
        lane_min_width_ratio: float = 0.22,
        lane_max_width_ratio: float = 0.48,
        lane_expected_width_ratio: float = 0.32,
        lane_center_bias_px: float = 0.0,
        fit_conf_threshold: float = 0.03,
        fallback_decay: float = 0.97,
        canny_low: int = 70,
        canny_high: int = 140,
        n_windows: int = 8,
        margin: int = 50,
        minpix: int = 20,
        warp_top_y_ratio: float = 0.25,
        warp_top_left_x_ratio: float = 0.32,
        warp_top_right_x_ratio: float = 0.68,
        warp_bottom_left_x_ratio: float = 0.0,
        warp_bottom_right_x_ratio: float = 1.0,
        warp_dst_left_x_ratio: float = 0.18,
        warp_dst_right_x_ratio: float = 0.82,
        warped_gap_close_kernel_height: int = 31,
        warped_gap_close_kernel_width: int = 5,
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

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(
            hsv,
            np.array(white_lower_hsv, dtype=np.uint8),
            np.array(white_upper_hsv, dtype=np.uint8),
        )
        white_mask = cv2.GaussianBlur(white_mask, (5, 5), 0)
        _, white_mask = cv2.threshold(white_mask, 80, 255, cv2.THRESH_BINARY)
        close_kernel_h = max(3, int(white_close_kernel_height))
        if close_kernel_h % 2 == 0:
            close_kernel_h += 1
        open_kernel_s = max(3, int(white_open_kernel_size))
        if open_kernel_s % 2 == 0:
            open_kernel_s += 1

        vertical_kernel = np.ones((close_kernel_h, 3), np.uint8)
        square_kernel = np.ones((open_kernel_s, open_kernel_s), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, vertical_kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, square_kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, square_kernel)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, canny_low, canny_high)
        lane_binary = cv2.bitwise_and(white_mask, edges)
        if np.count_nonzero(lane_binary) < 300:
            lane_binary = white_mask

        top_y = roi.shape[0] * float(warp_top_y_ratio)
        src = np.float32(
            [
                [roi.shape[1] * float(warp_bottom_left_x_ratio), roi.shape[0] - 1],
                [roi.shape[1] * float(warp_top_left_x_ratio), top_y],
                [roi.shape[1] * float(warp_top_right_x_ratio), top_y],
                [roi.shape[1] * float(warp_bottom_right_x_ratio), roi.shape[0] - 1],
            ]
        )
        dst = np.float32(
            [
                [roi.shape[1] * float(warp_dst_left_x_ratio), roi.shape[0] - 1],
                [roi.shape[1] * float(warp_dst_left_x_ratio), 0],
                [roi.shape[1] * float(warp_dst_right_x_ratio), 0],
                [roi.shape[1] * float(warp_dst_right_x_ratio), roi.shape[0] - 1],
            ]
        )
        mat = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(lane_binary, mat, (roi.shape[1], roi.shape[0]))
        binary = (warped > 0).astype(np.uint8)
        gap_close_h = max(3, int(warped_gap_close_kernel_height))
        gap_close_w = max(1, int(warped_gap_close_kernel_width))
        if gap_close_h % 2 == 0:
            gap_close_h += 1
        if gap_close_w % 2 == 0:
            gap_close_w += 1
        gap_kernel = np.ones((gap_close_h, gap_close_w), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, gap_kernel)

        row_weights = np.linspace(0.4, 1.0, binary.shape[0], dtype=np.float32).reshape(-1, 1)
        weighted_binary = binary.astype(np.float32) * row_weights
        histogram = np.sum(weighted_binary[binary.shape[0] // 3 :, :], axis=0)
        midpoint = histogram.shape[0] // 2 if histogram.size else binary.shape[1] // 2
        lane_half_width_px = max(20.0, float(binary.shape[1]) * float(lane_half_width_ratio))
        lane_min_width_px = max(40.0, float(binary.shape[1]) * float(lane_min_width_ratio))
        lane_max_width_px = max(lane_min_width_px + 10.0, float(binary.shape[1]) * float(lane_max_width_ratio))
        lane_expected_width_px = self._clamp(
            float(binary.shape[1]) * float(lane_expected_width_ratio),
            lane_min_width_px,
            lane_max_width_px,
        )
        image_center_px = binary.shape[1] / 2.0

        hist_for_peaks = histogram.astype(np.float32)
        if hist_for_peaks.size:
            hist_for_peaks = cv2.GaussianBlur(hist_for_peaks.reshape(1, -1), (1, 9), 0).reshape(-1)
        peak_threshold = max(10.0, float(np.max(hist_for_peaks)) * 0.2) if hist_for_peaks.size else 10.0
        peaks = self._find_histogram_peaks(hist_for_peaks, peak_threshold)

        center_left, center_right, adj_left, adj_right = self._select_strategic_lanes(peaks, midpoint)
        left_candidates = [int(p) for p in peaks if p < midpoint]
        right_candidates = [int(p) for p in peaks if p > midpoint]
        left_candidates.sort(key=lambda p: abs(midpoint - p))
        right_candidates.sort(key=lambda p: abs(p - midpoint))

        if center_left is not None:
            left_candidates.sort(key=lambda p: (p != center_left, abs(midpoint - p)))
        if center_right is not None:
            right_candidates.sort(key=lambda p: (p != center_right, abs(midpoint - p)))

        left_base_x = midpoint - int(lane_half_width_px)
        right_base_x = midpoint + int(lane_half_width_px)
        selected_pair = None
        selected_pair_score = -1e9
        preview_y = int(self._clamp(binary.shape[0] * float(preview_y_ratio), 0, binary.shape[0] - 1))
        far_preview_y = int(self._clamp(binary.shape[0] * float(far_preview_y_ratio), 0, binary.shape[0] - 1))
        far_preview_weight = self._clamp(float(far_preview_weight), 0.0, 0.8)
        for left_peak in left_candidates:
            for right_peak in right_candidates:
                pair_score = self._score_lane_pair(
                    left_peak,
                    right_peak,
                    image_width=binary.shape[1],
                    preview_y=preview_y,
                    lane_min_width_px=lane_min_width_px,
                    lane_max_width_px=lane_max_width_px,
                    expected_width_px=lane_expected_width_px,
                    image_center_px=image_center_px,
                )
                if pair_score > selected_pair_score:
                    selected_pair = (left_peak, right_peak)
                    selected_pair_score = pair_score

        if selected_pair is not None and selected_pair_score > -1e8:
            left_base_x, right_base_x = selected_pair
        else:
            fallback_left, fallback_right = self._get_fallback_seeds(binary, midpoint)
            if center_left is not None:
                left_base_x = center_left
            elif left_candidates:
                left_base_x = left_candidates[0]
            elif fallback_left is not None:
                left_base_x = fallback_left

            if center_right is not None:
                right_base_x = center_right
            elif right_candidates:
                right_base_x = right_candidates[0]
            elif fallback_right is not None:
                right_base_x = fallback_right

        nonzero_y, nonzero_x = binary.nonzero()
        left_current_x = left_base_x
        right_current_x = right_base_x
        left_lane_inds = []
        right_lane_inds = []
        window_h = max(1, binary.shape[0] // max(1, n_windows))
        sliding_vis = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

        for idx in range(n_windows):
            win_y_low = binary.shape[0] - (idx + 1) * window_h
            win_y_high = binary.shape[0] - idx * window_h
            left_x_low = max(0, left_current_x - margin)
            left_x_high = min(binary.shape[1], left_current_x + margin)
            right_x_low = max(0, right_current_x - margin)
            right_x_high = min(binary.shape[1], right_current_x + margin)

            left_good_inds = (
                (nonzero_y >= win_y_low)
                & (nonzero_y < win_y_high)
                & (nonzero_x >= left_x_low)
                & (nonzero_x < left_x_high)
            ).nonzero()[0]
            right_good_inds = (
                (nonzero_y >= win_y_low)
                & (nonzero_y < win_y_high)
                & (nonzero_x >= right_x_low)
                & (nonzero_x < right_x_high)
            ).nonzero()[0]
            left_lane_inds.append(left_good_inds)
            right_lane_inds.append(right_good_inds)

            if use_debug_visualization and use_sliding_window_visualization:
                cv2.rectangle(
                    sliding_vis,
                    (left_x_low, win_y_low),
                    (left_x_high, win_y_high),
                    (0, 255, 0),
                    2,
                )
                cv2.rectangle(
                    sliding_vis,
                    (right_x_low, win_y_low),
                    (right_x_high, win_y_high),
                    (255, 255, 0),
                    2,
                )

            if left_good_inds.size > minpix:
                left_current_x = int(np.mean(nonzero_x[left_good_inds]))
            if right_good_inds.size > minpix:
                right_current_x = int(np.mean(nonzero_x[right_good_inds]))

        left_xs = np.concatenate(left_lane_inds) if left_lane_inds else np.array([], dtype=np.int32)
        right_xs = np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=np.int32)
        left_lane_x = nonzero_x[left_xs] if left_xs.size else np.array([], dtype=np.int32)
        left_lane_y = nonzero_y[left_xs] if left_xs.size else np.array([], dtype=np.int32)
        right_lane_x = nonzero_x[right_xs] if right_xs.size else np.array([], dtype=np.int32)
        right_lane_y = nonzero_y[right_xs] if right_xs.size else np.array([], dtype=np.int32)

        left_confidence = min(1.0, left_lane_x.size / 2500.0)
        right_confidence = min(1.0, right_lane_x.size / 2500.0)
        confidence = 0.5 * (left_confidence + right_confidence)
        lane_curvature = 0.0
        lane_error = self._prev_lane_error
        left_fit = None
        right_fit = None
        if left_lane_x.size >= 80:
            left_fit = np.polyfit(left_lane_y.astype(np.float32), left_lane_x.astype(np.float32), 2)
        elif self._prev_left_fit is not None and left_confidence >= fit_conf_threshold:
            left_fit = self._prev_left_fit

        if right_lane_x.size >= 80:
            right_fit = np.polyfit(right_lane_y.astype(np.float32), right_lane_x.astype(np.float32), 2)
        elif self._prev_right_fit is not None and right_confidence >= fit_conf_threshold:
            right_fit = self._prev_right_fit

        left_fit, right_fit = self._validate_lane_geometry(
            left_fit,
            right_fit,
            preview_y=preview_y,
            lane_min_width_px=lane_min_width_px,
            lane_max_width_px=lane_max_width_px,
            image_width=binary.shape[1],
        )

        if left_fit is not None or right_fit is not None:
            if left_fit is not None and right_fit is not None:
                left_x_preview = self._poly_x(left_fit, preview_y)
                right_x_preview = self._poly_x(right_fit, preview_y)
                lane_center_px = 0.5 * (left_x_preview + right_x_preview)
                far_lane_center_px = 0.5 * (
                    self._poly_x(left_fit, far_preview_y) + self._poly_x(right_fit, far_preview_y)
                )
                lane_center_px = (1.0 - far_preview_weight) * lane_center_px + far_preview_weight * far_lane_center_px
                lane_curvature = float(0.5 * (left_fit[0] + right_fit[0]))
                confidence = min(left_confidence, right_confidence)
                lane_width_preview = right_x_preview - left_x_preview
                width_conf = 1.0 - min(
                    1.0,
                    abs(lane_width_preview - lane_expected_width_px) / max(1.0, lane_expected_width_px),
                )
                confidence *= max(0.25, width_conf)
                confidence = max(confidence, self._prev_confidence * 0.82)
            elif left_fit is not None:
                left_x_preview = self._poly_x(left_fit, preview_y)
                lane_center_px = left_x_preview + lane_half_width_px
                far_lane_center_px = self._poly_x(left_fit, far_preview_y) + lane_half_width_px
                lane_center_px = (1.0 - far_preview_weight) * lane_center_px + far_preview_weight * far_lane_center_px
                lane_curvature = float(left_fit[0])
                confidence = max(left_confidence * 0.7, self._prev_confidence * 0.72)
            else:
                right_x_preview = self._poly_x(right_fit, preview_y)
                lane_center_px = right_x_preview - lane_half_width_px
                far_lane_center_px = self._poly_x(right_fit, far_preview_y) - lane_half_width_px
                lane_center_px = (1.0 - far_preview_weight) * lane_center_px + far_preview_weight * far_lane_center_px
                lane_curvature = float(right_fit[0])
                confidence = max(right_confidence * 0.7, self._prev_confidence * 0.72)

            lane_center_px += float(lane_center_bias_px)
            lane_center_px = self._clamp(lane_center_px, 0.0, float(binary.shape[1] - 1))

            raw_lane_error = self._clamp(
                (lane_center_px - image_center_px) / max(1.0, image_center_px),
                -1.0,
                1.0,
            )
            alpha = self._clamp(float(lane_error_alpha), 0.0, 0.98)
            lane_error = alpha * self._smoothed_lane_error + (1.0 - alpha) * raw_lane_error
            lane_error = self._clamp(lane_error, -1.0, 1.0)
            self._prev_lane_error = lane_error
            self._smoothed_lane_error = lane_error
            self._prev_confidence = confidence
            if left_fit is not None:
                self._prev_left_fit = left_fit
            if right_fit is not None:
                self._prev_right_fit = right_fit

            if use_debug_visualization:
                plot_y = np.linspace(0, binary.shape[0] - 1, binary.shape[0]).astype(np.int32)
                if left_fit is not None:
                    left_fit_x = (
                        left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
                    ).astype(np.int32)
                    left_fit_x = np.clip(left_fit_x, 0, binary.shape[1] - 1)
                    sliding_vis[plot_y, left_fit_x] = (255, 0, 255)
                if right_fit is not None:
                    right_fit_x = (
                        right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]
                    ).astype(np.int32)
                    right_fit_x = np.clip(right_fit_x, 0, binary.shape[1] - 1)
                    sliding_vis[plot_y, right_fit_x] = (255, 0, 255)
                if left_xs.size:
                    sliding_vis[nonzero_y[left_xs], nonzero_x[left_xs]] = (0, 0, 255)
                if right_xs.size:
                    sliding_vis[nonzero_y[right_xs], nonzero_x[right_xs]] = (0, 255, 255)
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
                cv2.line(
                    sliding_vis,
                    (0, preview_y),
                    (sliding_vis.shape[1] - 1, preview_y),
                    (0, 165, 255),
                    2,
                )
                cv2.putText(
                    sliding_vis,
                    f"err={lane_error:.3f} conf={confidence:.3f} pair={left_base_x}:{right_base_x} score={selected_pair_score:.2f}",
                    (10, sliding_vis.shape[0] - 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    sliding_vis,
                    f"strategic=({center_left},{center_right}) adj=({adj_left},{adj_right})",
                    (10, sliding_vis.shape[0] - 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    sliding_vis,
                    f"hw={lane_half_width_px:.0f} w=[{lane_min_width_px:.0f},{lane_max_width_px:.0f}] exp={lane_expected_width_px:.0f}",
                    (10, sliding_vis.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        elif self._prev_left_fit is not None or self._prev_right_fit is not None:
            image_center_px = binary.shape[1] / 2.0
            if self._prev_left_fit is not None and self._prev_right_fit is not None:
                lane_center_px = 0.5 * (
                    self._poly_x(self._prev_left_fit, preview_y)
                    + self._poly_x(self._prev_right_fit, preview_y)
                )
                far_lane_center_px = 0.5 * (
                    self._poly_x(self._prev_left_fit, far_preview_y)
                    + self._poly_x(self._prev_right_fit, far_preview_y)
                )
                lane_center_px = (1.0 - far_preview_weight) * lane_center_px + far_preview_weight * far_lane_center_px
                lane_curvature = float(0.5 * (self._prev_left_fit[0] + self._prev_right_fit[0]))
            elif self._prev_left_fit is not None:
                lane_center_px = self._poly_x(self._prev_left_fit, preview_y) + lane_half_width_px
                far_lane_center_px = self._poly_x(self._prev_left_fit, far_preview_y) + lane_half_width_px
                lane_center_px = (1.0 - far_preview_weight) * lane_center_px + far_preview_weight * far_lane_center_px
                lane_curvature = float(self._prev_left_fit[0])
            else:
                lane_center_px = self._poly_x(self._prev_right_fit, preview_y) - lane_half_width_px
                far_lane_center_px = self._poly_x(self._prev_right_fit, far_preview_y) - lane_half_width_px
                lane_center_px = (1.0 - far_preview_weight) * lane_center_px + far_preview_weight * far_lane_center_px
                lane_curvature = float(self._prev_right_fit[0])
            lane_center_px += float(lane_center_bias_px)
            lane_center_px = self._clamp(lane_center_px, 0.0, float(binary.shape[1] - 1))
            raw_lane_error = self._clamp(
                (lane_center_px - image_center_px) / max(1.0, image_center_px),
                -1.0,
                1.0,
            )
            decay = self._clamp(float(fallback_decay), 0.0, 1.0)
            lane_error = decay * self._smoothed_lane_error + (1.0 - decay) * raw_lane_error
            lane_error = self._clamp(lane_error, -1.0, 1.0)
            confidence = self._prev_confidence * 0.96
            self._prev_lane_error = lane_error
            self._smoothed_lane_error = lane_error

            if use_debug_visualization:
                plot_y = np.linspace(0, binary.shape[0] - 1, binary.shape[0]).astype(np.int32)
                if self._prev_left_fit is not None:
                    left_fit_x = (
                        self._prev_left_fit[0] * plot_y ** 2
                        + self._prev_left_fit[1] * plot_y
                        + self._prev_left_fit[2]
                    ).astype(np.int32)
                    left_fit_x = np.clip(left_fit_x, 0, binary.shape[1] - 1)
                    sliding_vis[plot_y, left_fit_x] = (255, 0, 255)
                if self._prev_right_fit is not None:
                    right_fit_x = (
                        self._prev_right_fit[0] * plot_y ** 2
                        + self._prev_right_fit[1] * plot_y
                        + self._prev_right_fit[2]
                    ).astype(np.int32)
                    right_fit_x = np.clip(right_fit_x, 0, binary.shape[1] - 1)
                    sliding_vis[plot_y, right_fit_x] = (255, 0, 255)
                cv2.line(
                    sliding_vis,
                    (0, preview_y),
                    (sliding_vis.shape[1] - 1, preview_y),
                    (0, 165, 255),
                    2,
                )
                cv2.putText(
                    sliding_vis,
                    f"fallback err={lane_error:.3f} conf={confidence:.3f} hw={lane_half_width_px:.0f}",
                    (10, sliding_vis.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
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
            cv2.line(
                overlay,
                (w // 2, roi_y),
                (w // 2, h - 1),
                (0, 255, 255),
                2,
            )
            src_int = src.astype(np.int32)
            cv2.polylines(overlay, [src_int + np.array([0, roi_y])], True, (0, 200, 255), 2)

            debug_images["threshold_image"] = self._binary_to_bgr(lane_binary)
            debug_images["warp_image"] = warped_bgr
            debug_images["histogram_image"] = histogram_vis
            debug_images["sliding_window_image"] = sliding_vis
            debug_images["lane_overlay_image"] = overlay

        return LaneDetectionResult(lane_error, lane_curvature, confidence, debug_images)

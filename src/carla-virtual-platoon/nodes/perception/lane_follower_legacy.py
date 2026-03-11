#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import cv2
import numpy as np
from cv2 import COLOR_BGR2HSV, cvtColor


class LaneFollower:
    """Legacy sliding-window lane follower adapted for CARLA stack."""

    def __init__(self) -> None:
        self.img = None
        self.img_hsv = None
        self.y, self.x = None, None
        self.yellow_range = None
        self.white_range = None
        self.combined_range = None
        self.yellow_warped = None

        self.warp_img_size = None
        self.warp_img_zoomx = None
        self.warped_img = None

        self.out_img = None
        self.nwindows = 12
        self.margin = 60
        self.minpix = 5
        self.threshold = 100
        self.l_lane, self.r_lane = None, None

        self.pos = None
        self.alpha = 0.8
        self.jump_thresh = 150.0
        self.pos_min = 0.0
        self.pos_max = 1e9
        self.lane_half_px = 40.0
        self.center_boost_px = 12.0
        self.curve_offset_gain = 0.40
        self.preview_alpha_scale = 50.0
        self._curv_ema = 0.0

    def process(self, bgr_img):
        """
        Input: BGR image
        Output: (steer_0_to_1, pos_norm_0_to_1, debug_bgr, confidence_0_to_1)
        """
        if bgr_img is None or not hasattr(bgr_img, "shape"):
            return 0.5, 0.5, None, 0.0

        self.img_init(bgr_img)
        self.img_transform()

        try:
            self.img_warp(self.yellow_range, change_img=False, warp_img_zoomx=self.x // 2.5)
            self.yellow_warped = self.warped_img
        except Exception:
            pass
        self.img_warp()
        self.sliding_window()

        center_pos_raw = self.compute_center_pos()
        if center_pos_raw is None:
            center_pos_raw = self.go_yellow()
        if center_pos_raw is None or self.x is None or self.x <= 0:
            return 0.5, 0.5, self.out_img, 0.05

        self.pos = self._smooth_pos(self.pos, center_pos_raw)

        pos_norm = float(np.clip(self.pos / float(self.x), 0.0, 1.0))
        steer = float(np.clip(pos_norm, 0.0, 1.0))
        confidence = self._estimate_confidence()
        return steer, pos_norm, self.out_img, confidence

    def go_yellow(self):
        if self.yellow_warped is None:
            return None

        h, w = self.yellow_warped.shape[:2]
        col_sum = np.sum(self.yellow_warped, axis=0).astype(np.float64)
        s = col_sum.sum()
        if s < 1e-6:
            return None
        x_all = (np.arange(w) * col_sum).sum() / s

        def band_centroid(img, y0, y1, fallback):
            band = img[y0:y1, :]
            col = np.sum(band, axis=0).astype(np.float64)
            ss = col.sum()
            if ss < 1e-6:
                return fallback
            return (np.arange(col.shape[0]) * col).sum() / ss

        y_low0, y_low1 = int(0.78 * h), int(0.96 * h)
        y_high0, y_high1 = int(0.40 * h), int(0.58 * h)
        x_low = band_centroid(self.yellow_warped, y_low0, y_low1, x_all)
        x_high = band_centroid(self.yellow_warped, y_high0, y_high1, x_all)

        curv = abs(x_high - x_low)
        self._curv_ema = 0.6 * self._curv_ema + 0.4 * curv

        base_offset = float(self.lane_half_px + self.center_boost_px)
        extra = float(np.clip(self.curve_offset_gain * self._curv_ema, 0.0, 35.0))
        alpha = float(np.clip(self._curv_ema / max(1.0, self.preview_alpha_scale), 0.0, 0.9))
        x_preview = (1.0 - alpha) * x_low + alpha * x_high

        return float(np.clip(x_preview + base_offset + extra, 0, w - 1))

    def img_init(self, img):
        self.img = img
        self.y, self.x, _ = img.shape
        self.img_hsv = cvtColor(img, COLOR_BGR2HSV)
        self.warp_img_size = [self.x, self.x]
        self.warp_img_zoomx = self.x // 4

    def img_transform(self):
        img_hsv = self.img_hsv
        lower_yellow = np.array([15, 100, 140], dtype=np.uint8)
        upper_yellow = np.array([30, 200, 255], dtype=np.uint8)
        self.yellow_range = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

        lower_white = np.array([0, 0, 140], dtype=np.uint8)
        upper_white = np.array([50, 70, 255], dtype=np.uint8)
        self.white_range = cv2.inRange(img_hsv, lower_white, upper_white)

        self.combined_range = cv2.bitwise_or(self.yellow_range, self.white_range)
        self.img = self.combined_range

    def img_warp(self, img=None, change_img=True, warp_img_zoomx=None):
        y, x = self.y, self.x
        if img is None:
            img = self.img
        warp_img_size = self.warp_img_size
        if warp_img_zoomx is None:
            warp_img_zoomx = self.warp_img_zoomx

        src_points = np.float32([[0, 450], [285, 260], [x - 285, 260], [x, 450]])

        topx = warp_img_zoomx
        bottomx = warp_img_zoomx
        topy = x // 4
        bottomy = x
        dst_points = np.float32(
            [[bottomx, bottomy], [topx, topy], [x - topx, topy], [x - bottomx, bottomy]]
        )

        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.warped_img = cv2.warpPerspective(img, matrix, warp_img_size)

        if change_img:
            self.img = self.warped_img
            self.x, self.y = warp_img_size

    def sliding_window(self):
        lane = self.img
        midpoint = self.x // 2
        histogram = np.sum(lane, axis=0)
        leftx_current = int(np.argmax(histogram[:midpoint])) if midpoint > 0 else 0
        rightx_current = int(np.argmax(histogram[midpoint:]) + midpoint) if midpoint < len(histogram) else 0

        window_height = max(1, int(self.x / max(1, self.nwindows)))
        nz = lane.nonzero()

        lx, ly, rx, ry = [], [], [], []
        left_lane_inds = []
        right_lane_inds = []

        try:
            self.out_img = np.dstack((lane, lane, lane))
        except Exception:
            self.out_img = None

        l_err = [0, 0]
        r_err = [0, 0]
        foundr, foundl = (False, False)

        for window in range(self.nwindows):
            win_yl = self.x - (window + 1) * window_height
            win_yh = self.x - window * window_height

            win_xll = leftx_current - self.margin
            win_xlh = leftx_current + self.margin
            win_xrl = rightx_current - self.margin
            win_xrh = rightx_current + self.margin

            good_left_mask = (nz[0] >= win_yl) & (nz[0] < win_yh) & (nz[1] >= win_xll) & (nz[1] < win_xlh)
            good_right_mask = (nz[0] >= win_yl) & (nz[0] < win_yh) & (nz[1] >= win_xrl) & (nz[1] < win_xrh)

            good_left_inds = np.where(good_left_mask)[0]
            good_right_inds = np.where(good_right_mask)[0]
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > self.minpix:
                if self.threshold < leftx_current < (len(histogram) - self.threshold):
                    l_err[1] = 0
                    foundl = True
                else:
                    l_err[1] += 1
                leftx_current = int(np.mean(nz[1][good_left_inds]))
            else:
                l_err[1] += 1
            if not foundl:
                l_err[0] += 1

            if len(good_right_inds) > self.minpix:
                if self.threshold < rightx_current < (len(histogram) - self.threshold):
                    r_err[1] = 0
                    foundr = True
                else:
                    r_err[1] += 1
                rightx_current = int(np.mean(nz[1][good_right_inds]))
            else:
                r_err[1] += 1
            if not foundr:
                r_err[0] += 1

            lx.append(leftx_current)
            ly.append((win_yl + win_yh) / 2.0)
            rx.append(rightx_current)
            ry.append((win_yl + win_yh) / 2.0)

        try:
            left_lane_inds = np.concatenate(left_lane_inds) if len(left_lane_inds) else np.array([], dtype=int)
            right_lane_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) else np.array([], dtype=int)
            if self.out_img is not None:
                if left_lane_inds.size:
                    self.out_img[nz[0][left_lane_inds], nz[1][left_lane_inds]] = [255, 0, 0]
                if right_lane_inds.size:
                    self.out_img[nz[0][right_lane_inds], nz[1][right_lane_inds]] = [0, 0, 255]
        except Exception:
            pass

        self.l_lane, self.r_lane = (lx, ly, l_err), (rx, ry, r_err)

    def compute_center_pos(self):
        if not self.l_lane or not self.r_lane:
            return None
        try:
            if len(self.l_lane[0]) < 4 or len(self.r_lane[0]) < 4:
                return None
            posl = int(self.l_lane[0][3])
            posr = int(self.r_lane[0][3])
        except Exception:
            return None
        pos = (posl + posr) // 2
        return float(pos)

    def _smooth_pos(self, prev_pos, new_pos):
        if prev_pos is None:
            return float(new_pos)
        if abs(new_pos - prev_pos) > self.jump_thresh:
            new_pos = prev_pos
        pos = self.alpha * new_pos + (1.0 - self.alpha) * prev_pos
        pos = max(self.pos_min, min(self.pos_max, pos))
        return pos

    def _estimate_confidence(self) -> float:
        if self.yellow_warped is not None:
            density = float(np.count_nonzero(self.yellow_warped)) / float(max(1, self.yellow_warped.size))
        else:
            density = 0.0
        if not self.l_lane or not self.r_lane:
            return min(0.6, 0.05 + 8.0 * density)
        l_ok = float(sum(self.l_lane[2]) <= 6)
        r_ok = float(sum(self.r_lane[2]) <= 6)
        return min(1.0, 0.2 + 0.35 * l_ok + 0.35 * r_ok + min(0.1, 4.0 * density))

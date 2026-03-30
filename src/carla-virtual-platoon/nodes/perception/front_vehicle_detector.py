#!/usr/bin/env python3

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class FrontVehicleDetectionResult:
    distance: float


class FrontVehicleDetector:
    def detect(
        self,
        points,
        *,
        max_distance: float,
        roi_y_abs: float,
        min_z: float,
        max_z: float,
        roi_y_min: Optional[float] = None,
        roi_y_max: Optional[float] = None,
        max_angle_deg: float = 45.0,
    ):
        min_distance = float(max_distance)
        y_min = float(roi_y_min) if roi_y_min is not None else -float(roi_y_abs)
        y_max = float(roi_y_max) if roi_y_max is not None else float(roi_y_abs)
        max_angle_rad = math.radians(float(max_angle_deg))

        for point in points:
            x, y, z = float(point[0]), float(point[1]), float(point[2])
            if x <= 0.0:
                continue
            if y < y_min or y > y_max or z < min_z or z > max_z:
                continue
            if abs(math.atan2(y, x)) > max_angle_rad:
                continue
            distance = math.sqrt(x * x + y * y + z * z)
            if distance < min_distance:
                min_distance = distance
        return FrontVehicleDetectionResult(distance=min_distance)

#!/usr/bin/env python3

import math
from dataclasses import dataclass


@dataclass
class FrontVehicleDetectionResult:
    distance: float


class FrontVehicleDetector:
    def detect(self, points, *, max_distance: float, roi_y_abs: float, min_z: float, max_z: float):
        min_distance = float(max_distance)
        for point in points:
            x, y, z = float(point[0]), float(point[1]), float(point[2])
            if x <= 0.0:
                continue
            if abs(y) > roi_y_abs or z < min_z or z > max_z:
                continue
            distance = math.sqrt(x * x + y * y + z * z)
            if distance < min_distance:
                min_distance = distance
        return FrontVehicleDetectionResult(distance=min_distance)

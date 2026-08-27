from __future__ import annotations

import cv2
import numpy as np

from glove_chirality.config import DetectorConfig
from glove_chirality.detection.base import GloveDetector
from glove_chirality.types import Detection


class DarkContourDetector(GloveDetector):
    """CPU bootstrap detector for dark gloves on a bright fixed background."""

    name = "dark_contour"

    def __init__(self, config: DetectorConfig):
        self.config = config

    @staticmethod
    def _box(box: tuple[float, float, float, float], w: int, h: int):
        x1, y1, x2, y2 = box
        return (round(x1 * w), round(y1 * h), round(x2 * w), round(y2 * h))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        rx1, ry1, rx2, ry2 = self._box(self.config.roi, width, height)
        tx1, ty1, tx2, ty2 = self._box(self.config.trigger_zone, width, height)
        region = frame[ry1:ry2, rx1:rx2]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        blur = max(3, self.config.blur_kernel | 1)
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)
        _, mask = cv2.threshold(gray, self.config.dark_threshold, 255, cv2.THRESH_BINARY_INV)
        morph = max(3, self.config.morph_kernel | 1)
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, element)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, element)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_area = float(max(1, region.shape[0] * region.shape[1]))
        trigger_area = float(max(1, (tx2 - tx1) * (ty2 - ty1)))
        found: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not self.config.min_area_ratio <= area / roi_area <= self.config.max_area_ratio:
                continue
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = area / max(1.0, hull_area)
            if solidity < self.config.min_solidity:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            candidate = Detection(rx1 + x, ry1 + y, rx1 + x + w, ry1 + y + h, 0.0)
            cx, cy = candidate.center
            if not (tx1 <= cx <= tx2 and ty1 <= cy <= ty2):
                continue
            area_score = min(1.0, candidate.area / trigger_area * 3.0)
            confidence = 0.55 * area_score + 0.45 * min(1.0, solidity)
            found.append(Detection(candidate.x1, candidate.y1, candidate.x2, candidate.y2, confidence))
        return sorted(found, key=lambda d: d.confidence, reverse=True)

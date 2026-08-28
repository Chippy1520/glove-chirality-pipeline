from __future__ import annotations

from abc import abstractmethod

import cv2
import numpy as np

from glove_chirality.config import DetectorConfig
from glove_chirality.detection.base import GloveDetector
from glove_chirality.types import Detection


class _ContourDetector(GloveDetector):
    def __init__(self, config: DetectorConfig):
        self.config = config

    @staticmethod
    def _box(box: tuple[float, float, float, float], width: int, height: int):
        x1, y1, x2, y2 = box
        return (
            round(x1 * width),
            round(y1 * height),
            round(x2 * width),
            round(y2 * height),
        )

    @abstractmethod
    def _foreground_mask(self, region: np.ndarray) -> np.ndarray:
        """Return an 8-bit foreground mask for an ROI-local BGR image."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        rx1, ry1, rx2, ry2 = self._box(self.config.roi, width, height)
        tx1, ty1, tx2, ty2 = self._box(self.config.trigger_zone, width, height)
        region = frame[ry1:ry2, rx1:rx2]
        if region.size == 0:
            return []
        mask = self._foreground_mask(region)
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
            x, y, box_width, box_height = cv2.boundingRect(contour)
            candidate = Detection(
                rx1 + x,
                ry1 + y,
                rx1 + x + box_width,
                ry1 + y + box_height,
                0.0,
            )
            area_score = min(1.0, candidate.area / trigger_area * 3.0)
            confidence = 0.55 * area_score + 0.45 * min(1.0, solidity)
            found.append(
                Detection(
                    candidate.x1,
                    candidate.y1,
                    candidate.x2,
                    candidate.y2,
                    confidence,
                )
            )
        return sorted(found, key=lambda detection: detection.confidence, reverse=True)


class BeltForegroundDetector(_ContourDetector):
    """Color-agnostic glove foreground against a dominant, stable conveyor belt.

    Lab color distance detects any glove that differs visually from the belt.
    Valid temporal background subtraction is preferred while objects move;
    color distance is the fallback. A custom detector remains necessary for truly
    camouflaged or touching gloves.
    """

    name = "belt_foreground"

    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.mog_history,
            varThreshold=config.mog_var_threshold,
            detectShadows=False,
        )
        self._foreground_present = False

    def _foreground_mask(self, region: np.ndarray) -> np.ndarray:
        blur = max(3, self.config.blur_kernel | 1)
        smoothed = cv2.GaussianBlur(region, (blur, blur), 0)
        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB).astype(np.float32)
        sampled = lab[:: max(1, self.config.belt_sample_stride), :: max(1, self.config.belt_sample_stride)]
        belt_color = np.median(sampled.reshape(-1, 3), axis=0)
        distance = np.linalg.norm(lab - belt_color, axis=2)
        color_mask = np.where(distance >= self.config.color_distance_threshold, 255, 0).astype(np.uint8)

        if not self.config.motion_assist:
            return color_mask
        if self.config.adaptive_background:
            learning_rate = (
                self.config.mog_foreground_learning_rate
                if self._foreground_present
                else self.config.mog_empty_learning_rate
            )
        else:
            learning_rate = self.config.mog_learning_rate
        motion_mask = self.background_subtractor.apply(smoothed, learningRate=learning_rate)
        pixels = max(1, motion_mask.size)
        motion_ratio = cv2.countNonZero(motion_mask) / pixels
        color_ratio = cv2.countNonZero(color_mask) / pixels
        motion_valid = self.config.min_area_ratio <= motion_ratio <= self.config.max_area_ratio
        color_valid = self.config.min_area_ratio <= color_ratio <= self.config.max_area_ratio
        self._foreground_present = motion_valid or color_valid
        if motion_valid:
            return motion_mask
        return color_mask


class DarkContourDetector(_ContourDetector):
    """Legacy detector for dark gloves on a brighter fixed background."""

    name = "dark_contour"

    def _foreground_mask(self, region: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        blur = max(3, self.config.blur_kernel | 1)
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)
        return cv2.threshold(
            gray,
            self.config.dark_threshold,
            255,
            cv2.THRESH_BINARY_INV,
        )[1]

import cv2
import numpy as np
import pytest

from glove_chirality.config import DetectorConfig
from glove_chirality.detection import build_detector


@pytest.mark.parametrize(
    "glove_color",
    [
        (25, 25, 25),       # black
        (230, 230, 230),    # white
        (30, 30, 220),      # red
        (220, 30, 30),      # blue
        (30, 220, 220),     # yellow
        (180, 40, 180),     # purple
    ],
)
def test_belt_foreground_detects_multiple_glove_colors(glove_color):
    frame = np.full((240, 320, 3), (65, 175, 65), dtype=np.uint8)
    cv2.rectangle(frame, (115, 70), (205, 175), glove_color, -1)
    config = DetectorConfig(
        backend="belt_foreground",
        roi=(0, 0, 1, 1),
        trigger_zone=(0.1, 0.1, 0.9, 0.9),
        color_distance_threshold=22,
        motion_assist=False,
        blur_kernel=3,
        morph_kernel=3,
        min_area_ratio=0.01,
        max_area_ratio=0.5,
    )
    detections = build_detector(config).detect(frame)
    assert len(detections) == 1
    assert detections[0].x1 <= 120 < detections[0].x2
    assert detections[0].y1 <= 75 < detections[0].y2


def test_motion_assist_detects_low_color_contrast_motion():
    background = np.full((240, 320, 3), (65, 175, 65), dtype=np.uint8)
    config = DetectorConfig(
        backend="belt_foreground",
        roi=(0, 0, 1, 1),
        trigger_zone=(0.1, 0.1, 0.9, 0.9),
        color_distance_threshold=255,
        motion_assist=True,
        mog_history=20,
        blur_kernel=3,
        morph_kernel=3,
        min_area_ratio=0.01,
        max_area_ratio=0.5,
    )
    detector = build_detector(config)
    for _ in range(10):
        detector.detect(background)
    frame = background.copy()
    cv2.rectangle(frame, (120, 80), (200, 170), (70, 160, 70), -1)
    assert detector.detect(frame)

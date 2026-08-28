import cv2
import numpy as np
import pytest

from glove_chirality.config import DetectorConfig
from glove_chirality.detection import build_detector
from glove_chirality.detection.base import inside_trigger
from glove_chirality.types import Detection


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


def test_trigger_rejects_partial_glove_even_when_center_is_inside():
    config = DetectorConfig(trigger_zone=(0.2, 0.2, 0.8, 0.8))
    partial = Detection(40, 70, 160, 170, 0.9)
    complete = Detection(80, 70, 200, 170, 0.9)
    assert partial.center[0] > 0.2 * 320
    assert inside_trigger(partial, config, 320, 240) is False
    assert inside_trigger(complete, config, 320, 240) is True
    legacy = DetectorConfig(
        trigger_zone=config.trigger_zone,
        require_full_containment=False,
    )
    assert inside_trigger(partial, legacy, 320, 240) is True


def test_trigger_inner_margin_requires_extra_clearance():
    config = DetectorConfig(
        trigger_zone=(0.2, 0.2, 0.8, 0.8),
        trigger_inner_margin_ratio=0.1,
    )
    near_edge = Detection(70, 55, 240, 180, 0.9)
    clear = Detection(85, 65, 235, 175, 0.9)
    assert inside_trigger(near_edge, config, 320, 240) is False
    assert inside_trigger(clear, config, 320, 240) is True


def test_classical_detector_emits_only_after_full_trigger_entry():
    config = DetectorConfig(
        roi=(0, 0, 1, 1),
        trigger_zone=(0.2, 0.2, 0.8, 0.8),
        color_distance_threshold=22,
        motion_assist=False,
        blur_kernel=3,
        morph_kernel=3,
        min_area_ratio=0.01,
        max_area_ratio=0.5,
    )
    detector = build_detector(config)
    partial = np.full((240, 320, 3), (65, 175, 65), dtype=np.uint8)
    cv2.rectangle(partial, (40, 70), (160, 170), (30, 30, 220), -1)
    complete = np.full_like(partial, (65, 175, 65))
    cv2.rectangle(complete, (80, 70), (200, 170), (30, 30, 220), -1)
    partial_detections = detector.detect(partial)
    assert len(partial_detections) == 1
    assert inside_trigger(partial_detections[0], config, 320, 240) is False
    complete_detections = detector.detect(complete)
    assert len(complete_detections) == 1
    assert inside_trigger(complete_detections[0], config, 320, 240) is True

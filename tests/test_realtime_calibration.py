from datetime import datetime, timezone

import numpy as np
import pytest

from glove_chirality.config import DetectorConfig, ExtractionConfig
from glove_chirality.detection.yolo import SizeRejectedDetection, YoloDetectionDiagnostics
from glove_chirality.realtime_calibration import (
    draw_calibration_overlay,
    normalized_box_pixels,
    screenshot_path,
    validate_calibration_config,
)
from glove_chirality.types import Detection


def test_calibration_requires_yolo_backend():
    with pytest.raises(ValueError, match="detector.backend: yolo"):
        validate_calibration_config(ExtractionConfig())

    validate_calibration_config(
        ExtractionConfig(
            detector=DetectorConfig(backend="yolo", yolo_model="custom.pt")
        )
    )


def test_normalized_overlay_geometry_uses_full_frame_coordinates():
    assert normalized_box_pixels((0.1, 0.2, 0.8, 0.9), 200, 100) == (20, 20, 160, 90)


def test_calibration_overlay_draws_rejected_diagnostic_without_mutating_frame():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    rejected = Detection(10, 20, 30, 40, 0.8, 0, ((10, 20), (30, 20), (30, 40), (10, 40)))
    diagnostics = YoloDetectionDiagnostics(
        raw_yolo_count=1,
        size_rejected_count=1,
        returned_detection_count=0,
        size_rejected=(SizeRejectedDetection(rejected, 0.02),),
    )
    config = ExtractionConfig(
        detector=DetectorConfig(roi=(0, 0, 1, 1), trigger_zone=(0.1, 0.1, 0.9, 0.9))
    )

    display = draw_calibration_overlay(
        frame,
        config,
        [],
        diagnostics,
        fps=12.5,
        size_filter_enabled=True,
        show_size_rejected=True,
    )

    assert not np.any(frame)
    assert np.any(display)
    assert display[20, 10, 2] > 0


def test_screenshot_path_creates_versioned_output_directory(tmp_path):
    directory = tmp_path / "camera_a"
    path = screenshot_path(
        directory,
        datetime(2026, 8, 29, 14, 5, 6, 123456, tzinfo=timezone.utc),
    )
    assert directory.is_dir()
    assert path.name == "calibration_20260829_140506_123456.png"

from pathlib import Path

import numpy as np

from glove_chirality.camera import camera_geometry_status
from glove_chirality.config import ExtractionConfig
from glove_chirality.events import trigger_line_position
from glove_chirality.factory_live import GRIP_CAMERA, GRIP_GEOMETRY, apply_grip_geometry
from glove_chirality.overlay import draw_live_overlay
from glove_chirality.types import Detection


def test_grip_1080p_preset_and_horizontal_trigger():
    assert (GRIP_CAMERA["width"], GRIP_CAMERA["height"], GRIP_CAMERA["fourcc"]) == (1920, 1080, "MJPG")
    config = ExtractionConfig()
    apply_grip_geometry(config)
    axis, _position = trigger_line_position(config, 1920, 1080)
    assert config.event.belt_direction == "bottom_to_top"
    assert axis == "y"
    assert config.detector.roi == GRIP_GEOMETRY["roi"]
    assert config.detector.trigger_zone == GRIP_GEOMETRY["trigger_zone"]
    assert config.detector.yolo_imgsz == 640


def test_geometry_mismatch_is_explicit_and_does_not_claim_resize():
    status = camera_geometry_status(1920, 1080, 640, 480)
    assert status["matches"] is False
    assert "CAMERA GEOMETRY MISMATCH" in status["warning"]
    assert "640x480" in status["warning"]
    assert camera_geometry_status(1920, 1080, 1920, 1080)["matches"] is True
    assert camera_geometry_status(None, None, 640, 480)["matches"] is None


def test_overlay_keeps_full_frame_and_reports_full_frame_area():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    detection = Detection(100, 100, 292, 208, 0.96)
    config = ExtractionConfig()
    apply_grip_geometry(config)
    annotated = draw_live_overlay(frame, config, [detection])
    assert annotated.shape == frame.shape
    assert not np.array_equal(frame, annotated)
    ratio = detection.area / (1920 * 1080)
    assert ratio == (192 * 108) / (1920 * 1080)


def test_factory_css_contains_full_frame_not_cover():
    css = Path("src/glove_chirality/web/static/factory.css").read_text(encoding="utf-8")
    assert "object-fit: contain" in css
    assert "object-fit: cover" not in css
    assert "max-width: 1920px" in css


def test_factory_yaml_uses_grip_geometry_without_absolute_paths():
    config = ExtractionConfig.from_yaml("configs/factory.yaml")
    assert config.detector.roi == (0.12, 0.03, 0.98, 0.995)
    assert config.detector.trigger_zone == (0.39, 0.04, 0.92, 0.93)
    assert config.event.belt_direction == "bottom_to_top"
    assert config.detector.yolo_imgsz == 640
    assert config.event.output_size == 256
    assert config.event.letterbox_fill == 114
    assert ":" not in config.detector.yolo_model

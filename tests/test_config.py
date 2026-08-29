from pathlib import Path

import pytest
import yaml

from glove_chirality.config import DetectorConfig, EventConfig, ExtractionConfig


def test_default_config_loads():
    path = Path(__file__).parents[1] / "configs" / "default.yaml"
    config = ExtractionConfig.from_yaml(path)
    assert config.detector.backend == "belt_foreground"
    assert config.detector.require_full_containment is True
    assert config.detector.adaptive_background is True
    assert config.event.reject_multiple_detections is True
    assert config.event.output_size == 256
    assert config.event.crop_mode == "bbox"
    assert config.event.timing_mode == "frames"
    assert config.event.make_square is True
    assert config.detector.yolo_require_masks is False
    assert config.detector.yolo_min_box_area_ratio == 0.0
    assert config.detector.yolo_max_box_area_ratio == 1.0
    assert config.runtime.detect_every_n_frames == 1


def test_trigger_margin_must_leave_a_nonempty_inner_zone():
    with pytest.raises(ValueError, match="trigger_inner_margin_ratio"):
        DetectorConfig(trigger_inner_margin_ratio=0.5)


def test_output_size_must_be_positive():
    with pytest.raises(ValueError, match="output_size"):
        EventConfig(output_size=0)


def test_segmentation_config_validation_is_explicit():
    with pytest.raises(ValueError, match="Select a custom YOLO detector checkpoint"):
        DetectorConfig(backend="yolo")
    with pytest.raises(ValueError, match="requires yolo_use_masks"):
        DetectorConfig(yolo_use_masks=False, yolo_require_masks=True)
    with pytest.raises(ValueError, match="yolo_confidence"):
        DetectorConfig(yolo_confidence=1.1)
    with pytest.raises(ValueError, match="box area ratios"):
        DetectorConfig(yolo_min_box_area_ratio=0.4, yolo_max_box_area_ratio=0.4)
    with pytest.raises(ValueError, match="box area ratios"):
        DetectorConfig(yolo_min_box_area_ratio=-0.1)
    with pytest.raises(ValueError, match="roi"):
        DetectorConfig(roi=(0.8, 0.1, 0.2, 0.9))
    with pytest.raises(ValueError, match="yolo_class_id"):
        DetectorConfig(yolo_class_id=-1)
    with pytest.raises(ValueError, match="crop_mode"):
        EventConfig(crop_mode="unknown")
    with pytest.raises(ValueError, match="frame thresholds"):
        EventConfig(exit_missing_frames=0)


def test_production_config_requires_single_class_segmentation():
    path = Path(__file__).parents[1] / "configs" / "production.yaml"
    config = ExtractionConfig.from_yaml(path)
    assert config.detector.backend == "yolo"
    assert config.detector.yolo_class_id == 0
    assert config.detector.yolo_require_masks is True
    assert config.detector.yolo_crop_to_roi is True
    assert config.event.timing_mode == "time"


def test_grip_config_uses_measured_size_gate_and_tight_bbox():
    path = Path(__file__).parents[1] / "configs" / "grip_aug27_seed.yaml"
    config = ExtractionConfig.from_yaml(path)
    assert config.detector.yolo_min_box_area_ratio == 0.03
    assert config.detector.yolo_max_box_area_ratio == 0.40
    assert config.event.crop_padding == 0.0
    assert config.event.make_square is False
    assert config.event.crop_mode == "bbox"


def test_current_camera_config_is_explicitly_uncalibrated_and_uses_final_model_name():
    path = Path(__file__).parents[1] / "configs" / "grip_current_camera.yaml"
    config = ExtractionConfig.from_yaml(path)
    assert config.detector.yolo_model.endswith("yolo11n_seg_glove_final_v2.pt")
    assert config.detector.roi == (0.0, 0.0, 1.0, 1.0)
    assert config.detector.yolo_min_box_area_ratio == 0.0
    assert config.detector.yolo_max_box_area_ratio == 1.0


def test_unknown_config_key_fails(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"event": {"typo_setting": 1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="typo_setting"):
        ExtractionConfig.from_yaml(path)

    path.write_text(yaml.safe_dump({"typo_section": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="typo_section"):
        ExtractionConfig.from_yaml(path)


def test_config_yaml_round_trip(tmp_path):
    config = ExtractionConfig()
    config.detector.roi = (0.1, 0.2, 0.8, 0.9)
    config.detector.color_distance_threshold = 31.5
    config.event.crop_padding = 0.2
    path = config.to_yaml(tmp_path / "nested" / "gui.yaml")
    loaded = ExtractionConfig.from_yaml(path)
    assert loaded.detector.roi == (0.1, 0.2, 0.8, 0.9)
    assert loaded.detector.color_distance_threshold == 31.5
    assert loaded.event.crop_padding == 0.2

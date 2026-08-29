import pytest

from glove_chirality.gui import (
    custom_yolo_segmentation_preset,
    tight_detection_crop_preset,
)


def test_custom_yolo_preset_configures_safe_layer_one_defaults(tmp_path):
    model = tmp_path / "best.pt"
    model.write_bytes(b"test checkpoint placeholder")

    settings = custom_yolo_segmentation_preset(model)

    assert settings == {
        "backend": "yolo",
        "yolo_model": str(model.resolve()),
        "yolo_class_id": 0,
        "yolo_use_masks": True,
        "yolo_require_masks": True,
        "yolo_crop_to_roi": True,
    }


def test_tight_detection_crop_preset_uses_selected_box_without_expansion():
    assert tight_detection_crop_preset() == {
        "crop_padding": 0.0,
        "make_square": False,
        "crop_mode": "bbox",
    }


def test_custom_yolo_preset_rejects_missing_or_unsupported_files(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        custom_yolo_segmentation_preset(tmp_path / "missing.pt")

    unsupported = tmp_path / "model.txt"
    unsupported.write_text("not a detector", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported YOLO model format"):
        custom_yolo_segmentation_preset(unsupported)

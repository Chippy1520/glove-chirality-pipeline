from __future__ import annotations

from pathlib import Path

CUSTOM_YOLO_SUFFIXES = {".pt", ".pth", ".onnx", ".engine", ".torchscript"}


def tight_detection_crop_preset() -> dict[str, object]:
    """Use the selected detector box without padding or square expansion."""
    return {
        "crop_padding": 0.0,
        "make_square": False,
        "crop_mode": "bbox",
    }


def custom_yolo_segmentation_preset(model_path: str | Path) -> dict[str, object]:
    """Return conservative Layer-1 settings for a custom glove segmentation model."""
    path = Path(model_path).expanduser()
    if not path.is_file():
        raise ValueError(f"YOLO model not found: {path}")
    if path.suffix.lower() not in CUSTOM_YOLO_SUFFIXES:
        raise ValueError(
            "Unsupported YOLO model format. Choose .pt, .pth, .onnx, .engine, or .torchscript"
        )
    return {
        "backend": "yolo",
        "yolo_model": str(path.resolve()),
        "yolo_class_id": 0,
        "yolo_use_masks": True,
        "yolo_require_masks": True,
        "yolo_crop_to_roi": True,
    }

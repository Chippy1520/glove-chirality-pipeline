from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DetectorConfig:
    backend: str = "belt_foreground"
    roi: tuple[float, float, float, float] = (0.05, 0.05, 0.95, 0.95)
    trigger_zone: tuple[float, float, float, float] = (0.20, 0.15, 0.80, 0.85)
    require_full_containment: bool = True
    trigger_inner_margin_ratio: float = 0.0
    dark_threshold: int = 105
    color_distance_threshold: float = 28.0
    belt_sample_stride: int = 8
    motion_assist: bool = True
    mog_history: int = 250
    mog_var_threshold: float = 16.0
    mog_learning_rate: float = -1.0
    adaptive_background: bool = True
    mog_empty_learning_rate: float = 0.02
    mog_foreground_learning_rate: float = 0.0
    blur_kernel: int = 7
    morph_kernel: int = 11
    min_area_ratio: float = 0.015
    max_area_ratio: float = 0.55
    min_solidity: float = 0.35
    yolo_model: str = ""
    yolo_confidence: float = 0.35
    yolo_class_id: int | None = None
    yolo_device: str = "auto"
    yolo_half: bool = False
    yolo_use_masks: bool = True
    yolo_require_masks: bool = False
    yolo_imgsz: int = 640
    yolo_iou: float = 0.50
    yolo_max_det: int = 5
    yolo_crop_to_roi: bool = False
    yolo_min_box_area_ratio: float = 0.0
    yolo_max_box_area_ratio: float = 1.0

    def __post_init__(self):
        self.validate()

    def validate(self):
        for name in ("roi", "trigger_zone"):
            box = getattr(self, name)
            if (
                len(box) != 4
                or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in box)
                or box[0] >= box[2]
                or box[1] >= box[3]
            ):
                raise ValueError(f"{name} must be ordered finite normalized coordinates")
        if not 0.0 <= self.trigger_inner_margin_ratio < 0.5:
            raise ValueError("trigger_inner_margin_ratio must be in [0.0, 0.5)")
        if not 0.0 <= self.yolo_confidence <= 1.0:
            raise ValueError("yolo_confidence must be in [0.0, 1.0]")
        if self.yolo_class_id is not None and self.yolo_class_id < 0:
            raise ValueError("yolo_class_id must be non-negative or null")
        if self.backend == "yolo" and not self.yolo_model.strip():
            raise ValueError(
                "Select a custom YOLO detector checkpoint before using backend=yolo"
            )
        if self.yolo_require_masks and not self.yolo_use_masks:
            raise ValueError("yolo_require_masks requires yolo_use_masks=true")
        if self.yolo_imgsz <= 0:
            raise ValueError("yolo_imgsz must be positive")
        if not 0.0 < self.yolo_iou <= 1.0:
            raise ValueError("yolo_iou must be in (0.0, 1.0]")
        if self.yolo_max_det <= 0:
            raise ValueError("yolo_max_det must be positive")
        if not (
            0.0
            <= self.yolo_min_box_area_ratio
            < self.yolo_max_box_area_ratio
            <= 1.0
        ):
            raise ValueError("YOLO box area ratios must satisfy 0 <= min < max <= 1")


@dataclass
class EventConfig:
    min_detected_frames: int = 2
    reject_multiple_detections: bool = True
    exit_missing_frames: int = 5
    cooldown_frames: int = 8
    max_track_distance_ratio: float = 0.20
    crop_padding: float = 0.12
    make_square: bool = True
    output_size: int = 256
    crop_mode: str = "bbox"
    save_masks: bool = False
    min_sharpness: float = 0.0
    save_full_frames: bool = False
    timing_mode: str = "frames"
    min_detected_seconds: float = 0.08
    exit_missing_seconds: float = 0.20
    cooldown_seconds: float = 0.30
    association_iou_weight: float = 0.25
    association_mask_iou_weight: float = 0.0
    quality_mask_area_weight: float = 0.0
    quality_mask_stability_weight: float = 0.0
    quality_boundary_clearance_weight: float = 0.0
    quality_edge_penalty_weight: float = 0.0

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.min_detected_frames <= 0 or self.exit_missing_frames <= 0:
            raise ValueError("detection and exit frame thresholds must be positive")
        if self.cooldown_frames < 0:
            raise ValueError("cooldown_frames must be non-negative")
        if self.max_track_distance_ratio < 0 or self.crop_padding < 0:
            raise ValueError("track distance and crop padding must be non-negative")
        if self.output_size <= 0:
            raise ValueError("output_size must be positive")
        if self.crop_mode not in {"bbox", "masked", "masked_fill"}:
            raise ValueError("crop_mode must be bbox, masked, or masked_fill")
        if self.timing_mode not in {"frames", "time"}:
            raise ValueError("timing_mode must be frames or time")
        if self.min_detected_seconds <= 0 or self.exit_missing_seconds <= 0:
            raise ValueError("detection and exit time thresholds must be positive")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")
        weights = (
            self.association_iou_weight,
            self.association_mask_iou_weight,
            self.quality_mask_area_weight,
            self.quality_mask_stability_weight,
            self.quality_boundary_clearance_weight,
            self.quality_edge_penalty_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("association and quality weights must be non-negative")


@dataclass
class DiagnosticsConfig:
    show_masks: bool = True
    mask_alpha: float = 0.30

    def __post_init__(self):
        if not 0.0 <= self.mask_alpha <= 1.0:
            raise ValueError("mask_alpha must be in [0.0, 1.0]")


@dataclass
class RuntimeConfig:
    capture_queue_size: int = 2
    detect_every_n_frames: int = 1
    report_interval_seconds: float = 5.0
    warmup: bool = True

    def __post_init__(self):
        if self.capture_queue_size <= 0:
            raise ValueError("capture_queue_size must be positive")
        if self.detect_every_n_frames <= 0:
            raise ValueError("detect_every_n_frames must be positive")
        if self.report_interval_seconds <= 0:
            raise ValueError("report_interval_seconds must be positive")


@dataclass
class ExtractionConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    event: EventConfig = field(default_factory=EventConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_yaml(cls, path: str | Path | None) -> ExtractionConfig:
        if path is None:
            return cls()
        with Path(path).open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}
        unknown = sorted(set(raw) - {"detector", "event", "diagnostics", "runtime"})
        if unknown:
            raise ValueError(f"Unknown top-level settings: {', '.join(unknown)}")
        return cls(
            detector=_load(DetectorConfig, raw.get("detector", {})),
            event=_load(EventConfig, raw.get("event", {})),
            diagnostics=_load(DiagnosticsConfig, raw.get("diagnostics", {})),
            runtime=_load(RuntimeConfig, raw.get("runtime", {})),
        )

    def to_yaml(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(_lists(asdict(self)), stream, sort_keys=False)
        return path


def _load(cls: type, values: dict[str, Any]):
    values = dict(values)
    if cls is DetectorConfig:
        for key in ("roi", "trigger_zone"):
            if key in values:
                values[key] = tuple(values[key])
    allowed = set(cls.__dataclass_fields__)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {', '.join(unknown)}")
    return cls(**values)


def _lists(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _lists(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_lists(item) for item in value]
    return value

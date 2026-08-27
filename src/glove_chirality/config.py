from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DetectorConfig:
    backend: str = "belt_foreground"
    roi: tuple[float, float, float, float] = (0.05, 0.05, 0.95, 0.95)
    trigger_zone: tuple[float, float, float, float] = (0.20, 0.15, 0.80, 0.85)
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
    yolo_model: str = "yolo11n.pt"
    yolo_confidence: float = 0.35
    yolo_class_id: int | None = None
    yolo_device: str = "auto"
    yolo_half: bool = False


@dataclass
class EventConfig:
    min_detected_frames: int = 2
    exit_missing_frames: int = 5
    cooldown_frames: int = 8
    max_track_distance_ratio: float = 0.20
    crop_padding: float = 0.12
    make_square: bool = True
    min_sharpness: float = 0.0
    save_full_frames: bool = False


@dataclass
class ExtractionConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    event: EventConfig = field(default_factory=EventConfig)

    @classmethod
    def from_yaml(cls, path: str | Path | None) -> ExtractionConfig:
        if path is None:
            return cls()
        with Path(path).open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}
        return cls(
            detector=_load(DetectorConfig, raw.get("detector", {})),
            event=_load(EventConfig, raw.get("event", {})),
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

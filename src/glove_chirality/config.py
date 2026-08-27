from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DetectorConfig:
    backend: str = "dark_contour"
    roi: tuple[float, float, float, float] = (0.05, 0.05, 0.95, 0.95)
    trigger_zone: tuple[float, float, float, float] = (0.20, 0.15, 0.80, 0.85)
    dark_threshold: int = 105
    blur_kernel: int = 7
    morph_kernel: int = 11
    min_area_ratio: float = 0.015
    max_area_ratio: float = 0.55
    min_solidity: float = 0.35
    yolo_model: str = "yolo11n.pt"
    yolo_confidence: float = 0.35
    yolo_class_id: int | None = None


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


def _load(cls: type, values: dict[str, Any]):
    allowed = set(cls.__dataclass_fields__)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {', '.join(unknown)}")
    return cls(**values)

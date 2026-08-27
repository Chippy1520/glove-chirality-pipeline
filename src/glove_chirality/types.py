from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int | None = None

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(frozen=True)
class ExtractedEvent:
    event_id: str
    image_path: Path
    source_video: str
    label: str
    frame_index: int
    timestamp_s: float
    detection: Detection
    quality_score: float
    detector: str

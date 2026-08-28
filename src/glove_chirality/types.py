from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

Polygon = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int | None = None
    polygon: Polygon | None = None

    def __post_init__(self) -> None:
        if self.polygon is None:
            return
        polygon = tuple((float(x), float(y)) for x, y in self.polygon)
        if len(polygon) < 3 or any(
            not math.isfinite(coordinate)
            for point in polygon
            for coordinate in point
        ):
            raise ValueError("polygon must contain at least three finite points")
        object.__setattr__(self, "polygon", polygon)

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

    @property
    def mask_area(self) -> float | None:
        if not self.polygon or len(self.polygon) < 3:
            return None
        points = self.polygon
        return abs(
            sum(
                x1 * y2 - x2 * y1
                for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
            )
        ) / 2.0

    @property
    def mask_bbox_fill_ratio(self) -> float | None:
        area = self.mask_area
        return None if area is None else area / max(1, self.area)


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
    candidate_count: int = 1
    mask_path: Path | None = None


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    source_video: str
    frame_index: int
    timestamp_s: float
    status: str
    reject_reason: str
    candidate_count: int
    detection: Detection | None = None
    detector: str = ""

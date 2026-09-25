"""Count a glove when its center crosses the line.

A sighting lives only for the reentry window. It is not an identity that
follows the glove down the belt. A second count is suppressed only when it
is the trailing boxes of a crossing already counted: same lane, within one
glove length, inside that window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from glove_chirality.config import ExtractionConfig
from glove_chirality.events import (
    PassageOutcome,
    _direction_increasing,
    create_event_crop,
    directional_crossing_alpha,
    trigger_line_position,
)
from glove_chirality.types import Detection


@dataclass
class _Shot:
    detection: Detection
    pixels: np.ndarray
    quality: float
    frame_index: int


@dataclass
class _Sighting:
    sighting_id: int
    detection: Detection
    first_seen_s: float
    last_seen_s: float
    hits: int = 1
    armed: bool = False
    emitted: bool = False
    latched: bool = False
    crossing_s: float | None = None
    crossing_px: tuple[float, float] | None = None
    shots: list[_Shot] = field(default_factory=list)


class LineCounter:
    """One crossing, one crop. No track survives the duplicate window."""

    def __init__(self, config: ExtractionConfig, source_video: str, label: str, next_event_id):
        self.config = config
        self.source_video = source_video
        self.label = label
        self._next_event_id = next_event_id
        self._sightings: list[_Sighting] = []
        self._next_id = 1
        self._width = 1
        self._height = 1
        self._lost: list[PassageOutcome] = []
        self.debug_events: list[dict[str, object]] = []

    def close(self) -> list[PassageOutcome]:
        lost = [self._terminal(item, self._death_reason(item)) for item in self._sightings if item.hits >= 2 and not item.emitted]
        self._sightings.clear()
        return lost

    def update(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        frame_index: int,
        timestamp_s: float,
        tracking_only: list[Detection] | None = None,
    ) -> list[PassageOutcome]:
        del tracking_only
        self._height, self._width = frame.shape[:2]
        self._lost = []
        boxes = [
            item
            for item in detections
            if _inside(item, self.config, self._width, self._height)
            and item.confidence >= self.config.event.track_high_conf
        ]
        used: set[int] = set()
        for detection in boxes:
            match = self._nearest(detection, timestamp_s, used)
            if match is None:
                self._birth(detection, frame, frame_index, timestamp_s)
                continue
            used.add(match.sighting_id)
            self._advance(match, detection, frame, frame_index, timestamp_s, boxes)
        self._expire(timestamp_s)
        return self._emit(timestamp_s) + self._lost

    def _birth(self, detection: Detection, frame: np.ndarray, frame_index: int, timestamp_s: float) -> None:
        sighting = _Sighting(
            sighting_id=self._next_id,
            detection=detection,
            first_seen_s=timestamp_s,
            last_seen_s=timestamp_s,
        )
        self._next_id += 1
        axis, line = trigger_line_position(self.config, self._width, self._height)
        along = detection.center[1] if axis == "y" else detection.center[0]
        increasing = _direction_increasing(self.config.event.belt_direction)
        sighting.armed = along <= line if increasing else along >= line
        self.debug_events.append({
            "kind": "birth",
            "armed": sighting.armed,
            "timestamp_s": timestamp_s,
            "center": detection.center,
        })
        self._remember(sighting, detection, frame, frame_index, boxes=[])
        self._sightings.append(sighting)

    def _advance(self, sighting, detection, frame, frame_index, timestamp_s, boxes) -> None:
        previous = sighting.detection.center
        self._latch(sighting, previous, detection.center, sighting.last_seen_s, timestamp_s)
        sighting.detection = detection
        sighting.last_seen_s = timestamp_s
        sighting.hits += 1
        self._remember(sighting, detection, frame, frame_index, boxes)

    def _nearest(self, detection: Detection, timestamp_s: float, used: set[int]) -> _Sighting | None:
        axis = "y" if self.config.event.belt_direction in {"bottom_to_top", "top_to_bottom"} else "x"
        along = 1 if axis == "y" else 0
        cross = 1 - along
        best: _Sighting | None = None
        best_cost = 1e9
        for sighting in self._sightings:
            if sighting.sighting_id in used:
                continue
            if timestamp_s - sighting.last_seen_s > self.config.event.reentry_time_s:
                continue
            glove = max(sighting.detection.height, sighting.detection.width, detection.height, detection.width, 1)
            along_error = abs(sighting.detection.center[along] - detection.center[along])
            cross_error = abs(sighting.detection.center[cross] - detection.center[cross])
            if along_error > 1.5 * glove or cross_error > 0.75 * glove:
                continue
            cost = along_error / glove + 0.5 * cross_error / glove
            if cost < best_cost:
                best = sighting
                best_cost = cost
        return best

    def _latch(self, sighting, previous, current, previous_s, timestamp_s) -> None:
        if sighting.emitted or sighting.latched or not sighting.armed:
            return
        axis, line = trigger_line_position(self.config, self._width, self._height)
        increasing = _direction_increasing(self.config.event.belt_direction)
        prev = previous[1] if axis == "y" else previous[0]
        now = current[1] if axis == "y" else current[0]
        alpha = directional_crossing_alpha(prev, now, line, increasing)
        if alpha is None:
            return
        if axis == "y":
            cross = (previous[0] + alpha * (current[0] - previous[0]), line)
        else:
            cross = (line, previous[1] + alpha * (current[1] - previous[1]))
        sighting.latched = True
        sighting.crossing_s = previous_s + alpha * (timestamp_s - previous_s)
        sighting.crossing_px = (float(cross[0]), float(cross[1]))

    def _remember(self, sighting, detection, frame, frame_index, boxes) -> None:
        if detection.x1 <= 2 or detection.y1 <= 2 or detection.x2 >= self._width - 2 or detection.y2 >= self._height - 2:
            return
        if any(other is not detection and _iou(detection, other) > 0.35 for other in boxes):
            return
        pixels = frame[detection.y1:detection.y2, detection.x1:detection.x2].copy()
        if not pixels.size:
            return
        gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
        sharp = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 500.0)
        sighting.shots.append(_Shot(detection, pixels, 0.7 * detection.confidence + 0.3 * sharp, frame_index))
        sighting.shots = sorted(sighting.shots, key=lambda item: item.quality, reverse=True)[:5]

    def _expire(self, timestamp_s: float) -> None:
        alive = []
        limit = self.config.event.reentry_time_s
        for sighting in self._sightings:
            if timestamp_s - sighting.last_seen_s <= limit:
                alive.append(sighting)
                continue
            if sighting.hits >= 2 and not sighting.emitted:
                self._lost.append(self._terminal(sighting, self._death_reason(sighting)))
        self._sightings = alive

    def _emit(self, timestamp_s: float) -> list[PassageOutcome]:
        outcomes = []
        for sighting in self._sightings:
            if not sighting.latched or sighting.emitted or sighting.hits < 2 or not sighting.shots:
                continue
            shot = max(sighting.shots, key=lambda item: item.quality)
            local = Detection(0, 0, shot.pixels.shape[1], shot.pixels.shape[0], shot.detection.confidence)
            outcomes.append(
                PassageOutcome(
                    self._next_event_id(),
                    self.source_video,
                    self.label,
                    "accepted",
                    "",
                    shot.frame_index,
                    timestamp_s,
                    sighting.hits,
                    shot.detection,
                    shot.quality,
                    create_event_crop(shot.pixels, local, self.config),
                    None,
                    sighting.first_seen_s,
                    sighting.crossing_s,
                    sighting.crossing_px,
                    None,
                    shot.detection.center,
                    None,
                )
            )
            sighting.emitted = True
        return outcomes

    def _death_reason(self, sighting: _Sighting) -> str:
        if not sighting.armed:
            return "born_past_line"
        if sighting.latched and not sighting.shots:
            return "crossed_without_crop"
        return "lost_before_trigger"

    def _terminal(self, sighting: _Sighting, reason: str) -> PassageOutcome:
        self.debug_events.append({"kind": "death", "reason": reason, "armed": sighting.armed, "timestamp_s": sighting.last_seen_s})
        return PassageOutcome(
            self._next_event_id(),
            self.source_video,
            self.label,
            "rejected",
            reason,
            0,
            sighting.last_seen_s,
            sighting.hits,
            sighting.detection,
            0.0,
            None,
            None,
            sighting.first_seen_s,
            sighting.crossing_s,
            sighting.crossing_px,
            None,
            sighting.detection.center,
            None,
        )


def _inside(detection: Detection, config: ExtractionConfig, width: int, height: int) -> bool:
    x1, y1, x2, y2 = config.detector.roi
    cx, cy = detection.center
    return x1 * width <= cx <= x2 * width and y1 * height <= cy <= y2 * height


def _iou(left: Detection, right: Detection) -> float:
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    union = left.area + right.area - intersection
    return intersection / union if union else 0.0

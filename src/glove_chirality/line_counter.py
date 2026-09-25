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
    remove_stray_glove_fragments,
    suppress_other_gloves,
    trigger_line_position,
)
from glove_chirality.types import Detection

_HOLD_S = 1.0


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
    velocity: tuple[float, float] = (0.0, 0.0)


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
        self.min_mask_iou = 0.0

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
        pairs = self._assign(boxes, timestamp_s)
        matched_ids = {id(detection) for _, detection in pairs}
        for sighting, detection in pairs:
            self._advance(sighting, detection, frame, frame_index, timestamp_s, boxes)
        for detection in boxes:
            if id(detection) in matched_ids:
                continue
            self._birth(detection, frame, frame_index, timestamp_s)
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
        dt = max(1e-3, timestamp_s - sighting.last_seen_s)
        sighting.velocity = (
            (detection.center[0] - previous[0]) / dt,
            (detection.center[1] - previous[1]) / dt,
        )
        self._latch(sighting, previous, detection.center, sighting.last_seen_s, timestamp_s)
        sighting.detection = detection
        sighting.last_seen_s = timestamp_s
        sighting.hits += 1
        self._remember(sighting, detection, frame, frame_index, boxes)

    def _assign(self, detections: list[Detection], timestamp_s: float) -> list[tuple[_Sighting, Detection]]:
        scored: list[tuple[float, _Sighting, Detection]] = []
        for sighting in self._sightings:
            if timestamp_s - sighting.last_seen_s > _HOLD_S:
                continue
            predicted = self._shifted(sighting, timestamp_s)
            for detection in detections:
                overlap = _mask_overlap(predicted, detection)
                if overlap <= self.min_mask_iou:
                    continue
                scored.append((overlap, sighting, detection))
        scored.sort(key=lambda item: item[0], reverse=True)
        used_sightings: set[int] = set()
        used_detections: set[int] = set()
        pairs = []
        for _overlap, sighting, detection in scored:
            if sighting.sighting_id in used_sightings or id(detection) in used_detections:
                continue
            used_sightings.add(sighting.sighting_id)
            used_detections.add(id(detection))
            pairs.append((sighting, detection))
        return pairs

    def _shifted(self, sighting: _Sighting, timestamp_s: float) -> Detection:
        dt = max(0.0, timestamp_s - sighting.last_seen_s)
        dx = sighting.velocity[0] * dt
        dy = sighting.velocity[1] * dt
        detection = sighting.detection
        polygon = None if detection.polygon is None else tuple((x + dx, y + dy) for x, y in detection.polygon)
        return Detection(
            round(detection.x1 + dx),
            round(detection.y1 + dy),
            round(detection.x2 + dx),
            round(detection.y2 + dy),
            detection.confidence,
            detection.class_id,
            polygon,
        )

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
        if not self._near_line(detection) or self._reaches_led(detection):
            return
        pixels = frame[detection.y1:detection.y2, detection.x1:detection.x2].copy()
        if not pixels.size:
            return
        pixels = suppress_other_gloves(
            pixels,
            (detection.x1, detection.y1),
            detection,
            boxes,
            self.config.event.letterbox_fill,
        )
        pixels = remove_stray_glove_fragments(pixels, self.config.event.letterbox_fill)
        gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
        sharp = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 500.0)
        sighting.shots.append(_Shot(detection, pixels, 0.7 * detection.confidence + 0.3 * sharp, frame_index))
        sighting.shots = sorted(sighting.shots, key=lambda item: item.quality, reverse=True)[:5]

    def _expire(self, timestamp_s: float) -> None:
        alive = []
        limit = _HOLD_S
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
            near = [shot for shot in sighting.shots if self._near_line(shot.detection) and not self._reaches_led(shot.detection)]
            if not near:
                continue
            shot = min(near, key=lambda item: self._line_distance(item.detection))
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

    def _line_distance(self, detection: Detection) -> float:
        axis, line = trigger_line_position(self.config, self._width, self._height)
        along = detection.center[1] if axis == "y" else detection.center[0]
        return abs(along - line)

    def _near_line(self, detection: Detection) -> bool:
        span = self._height if self._belt_axis() == "y" else self._width
        return self._line_distance(detection) <= 0.12 * span

    def _reaches_led(self, detection: Detection) -> bool:
        return detection.y2 > self._height * 0.94 or detection.x1 < self._width * 0.05

    def _belt_axis(self) -> str:
        return "y" if self.config.event.belt_direction in {"bottom_to_top", "top_to_bottom"} else "x"

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


def _mask_overlap(left: Detection, right: Detection) -> float:
    """Positive only when the glove shapes overlap, not merely their boxes."""
    if left.polygon and right.polygon:
        return _polygon_iou(left.polygon, right.polygon)
    return _iou(left, right)


def _polygon_iou(left: tuple, right: tuple) -> float:
    xs = [point[0] for point in left + right]
    ys = [point[1] for point in left + right]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return 0.0
    scale = 1.0 if width * height <= 80_000 else (80_000 / (width * height)) ** 0.5

    def raster(polygon) -> np.ndarray:
        points = np.array([[(x - x1) * scale, (y - y1) * scale] for x, y in polygon], dtype=np.int32)
        mask = np.zeros((max(1, int(height * scale)), max(1, int(width * scale))), dtype=np.uint8)
        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 1)
        return mask

    left_mask = raster(left)
    right_mask = raster(right)
    union = int(np.logical_or(left_mask, right_mask).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(left_mask, right_mask).sum()) / union


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

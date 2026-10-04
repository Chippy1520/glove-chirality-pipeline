"""Count gloves by predicted-box overlap.

This is a separate system from passage_v2. A detection continues a track only
when it overlaps that track's predicted box. Two separated gloves in one frame
are two tracks, even in the same lane. There is no lane-ownership rule.
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
    timestamp_s: float
    quality: float
    frame_index: int


@dataclass
class _Glove:
    track_id: int
    detection: Detection
    velocity: tuple[float, float] = (0.0, 0.0)
    hits: int = 1
    first_seen_s: float = 0.0
    last_seen_s: float = 0.0
    missing_since_s: float | None = None
    armed: bool = False
    latched: bool = False
    emitted: bool = False
    crossing_s: float | None = None
    crossing_px: tuple[float, float] | None = None
    shots: list[_Shot] = field(default_factory=list)


class GloveCounter:
    """One physical glove, one ID, one crop. Match by overlap, not by lane."""

    def __init__(self, config: ExtractionConfig, source_video: str, label: str, next_event_id):
        self.config = config
        self.source_video = source_video
        self.label = label
        self._next_event_id = next_event_id
        self._gloves: list[_Glove] = []
        self._next_id = 1
        self._width = 1
        self._height = 1
        self._lost: list[PassageOutcome] = []

    def close(self) -> list[PassageOutcome]:
        lost = [
            self._terminal(glove, "lost_before_trigger")
            for glove in self._gloves
            if glove.hits >= 2 and not glove.emitted
        ]
        self._gloves.clear()
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
        boxes = [item for item in detections if _inside(item, self.config, self._width, self._height)]
        high = [item for item in boxes if item.confidence >= self.config.event.track_high_conf]
        low = [item for item in boxes if self.config.event.track_low_conf <= item.confidence < self.config.event.track_high_conf]
        matched = _match(self._gloves, high, timestamp_s)
        used_gloves = {index for index, _det in matched}
        used_high = {id(det) for _index, det in matched}
        if low:
            rest = [glove for index, glove in enumerate(self._gloves) if index not in used_gloves]
            extra = _match(rest, low, timestamp_s)
            for local_index, detection in extra:
                glove = rest[local_index]
                matched.append((self._gloves.index(glove), detection))
                used_gloves.add(self._gloves.index(glove))
        for index, detection in matched:
            self._advance(self._gloves[index], detection, frame, frame_index, timestamp_s, high)
        for detection in high:
            if id(detection) in used_high or detection.confidence < self.config.event.new_track_conf:
                continue
            self._birth(detection, frame, frame_index, timestamp_s)
        self._drop_missing(timestamp_s)
        return self._emit(timestamp_s) + self._lost

    def _birth(self, detection: Detection, frame: np.ndarray, frame_index: int, timestamp_s: float) -> None:
        glove = _Glove(
            track_id=self._next_id,
            detection=detection,
            first_seen_s=timestamp_s,
            last_seen_s=timestamp_s,
        )
        self._next_id += 1
        axis, line = trigger_line_position(self.config, self._width, self._height)
        along = detection.center[1] if axis == "y" else detection.center[0]
        increasing = _direction_increasing(self.config.event.belt_direction)
        glove.armed = along <= line if increasing else along >= line
        self._remember(glove, detection, frame, frame_index, timestamp_s, [])
        self._gloves.append(glove)

    def _advance(
        self,
        glove: _Glove,
        detection: Detection,
        frame: np.ndarray,
        frame_index: int,
        timestamp_s: float,
        siblings: list[Detection],
    ) -> None:
        previous = glove.detection.center
        dt = max(1e-3, timestamp_s - glove.last_seen_s)
        glove.velocity = (
            0.6 * glove.velocity[0] + 0.4 * ((detection.center[0] - previous[0]) / dt),
            0.6 * glove.velocity[1] + 0.4 * ((detection.center[1] - previous[1]) / dt),
        )
        self._latch(glove, previous, detection.center, glove.last_seen_s, timestamp_s)
        glove.detection = detection
        glove.last_seen_s = timestamp_s
        glove.missing_since_s = None
        glove.hits += 1
        if detection.confidence >= self.config.event.track_high_conf:
            self._remember(glove, detection, frame, frame_index, timestamp_s, siblings)

    def _latch(self, glove: _Glove, previous, current, previous_s: float, timestamp_s: float) -> None:
        if glove.emitted or glove.latched or not glove.armed or glove.hits < 1:
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
        glove.latched = True
        glove.crossing_s = previous_s + alpha * (timestamp_s - previous_s)
        glove.crossing_px = (float(cross[0]), float(cross[1]))

    def _remember(self, glove, detection, frame, frame_index, timestamp_s, siblings) -> None:
        if detection.x1 <= 2 or detection.y1 <= 2:
            return
        if detection.x2 >= self._width - 2 or detection.y2 >= self._height - 2:
            return
        if any(other is not detection and _iou(detection, other) > 0.35 for other in siblings):
            return
        pixels = frame[detection.y1:detection.y2, detection.x1:detection.x2].copy()
        if not pixels.size:
            return
        gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
        sharp = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 500.0)
        glove.shots.append(_Shot(detection, pixels, timestamp_s, 0.7 * detection.confidence + 0.3 * sharp, frame_index))
        glove.shots = sorted(glove.shots, key=lambda item: item.quality, reverse=True)[:5]

    def _drop_missing(self, timestamp_s: float) -> None:
        alive = []
        for glove in self._gloves:
            if glove.last_seen_s == timestamp_s:
                alive.append(glove)
                continue
            if glove.missing_since_s is None:
                glove.missing_since_s = timestamp_s
            if timestamp_s - glove.missing_since_s <= self.config.event.reentry_time_s:
                alive.append(glove)
                continue
            if glove.hits >= 2 and not glove.emitted:
                self._lost.append(self._terminal(glove, "lost_before_trigger"))
                glove.emitted = True
        self._gloves = alive

    def _emit(self, timestamp_s: float) -> list[PassageOutcome]:
        outcomes = []
        for glove in self._gloves:
            if not glove.latched or glove.emitted or glove.hits < 2 or not glove.shots:
                continue
            shot = max(glove.shots, key=lambda item: item.quality)
            local = Detection(0, 0, shot.pixels.shape[1], shot.pixels.shape[0], shot.detection.confidence)
            outcomes.append(PassageOutcome(
                self._next_event_id(),
                self.source_video,
                self.label,
                "accepted",
                "",
                shot.frame_index,
                timestamp_s,
                1,
                shot.detection,
                shot.quality,
                create_event_crop(shot.pixels, local, self.config),
                None,
                glove.first_seen_s,
                glove.crossing_s,
                glove.crossing_px,
                None,
                shot.detection.center,
                None,
            ))
            glove.emitted = True
        return outcomes

    def _terminal(self, glove: _Glove, reason: str) -> PassageOutcome:
        return PassageOutcome(
            self._next_event_id(),
            self.source_video,
            self.label,
            "rejected",
            reason,
            0,
            glove.last_seen_s,
            glove.hits,
            glove.detection,
            0.0,
            None,
            None,
            glove.first_seen_s,
            glove.crossing_s,
            glove.crossing_px,
            None,
            glove.detection.center,
            None,
        )


def _match(gloves: list[_Glove], detections: list[Detection], timestamp_s: float) -> list[tuple[int, Detection]]:
    scored = []
    for index, glove in enumerate(gloves):
        if glove.emitted:
            continue
        predicted = _predicted(glove, timestamp_s)
        for detection in detections:
            score = _iou(_expanded(predicted, 0.5), _expanded(detection, 0.5))
            if score >= 0.05:
                scored.append((score, index, detection))
    scored.sort(key=lambda item: item[0], reverse=True)
    used_gloves: set[int] = set()
    used_detections: set[int] = set()
    pairs = []
    for _score, index, detection in scored:
        if index in used_gloves or id(detection) in used_detections:
            continue
        used_gloves.add(index)
        used_detections.add(id(detection))
        pairs.append((index, detection))
    return pairs


def _predicted(glove: _Glove, timestamp_s: float) -> Detection:
    dt = max(0.0, timestamp_s - glove.last_seen_s)
    shift_x = glove.velocity[0] * dt
    shift_y = glove.velocity[1] * dt
    box = glove.detection
    return Detection(
        round(box.x1 + shift_x),
        round(box.y1 + shift_y),
        round(box.x2 + shift_x),
        round(box.y2 + shift_y),
        box.confidence,
    )


def _expanded(detection: Detection, fraction: float) -> Detection:
    pad_x = detection.width * fraction
    pad_y = detection.height * fraction
    return Detection(
        round(detection.x1 - pad_x),
        round(detection.y1 - pad_y),
        round(detection.x2 + pad_x),
        round(detection.y2 + pad_y),
        detection.confidence,
    )


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


def _inside(detection: Detection, config: ExtractionConfig, width: int, height: int) -> bool:
    x1, y1, x2, y2 = config.detector.roi
    cx, cy = detection.center
    return x1 * width <= cx <= x2 * width and y1 * height <= cy <= y2 * height

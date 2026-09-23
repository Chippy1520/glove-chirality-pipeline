"""Conveyor passage tracker.

One YOLO detection list in, one crop per physical glove out. A weak box may
keep an existing identity. It cannot create one, and it cannot become the
Layer 2 crop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from glove_chirality.config import ExtractionConfig
from glove_chirality.events import (
    PassageOutcome,
    _box_iou,
    _direction_increasing,
    _polygon_iou,
    create_event_crop,
    directional_crossing_alpha,
    trigger_line_position,
)
from glove_chirality.types import Detection


@dataclass
class _Candidate:
    detection: Detection
    pixels: np.ndarray
    timestamp_s: float
    quality: float
    frame_index: int


@dataclass
class _Passage:
    passage_id: int
    detection: Detection
    observed: tuple[float, float, float]
    velocity: tuple[float, float] = (0.0, 0.0)
    predicted: tuple[float, float] = (0.0, 0.0)
    first_seen_s: float = 0.0
    last_seen_s: float = 0.0
    strong_hits: int = 0
    weak_hits: int = 0
    missing_since_s: float | None = None
    armed: bool = False
    crossing_latched: bool = False
    event_emitted: bool = False
    crossing_s: float | None = None
    crossing_px: tuple[float, float] | None = None
    candidates: list[_Candidate] = field(default_factory=list)


class PassageTracker:
    """Track ROI gloves, latch one directional crossing, emit one bbox crop."""

    def __init__(self, config: ExtractionConfig, source_video: str, label: str, next_event_id):
        self.config = config
        self.source_video = source_video
        self.label = label
        self._next_event_id = next_event_id
        self._passages: list[_Passage] = []
        self._next_passage_id = 1
        self._ledger: list[_Passage] = []

    def close(self) -> None:
        self._passages.clear()

    def update(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        frame_index: int,
        timestamp_s: float,
    ) -> list[PassageOutcome]:
        height, width = frame.shape[:2]
        observed = [item for item in detections if _center_in_roi(item, self.config, width, height)]
        strong, weak = self._split(observed)
        self._predict(timestamp_s)
        pairs, unmatched, unmatched_strong = _associate(
            self._passages, strong, width, height, self.config, relaxed=False
        )
        for track, detection in pairs:
            self._update_track(track, detection, timestamp_s, frame, frame_index, width, height, weak=False)
        if self.config.event.low_conf_recovery and weak and unmatched:
            rescued, unmatched, _ignored = _associate(
                unmatched, weak, width, height, self.config, relaxed=True
            )
            for track, detection in rescued:
                self._update_track(track, detection, timestamp_s, frame, frame_index, width, height, weak=True)
        for detection in unmatched_strong:
            if detection.confidence < self.config.event.new_track_conf:
                continue
            self._birth(detection, timestamp_s, frame, frame_index)
        self._retire(timestamp_s)
        return self._emit_ready(timestamp_s, width, height)

    def _split(self, detections: list[Detection]) -> tuple[list[Detection], list[Detection]]:
        event = self.config.event
        strong = [item for item in detections if item.confidence >= event.track_high_conf]
        weak = [
            item
            for item in detections
            if event.low_conf_recovery and event.track_low_conf <= item.confidence < event.track_high_conf
        ]
        return strong, weak

    def _predict(self, timestamp_s: float) -> None:
        for track in self._passages:
            dt = max(0.0, timestamp_s - track.last_seen_s)
            cx, cy, _seen = track.observed
            track.predicted = (cx + track.velocity[0] * dt, cy + track.velocity[1] * dt)

    def _update_track(
        self,
        track: _Passage,
        detection: Detection,
        timestamp_s: float,
        frame: np.ndarray,
        frame_index: int,
        width: int,
        height: int,
        *,
        weak: bool,
    ) -> None:
        prev_x, prev_y, prev_t = track.observed
        cx, cy = (float(detection.center[0]), float(detection.center[1]))
        dt = max(1e-3, timestamp_s - track.last_seen_s)
        track.velocity = (
            0.7 * track.velocity[0] + 0.3 * ((cx - prev_x) / dt),
            0.7 * track.velocity[1] + 0.3 * ((cy - prev_y) / dt),
        )
        track.detection = detection
        track.observed = (cx, cy, timestamp_s)
        track.predicted = (cx, cy)
        track.last_seen_s = timestamp_s
        track.missing_since_s = None
        if weak:
            track.weak_hits += 1
            return
        track.strong_hits += 1
        self._advance_trigger(track, prev_x, prev_y, prev_t, cx, cy, timestamp_s, width, height)
        pixels = frame[detection.y1:detection.y2, detection.x1:detection.x2].copy()
        if pixels.size:
            track.candidates.append(_Candidate(
                detection,
                pixels,
                timestamp_s,
                _quality(pixels, detection.confidence),
                frame_index,
            ))
            track.candidates = sorted(track.candidates, key=lambda item: item.quality, reverse=True)[:5]

    def _advance_trigger(
        self,
        track: _Passage,
        prev_x: float,
        prev_y: float,
        prev_t: float,
        cx: float,
        cy: float,
        timestamp_s: float,
        width: int,
        height: int,
    ) -> None:
        if track.event_emitted or track.crossing_latched:
            return
        axis, line = trigger_line_position(self.config, width, height)
        hysteresis = _hysteresis(self.config, width, height, axis)
        increasing = _direction_increasing(self.config.event.belt_direction)
        current = cy if axis == "y" else cx
        if not track.armed:
            if _upstream(current, line, hysteresis, increasing):
                track.armed = True
            return
        if not _downstream(current, line, hysteresis, increasing):
            return
        previous = prev_y if axis == "y" else prev_x
        alpha = directional_crossing_alpha(previous, current, line, increasing)
        if alpha is None:
            return
        if axis == "y":
            cross_x = prev_x + alpha * (cx - prev_x)
            cross_y = line
        else:
            cross_x = line
            cross_y = prev_y + alpha * (cy - prev_y)
        track.crossing_latched = True
        track.crossing_s = prev_t + alpha * (timestamp_s - prev_t)
        track.crossing_px = (float(cross_x), float(cross_y))

    def _birth(self, detection: Detection, timestamp_s: float, frame: np.ndarray, frame_index: int) -> None:
        cx, cy = (float(detection.center[0]), float(detection.center[1]))
        track = _Passage(
            passage_id=self._next_passage_id,
            detection=detection,
            observed=(cx, cy, timestamp_s),
            predicted=(cx, cy),
            first_seen_s=timestamp_s,
            last_seen_s=timestamp_s,
            strong_hits=1,
        )
        self._next_passage_id += 1
        pixels = frame[detection.y1:detection.y2, detection.x1:detection.x2].copy()
        if pixels.size:
            track.candidates.append(_Candidate(
                detection, pixels, timestamp_s, _quality(pixels, detection.confidence), frame_index
            ))
        axis, line = trigger_line_position(self.config, frame.shape[1], frame.shape[0])
        hysteresis = _hysteresis(self.config, frame.shape[1], frame.shape[0], axis)
        increasing = _direction_increasing(self.config.event.belt_direction)
        current = cy if axis == "y" else cx
        track.armed = _upstream(current, line, hysteresis, increasing)
        self._passages.append(track)

    def _near_existing(self, detection: Detection, width: int, height: int) -> bool:
        return any(
            _pair_cost(track, detection, width, height, self.config, relaxed=True) < float("inf")
            for track in self._passages
        )

    def _retire(self, timestamp_s: float) -> None:
        alive = []
        for track in self._passages:
            if track.last_seen_s == timestamp_s:
                alive.append(track)
                continue
            if track.missing_since_s is None:
                track.missing_since_s = timestamp_s
            if timestamp_s - track.missing_since_s <= self.config.event.reentry_time_s:
                alive.append(track)
            elif track.event_emitted:
                self._ledger.append(track)
        self._passages = alive
        cutoff = timestamp_s - 2.0
        self._ledger = [item for item in self._ledger if (item.crossing_s or item.last_seen_s) >= cutoff]

    def _emit_ready(self, timestamp_s: float, width: int, height: int) -> list[PassageOutcome]:
        outcomes = []
        for track in self._passages:
            if not track.crossing_latched or track.event_emitted or track.strong_hits < 2:
                continue
            if self._duplicate(track, width, height):
                track.event_emitted = True
                continue
            selected = _select_candidate(track)
            if selected is None or selected.pixels.size == 0:
                continue
            local = Detection(
                0,
                0,
                selected.pixels.shape[1],
                selected.pixels.shape[0],
                selected.detection.confidence,
            )
            cx, cy = selected.detection.center
            outcomes.append(PassageOutcome(
                self._next_event_id(),
                self.source_video,
                self.label,
                "accepted",
                "",
                selected.frame_index,
                timestamp_s,
                1,
                selected.detection,
                selected.quality,
                create_event_crop(selected.pixels, local, self.config),
                None,
                track.first_seen_s,
                track.crossing_s,
                track.crossing_px,
                None if track.crossing_px is None else (
                    track.crossing_px[0] / max(width, 1),
                    track.crossing_px[1] / max(height, 1),
                ),
                (float(cx), float(cy)),
                (float(cx) / max(width, 1), float(cy) / max(height, 1)),
            ))
            track.event_emitted = True
            self._ledger.append(track)
        return outcomes

    def _duplicate(self, track: _Passage, width: int, height: int) -> bool:
        if track.crossing_px is None or track.crossing_s is None:
            return False
        axis, _line = trigger_line_position(self.config, width, height)
        current = track.crossing_px[0] if axis == "y" else track.crossing_px[1]
        limit = 0.75 * max(track.detection.width, track.detection.height, 1)
        for prior in self._ledger:
            if prior is track or prior.crossing_s is None or prior.crossing_px is None:
                continue
            if abs(track.crossing_s - prior.crossing_s) > 0.75:
                continue
            previous = prior.crossing_px[0] if axis == "y" else prior.crossing_px[1]
            if abs(current - previous) <= limit:
                track.passage_id = prior.passage_id
                return True
        return False


def _associate(tracks, detections, width, height, config: ExtractionConfig, *, relaxed: bool):
    if not tracks or not detections:
        return [], list(tracks), list(detections)
    costs = [
        [_pair_cost(track, detection, width, height, config, relaxed=relaxed) for detection in detections]
        for track in tracks
    ]
    pairs = _assign(costs)
    matched_tracks = {track_index for track_index, _detection_index in pairs}
    matched_detections = {detection_index for _track_index, detection_index in pairs}
    return (
        [(tracks[track_index], detections[detection_index]) for track_index, detection_index in pairs],
        [track for index, track in enumerate(tracks) if index not in matched_tracks],
        [item for index, item in enumerate(detections) if index not in matched_detections],
    )


def _pair_cost(track: _Passage, detection: Detection, width: int, height: int, config: ExtractionConfig, *, relaxed: bool) -> float:
    diagonal = max(1.0, float(np.hypot(width, height)))
    gate = config.event.max_track_distance_ratio * diagonal * (1.35 if relaxed else 1.0)
    distance = float(np.hypot(detection.center[0] - track.predicted[0], detection.center[1] - track.predicted[1]))
    if distance > gate:
        return float("inf")
    axis, _line = trigger_line_position(config, width, height)
    increasing = _direction_increasing(config.event.belt_direction)
    delta = (detection.center[1] - track.predicted[1]) if axis == "y" else (detection.center[0] - track.predicted[0])
    opposed = (delta > 0) != increasing and abs(delta) > max(8.0, 0.5 * max(detection.width, detection.height))
    mask = _polygon_iou(track.detection, detection) if track.detection.polygon and detection.polygon else 0.0
    return (
        0.45 * distance / diagonal
        + 0.25 * (1.0 - _box_iou(track.detection, detection))
        + 0.15 * (1.0 - mask)
        + (0.35 if opposed else 0.0)
        + 0.05 * abs(detection.area - track.detection.area) / max(detection.area, track.detection.area, 1)
    )


def _assign(costs: list[list[float]]) -> list[tuple[int, int]]:
    best = (0, 0.0, [])

    def search(track_index: int, used: set[int], score: float, chosen: list[tuple[int, int]]) -> None:
        nonlocal best
        if track_index == len(costs):
            candidate = (len(chosen), -score, list(chosen))
            if candidate[:2] > best[:2]:
                best = candidate
            return
        search(track_index + 1, used, score, chosen)
        for detection_index, cost in enumerate(costs[track_index]):
            if detection_index in used or cost == float("inf"):
                continue
            used.add(detection_index)
            chosen.append((track_index, detection_index))
            search(track_index + 1, used, score + cost, chosen)
            chosen.pop()
            used.remove(detection_index)

    search(0, set(), 0.0, [])
    return best[2]


def _center_in_roi(detection: Detection, config: ExtractionConfig, width: int, height: int) -> bool:
    x1, y1, x2, y2 = config.detector.roi
    cx, cy = detection.center
    return x1 * width <= cx <= x2 * width and y1 * height <= cy <= y2 * height


def _hysteresis(config: ExtractionConfig, width: int, height: int, axis: str) -> float:
    tx1, ty1, tx2, ty2 = config.detector.trigger_zone
    span = (ty2 - ty1) * height if axis == "y" else (tx2 - tx1) * width
    return float(config.event.trigger_hysteresis_ratio * span)


def _upstream(coord: float, line: float, hysteresis: float, increasing: bool) -> bool:
    return coord <= line - hysteresis if increasing else coord >= line + hysteresis


def _downstream(coord: float, line: float, hysteresis: float, increasing: bool) -> bool:
    return coord >= line + hysteresis if increasing else coord <= line - hysteresis


def _quality(pixels: np.ndarray, confidence: float) -> float:
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    sharp = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 500.0)
    return 0.65 * confidence + 0.35 * sharp


def _select_candidate(track: _Passage) -> _Candidate | None:
    if not track.candidates or track.crossing_s is None:
        return None
    eligible = [item for item in track.candidates if item.timestamp_s <= track.crossing_s + 1e-6]
    pool = eligible or track.candidates
    return max(pool, key=lambda item: item.quality)

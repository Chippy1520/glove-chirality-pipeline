"""Conveyor passage tracker.

One YOLO detection list in, one crop per physical glove out. Identity is the
lane and the belt direction, not a distance cutoff. A weak box may keep an
existing identity. It cannot create one, and it cannot become the Layer 2 crop.
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
    first_along: float = 0.0
    separated: bool = False
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
        self._dropped: list[PassageOutcome] = []
        self._width = 1
        self._height = 1

    def close(self) -> list[PassageOutcome]:
        outcomes = [self._terminal(track, self._terminal_reason(track)) for track in self._passages if not track.event_emitted]
        self._passages.clear()
        return outcomes

    def update(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        frame_index: int,
        timestamp_s: float,
        tracking_only: list[Detection] | None = None,
    ) -> list[PassageOutcome]:
        height, width = frame.shape[:2]
        self._width, self._height = width, height
        self._dropped = []
        observed = [item for item in detections if _center_in_roi(item, self.config, width, height)]
        strong, weak = self._split(observed)
        self._predict(timestamp_s)
        pairs, unmatched, unmatched_strong = _associate(
            self._passages, strong, width, height, self.config, relaxed=False
        )
        self._mark_separated(pairs)
        if self.config.event.merge_recovery:
            pairs, unmatched_strong = self._recover_merged(pairs, unmatched_strong, strong)
            matched = {id(track) for track, _detection in pairs}
            unmatched = [track for track in self._passages if id(track) not in matched]
        for track, detection in pairs:
            self._update_track(
                track, detection, timestamp_s, frame, frame_index, width, height, weak=False, siblings=strong
            )
        if self.config.event.low_conf_recovery and weak and unmatched:
            rescued, unmatched, _ignored = _associate(
                unmatched, weak, width, height, self.config, relaxed=True
            )
            for track, detection in rescued:
                self._update_track(track, detection, timestamp_s, frame, frame_index, width, height, weak=True)
        partials = []
        frame_area = max(1, width * height)
        maximum = self.config.detector.yolo_max_box_area_ratio
        for item in tracking_only or []:
            if not _center_in_roi(item, self.config, width, height):
                continue
            if item.area / frame_area > maximum:
                continue
            partials.append(item)
        if partials and unmatched:
            rescued, unmatched, _ignored = _associate(
                unmatched, partials, width, height, self.config, relaxed=True
            )
            for track, detection in rescued:
                self._update_track(
                    track, detection, timestamp_s, frame, frame_index, width, height, weak=True, siblings=strong
                )
        for detection in unmatched_strong:
            if detection.confidence < self.config.event.new_track_conf:
                continue
            if not self._eligible_birth(detection, width, height, timestamp_s):
                continue
            self._birth(detection, timestamp_s, frame, frame_index)
        self._retire(timestamp_s, width, height)
        return self._emit_ready(timestamp_s, width, height) + self._dropped

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
        siblings: list[Detection] | None = None,
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
            if track.strong_hits >= 2:
                self._advance_trigger(track, prev_x, prev_y, prev_t, cx, cy, timestamp_s, width, height)
            return
        track.strong_hits += 1
        self._advance_trigger(track, prev_x, prev_y, prev_t, cx, cy, timestamp_s, width, height)
        pixels = frame[detection.y1:detection.y2, detection.x1:detection.x2].copy()
        siblings = [item for item in (siblings or []) if item is not detection]
        if pixels.size and not _crop_rejected(detection, siblings, width, height):
            track.candidates.append(_Candidate(
                detection,
                pixels,
                timestamp_s,
                _quality(pixels, detection, siblings, width, height),
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
            first_along=cy if _motion_axis(self.config) == "y" else cx,
        )
        self._next_passage_id += 1
        pixels = frame[detection.y1:detection.y2, detection.x1:detection.x2].copy()
        width, height = frame.shape[1], frame.shape[0]
        if pixels.size and not _crop_rejected(detection, [], width, height):
            track.candidates.append(_Candidate(
                detection,
                pixels,
                timestamp_s,
                _quality(pixels, detection, [], width, height),
                frame_index,
            ))
        axis, line = trigger_line_position(self.config, frame.shape[1], frame.shape[0])
        hysteresis = _hysteresis(self.config, frame.shape[1], frame.shape[0], axis)
        increasing = _direction_increasing(self.config.event.belt_direction)
        current = cy if axis == "y" else cx
        track.armed = _upstream(current, line, hysteresis, increasing)
        self._passages.append(track)

    def _absorb_same_lane(self, tracks, detections, timestamp_s, frame, frame_index, width, height, siblings):
        if not tracks or not detections:
            return [], list(detections)
        pairs, _left, remaining = _associate(
            tracks, detections, width, height, self.config, relaxed=True, lane_only=True
        )
        for track, detection in pairs:
            self._update_track(
                track, detection, timestamp_s, frame, frame_index, width, height, weak=False, siblings=siblings
            )
        return pairs, remaining

    def _eligible_birth(self, detection: Detection, width: int, height: int, timestamp_s: float) -> bool:
        axis, line = trigger_line_position(self.config, width, height)
        hysteresis = _hysteresis(self.config, width, height, axis)
        increasing = _direction_increasing(self.config.event.belt_direction)
        current = detection.center[1] if axis == "y" else detection.center[0]
        if not _upstream(current, line, hysteresis, increasing):
            return False
        for track in self._passages:
            if track.event_emitted or not _same_lane(track, detection, self.config):
                continue
            if track.last_seen_s < timestamp_s or not _is_behind(track, detection, self.config):
                return False
        return True

    def _near_existing(self, detection: Detection, width: int, height: int) -> bool:
        return any(
            _pair_cost(track, detection, width, height, self.config, relaxed=True) < float("inf")
            for track in self._passages
        )

    def _retire(self, timestamp_s: float, width: int, height: int) -> None:
        alive = []
        for track in self._passages:
            if track.last_seen_s == timestamp_s:
                alive.append(track)
                continue
            if track.missing_since_s is None:
                track.missing_since_s = timestamp_s
            if timestamp_s - track.missing_since_s <= self.config.event.reentry_time_s:
                alive.append(track)
                continue
            if track.event_emitted:
                self._ledger.append(track)
                continue
            self._dropped.append(self._terminal(track, self._terminal_reason(track), width, height))
            track.event_emitted = True
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
                self._dropped.append(self._terminal(track, "duplicate_suppressed", width, height))
                continue
            selected = _select_candidate(track)
            if selected is None or selected.pixels.size == 0:
                track.event_emitted = True
                self._dropped.append(self._terminal(track, "no_valid_crop", width, height))
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

    def _terminal_reason(self, track: _Passage) -> str:
        if track.strong_hits < 2:
            return "insufficient_confirmation"
        if track.crossing_latched and not track.candidates:
            return "no_valid_crop"
        if not track.crossing_latched:
            return "lost_before_trigger"
        return "no_valid_crop"

    def _terminal(self, track: _Passage, reason: str, width: int | None = None, height: int | None = None) -> PassageOutcome:
        width = self._width if width is None else width
        height = self._height if height is None else height
        cx, cy, _seen = track.observed
        return PassageOutcome(
            self._next_event_id(),
            self.source_video,
            self.label,
            "rejected",
            reason,
            0,
            track.last_seen_s,
            track.strong_hits + track.weak_hits,
            track.detection,
            0.0,
            None,
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
        )

    def _mark_separated(self, pairs) -> None:
        axis = _cross_axis(self.config)
        for index, (left, left_det) in enumerate(pairs):
            for right, right_det in pairs[index + 1 :]:
                if left_det is right_det:
                    continue
                gap = abs(_cross(left_det.center, axis) - _cross(right_det.center, axis))
                width = max(left_det.width, right_det.width, 1)
                if gap >= 0.85 * width:
                    left.separated = True
                    right.separated = True

    def _recover_merged(self, pairs, unmatched_strong, strong):
        extra: list[tuple[_Passage, Detection]] = []
        consumed: set[int] = set()
        replaced: set[int] = set()
        for detection in strong:
            covered = [
                track
                for track in self._passages
                if track.strong_hits >= 2
                and not track.event_emitted
                and track.separated
                and _claims(track, detection, self.config)
            ]
            if len(covered) < 2 or not _still_apart(covered, self.config):
                continue
            pieces = [_slice_for_track(track, detection, self.config) for track in covered]
            if any(piece is None for piece in pieces):
                continue
            if any(
                _box_iou(left, right) > 0.85
                for index, left in enumerate(pieces)
                for right in pieces[index + 1 :]
            ):
                continue
            consumed.add(id(detection))
            replaced.update(id(track) for track in covered)
            extra.extend(zip(covered, pieces, strict=True))
        if not extra:
            return pairs, unmatched_strong
        kept = [
            (track, detection)
            for track, detection in pairs
            if id(track) not in replaced and id(detection) not in consumed
        ]
        kept.extend(extra)
        return kept, [item for item in unmatched_strong if id(item) not in consumed]

    def _duplicate(self, track: _Passage, width: int, height: int) -> bool:
        if track.crossing_px is None or track.crossing_s is None:
            return False
        axis, line = trigger_line_position(self.config, width, height)
        increasing = _direction_increasing(self.config.event.belt_direction)
        current = track.crossing_px[0] if axis == "y" else track.crossing_px[1]
        limit = 0.75 * max(track.detection.width, track.detection.height, 1)
        for prior in self._ledger:
            if prior is track or prior.crossing_s is None or prior.crossing_px is None:
                continue
            if track.first_seen_s >= prior.crossing_s and _upstream(
                track.first_along, line, 0.0, increasing
            ):
                continue
            if abs(track.crossing_s - prior.crossing_s) > 0.75:
                continue
            previous = prior.crossing_px[0] if axis == "y" else prior.crossing_px[1]
            if abs(current - previous) <= limit:
                track.passage_id = prior.passage_id
                return True
        return False


def _motion_axis(config: ExtractionConfig) -> str:
    if config.event.belt_direction in {"bottom_to_top", "top_to_bottom"}:
        return "y"
    return "x"


def _cross_axis(config: ExtractionConfig) -> str:
    return "x" if _motion_axis(config) == "y" else "y"


def _cross(point, axis: str) -> float:
    return float(point[0] if axis == "x" else point[1])


def _claims(track: _Passage, detection: Detection, config: ExtractionConfig) -> bool:
    cross = _cross(track.predicted, _cross_axis(config))
    if _cross_axis(config) == "x":
        return detection.x1 <= cross <= detection.x2
    return detection.y1 <= cross <= detection.y2


def _slice_for_track(track: _Passage, merged: Detection, config: ExtractionConfig) -> Detection | None:
    cross = _cross(track.predicted, _cross_axis(config))
    half = max(track.detection.width, track.detection.height, 8) / 2
    if _cross_axis(config) == "x":
        x1 = max(merged.x1, round(cross - half))
        x2 = min(merged.x2, round(cross + half))
        y1, y2 = merged.y1, merged.y2
    else:
        y1 = max(merged.y1, round(cross - half))
        y2 = min(merged.y2, round(cross + half))
        x1, x2 = merged.x1, merged.x2
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return Detection(x1, y1, x2, y2, merged.confidence)


def _still_apart(tracks: list[_Passage], config: ExtractionConfig) -> bool:
    axis = _cross_axis(config)
    points = [_cross(track.predicted, axis) for track in tracks]
    width = max(track.detection.width for track in tracks)
    return max(points) - min(points) >= 0.45 * max(width, 1)


def _associate(tracks, detections, width, height, config: ExtractionConfig, *, relaxed: bool, lane_only: bool = False):
    if not tracks or not detections:
        return [], list(tracks), list(detections)
    costs = [
        [
            _pair_cost(track, detection, width, height, config, relaxed=relaxed, lane_only=lane_only)
            for detection in detections
        ]
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


def _pair_cost(track: _Passage, detection: Detection, width: int, height: int, config: ExtractionConfig, *, relaxed: bool, lane_only: bool = False) -> float:
    del relaxed, lane_only, width, height
    if not _owns(track, detection, config):
        return float("inf")
    return _along_gap(track, detection, config)


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


def _owns(track: _Passage, detection: Detection, config: ExtractionConfig) -> bool:
    """A live glove owns every later box in its lane. A jump is not a new glove."""
    return _same_lane(track, detection, config)


def _is_behind(track: _Passage, detection: Detection, config: ExtractionConfig) -> bool:
    axis = _motion_axis(config)
    increasing = _direction_increasing(config.event.belt_direction)
    track_along = _cross(track.predicted, axis)
    det_along = _cross(detection.center, axis)
    glove = max(
        track.detection.height if axis == "y" else track.detection.width,
        detection.height if axis == "y" else detection.width,
        1,
    )
    if increasing:
        return det_along < track_along - glove
    return det_along > track_along + glove


def _same_lane(track: _Passage, detection: Detection, config: ExtractionConfig) -> bool:
    axis = _cross_axis(config)
    gap = abs(_cross(detection.center, axis) - _cross(track.predicted, axis))
    return gap <= 0.75 * max(detection.width, track.detection.width, 1)


def _along_gap(track: _Passage, detection: Detection, config: ExtractionConfig) -> float:
    axis = _motion_axis(config)
    return abs(_cross(detection.center, axis) - _cross(track.predicted, axis))


def _crop_rejected(detection: Detection, siblings: list[Detection], width: int, height: int) -> bool:
    margin = 2
    if detection.x1 <= margin or detection.y1 <= margin:
        return True
    if detection.x2 >= width - margin or detection.y2 >= height - margin:
        return True
    return any(_box_iou(detection, other) > 0.35 for other in siblings)


def _quality(pixels: np.ndarray, detection: Detection, siblings: list[Detection], width: int, height: int) -> float:
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    sharp = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 500.0)
    cx, cy = detection.center
    clearance = min(cx, cy, width - cx, height - cy) / max(min(width, height), 1)
    edge = min(1.0, clearance / 0.12)
    overlap = max((_box_iou(detection, other) for other in siblings), default=0.0)
    return 0.25 * detection.confidence + 0.15 * sharp + 0.30 * edge + 0.30 * (1.0 - overlap)


def _select_candidate(track: _Passage) -> _Candidate | None:
    if not track.candidates or track.crossing_s is None:
        return None
    eligible = [item for item in track.candidates if item.timestamp_s <= track.crossing_s + 1e-6]
    pool = eligible or track.candidates
    return max(pool, key=lambda item: item.quality)

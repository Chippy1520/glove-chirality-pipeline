from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from glove_chirality.config import ExtractionConfig
from glove_chirality.detection.base import GloveDetector, inside_trigger
from glove_chirality.types import Detection


@dataclass
class _GloveTrack:
    track_id: int
    detection: Detection
    center: tuple[float, float, float]
    first_seen_s: float
    seen: int = 1
    missing: int = 0
    missing_since_s: float | None = None
    emitted: bool = False


@dataclass
class _Candidate:
    frame: np.ndarray
    detection: Detection
    frame_index: int
    timestamp_s: float
    quality: float
    sharpness: float
    candidate_count: int


@dataclass
class PassageOutcome:
    event_id: str
    source_video: str
    label: str
    status: str
    reject_reason: str
    frame_index: int
    timestamp_s: float
    candidate_count: int
    detection: Detection | None
    quality_score: float = 0.0
    crop: np.ndarray | None = None
    full_frame: np.ndarray | None = None
    passage_started_s: float | None = None
    trigger_crossing_s: float | None = None
    trigger_crossing_position_px: tuple[float, float] | None = None
    trigger_crossing_position_norm: tuple[float, float] | None = None
    center_px: tuple[float, float] | None = None
    center_norm: tuple[float, float] | None = None

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"


@dataclass(frozen=True)
class FrameResult:
    outcomes: tuple[PassageOutcome, ...]
    detections: tuple[Detection, ...]
    detector_latency_ms: float
    event_latency_ms: float


def _sharpness(frame: np.ndarray, box: Detection) -> float:
    crop = frame[box.y1:box.y2, box.x1:box.x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _box_iou(first: Detection, second: Detection) -> float:
    x1, y1 = max(first.x1, second.x1), max(first.y1, second.y1)
    x2, y2 = min(first.x2, second.x2), min(first.y2, second.y2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    return intersection / max(1, first.area + second.area - intersection)


def _polygon_iou(first: Detection, second: Detection) -> float:
    if first.polygon is None or second.polygon is None:
        return 0.0
    x1, y1 = min(first.x1, second.x1), min(first.y1, second.y1)
    x2, y2 = max(first.x2, second.x2), max(first.y2, second.y2)
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return 0.0
    first_mask = np.zeros((height, width), dtype=np.uint8)
    second_mask = np.zeros_like(first_mask)
    first_points = np.rint(np.asarray(first.polygon) - (x1, y1)).astype(np.int32)
    second_points = np.rint(np.asarray(second.polygon) - (x1, y1)).astype(np.int32)
    cv2.fillPoly(first_mask, [first_points], 1)
    cv2.fillPoly(second_mask, [second_points], 1)
    intersection = np.count_nonzero(first_mask & second_mask)
    union = np.count_nonzero(first_mask | second_mask)
    return float(intersection / union) if union else 0.0


def _boundary_clearance(detection: Detection, config: ExtractionConfig, width: int, height: int) -> float:
    tx1, ty1, tx2, ty2 = config.detector.trigger_zone
    left = detection.x1 - tx1 * width
    top = detection.y1 - ty1 * height
    right = tx2 * width - detection.x2
    bottom = ty2 * height - detection.y2
    scale = max(1.0, min((tx2 - tx1) * width, (ty2 - ty1) * height) / 2.0)
    return float(np.clip(min(left, top, right, bottom) / scale, 0.0, 1.0))


def _touches_edge(detection: Detection, config: ExtractionConfig, width: int, height: int) -> bool:
    rx1, ry1, rx2, ry2 = config.detector.roi
    return (
        detection.x1 <= round(rx1 * width)
        or detection.y1 <= round(ry1 * height)
        or detection.x2 >= round(rx2 * width)
        or detection.y2 >= round(ry2 * height)
        or detection.x1 <= 0
        or detection.y1 <= 0
        or detection.x2 >= width
        or detection.y2 >= height
    )


def _quality(
    frame: np.ndarray,
    detection: Detection,
    config: ExtractionConfig,
    previous: Detection | None,
) -> tuple[float, float]:
    height, width = frame.shape[:2]
    tx1, ty1, tx2, ty2 = config.detector.trigger_zone
    target_x, target_y = (tx1 + tx2) * width / 2, (ty1 + ty2) * height / 2
    cx, cy = detection.center
    distance = np.hypot(cx - target_x, cy - target_y)
    diagonal = max(1.0, np.hypot((tx2 - tx1) * width, (ty2 - ty1) * height) / 2)
    centrality = max(0.0, 1.0 - distance / diagonal)
    raw_sharpness = _sharpness(frame, detection)
    sharpness = min(1.0, raw_sharpness / 500.0)
    score = 0.50 * centrality + 0.35 * detection.confidence + 0.15 * sharpness

    event = config.event
    mask_area = detection.mask_area
    if mask_area is not None:
        trigger_area = max(1.0, (tx2 - tx1) * width * (ty2 - ty1) * height)
        score += event.quality_mask_area_weight * min(1.0, mask_area / trigger_area)
        if previous is not None and previous.mask_area is not None:
            stability = 1.0 - abs(mask_area - previous.mask_area) / max(mask_area, previous.mask_area, 1.0)
            score += event.quality_mask_stability_weight * max(0.0, stability)
        score += event.quality_boundary_clearance_weight * _boundary_clearance(
            detection, config, width, height
        )
        if _touches_edge(detection, config, width, height):
            score -= event.quality_edge_penalty_weight
    return float(score), raw_sharpness


def _crop_bounds(
    frame: np.ndarray,
    box: Detection,
    padding: float,
    square: bool,
) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    center_x, center_y = box.center
    expanded_width = box.width * (1 + 2 * padding)
    expanded_height = box.height * (1 + 2 * padding)
    if square:
        side = max(1, min(width, height, int(np.ceil(max(expanded_width, expanded_height)))))
        x1 = max(0, min(width - side, round(center_x - side / 2)))
        y1 = max(0, min(height - side, round(center_y - side / 2)))
        return x1, y1, x1 + side, y1 + side
    return (
        max(0, int(np.floor(center_x - expanded_width / 2))),
        max(0, int(np.floor(center_y - expanded_height / 2))),
        min(width, int(np.ceil(center_x + expanded_width / 2))),
        min(height, int(np.ceil(center_y + expanded_height / 2))),
    )


def _letterbox(image: np.ndarray, size: int, fill_value: int = 114) -> np.ndarray:
    """Resize without aspect distortion and center on a fixed square canvas."""
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return image
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.empty((size, size, image.shape[2]), dtype=image.dtype)
    canvas[:] = fill_value
    x1 = (size - resized_width) // 2
    y1 = (size - resized_height) // 2
    canvas[y1:y1 + resized_height, x1:x1 + resized_width] = resized
    return canvas


def create_event_crop(
    frame: np.ndarray,
    detection: Detection,
    config: ExtractionConfig,
) -> np.ndarray:
    """Create the canonical offline/live crop for one selected passage frame."""
    event = config.event
    x1, y1, x2, y2 = _crop_bounds(frame, detection, event.crop_padding, event.make_square)
    crop = frame[y1:y2, x1:x2].copy()
    if event.crop_mode != "bbox":
        if detection.polygon is None:
            raise RuntimeError(f"crop_mode={event.crop_mode} requires a segmentation polygon")
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        points = np.rint(np.asarray(detection.polygon) - (x1, y1)).astype(np.int32)
        cv2.fillPoly(mask, [points], 255)
        outside = mask == 0
        if event.crop_mode == "masked":
            crop[outside] = 0
        else:
            background = crop[outside]
            source = background if background.size else crop.reshape(-1, crop.shape[2])
            crop[outside] = np.median(source, axis=0).astype(crop.dtype)
    return _letterbox(crop, event.output_size, event.letterbox_fill)


def _canonical_detection_key(detection: Detection) -> tuple[float, ...]:
    return (
        -detection.confidence,
        -detection.area,
        detection.x1,
        detection.y1,
        detection.x2,
        detection.y2,
        -1 if detection.class_id is None else detection.class_id,
    )


def directional_crossing_alpha(
    previous: float,
    current: float,
    line: float,
    increasing: bool,
) -> float | None:
    """Interpolation alpha for one directional line crossing, or None if it did not happen."""
    if increasing:
        crossed = previous < line <= current
    else:
        crossed = previous > line >= current
    if not crossed or current == previous:
        return None
    alpha = (line - previous) / (current - previous)
    return float(min(1.0, max(0.0, alpha)))


def trigger_line_axis(direction: str) -> str:
    """Motion axis for a belt direction. The trigger line is perpendicular to it."""
    if direction in {"left_to_right", "right_to_left"}:
        return "x"
    if direction in {"top_to_bottom", "bottom_to_top"}:
        return "y"
    raise ValueError(
        "belt_direction must be left_to_right, right_to_left, "
        "top_to_bottom, or bottom_to_top"
    )


def trigger_line_position(
    config: ExtractionConfig,
    width: int,
    height: int,
) -> tuple[str, float]:
    """Return the trigger-line axis and its pixel position inside the trigger zone."""
    axis = trigger_line_axis(config.event.belt_direction)
    tx1, ty1, tx2, ty2 = config.detector.trigger_zone
    fraction = config.event.trigger_line_fraction
    if axis == "x":
        position = (tx1 + fraction * (tx2 - tx1)) * width
    else:
        position = (ty1 + fraction * (ty2 - ty1)) * height
    return axis, float(position)


def _direction_increasing(direction: str) -> bool:
    if direction in {"left_to_right", "top_to_bottom"}:
        return True
    if direction in {"right_to_left", "bottom_to_top"}:
        return False
    raise ValueError(
        "belt_direction must be left_to_right, right_to_left, "
        "top_to_bottom, or bottom_to_top"
    )


def _candidate_key(candidate: _Candidate) -> tuple[float, ...]:
    return (
        candidate.quality,
        candidate.detection.confidence,
        candidate.sharpness,
        candidate.detection.area,
        -candidate.frame_index,
    )


class PassageProcessor:
    """Shared detector, per-glove tracker, crossing crop, and canonical crop path."""

    def __init__(
        self,
        detector: GloveDetector,
        config: ExtractionConfig,
        source_video: str,
        label: str = "unknown",
    ):
        self.detector = detector
        self.config = config
        self.source_video = source_video
        self.label = label
        self.active = False
        self.seen = 0
        self.missing = 0
        self.first_seen_s: float | None = None
        self.missing_since_s: float | None = None
        self.cooldown = 0
        self.cooldown_until_s = 0.0
        self.rearming = False
        self.last_timestamp_s: float | None = None
        self.best: _Candidate | None = None
        self.last_detection: Detection | None = None
        self.pending_partial: tuple[int, float, Detection, int] | None = None
        self.ambiguity_latched = False
        self.sequence = 0
        self._previous_center: tuple[float, float, float] | None = None
        self._trigger_crossing_s: float | None = None
        self._trigger_crossing_position_px: tuple[float, float] | None = None
        self._trigger_crossing_position_norm: tuple[float, float] | None = None
        self._frame_width = 0
        self._frame_height = 0
        self._tracks: list[_GloveTrack] = []
        self._next_track_id = 1
        self._recent_emits: list[tuple[float, float, float]] = []
        self._passage_tracker = None

    def _event_id(self) -> str:
        self.sequence += 1
        return f"{self.label}__{self.source_video.rsplit('.', 1)[0]}__e{self.sequence:06d}"

    def _passage_v2(self):
        if self._passage_tracker is None:
            from glove_chirality.tracking import PassageTracker

            self._passage_tracker = PassageTracker(
                self.config,
                self.source_video,
                self.label,
                self._event_id,
            )
        return self._passage_tracker

    def _reset(self, timestamp_s: float, cooldown: bool = False) -> None:
        self.active = False
        self.seen = 0
        self.missing = 0
        self.first_seen_s = None
        self.missing_since_s = None
        self.best = None
        self.last_detection = None
        self._previous_center = None
        self._trigger_crossing_s = None
        self._trigger_crossing_position_px = None
        self._trigger_crossing_position_norm = None
        if cooldown:
            self.rearming = True
            self.cooldown = self.config.event.cooldown_frames
            self.cooldown_until_s = timestamp_s + self.config.event.cooldown_seconds
        else:
            self.rearming = False

    def _apply_geometry(self, outcome: PassageOutcome) -> PassageOutcome:
        outcome.trigger_crossing_s = self._trigger_crossing_s
        position = self._trigger_crossing_position_px
        normalized = self._trigger_crossing_position_norm
        outcome.trigger_crossing_position_px = None if position is None else tuple(position)
        outcome.trigger_crossing_position_norm = None if normalized is None else tuple(normalized)
        detection = outcome.detection
        if detection is None:
            return outcome
        cx, cy = detection.center
        outcome.center_px = (float(cx), float(cy))
        width, height = self._frame_width, self._frame_height
        if width > 0 and height > 0:
            outcome.center_norm = (float(cx) / width, float(cy) / height)
        return outcome

    def _note_trigger_crossing(
        self,
        previous: tuple[float, float, float],
        current_x: float,
        current_y: float,
        timestamp_s: float,
        width: int,
        height: int,
    ) -> None:
        event = self.config.event
        if not event.trigger_line_enabled or self._trigger_crossing_s is not None:
            return
        if width <= 0 or height <= 0:
            return
        axis, line = trigger_line_position(self.config, width, height)
        increasing = _direction_increasing(event.belt_direction)
        prev_x, prev_y, prev_t = previous
        if axis == "x":
            alpha = directional_crossing_alpha(prev_x, current_x, line, increasing)
            if alpha is None:
                return
            cross_x = line
            cross_y = prev_y + alpha * (current_y - prev_y)
        else:
            alpha = directional_crossing_alpha(prev_y, current_y, line, increasing)
            if alpha is None:
                return
            cross_x = prev_x + alpha * (current_x - prev_x)
            cross_y = line
        self._trigger_crossing_s = prev_t + alpha * (timestamp_s - prev_t)
        self._trigger_crossing_position_px = (float(cross_x), float(cross_y))
        self._trigger_crossing_position_norm = (float(cross_x) / width, float(cross_y) / height)

    def _record(
        self,
        reason: str,
        frame_index: int,
        timestamp_s: float,
        candidate_count: int,
        detection: Detection | None,
    ) -> PassageOutcome:
        outcome = PassageOutcome(
            self._event_id(),
            self.source_video,
            self.label,
            reason,
            reason,
            frame_index,
            timestamp_s,
            candidate_count,
            detection,
        )
        return self._apply_geometry(outcome)

    def _confirmed(self, timestamp_s: float) -> bool:
        if self.config.event.timing_mode == "time":
            return (
                self.first_seen_s is not None
                and timestamp_s - self.first_seen_s >= self.config.event.min_detected_seconds
            )
        return self.seen >= self.config.event.min_detected_frames

    def _cooling_down(self, timestamp_s: float) -> bool:
        return self.rearming

    def _missing_complete(self, timestamp_s: float) -> bool:
        if self.config.event.timing_mode == "time":
            return (
                self.missing_since_s is not None
                and timestamp_s - self.missing_since_s >= self.config.event.exit_missing_seconds
            )
        return self.missing >= self.config.event.exit_missing_frames

    def _finalize(self, reason: str, timestamp_s: float) -> PassageOutcome | None:
        best = self.best
        if best is None:
            self._reset(timestamp_s, cooldown=True)
            return None
        if not self.active:
            outcome = self._record(
                "insufficient_confirmation",
                best.frame_index,
                best.timestamp_s,
                best.candidate_count,
                best.detection,
            )
        elif best.sharpness < self.config.event.min_sharpness:
            outcome = self._record(
                "low_sharpness",
                best.frame_index,
                best.timestamp_s,
                best.candidate_count,
                best.detection,
            )
        else:
            outcome = PassageOutcome(
                self._event_id(),
                self.source_video,
                self.label,
                "accepted",
                "",
                best.frame_index,
                best.timestamp_s,
                best.candidate_count,
                best.detection,
                best.quality,
                create_event_crop(best.frame, best.detection, self.config),
                best.frame,
                self.first_seen_s,
            )
        outcome = self._apply_geometry(outcome)
        self._reset(timestamp_s, cooldown=True)
        return outcome

    def _associate(self, detections: list[Detection], width: int, height: int) -> Detection | None:
        if not detections:
            return None
        if self.last_detection is None:
            return min(detections, key=_canonical_detection_key)
        last = self.last_detection
        diagonal = max(1.0, np.hypot(width, height))
        maximum = self.config.event.max_track_distance_ratio * diagonal

        def rank(item: Detection) -> tuple[float, ...]:
            distance = np.hypot(item.center[0] - last.center[0], item.center[1] - last.center[1])
            box_iou = _box_iou(last, item)
            mask_iou = _polygon_iou(last, item)
            cost = float(
                distance / diagonal
                + self.config.event.association_iou_weight * (1.0 - box_iou)
                + self.config.event.association_mask_iou_weight * (1.0 - mask_iou)
            )
            return (
                cost,
                float(distance),
                -box_iou,
                -mask_iou,
                -item.confidence,
                -item.area,
                item.x1,
                item.y1,
                item.x2,
                item.y2,
                -1 if item.class_id is None else item.class_id,
            )

        chosen = min(detections, key=rank)
        distance = np.hypot(chosen.center[0] - last.center[0], chosen.center[1] - last.center[1])
        return chosen if distance <= maximum else None

    def _track_gate(self, detection: Detection, width: int, height: int) -> float:
        del detection
        diagonal = max(1.0, float(np.hypot(width, height)))
        return float(self.config.event.max_track_distance_ratio * diagonal)

    def _match_instances(
        self,
        detections: list[Detection],
        width: int,
        height: int,
    ) -> dict[int, int]:
        diagonal = max(1.0, float(np.hypot(width, height)))
        pairs: list[tuple[float, float, int, int]] = []
        for track_index, track in enumerate(self._tracks):
            gate = self._track_gate(track.detection, width, height)
            for detection_index, detection in enumerate(detections):
                distance = float(np.hypot(
                    detection.center[0] - track.center[0],
                    detection.center[1] - track.center[1],
                ))
                if distance > gate:
                    continue
                cost = distance / diagonal + self.config.event.association_iou_weight * (
                    1.0 - _box_iou(track.detection, detection)
                )
                pairs.append((cost, distance, track_index, detection_index))
        pairs.sort()
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        matches: dict[int, int] = {}
        for cost, _distance, track_index, detection_index in pairs:
            del cost
            if track_index in used_tracks or detection_index in used_detections:
                continue
            used_tracks.add(track_index)
            used_detections.add(detection_index)
            matches[track_index] = detection_index
        return matches

    def _overlapping(self, detections: list[Detection]) -> set[int]:
        if not self.config.event.reject_multiple_detections:
            return set()
        blocked: set[int] = set()
        for index, first in enumerate(detections):
            for other in range(index + 1, len(detections)):
                if _box_iou(first, detections[other]) >= 0.45:
                    blocked.add(index)
                    blocked.add(other)
        return blocked

    def _recently_emitted(self, detection: Detection, timestamp_s: float, width: int, height: int) -> bool:
        cutoff = timestamp_s - self.config.event.cooldown_seconds
        self._recent_emits = [item for item in self._recent_emits if item[2] >= cutoff]
        gate = self._track_gate(detection, width, height)
        return any(
            float(np.hypot(detection.center[0] - x, detection.center[1] - y)) <= gate
            for x, y, _seen in self._recent_emits
        )

    def _emit_crossing(
        self,
        track: _GloveTrack,
        frame: np.ndarray,
        detection: Detection,
        frame_index: int,
        timestamp_s: float,
        width: int,
        height: int,
    ) -> PassageOutcome | None:
        if track.emitted or width <= 0 or height <= 0:
            return None
        axis, line = trigger_line_position(self.config, width, height)
        increasing = _direction_increasing(self.config.event.belt_direction)
        prev_x, prev_y, prev_t = track.center
        current_x, current_y = detection.center
        if axis == "y":
            alpha = directional_crossing_alpha(prev_y, float(current_y), line, increasing)
            cross_x = prev_x + (alpha or 0.0) * (float(current_x) - prev_x)
            cross_y = line
        else:
            alpha = directional_crossing_alpha(prev_x, float(current_x), line, increasing)
            cross_x = line
            cross_y = prev_y + (alpha or 0.0) * (float(current_y) - prev_y)
        if alpha is None:
            return None
        track.emitted = True
        self._recent_emits.append((float(current_x), float(current_y), timestamp_s))
        return PassageOutcome(
            self._event_id(),
            self.source_video,
            self.label,
            "accepted",
            "",
            frame_index,
            timestamp_s,
            1,
            detection,
            0.0,
            create_event_crop(frame, detection, self.config),
            None,
            track.first_seen_s,
            prev_t + alpha * (timestamp_s - prev_t),
            (float(cross_x), float(cross_y)),
            (float(cross_x) / width, float(cross_y) / height),
            (float(current_x), float(current_y)),
            (float(current_x) / width, float(current_y) / height),
        )

    def _process_instances(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        frame_index: int,
        timestamp_s: float,
        width: int,
        height: int,
    ) -> list[PassageOutcome]:
        eligible = [
            detection
            for detection in detections
            if inside_trigger(detection, self.config.detector, width, height)
        ]
        blocked = self._overlapping(eligible)
        outcomes: list[PassageOutcome] = []
        if blocked and not self.ambiguity_latched:
            outcomes.append(self._record(
                "multiple_candidates", frame_index, timestamp_s, len(blocked), None
            ))
        self.ambiguity_latched = bool(blocked)
        usable = [detection for index, detection in enumerate(eligible) if index not in blocked]
        matches = self._match_instances(usable, width, height)
        for track_index, detection_index in matches.items():
            track = self._tracks[track_index]
            detection = usable[detection_index]
            outcome = self._emit_crossing(
                track, frame, detection, frame_index, timestamp_s, width, height
            )
            if outcome is not None:
                outcomes.append(outcome)
            track.detection = detection
            track.center = (float(detection.center[0]), float(detection.center[1]), timestamp_s)
            track.seen += 1
            track.missing = 0
            track.missing_since_s = None
        matched = set(matches.values())
        born: list[_GloveTrack] = []
        for index, detection in enumerate(usable):
            if index in matched or self._recently_emitted(detection, timestamp_s, width, height):
                continue
            born.append(_GloveTrack(
                self._next_track_id,
                detection,
                (float(detection.center[0]), float(detection.center[1]), timestamp_s),
                timestamp_s,
            ))
            self._next_track_id += 1
        survivors: list[_GloveTrack] = []
        for track_index, track in enumerate(self._tracks):
            if track_index in matches:
                survivors.append(track)
                continue
            track.missing += 1
            if track.missing_since_s is None:
                track.missing_since_s = timestamp_s
            if self.config.event.timing_mode == "time":
                lost = timestamp_s - track.missing_since_s >= self.config.event.exit_missing_seconds
            else:
                lost = track.missing >= self.config.event.exit_missing_frames
            if not lost:
                survivors.append(track)
        self._tracks = survivors + born
        return outcomes

    def _handle_missing(self, frame_index: int, timestamp_s: float) -> list[PassageOutcome]:
        outcomes: list[PassageOutcome] = []
        if self.active or self.seen:
            self.missing += 1
            if self.missing_since_s is None:
                self.missing_since_s = timestamp_s
            if self._missing_complete(timestamp_s):
                outcome = self._finalize("track_lost", timestamp_s)
                if outcome is not None:
                    outcomes.append(outcome)
        if self.pending_partial is not None:
            partial_frame, partial_time, detection, count = self.pending_partial
            outcomes.append(self._record("partial", partial_frame, partial_time, count, detection))
            self.pending_partial = None
        return outcomes

    def process(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_s: float,
        run_detection: bool = True,
    ) -> FrameResult:
        if self.last_timestamp_s is not None and timestamp_s < self.last_timestamp_s:
            raise ValueError("passage timestamps must be nondecreasing")
        self.last_timestamp_s = timestamp_s
        if not run_detection:
            return FrameResult((), (), 0.0, 0.0)
        detect_start = time.perf_counter()
        detections = self.detector.detect(frame)
        detector_latency = (time.perf_counter() - detect_start) * 1000.0
        event_start = time.perf_counter()
        height, width = frame.shape[:2]
        self._frame_width = int(width)
        self._frame_height = int(height)
        if (
            self.config.event.tracker_mode == "passage_v2"
            and self.config.event.trigger_line_enabled
        ):
            outcomes = self._passage_v2().update(frame, detections, frame_index, timestamp_s)
            event_latency = (time.perf_counter() - event_start) * 1000.0
            return FrameResult(tuple(outcomes), tuple(detections), detector_latency, event_latency)
        if self.config.event.trigger_line_enabled:
            outcomes = self._process_instances(
                frame, detections, frame_index, timestamp_s, width, height
            )
            event_latency = (time.perf_counter() - event_start) * 1000.0
            return FrameResult(tuple(outcomes), tuple(detections), detector_latency, event_latency)
        outcomes: list[PassageOutcome] = []

        eligible = [
            detection
            for detection in detections
            if inside_trigger(
                detection,
                self.config.detector,
                width,
                height,
            )
        ]

        if self.rearming:
            if eligible:
                self.cooldown = self.config.event.cooldown_frames
                self.cooldown_until_s = (timestamp_s+ self.config.event.cooldown_seconds)
            elif self.config.event.timing_mode == "frames":
                self.cooldown = max(0, self.cooldown - 1)
                self.rearming = self.cooldown > 0
            elif timestamp_s >= self.cooldown_until_s:
                self.rearming = False

        if self.rearming:
            event_latency = (time.perf_counter() - event_start) * 1000.0
            return FrameResult((), tuple(detections), detector_latency, event_latency)

        if self.config.event.reject_multiple_detections and len(eligible) > 1:
            if not self.ambiguity_latched:
                outcomes.append(
                    self._record(
                        "multiple_candidates",
                        frame_index,
                        timestamp_s,
                        len(eligible),
                        None,
                    )
                )
            self.ambiguity_latched = True
            self.pending_partial = None
            self._reset(timestamp_s, cooldown=True)
        else:
            self.ambiguity_latched = False
            if detections and not eligible:
                if not self.active and not self.seen:
                    self.pending_partial = (
                        frame_index,
                        timestamp_s,
                        min(detections, key=_canonical_detection_key),
                        len(detections),
                    )
                if self.active or self.seen:
                    self.missing += 1
                    if self.missing_since_s is None:
                        self.missing_since_s = timestamp_s
                    if self._missing_complete(timestamp_s):
                        outcome = self._finalize("track_lost", timestamp_s)
                        if outcome is not None:
                            outcomes.append(outcome)
            elif not detections:
                outcomes.extend(self._handle_missing(frame_index, timestamp_s))
            else:
                self.pending_partial = None
                chosen = self._associate(eligible, width, height)
                if chosen is None:
                    if self.active or self.seen:
                        outcomes.append(
                            self._record(
                                "track_lost",
                                frame_index,
                                timestamp_s,
                                len(detections),
                                None,
                            )
                        )
                        self._reset(timestamp_s, cooldown=True)
                elif self.active or not self._cooling_down(timestamp_s):
                    previous = self.last_detection
                    if self.seen == 0:
                        self.first_seen_s = timestamp_s
                    self.seen += 1
                    self.missing = 0
                    self.missing_since_s = None
                    score, sharpness = _quality(frame, chosen, self.config, previous)
                    cx, cy = chosen.center
                    if self._previous_center is not None:
                        self._note_trigger_crossing(
                            self._previous_center,
                            float(cx),
                            float(cy),
                            float(timestamp_s),
                            width,
                            height,
                        )
                    self._previous_center = (float(cx), float(cy), float(timestamp_s))
                    self.last_detection = chosen
                    if self._confirmed(timestamp_s):
                        self.active = True
                    candidate = _Candidate(
                        frame.copy(),
                        chosen,
                        frame_index,
                        timestamp_s,
                        score,
                        sharpness,
                        len(detections),
                    )
                    if self.best is None or _candidate_key(candidate) > _candidate_key(self.best):
                        self.best = candidate

        event_latency = (time.perf_counter() - event_start) * 1000.0
        return FrameResult(tuple(outcomes), tuple(detections), detector_latency, event_latency)

    def close(self, timestamp_s: float) -> tuple[PassageOutcome, ...]:
        if self.last_timestamp_s is not None and timestamp_s < self.last_timestamp_s:
            raise ValueError("passage timestamps must be nondecreasing")
        self.last_timestamp_s = timestamp_s
        if self.config.event.trigger_line_enabled:
            if self._passage_tracker is not None:
                self._passage_tracker.close()
            self._tracks.clear()
            return ()
        outcomes: list[PassageOutcome] = []
        if self.active:
            outcome = self._finalize("end_of_stream", timestamp_s)
            if outcome is not None:
                outcomes.append(outcome)
        elif self.seen:
            best = self.best
            if best is not None:
                outcomes.append(
                    self._record(
                        "insufficient_confirmation",
                        best.frame_index,
                        best.timestamp_s,
                        best.candidate_count,
                        best.detection,
                    )
                )
            self._reset(timestamp_s)
        elif self.pending_partial is not None:
            frame_index, partial_time, detection, count = self.pending_partial
            outcomes.append(
                self._record("end_of_stream", frame_index, partial_time, count, detection)
            )
            self.pending_partial = None
        return tuple(outcomes)

"""Motion-only ByteTrack for conveyor gloves.

One detection list in, one track id per physical glove. Each track is a
constant-velocity Kalman filter. Assignment is Hungarian on distance to the
predicted center, so two gloves that keep the same order on the belt are not
glued together by the first overlapping pair. There is no appearance model.
A track emits one crop when its center crosses the trigger line.
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

_INF = 1e6


@dataclass
class _Shot:
    detection: Detection
    pixels: np.ndarray
    timestamp_s: float
    quality: float
    frame_index: int


@dataclass
class _Track:
    track_id: int
    detection: Detection
    kalman: _Kalman
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


class MotionByteTrack:
    """ByteTrack association: Kalman prediction, Hungarian assignment."""

    def __init__(self, config: ExtractionConfig, source_video: str, label: str, next_event_id):
        self.config = config
        self.source_video = source_video
        self.label = label
        self._next_event_id = next_event_id
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._width = 1
        self._height = 1
        self._lost: list[PassageOutcome] = []

    def close(self) -> list[PassageOutcome]:
        lost = [
            self._terminal(track, "lost_before_trigger")
            for track in self._tracks
            if track.hits >= 2 and not track.emitted
        ]
        self._tracks.clear()
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
        event = self.config.event
        high = [item for item in boxes if item.confidence >= event.track_high_conf]
        low = [
            item
            for item in boxes
            if event.low_conf_recovery and event.track_low_conf <= item.confidence < event.track_high_conf
        ]
        matched, used_tracks, used_high = _match(self._tracks, high, timestamp_s, self.config)
        if low:
            rest = [track for index, track in enumerate(self._tracks) if index not in used_tracks]
            extra, _, _ = _match(rest, low, timestamp_s, self.config)
            for local_index, detection in extra:
                track = rest[local_index]
                matched.append((self._tracks.index(track), detection))
                used_tracks.add(self._tracks.index(track))
        for index, detection in matched:
            self._advance(self._tracks[index], detection, frame, frame_index, timestamp_s, high)
        for detection in high:
            if id(detection) in used_high or detection.confidence < event.new_track_conf:
                continue
            self._birth(detection, frame, frame_index, timestamp_s)
        self._drop_missing(timestamp_s)
        return self._emit(timestamp_s) + self._lost

    def _birth(self, detection: Detection, frame: np.ndarray, frame_index: int, timestamp_s: float) -> None:
        track = _Track(
            track_id=self._next_id,
            detection=detection,
            kalman=_Kalman(*detection.center, *self._belt_velocity()),
            first_seen_s=timestamp_s,
            last_seen_s=timestamp_s,
        )
        self._next_id += 1
        axis, line = trigger_line_position(self.config, self._width, self._height)
        along = detection.center[1] if axis == "y" else detection.center[0]
        increasing = _direction_increasing(self.config.event.belt_direction)
        track.armed = along <= line if increasing else along >= line
        self._remember(track, detection, frame, frame_index, timestamp_s, [])
        self._tracks.append(track)

    def _belt_velocity(self) -> tuple[float, float]:
        speeds = [track.kalman.velocity for track in self._tracks if track.hits >= 2]
        if not speeds:
            return (0.0, 0.0)
        stacked = np.stack(speeds)
        return float(np.median(stacked[:, 0])), float(np.median(stacked[:, 1]))

    def _advance(
        self,
        track: _Track,
        detection: Detection,
        frame: np.ndarray,
        frame_index: int,
        timestamp_s: float,
        siblings: list[Detection],
    ) -> None:
        previous = track.detection.center
        track.kalman.predict(max(1e-3, timestamp_s - track.last_seen_s))
        track.kalman.update(*detection.center)
        self._latch(track, previous, detection.center, track.last_seen_s, timestamp_s)
        track.detection = detection
        track.last_seen_s = timestamp_s
        track.missing_since_s = None
        track.hits += 1
        if detection.confidence >= self.config.event.track_high_conf:
            self._remember(track, detection, frame, frame_index, timestamp_s, siblings)

    def _latch(self, track: _Track, previous, current, previous_s: float, timestamp_s: float) -> None:
        if track.emitted or track.latched or not track.armed or track.hits < 1:
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
        track.latched = True
        track.crossing_s = previous_s + alpha * (timestamp_s - previous_s)
        track.crossing_px = (float(cross[0]), float(cross[1]))

    def _remember(self, track, detection, frame, frame_index, timestamp_s, siblings) -> None:
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
        track.shots.append(
            _Shot(detection, pixels, timestamp_s, 0.7 * detection.confidence + 0.3 * sharp, frame_index)
        )
        track.shots = sorted(track.shots, key=lambda item: item.quality, reverse=True)[:5]

    def _drop_missing(self, timestamp_s: float) -> None:
        alive = []
        limit = self.config.event.reentry_time_s
        for track in self._tracks:
            if track.last_seen_s == timestamp_s or track.emitted:
                if not track.emitted:
                    alive.append(track)
                continue
            if track.missing_since_s is None:
                track.missing_since_s = timestamp_s
            if timestamp_s - track.missing_since_s <= limit:
                alive.append(track)
                continue
            if track.hits >= 2 and not track.emitted:
                self._lost.append(self._terminal(track, "lost_before_trigger"))
        self._tracks = alive

    def _emit(self, timestamp_s: float) -> list[PassageOutcome]:
        outcomes = []
        for track in self._tracks:
            if not track.latched or track.emitted or track.hits < 2 or not track.shots:
                continue
            shot = max(track.shots, key=lambda item: item.quality)
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
                    1,
                    shot.detection,
                    shot.quality,
                    create_event_crop(shot.pixels, local, self.config),
                    None,
                    track.first_seen_s,
                    track.crossing_s,
                    track.crossing_px,
                    None,
                    shot.detection.center,
                    None,
                )
            )
            track.emitted = True
        return outcomes

    def _terminal(self, track: _Track, reason: str) -> PassageOutcome:
        return PassageOutcome(
            self._next_event_id(),
            self.source_video,
            self.label,
            "rejected",
            reason,
            0,
            track.last_seen_s,
            track.hits,
            track.detection,
            0.0,
            None,
            None,
            track.first_seen_s,
            track.crossing_s,
            track.crossing_px,
            None,
            track.detection.center,
            None,
        )


def _match(
    tracks: list[_Track],
    detections: list[Detection],
    timestamp_s: float,
    config: ExtractionConfig,
) -> tuple[list[tuple[int, Detection]], set[int], set[int]]:
    live = [(index, track) for index, track in enumerate(tracks) if not track.emitted]
    if not live or not detections:
        return [], set(), set()
    cost = np.full((len(live), len(detections)), _INF)
    for row, (_index, track) in enumerate(live):
        predicted = track.kalman.center_at(max(0.0, timestamp_s - track.last_seen_s))
        for col, detection in enumerate(detections):
            value = _assignment_cost(predicted, track.detection, detection, config)
            if value < _INF:
                cost[row, col] = value
    pairs = []
    used_tracks: set[int] = set()
    used_detections: set[int] = set()
    for row, col in _hungarian(cost):
        if cost[row, col] >= _INF:
            continue
        index = live[row][0]
        used_tracks.add(index)
        used_detections.add(id(detections[col]))
        pairs.append((index, detections[col]))
    return pairs, used_tracks, used_detections


def _assignment_cost(
    predicted: tuple[float, float],
    previous: Detection,
    detection: Detection,
    config: ExtractionConfig,
) -> float:
    axis = "y" if config.event.belt_direction in {"bottom_to_top", "top_to_bottom"} else "x"
    along = 1 if axis == "y" else 0
    cross = 1 - along
    glove = max(previous.height, previous.width, detection.height, detection.width, 1)
    along_error = abs(predicted[along] - detection.center[along])
    cross_error = abs(predicted[cross] - detection.center[cross])
    if along_error > glove or cross_error > 0.75 * glove:
        return _INF
    return along_error / glove + 0.5 * cross_error / glove


class _Kalman:
    """Constant-velocity filter. Position is measured. Velocity is inferred."""

    def __init__(self, cx: float, cy: float, vx: float, vy: float) -> None:
        self.x = np.array([cx, cy, vx, vy], dtype=float)
        self.p = np.diag([25.0, 25.0, 1.0e4, 1.0e4])

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:4].copy()

    def center_at(self, dt: float) -> tuple[float, float]:
        return float(self.x[0] + self.x[2] * dt), float(self.x[1] + self.x[3] * dt)

    def predict(self, dt: float) -> None:
        transition = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        process = np.diag([1.0, 1.0, 25.0, 25.0]) * max(dt, 1e-3)
        self.x = transition @ self.x
        self.p = transition @ self.p @ transition.T + process

    def update(self, cx: float, cy: float) -> None:
        measurement = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        noise = np.diag([9.0, 9.0])
        innovation = np.array([cx, cy]) - measurement @ self.x
        gain = self.p @ measurement.T @ np.linalg.inv(measurement @ self.p @ measurement.T + noise)
        self.x = self.x + gain @ innovation
        self.p = (np.eye(4) - gain @ measurement) @ self.p


def _hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost assignment. Infinite entries are left unmatched."""
    if cost.size == 0:
        return []
    finite = cost[np.isfinite(cost)]
    if finite.size == 0:
        return []
    original = cost
    transposed = cost.shape[0] > cost.shape[1]
    matrix = cost.T.copy() if transposed else cost.copy()
    rows, cols = matrix.shape
    matrix = np.where(np.isfinite(matrix), matrix, finite.max() + 1.0)
    potential_row = np.zeros(rows + 1)
    potential_col = np.zeros(cols + 1)
    matched_row = np.zeros(cols + 1, dtype=int)
    way = np.zeros(cols + 1, dtype=int)
    for row in range(1, rows + 1):
        matched_row[0] = row
        column = 0
        minimum = np.full(cols + 1, np.inf)
        used = np.zeros(cols + 1, dtype=bool)
        while True:
            used[column] = True
            current_row = matched_row[column]
            delta = np.inf
            next_column = 0
            for candidate in range(1, cols + 1):
                if used[candidate]:
                    continue
                value = matrix[current_row - 1, candidate - 1] - potential_row[current_row] - potential_col[candidate]
                if value < minimum[candidate]:
                    minimum[candidate] = value
                    way[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(cols + 1):
                if used[candidate]:
                    potential_row[matched_row[candidate]] += delta
                    potential_col[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = way[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break
    assigned = []
    for column in range(1, cols + 1):
        if matched_row[column] == 0:
            continue
        row_index, col_index = matched_row[column] - 1, column - 1
        if transposed:
            row_index, col_index = col_index, row_index
        if row_index < original.shape[0] and col_index < original.shape[1] and np.isfinite(original[row_index, col_index]):
            assigned.append((int(row_index), int(col_index)))
    return assigned


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

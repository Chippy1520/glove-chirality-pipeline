"""Find why a glove is first seen past the trigger line.

This does not change association. It records every kept box and every
size-rejected box, then asks whether a born-past-line track had a discarded
box upstream in the same lane.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from glove_chirality.config import ExtractionConfig
from glove_chirality.detection.factory import build_detector
from glove_chirality.events import PassageProcessor, trigger_line_position
from glove_chirality.types import Detection


@dataclass
class _BoxNote:
    frame_index: int
    timestamp_s: float
    center: tuple[float, float]
    area: int
    upstream: bool
    kept: bool


def debug_video(
    video_path: str | Path,
    config: ExtractionConfig,
    *,
    label: str = "right",
) -> str:
    video_path = Path(video_path)
    detector = build_detector(config.detector)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    processor = PassageProcessor(detector, config, video_path.name, label)
    notes: list[_BoxNote] = []
    outcomes = []
    frame_index = -1
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            timestamp_s = frame_index / fps
            height, width = frame.shape[:2]
            reader = getattr(detector, "detect_with_diagnostics", None)
            if reader is None:
                kept = detector.detect(frame)
                rejected: list[Detection] = []
            else:
                kept, diagnostics = reader(frame)
                rejected = [item.detection for item in diagnostics.size_rejected if item.box_area_ratio <= config.detector.yolo_max_box_area_ratio]
            _note(notes, kept, True, frame_index, timestamp_s, config, width, height)
            _note(notes, rejected, False, frame_index, timestamp_s, config, width, height)
            result = processor.process(
                frame,
                frame_index,
                timestamp_s,
                detections=kept,
                tracking_only=rejected,
            )
            outcomes.extend(result.outcomes)
        outcomes.extend(processor.close(frame_index / fps if frame_index >= 0 else 0.0))
    finally:
        capture.release()
    tracker = processor._byte_tracker
    events = [] if tracker is None else list(tracker.debug_events)
    return _report(notes, events, config, outcomes)


def _note(notes, boxes, kept, frame_index, timestamp_s, config, width, height) -> None:
    axis, line = trigger_line_position(config, width, height)
    increasing = config.event.belt_direction in {"left_to_right", "top_to_bottom"}
    for box in boxes:
        along = box.center[1] if axis == "y" else box.center[0]
        upstream = along <= line if increasing else along >= line
        notes.append(_BoxNote(frame_index, timestamp_s, box.center, box.area, upstream, kept))


def _report(notes: list[_BoxNote], events: list[dict[str, object]], config: ExtractionConfig, outcomes: list) -> str:
    births = [item for item in events if item["kind"] == "birth" and not item["armed"]]
    explained = 0
    for birth in births:
        if _had_upstream_partial(birth, notes, config):
            explained += 1
    kept = [item for item in notes if item.kept]
    rejected = [item for item in notes if not item.kept]
    lines = [
        f"accepted_crops {sum(1 for item in outcomes if item.accepted)}",
        f"frames_with_boxes {len({item.frame_index for item in notes})}",
        f"kept_boxes {len(kept)}",
        f"kept_upstream {sum(item.upstream for item in kept)}",
        f"size_rejected {len(rejected)}",
        f"size_rejected_upstream {sum(item.upstream for item in rejected)}",
        f"born_past_line {len(births)}",
        f"born_past_line_with_upstream_partial {explained}",
        f"born_past_line_with_no_upstream_box {len(births) - explained}",
    ]
    reasons: dict[str, int] = {}
    for item in events:
        if item["kind"] != "death":
            continue
        reason = str(item["reason"])
        reasons[reason] = reasons.get(reason, 0) + 1
    for reason, count in sorted(reasons.items()):
        lines.append(f"death_{reason} {count}")
    return "\n".join(lines)


def _had_upstream_partial(birth: dict[str, object], notes: list[_BoxNote], config: ExtractionConfig) -> bool:
    center = birth["center"]
    assert isinstance(center, tuple)
    timestamp_s = float(birth["timestamp_s"])
    axis = "y" if config.event.belt_direction in {"bottom_to_top", "top_to_bottom"} else "x"
    along = 1 if axis == "y" else 0
    cross = 1 - along
    for note in notes:
        if note.kept or not note.upstream:
            continue
        if not timestamp_s - 2.0 <= note.timestamp_s <= timestamp_s:
            continue
        glove = max(24.0, note.area ** 0.5)
        if abs(note.center[cross] - center[cross]) <= 0.75 * glove:
            return True
    return False

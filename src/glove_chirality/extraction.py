from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from glove_chirality.config import ExtractionConfig
from glove_chirality.detection import build_detector
from glove_chirality.detection.base import GloveDetector
from glove_chirality.events import (
    PassageOutcome,
    PassageProcessor,
    _crop_bounds,
    _letterbox,
    create_event_crop,
)
from glove_chirality.types import Detection, EventRecord, ExtractedEvent

VIDEO_EXTENSIONS = {".mkv", ".avi", ".mp4", ".mov", ".m4v"}
MANIFEST_FIELDS = [
    "event_id",
    "image_path",
    "label",
    "label_provenance",
    "source_video",
    "frame_index",
    "timestamp_s",
    "x1",
    "y1",
    "x2",
    "y2",
    "detector",
    "detector_confidence",
    "used_segmentation",
    "mask_area_px",
    "mask_bbox_fill_ratio",
    "candidate_count",
    "status",
    "reject_reason",
    "mask_path",
    "quality_score",
    "config_hash",
]
EVENT_REPORT_FIELDS = [
    "event_id",
    "source_video",
    "frame_index",
    "timestamp_s",
    "status",
    "reject_reason",
    "candidate_count",
    "detector",
    "detector_confidence",
    "used_segmentation",
    "mask_area_px",
    "mask_bbox_fill_ratio",
    "x1",
    "y1",
    "x2",
    "y2",
]


@dataclass(frozen=True)
class ExtractionRun:
    events: list[ExtractedEvent]
    records: list[EventRecord]


def config_hash(config: ExtractionConfig) -> str:
    payload = {
        "detector": asdict(config.detector),
        "event": asdict(config.event),
        "detect_every_n_frames": config.runtime.detect_every_n_frames,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def discover_videos(path: str | Path) -> list[Path]:
    source = Path(path)
    if source.is_file():
        if source.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video extension: {source.suffix}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    return sorted(item for item in source.rglob("*") if item.suffix.lower() in VIDEO_EXTENSIONS)


def _crop(frame, box: Detection, padding: float, square: bool):
    """Backward-compatible bbox crop helper used by older callers/tests."""
    x1, y1, x2, y2 = _crop_bounds(frame, box, padding, square)
    return frame[y1:y2, x1:x2]


def _mask_path(
    outcome: PassageOutcome,
    output_dir: Path,
    config: ExtractionConfig,
) -> Path | None:
    detection = outcome.detection
    if not config.event.save_masks or detection is None or detection.polygon is None:
        return None
    path = output_dir / "masks" / f"{outcome.event_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": "polygon_xy_full_frame", "points": detection.polygon}
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return path


def _store_outcome(
    outcome: PassageOutcome,
    output_dir: Path,
    detector_name: str,
    config: ExtractionConfig,
) -> tuple[ExtractedEvent | None, EventRecord]:
    detection = outcome.detection
    record = EventRecord(
        outcome.event_id,
        outcome.source_video,
        outcome.frame_index,
        outcome.timestamp_s,
        outcome.status,
        outcome.reject_reason,
        outcome.candidate_count,
        detection,
        detector_name,
    )
    if not outcome.accepted or detection is None or outcome.crop is None:
        return None, record

    image_dir = output_dir / "images" / outcome.label
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{outcome.event_id}.jpg"
    if not cv2.imwrite(str(image_path), outcome.crop):
        raise RuntimeError(f"Could not write crop: {image_path}")
    if config.event.save_full_frames and outcome.full_frame is not None:
        full_dir = output_dir / "full_frames"
        full_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(full_dir / f"{outcome.event_id}.jpg"), outcome.full_frame)
    mask_path = _mask_path(outcome, output_dir, config)
    event = ExtractedEvent(
        outcome.event_id,
        image_path,
        outcome.source_video,
        outcome.label,
        outcome.frame_index,
        outcome.timestamp_s,
        detection,
        outcome.quality_score,
        detector_name,
        outcome.candidate_count,
        mask_path,
    )
    return event, record


def extract_video_with_report(
    video_path: str | Path,
    output_dir: str | Path,
    label: str = "unknown",
    config: ExtractionConfig | None = None,
    detector: GloveDetector | None = None,
) -> ExtractionRun:
    """Sequential offline adapter around the shared passage processor."""
    config = config or ExtractionConfig()
    config.detector.validate()
    config.event.validate()
    if label not in {"left", "right", "unknown"}:
        raise ValueError("label must be left, right, or unknown")
    video_path, output_dir = Path(video_path), Path(output_dir)
    detector = detector or build_detector(config.detector)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    processor = PassageProcessor(detector, config, video_path.name, label)
    events: list[ExtractedEvent] = []
    records: list[EventRecord] = []
    frame_index = -1
    timestamp_s = 0.0

    def store(outcome: PassageOutcome) -> None:
        event, record = _store_outcome(outcome, output_dir, detector.name, config)
        records.append(record)
        if event is not None:
            events.append(event)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            timestamp_s = frame_index / fps
            run_detection = frame_index % config.runtime.detect_every_n_frames == 0
            result = processor.process(frame, frame_index, timestamp_s, run_detection)
            for outcome in result.outcomes:
                store(outcome)
        for outcome in processor.close(timestamp_s):
            store(outcome)
    finally:
        capture.release()
    return ExtractionRun(events, records)


def extract_video(
    video_path: str | Path,
    output_dir: str | Path,
    label: str = "unknown",
    config: ExtractionConfig | None = None,
) -> list[ExtractedEvent]:
    """Backward-compatible accepted-event API."""
    return extract_video_with_report(video_path, output_dir, label, config).events


def event_rows(
    events: list[ExtractedEvent],
    root: str | Path,
    cfg_hash: str,
) -> list[dict[str, object]]:
    root = Path(root)
    rows = []
    for event in events:
        detection = event.detection
        rows.append(
            {
                "event_id": event.event_id,
                "image_path": event.image_path.relative_to(root).as_posix(),
                "label": event.label,
                "label_provenance": (
                    "known_stream" if event.label != "unknown" else "unlabeled"
                ),
                "source_video": event.source_video,
                "frame_index": event.frame_index,
                "timestamp_s": f"{event.timestamp_s:.6f}",
                "x1": detection.x1,
                "y1": detection.y1,
                "x2": detection.x2,
                "y2": detection.y2,
                "detector": event.detector,
                "detector_confidence": f"{detection.confidence:.6f}",
                "used_segmentation": detection.polygon is not None,
                "mask_area_px": _optional_float(detection.mask_area),
                "mask_bbox_fill_ratio": _optional_float(detection.mask_bbox_fill_ratio),
                "candidate_count": event.candidate_count,
                "status": "accepted",
                "reject_reason": "",
                "mask_path": (
                    event.mask_path.relative_to(root).as_posix() if event.mask_path else ""
                ),
                "quality_score": f"{event.quality_score:.6f}",
                "config_hash": cfg_hash,
            }
        )
    return rows


def event_report_rows(records: list[EventRecord]) -> list[dict[str, object]]:
    rows = []
    for record in records:
        detection = record.detection
        rows.append(
            {
                "event_id": record.event_id,
                "source_video": record.source_video,
                "frame_index": record.frame_index,
                "timestamp_s": f"{record.timestamp_s:.6f}",
                "status": record.status,
                "reject_reason": record.reject_reason,
                "candidate_count": record.candidate_count,
                "detector": record.detector,
                "detector_confidence": (
                    f"{detection.confidence:.6f}" if detection is not None else ""
                ),
                "used_segmentation": detection is not None and detection.polygon is not None,
                "mask_area_px": _optional_float(detection.mask_area if detection else None),
                "mask_bbox_fill_ratio": _optional_float(
                    detection.mask_bbox_fill_ratio if detection else None
                ),
                "x1": detection.x1 if detection else "",
                "y1": detection.y1 if detection else "",
                "x2": detection.x2 if detection else "",
                "y2": detection.y2 if detection else "",
            }
        )
    return rows


def _optional_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def _write_rows(
    rows: list[dict[str, object]],
    path: str | Path,
    fields: list[str],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_manifest(rows: list[dict[str, object]], path: str | Path) -> Path:
    return _write_rows(rows, path, MANIFEST_FIELDS)


def write_event_report(rows: list[dict[str, object]], path: str | Path) -> Path:
    return _write_rows(rows, path, EVENT_REPORT_FIELDS)


__all__ = [
    "ExtractionRun",
    "_crop",
    "_letterbox",
    "config_hash",
    "create_event_crop",
    "discover_videos",
    "event_report_rows",
    "event_rows",
    "extract_video",
    "extract_video_with_report",
    "write_event_report",
    "write_manifest",
]

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from glove_chirality.config import ExtractionConfig
from glove_chirality.detection import build_detector
from glove_chirality.types import Detection, ExtractedEvent

VIDEO_EXTENSIONS = {".mkv", ".avi", ".mp4", ".mov", ".m4v"}
MANIFEST_FIELDS = [
    "event_id", "image_path", "label", "label_provenance", "source_video",
    "frame_index", "timestamp_s", "x1", "y1", "x2", "y2", "detector",
    "quality_score", "config_hash",
]


@dataclass
class _Candidate:
    frame: np.ndarray
    detection: Detection
    frame_index: int
    timestamp_s: float
    quality: float


def config_hash(config: ExtractionConfig) -> str:
    raw = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def discover_videos(path: str | Path) -> list[Path]:
    source = Path(path)
    if source.is_file():
        if source.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video extension: {source.suffix}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)


def _sharpness(frame: np.ndarray, box: Detection) -> float:
    crop = frame[box.y1:box.y2, box.x1:box.x2]
    if crop.size == 0:
        return 0.0
    return float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def _quality(frame: np.ndarray, detection: Detection, config: ExtractionConfig) -> float:
    h, w = frame.shape[:2]
    tx1, ty1, tx2, ty2 = config.detector.trigger_zone
    target_x, target_y = (tx1 + tx2) * w / 2, (ty1 + ty2) * h / 2
    cx, cy = detection.center
    distance = np.hypot(cx - target_x, cy - target_y)
    diagonal = max(1.0, np.hypot((tx2 - tx1) * w, (ty2 - ty1) * h) / 2)
    centrality = max(0.0, 1.0 - distance / diagonal)
    sharpness = min(1.0, _sharpness(frame, detection) / 500.0)
    return float(0.50 * centrality + 0.35 * detection.confidence + 0.15 * sharpness)


def _crop(frame: np.ndarray, box: Detection, padding: float, square: bool) -> np.ndarray:
    h, w = frame.shape[:2]
    cx, cy = box.center
    expanded_w = box.width * (1 + 2 * padding)
    expanded_h = box.height * (1 + 2 * padding)
    if square:
        side = max(1, min(w, h, int(np.ceil(max(expanded_w, expanded_h)))))
        x1 = max(0, min(w - side, round(cx - side / 2)))
        y1 = max(0, min(h - side, round(cy - side / 2)))
        return frame[y1:y1 + side, x1:x1 + side]
    x1 = max(0, int(np.floor(cx - expanded_w / 2)))
    y1 = max(0, int(np.floor(cy - expanded_h / 2)))
    x2 = min(w, int(np.ceil(cx + expanded_w / 2)))
    y2 = min(h, int(np.ceil(cy + expanded_h / 2)))
    return frame[y1:y2, x1:x2]


def extract_video(
    video_path: str | Path,
    output_dir: str | Path,
    label: str = "unknown",
    config: ExtractionConfig | None = None,
) -> list[ExtractedEvent]:
    """Sequentially decode a video and emit one best crop per accepted passage."""
    config = config or ExtractionConfig()
    if label not in {"left", "right", "unknown"}:
        raise ValueError("label must be left, right, or unknown")
    video_path, output_dir = Path(video_path), Path(output_dir)
    detector = build_detector(config.detector)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    image_dir = output_dir / "images" / label
    image_dir.mkdir(parents=True, exist_ok=True)
    full_dir = output_dir / "full_frames"
    if config.event.save_full_frames:
        full_dir.mkdir(parents=True, exist_ok=True)

    events: list[ExtractedEvent] = []
    active = False
    seen = 0
    missing = 0
    cooldown = 0
    best: _Candidate | None = None
    last_detection: Detection | None = None
    frame_index = -1
    h = w = 1

    def finalize() -> None:
        nonlocal active, seen, missing, cooldown, best, last_detection
        if active and best is not None and _sharpness(best.frame, best.detection) >= config.event.min_sharpness:
            sequence = len(events) + 1
            event_id = f"{label}__{video_path.stem}__e{sequence:06d}"
            image_path = image_dir / f"{event_id}.jpg"
            crop = _crop(best.frame, best.detection, config.event.crop_padding, config.event.make_square)
            if crop.size and cv2.imwrite(str(image_path), crop):
                if config.event.save_full_frames:
                    cv2.imwrite(str(full_dir / f"{event_id}.jpg"), best.frame)
                events.append(ExtractedEvent(
                    event_id, image_path, video_path.name, label, best.frame_index,
                    best.timestamp_s, best.detection, best.quality, detector.name,
                ))
        active, seen, missing, best, last_detection = False, 0, 0, None, None
        cooldown = config.event.cooldown_frames

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            h, w = frame.shape[:2]
            cooldown = max(0, cooldown - 1)
            detections = detector.detect(frame)
            chosen: Detection | None = None
            if detections:
                if last_detection is None:
                    chosen = detections[0]
                else:
                    lx, ly = last_detection.center
                    ranked = sorted(detections, key=lambda d: np.hypot(d.center[0] - lx, d.center[1] - ly))
                    distance = np.hypot(ranked[0].center[0] - lx, ranked[0].center[1] - ly)
                    if distance <= config.event.max_track_distance_ratio * np.hypot(w, h):
                        chosen = ranked[0]
            if chosen is not None and (active or cooldown == 0):
                seen += 1
                missing = 0
                last_detection = chosen
                if seen >= config.event.min_detected_frames:
                    active = True
                score = _quality(frame, chosen, config)
                candidate = _Candidate(frame.copy(), chosen, frame_index, frame_index / fps, score)
                if best is None or candidate.quality > best.quality:
                    best = candidate
            elif active:
                missing += 1
                if missing >= config.event.exit_missing_frames:
                    finalize()
            elif seen:
                seen, best, last_detection = 0, None, None
        if active:
            finalize()
    finally:
        capture.release()
    return events


def event_rows(events: list[ExtractedEvent], root: str | Path, cfg_hash: str) -> list[dict[str, object]]:
    root = Path(root)
    rows = []
    for event in events:
        d = event.detection
        rows.append({
            "event_id": event.event_id,
            "image_path": event.image_path.relative_to(root).as_posix(),
            "label": event.label,
            "label_provenance": "known_stream" if event.label != "unknown" else "unlabeled",
            "source_video": event.source_video,
            "frame_index": event.frame_index,
            "timestamp_s": f"{event.timestamp_s:.6f}",
            "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
            "detector": event.detector,
            "quality_score": f"{event.quality_score:.6f}",
            "config_hash": cfg_hash,
        })
    return rows


def write_manifest(rows: list[dict[str, object]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path

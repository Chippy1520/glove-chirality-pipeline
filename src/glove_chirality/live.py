from __future__ import annotations

import json
import queue
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np

from glove_chirality.camera import open_camera
from glove_chirality.config import ExtractionConfig
from glove_chirality.detection import build_detector
from glove_chirality.events import FrameResult, PassageOutcome, PassageProcessor
from glove_chirality.inference import TorchClassifier


@dataclass(frozen=True)
class CapturedFrame:
    index: int
    captured_at: float
    image: np.ndarray


class LatestFrameCapture:
    """Background OpenCV capture that drops stale frames when its queue is full."""

    def __init__(self, source: int | str, queue_size: int = 2):
        self.source = source
        self._first_frame: np.ndarray | None = None
        if isinstance(source, int):
            opened = open_camera(source, capture_factory=cv2.VideoCapture)
            self.capture = opened.capture
            self._first_frame = opened.first_frame
            print(
                f"camera index={source} backend={opened.backend} "
                f"resolution={opened.width}x{opened.height} fps={opened.fps:.2f}",
                file=sys.stderr,
            )
        else:
            self.capture = cv2.VideoCapture(source)
            if not self.capture.isOpened():
                raise RuntimeError(f"Could not open live source: {source}")
        self.queue: queue.Queue[CapturedFrame] = queue.Queue(maxsize=queue_size)
        self.stop_requested = threading.Event()
        self.finished = threading.Event()
        self.captured_frames = 0
        self.dropped_frames = 0
        self._thread: threading.Thread | None = None

    def start(self) -> LatestFrameCapture:
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def _capture_loop(self) -> None:
        def enqueue(frame: np.ndarray) -> None:
            packet = CapturedFrame(self.captured_frames, time.monotonic(), frame)
            self.captured_frames += 1
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                    self.dropped_frames += 1
                except queue.Empty:
                    pass
            try:
                self.queue.put_nowait(packet)
            except queue.Full:
                self.dropped_frames += 1

        try:
            if self._first_frame is not None:
                enqueue(self._first_frame)
                self._first_frame = None
            while not self.stop_requested.is_set():
                ok, frame = self.capture.read()
                if not ok:
                    break
                enqueue(frame)
        finally:
            self.capture.release()
            self.finished.set()

    def read(self, timeout: float = 0.1) -> CapturedFrame | None:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def exhausted(self) -> bool:
        return self.finished.is_set() and self.queue.empty()

    def stop(self) -> None:
        self.stop_requested.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class JsonlEventSink:
    """Machine-readable event sink; callbacks can replace it for future integrations."""

    def __init__(self, output: str | Path | None):
        self._owned = output not in {None, "-"}
        if self._owned:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.stream: TextIO = path.open("w", encoding="utf-8")
        else:
            self.stream = sys.stdout

    def emit(self, payload: dict[str, object]) -> None:
        self.stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.stream.flush()

    def close(self) -> None:
        if self._owned:
            self.stream.close()


@dataclass
class LiveMetrics:
    captured_frames: int = 0
    processed_frames: int = 0
    dropped_frames: int = 0
    accepted_passages: int = 0
    rejected_passages: int = 0


class _RollingMetrics:
    def __init__(self):
        self.yolo_ms: deque[float] = deque(maxlen=100)
        self.event_ms: deque[float] = deque(maxlen=100)
        self.classifier_ms: deque[float] = deque(maxlen=100)
        self.accepted_latency_ms: deque[float] = deque(maxlen=100)

    @staticmethod
    def average(values: deque[float]) -> float:
        return sum(values) / len(values) if values else 0.0


def parse_capture_source(source: str | int) -> int | str:
    if isinstance(source, int):
        return source
    stripped = source.strip()
    return int(stripped) if stripped.isdigit() else stripped


def _wall_iso(relative_s: float | None, wall_start: datetime) -> str | None:
    if relative_s is None:
        return None
    return (wall_start + timedelta(seconds=float(relative_s))).isoformat(timespec="milliseconds")


def _pair(value) -> list[float] | None:
    if not value:
        return None
    return [float(value[0]), float(value[1])]


def _event_payload(
    outcome: PassageOutcome,
    prediction: str | None = None,
    confidence: float | None = None,
    *,
    wall_start: datetime | None = None,
    session_id: str | None = None,
    checkpoint: str | None = None,
    model_name: str | None = None,
    config_path: str | None = None,
    classifier_ms: float | None = None,
) -> dict[str, object]:
    detection = outcome.detection
    timestamp_s = round(float(outcome.timestamp_s), 6)
    passage_started = getattr(outcome, "passage_started_s", None)
    crossing_s = getattr(outcome, "trigger_crossing_s", None)
    payload: dict[str, object] = {
        "event_id": outcome.event_id,
        "timestamp": timestamp_s,
        "timestamp_s": timestamp_s,
        "wall_time_iso": _wall_iso(outcome.timestamp_s, wall_start) if wall_start else None,
        "passage_started_s": passage_started,
        "passage_started_wall_time_iso": _wall_iso(passage_started, wall_start) if wall_start else None,
        "trigger_crossing_s": crossing_s,
        "trigger_crossing_wall_time_iso": _wall_iso(crossing_s, wall_start) if wall_start else None,
        "trigger_crossing_position_px": _pair(getattr(outcome, "trigger_crossing_position_px", None)),
        "trigger_crossing_position_norm": _pair(
            getattr(outcome, "trigger_crossing_position_norm", None)
        ),
        "prediction": prediction,
        "confidence": None if prediction is None else confidence,
        "detector_confidence": detection.confidence if detection else None,
        "bbox": (
            [detection.x1, detection.y1, detection.x2, detection.y2] if detection else None
        ),
        "center_px": _pair(getattr(outcome, "center_px", None)),
        "center_norm": _pair(getattr(outcome, "center_norm", None)),
        "used_segmentation": detection is not None and detection.polygon is not None,
        "mask_area_px": detection.mask_area if detection else None,
        "candidate_count": outcome.candidate_count,
        "status": outcome.status,
        "reject_reason": outcome.reject_reason,
        "session_id": session_id,
        "model_name": model_name,
        "checkpoint": checkpoint,
        "extraction_config": config_path,
        "classifier_ms": None if classifier_ms is None else round(float(classifier_ms), 3),
    }
    return payload


def _metrics_snapshot(metrics: LiveMetrics, rolling: _RollingMetrics, started: float, capture) -> dict:
    elapsed = max(time.monotonic() - started, 1e-9)
    metrics.captured_frames = capture.captured_frames
    metrics.dropped_frames = capture.dropped_frames
    return {
        "capture_fps": metrics.captured_frames / elapsed,
        "processed_fps": metrics.processed_frames / elapsed,
        "yolo_ms": rolling.average(rolling.yolo_ms),
        "event_ms": rolling.average(rolling.event_ms),
        "classifier_ms": rolling.average(rolling.classifier_ms),
        "accepted_latency_ms": rolling.average(rolling.accepted_latency_ms),
        "captured_frames": metrics.captured_frames,
        "processed_frames": metrics.processed_frames,
        "dropped_frames": metrics.dropped_frames,
        "accepted_passages": metrics.accepted_passages,
        "rejected_passages": metrics.rejected_passages,
    }


def run_live_inference(
    source: str | int,
    checkpoint: str | Path,
    config: ExtractionConfig,
    device: str = "auto",
    amp: bool = False,
    output: str | Path | None = None,
    decision_class: str = "argmax",
    decision_threshold: float = 0.5,
    event_callback: Callable[[dict[str, object]], None] | None = None,
    max_processed_frames: int | None = None,
    detector=None,
    classifier=None,
    capture=None,
    frame_callback: Callable[[np.ndarray, FrameResult, float], None] | None = None,
    metrics_callback: Callable[[dict], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
    stop_event: threading.Event | None = None,
    session_id: str | None = None,
    config_path: str | None = None,
    should_classify: Callable[[], bool] | None = None,
) -> LiveMetrics:
    """Run event-driven inference with one classifier call per accepted passage."""
    detector = detector or build_detector(config.detector)
    classifier = classifier or TorchClassifier(
        checkpoint,
        device=device,
        amp=amp,
        decision_class=decision_class,
        decision_threshold=decision_threshold,
    )
    capture = capture or LatestFrameCapture(
        parse_capture_source(source),
        config.runtime.capture_queue_size,
    )
    capture.start()
    sink = JsonlEventSink(output) if event_callback is None else None
    emit = event_callback or sink.emit
    source_name = f"camera_{source}" if str(source).isdigit() else Path(str(source)).stem
    processor = PassageProcessor(detector, config, source_name, "live")
    metrics = LiveMetrics()
    rolling = _RollingMetrics()
    started = time.monotonic()
    wall_start = datetime.now().astimezone()
    session_id = session_id or wall_start.strftime("live_%Y%m%d_%H%M%S")
    checkpoint_text = None if checkpoint in {None, ""} else str(checkpoint)
    model_name = getattr(classifier, "model_name", None)
    last_report = started
    warmed = False
    announced = False
    processed_sequence = 0
    last_timestamp = 0.0

    def classify_now() -> bool:
        return True if should_classify is None else bool(should_classify())

    def handle(outcome: PassageOutcome, now_relative: float) -> None:
        prediction = None
        confidence = None
        classifier_ms = None
        if outcome.accepted and classify_now():
            classifier_start = time.perf_counter()
            prediction, confidence = classifier.predict_array(outcome.crop)
            classifier_ms = (time.perf_counter() - classifier_start) * 1000.0
            rolling.classifier_ms.append(classifier_ms)
            metrics.accepted_passages += 1
            if outcome.passage_started_s is not None:
                rolling.accepted_latency_ms.append(
                    max(0.0, now_relative - outcome.passage_started_s) * 1000.0
                )
        elif outcome.accepted:
            metrics.accepted_passages += 1
        else:
            metrics.rejected_passages += 1
        payload = _event_payload(
            outcome,
            prediction,
            confidence,
            wall_start=wall_start,
            session_id=session_id,
            checkpoint=checkpoint_text,
            model_name=model_name,
            config_path=config_path,
            classifier_ms=classifier_ms,
        )
        if event_callback is not None and outcome.crop is not None:
            payload["crop"] = outcome.crop
        emit(payload)

    def publish_metrics() -> None:
        if metrics_callback is not None:
            metrics_callback(_metrics_snapshot(metrics, rolling, started, capture))

    try:
        while stop_event is None or not stop_event.is_set():
            packet = capture.read(timeout=0.1)
            if packet is None:
                if capture.exhausted or (stop_event is not None and stop_event.is_set()):
                    break
                continue
            if not warmed and config.runtime.warmup:
                if status_callback is not None:
                    status_callback("Warming models...")
                detector.warmup(np.zeros_like(packet.image))
                classifier.warmup()
                warmed = True
            if not announced and status_callback is not None:
                status_callback("RUNNING")
                announced = True
            timestamp_s = packet.captured_at - started
            last_timestamp = timestamp_s
            run_detection = processed_sequence % config.runtime.detect_every_n_frames == 0
            result = processor.process(
                packet.image,
                packet.index,
                timestamp_s,
                run_detection,
            )
            processed_sequence += 1
            metrics.processed_frames += 1
            rolling.yolo_ms.append(result.detector_latency_ms)
            rolling.event_ms.append(result.event_latency_ms)
            if frame_callback is not None:
                frame_callback(packet.image, result, timestamp_s)
            now_relative = time.monotonic() - started
            for outcome in result.outcomes:
                handle(outcome, now_relative)
            publish_metrics()

            now = time.monotonic()
            if now - last_report >= config.runtime.report_interval_seconds:
                snapshot = _metrics_snapshot(metrics, rolling, started, capture)
                print(
                    "live "
                    f"capture_fps={snapshot['capture_fps']:.1f} "
                    f"processed_fps={snapshot['processed_fps']:.1f} "
                    f"yolo_ms={snapshot['yolo_ms']:.1f} "
                    f"event_ms={snapshot['event_ms']:.2f} "
                    f"classifier_ms={snapshot['classifier_ms']:.1f} "
                    f"accepted_latency_ms={snapshot['accepted_latency_ms']:.1f} "
                    f"dropped={snapshot['dropped_frames']} "
                    f"accepted={snapshot['accepted_passages']} "
                    f"rejected={snapshot['rejected_passages']}",
                    file=sys.stderr,
                    flush=True,
                )
                last_report = now
            if max_processed_frames is not None and metrics.processed_frames >= max_processed_frames:
                break

        now_relative = time.monotonic() - started
        for outcome in processor.close(last_timestamp):
            handle(outcome, now_relative)
        publish_metrics()
    except KeyboardInterrupt:
        now_relative = time.monotonic() - started
        for outcome in processor.close(last_timestamp):
            handle(outcome, now_relative)
    finally:
        capture.stop()
        if sink is not None:
            sink.close()
    metrics.captured_frames = capture.captured_frames
    metrics.dropped_frames = capture.dropped_frames
    return metrics

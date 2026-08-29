from __future__ import annotations

import json
import queue
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np

from glove_chirality.camera import open_camera
from glove_chirality.config import ExtractionConfig
from glove_chirality.detection import build_detector
from glove_chirality.events import PassageOutcome, PassageProcessor
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


def _event_payload(
    outcome: PassageOutcome,
    prediction: str | None = None,
    confidence: float | None = None,
) -> dict[str, object]:
    detection = outcome.detection
    payload: dict[str, object] = {
        "event_id": outcome.event_id,
        "timestamp": round(outcome.timestamp_s, 6),
        "prediction": prediction,
        "confidence": confidence,
        "detector_confidence": detection.confidence if detection else None,
        "bbox": (
            [detection.x1, detection.y1, detection.x2, detection.y2]
            if detection
            else None
        ),
        "used_segmentation": detection is not None and detection.polygon is not None,
        "mask_area_px": detection.mask_area if detection else None,
        "candidate_count": outcome.candidate_count,
        "status": outcome.status,
        "reject_reason": outcome.reject_reason,
    }
    return payload


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
    last_report = started
    warmed = False
    processed_sequence = 0
    last_timestamp = 0.0

    def handle(outcome: PassageOutcome, now_relative: float) -> None:
        if outcome.accepted:
            classifier_start = time.perf_counter()
            prediction, confidence = classifier.predict_array(outcome.crop)
            rolling.classifier_ms.append((time.perf_counter() - classifier_start) * 1000.0)
            metrics.accepted_passages += 1
            if outcome.passage_started_s is not None:
                rolling.accepted_latency_ms.append(
                    max(0.0, now_relative - outcome.passage_started_s) * 1000.0
                )
            emit(_event_payload(outcome, prediction, confidence))
        else:
            metrics.rejected_passages += 1
            emit(_event_payload(outcome))

    try:
        while True:
            packet = capture.read(timeout=0.1)
            if packet is None:
                if capture.exhausted:
                    break
                continue
            if not warmed and config.runtime.warmup:
                detector.warmup(np.zeros_like(packet.image))
                classifier.warmup()
                warmed = True
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
            now_relative = time.monotonic() - started
            for outcome in result.outcomes:
                handle(outcome, now_relative)

            now = time.monotonic()
            if now - last_report >= config.runtime.report_interval_seconds:
                elapsed = max(now - started, 1e-9)
                metrics.captured_frames = capture.captured_frames
                metrics.dropped_frames = capture.dropped_frames
                print(
                    "live "
                    f"capture_fps={metrics.captured_frames / elapsed:.1f} "
                    f"processed_fps={metrics.processed_frames / elapsed:.1f} "
                    f"yolo_ms={rolling.average(rolling.yolo_ms):.1f} "
                    f"event_ms={rolling.average(rolling.event_ms):.2f} "
                    f"classifier_ms={rolling.average(rolling.classifier_ms):.1f} "
                    f"accepted_latency_ms={rolling.average(rolling.accepted_latency_ms):.1f} "
                    f"dropped={metrics.dropped_frames} "
                    f"accepted={metrics.accepted_passages} "
                    f"rejected={metrics.rejected_passages}",
                    file=sys.stderr,
                    flush=True,
                )
                last_report = now
            if max_processed_frames is not None and metrics.processed_frames >= max_processed_frames:
                break

        now_relative = time.monotonic() - started
        for outcome in processor.close(last_timestamp):
            handle(outcome, now_relative)
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

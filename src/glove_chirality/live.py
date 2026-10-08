from __future__ import annotations

import json
import queue
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
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
from glove_chirality.types import Detection


@dataclass(frozen=True)
class CapturedFrame:
    index: int
    captured_at: float
    image: np.ndarray
    received_at: float | None = None


@dataclass(frozen=True)
class _DetectedFrame:
    packet: CapturedFrame
    detections: tuple[Detection, ...]
    tracking_only: tuple[Detection, ...]
    detector_ms: float
    ready_at: float
    diagnostics: object | None
    detection_ran: bool = True


class _LatestObserver:
    """One in-flight callback plus one latest observation; never blocks tracking."""

    def __init__(self, frame_callback, metrics_callback, on_error):
        self.frame_callback = frame_callback
        self.metrics_callback = metrics_callback
        self.on_error = on_error
        self.condition = threading.Condition()
        self.frame = None
        self.snapshot = None
        self.done = False
        self.skipped = 0
        self.error = None
        self.thread = threading.Thread(target=self._run, name="pipeline-observer", daemon=True)

    def offer(self, frame=None, snapshot=None):
        with self.condition:
            if frame is not None and self.frame_callback is not None:
                if self.frame is not None:
                    self.skipped += 1
                self.frame = frame
            if snapshot is not None:
                self.snapshot = snapshot
            self.condition.notify()

    def _run(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.done or self.frame is not None or self.snapshot is not None)
                    frame, snapshot = self.frame, self.snapshot
                    self.frame = self.snapshot = None
                    if frame is None and snapshot is None and self.done:
                        return
                if frame is not None:
                    self.frame_callback(*frame)
                if snapshot is not None:
                    self.metrics_callback(snapshot)
        except Exception as exc:  # noqa: BLE001 - observer faults must stop the session
            self.error = exc
            self.on_error()

    def close(self):
        with self.condition:
            self.done = True
            self.condition.notify()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            self.on_error()
            raise RuntimeError("Pipeline observer did not stop")
        if self.error is not None:
            raise self.error


class LatestFrameCapture:
    """Background OpenCV capture that drops stale frames when its queue is full."""

    def __init__(self, source: int | str, queue_size: int = 2, *, camera_mode: dict | None = None):
        self.source = source
        self.camera_mode = dict(camera_mode or {})
        self.opened = None
        self._first_frame: np.ndarray | None = None
        if isinstance(source, int):
            opened = open_camera(
                source,
                capture_factory=cv2.VideoCapture,
                preferred_backend=self.camera_mode.get("backend"),
                requested_width=self.camera_mode.get("width"),
                requested_height=self.camera_mode.get("height"),
                requested_fps=self.camera_mode.get("fps"),
                preferred_fourcc=self.camera_mode.get("fourcc"),
                low_latency=True,
            )
            self.opened = opened
            self.capture = opened.capture
            self._first_frame = opened.first_frame
            print(
                f"camera index={source} backend={opened.backend} "
                f"requested={opened.requested_width}x{opened.requested_height} "
                f"actual={opened.width}x{opened.height} fps={opened.fps:.2f} "
                f"fourcc={opened.actual_fourcc or opened.requested_fourcc or 'auto'}",
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


class SequentialVideoCapture:
    """Lossless file playback; media timestamps are independent of playback speed."""

    def __init__(self, source, *, capture_factory=None):
        self.capture = (capture_factory or cv2.VideoCapture)(str(source))
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError("Could not open video source")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(self.fps) or self.fps <= 0:
            self.capture.release()
            raise ValueError("Video source must report a positive finite frame rate")
        self.captured_frames = 0
        self.dropped_frames = 0
        self.finished = threading.Event()
        self._condition = threading.Condition()
        self._paused = False
        self._speed = 1.0
        self._due = None
        self.time_origin = time.monotonic()
        self._last_media_time = -1.0

    def start(self):
        self.time_origin = time.monotonic()
        return self

    def playback(self, pause=None, speed=None):
        if pause is not None and not isinstance(pause, bool):
            raise ValueError("pause must be a boolean")
        if speed is not None and (isinstance(speed, bool) or speed not in {0.25, 0.5, 1.0, 2.0}):
            raise ValueError("speed must be 0.25, 0.5, 1, or 2")
        with self._condition:
            if pause is not None:
                self._paused = pause
            if speed is not None:
                self._speed = float(speed)
            self._due = None
            self._condition.notify_all()
            return {"paused": self._paused, "speed": self._speed}

    def read(self, timeout=0.1):
        with self._condition:
            if self.finished.is_set():
                return None
            if self._paused:
                self._condition.wait(timeout)
                return None
            if self._due is not None:
                remaining = self._due - time.monotonic()
                if remaining > 0:
                    self._condition.wait(min(timeout, remaining))
                    return None
            ok, image = self.capture.read()
            if not ok or image is None:
                self.finished.set()
                self.capture.release()
                return None
            index = self.captured_frames
            self.captured_frames += 1
            self._due = time.monotonic() + 1.0 / (self.fps * self._speed)
            media_time = float(self.capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if not np.isfinite(media_time) or media_time < 0 or (index > 0 and media_time <= self._last_media_time):
                media_time = max(index / self.fps, self._last_media_time + 1.0 / self.fps)
            if index == 0:
                media_time = 0.0
            self._last_media_time = media_time
            return CapturedFrame(index, self.time_origin + media_time, image, time.monotonic())

    @property
    def exhausted(self):
        return self.finished.is_set()

    def stop(self):
        with self._condition:
            self.finished.set()
            self._condition.notify_all()
            self.capture.release()


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
    source_exhausted: bool = False
    processed_frames: int = 0
    dropped_frames: int = 0
    accepted_passages: int = 0
    rejected_passages: int = 0
    classifier_overload: int = 0
    classifier_queue_depth: int = 0


class _RollingMetrics:
    def __init__(self):
        self.yolo_ms: deque[float] = deque(maxlen=100)
        self.event_ms: deque[float] = deque(maxlen=100)
        self.classifier_ms: deque[float] = deque(maxlen=100)
        self.accepted_latency_ms: deque[float] = deque(maxlen=100)
        self.classifier_queue_wait_ms: deque[float] = deque(maxlen=100)
        self.detection_queue_wait_ms: deque[float] = deque(maxlen=100)
        self.frame_age_ms: deque[float] = deque(maxlen=100)

    @staticmethod
    def average(values: deque[float]) -> float:
        return sum(values) / len(values) if values else 0.0


LAYER2_QUEUE_WARN = 4


def note_classifier_backlog(waiting: int, metrics: LiveMetrics) -> None:
    """Count a deep Layer 3 queue. Never silently discard a passage."""
    if waiting >= LAYER2_QUEUE_WARN:
        metrics.classifier_overload += 1


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
        "classifier_queue_wait_ms": None,
        "crossing_to_decision_ms": None,
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
        "layer1_ms": rolling.average(rolling.yolo_ms),
        "layer2_ms": rolling.average(rolling.event_ms),
        "layer3_ms": rolling.average(rolling.classifier_ms),
        "detection_queue_wait_ms": rolling.average(rolling.detection_queue_wait_ms),
        "frame_age_ms": rolling.average(rolling.frame_age_ms),
        "classifier_ms": rolling.average(rolling.classifier_ms),
        "accepted_latency_ms": rolling.average(rolling.accepted_latency_ms),
        "classifier_queue_wait_ms": rolling.average(rolling.classifier_queue_wait_ms),
        "classifier_queue_depth": metrics.classifier_queue_depth,
        "classifier_overload": metrics.classifier_overload,
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
    before_classify: Callable[[], None] | None = None,
    stop_event: threading.Event | None = None,
    session_id: str | None = None,
    config_path: str | None = None,
    should_classify: Callable[[], bool] | None = None,
    on_stage: Callable[[str, str], None] | None = None,
    on_firstinference: Callable[[], None] | None = None,
    classify_callback=None,
) -> LiveMetrics:
    """Segmentation -> ordered passage/crop extraction -> event-rate classification."""
    detector = detector or build_detector(config.detector)
    classifier = classifier or TorchClassifier(
        checkpoint, device=device, amp=amp, decision_class=decision_class,
        decision_threshold=decision_threshold,
    )
    capture = capture or LatestFrameCapture(
        parse_capture_source(source), config.runtime.capture_queue_size,
    )
    capture.start()
    sink = JsonlEventSink(output) if event_callback is None else None
    emit = event_callback or sink.emit
    source_name = "stream" if "://" in str(source) else (f"camera_{source}" if str(source).isdigit() else Path(str(source)).stem)
    processor = PassageProcessor(detector, config, source_name, "live")
    metrics, rolling = LiveMetrics(), _RollingMetrics()
    metrics_lock = threading.Lock()
    classify_queue: queue.Queue = queue.Queue(maxsize=config.runtime.classifier_queue_size)
    detect_queue: queue.Queue[_DetectedFrame | None] = queue.Queue(maxsize=2)
    pipeline_stop = threading.Event()
    detect_error, classify_error = [], []
    started = time.monotonic()
    wall_start = datetime.now().astimezone()
    session_id = session_id or wall_start.strftime("live_%Y%m%d_%H%M%S")
    checkpoint_text = None if checkpoint in {None, ""} else str(checkpoint)
    model_name = getattr(classifier, "model_name", None)
    last_report = started
    last_timestamp = 0.0
    processed_sequence = 0
    announced = False

    def fail_pipeline() -> None:
        pipeline_stop.set()
        if stop_event is not None:
            stop_event.set()

    def notify_metrics(snapshot: dict) -> None:
        nonlocal last_report
        if metrics_callback is not None:
            metrics_callback(snapshot)
        now = time.monotonic()
        if now - last_report >= config.runtime.report_interval_seconds:
            print(
                "live "
                f"capture_fps={snapshot['capture_fps']:.1f} "
                f"processed_fps={snapshot['processed_fps']:.1f} "
                f"layer1_ms={snapshot['layer1_ms']:.1f} "
                f"layer2_ms={snapshot['layer2_ms']:.2f} "
                f"layer3_ms={snapshot['layer3_ms']:.1f} "
                f"frame_age_ms={snapshot['frame_age_ms']:.1f} "
                f"dropped={snapshot['dropped_frames']} "
                f"accepted={snapshot['accepted_passages']} "
                f"rejected={snapshot['rejected_passages']}",
                file=sys.stderr, flush=True,
            )
            last_report = now

    observer = _LatestObserver(frame_callback, notify_metrics, fail_pipeline)

    def metrics_snapshot() -> dict:
        with metrics_lock:
            metrics.classifier_queue_depth = classify_queue.qsize()
            result = _metrics_snapshot(metrics, rolling, started, capture)
        result.update(observer_frames_skipped=observer.skipped,
                      detection_queue_depth=detect_queue.qsize())
        return result

    def check_errors() -> None:
        for error in [*detect_error, *classify_error, observer.error]:
            if error is not None:
                raise error

    def emit_outcome(outcome, prediction=None, confidence=None, classifier_ms=None,
                     queue_wait_ms=None, controls=None):
        completed = time.monotonic() - started
        anchor = outcome.trigger_crossing_s
        if anchor is None:
            anchor = outcome.passage_started_s
        latency = None
        if outcome.accepted and classifier_ms is not None and anchor is not None:
            latency = max(0.0, completed - float(anchor)) * 1000
        with metrics_lock:
            if outcome.accepted:
                metrics.accepted_passages += 1
                if classifier_ms is not None:
                    rolling.classifier_ms.append(classifier_ms)
                if latency is not None:
                    rolling.accepted_latency_ms.append(latency)
                if queue_wait_ms is not None:
                    rolling.classifier_queue_wait_ms.append(queue_wait_ms)
            else:
                metrics.rejected_passages += 1
        payload = _event_payload(
            outcome, prediction, confidence, wall_start=wall_start, session_id=session_id,
            checkpoint=checkpoint_text, model_name=model_name, config_path=config_path,
            classifier_ms=classifier_ms,
        )
        if queue_wait_ms is not None:
            payload["classifier_queue_wait_ms"] = round(float(queue_wait_ms), 3)
        if latency is not None:
            payload["crossing_to_decision_ms"] = round(latency, 3)
        if event_callback is not None and outcome.crop is not None:
            payload["crop"] = outcome.crop
        if controls is not None:
            payload.update(controls)
        emit(payload)

    def classify_worker():
        try:
            if on_stage is not None:
                on_stage("classifier_warmup", "running")
            if config.runtime.warmup or on_stage is not None:
                classifier.warmup()
            if on_stage is not None:
                on_stage("classifier_warmup", "passed")
            while True:
                item = classify_queue.get()
                try:
                    if item is None:
                        return
                    outcome, queued_at, run_classifier = item
                    if not run_classifier:
                        emit_outcome(outcome)
                        continue
                    if before_classify is not None:
                        before_classify()
                    began = time.monotonic()
                    controls = None
                    duration_start = time.perf_counter()
                    if classify_callback is None:
                        prediction, confidence = classifier.predict_array(outcome.crop)
                    else:
                        prediction, confidence, controls = classify_callback(outcome.crop, classifier)
                    elapsed = (time.perf_counter() - duration_start) * 1000
                    if on_firstinference is not None:
                        on_firstinference()
                    emit_outcome(outcome, prediction, confidence, elapsed,
                                 max(0.0, began - queued_at) * 1000, controls)
                finally:
                    classify_queue.task_done()
        except Exception as exc:  # noqa: BLE001 - propagated to the session owner
            classify_error.append(exc)
            fail_pipeline()

    def handoff(outcome):
        run_classifier = outcome.accepted and outcome.crop is not None and (
            should_classify is None or bool(should_classify())
        )
        with metrics_lock:
            note_classifier_backlog(classify_queue.qsize(), metrics)
        try:
            classify_queue.put_nowait((outcome, time.monotonic(), run_classifier))
        except queue.Full as exc:
            with metrics_lock:
                metrics.classifier_overload += 1
            fail_pipeline()
            raise RuntimeError(f"Layer 3 passage queue is full for event {outcome.event_id}") from exc

    def consume(packet, found=None, elapsed=None, partials=None, diagnostics=None, ready_at=None, detection_ran=True):
        nonlocal last_timestamp, processed_sequence, announced
        last_timestamp = packet.captured_at - getattr(capture, "time_origin", started)
        tracking_start = time.monotonic()
        if not detection_ran:
            result = processor.process(packet.image, packet.index, last_timestamp, run_detection=False)
        elif found is None:
            run_detection = processed_sequence % config.runtime.detect_every_n_frames == 0
            result = processor.process(packet.image, packet.index, last_timestamp, run_detection)
            diagnostics = getattr(detector, "last_diagnostics", None) if run_detection else None
        else:
            result = processor.process(packet.image, packet.index, last_timestamp, detections=found,
                                       detector_latency_ms=elapsed, tracking_only=partials)
        processed_at = time.monotonic()
        result = replace(result, detector_diagnostics=diagnostics,
                         processed_at_monotonic=time.perf_counter())
        processed_sequence += 1
        with metrics_lock:
            metrics.processed_frames += 1
            rolling.yolo_ms.append(result.detector_latency_ms)
            rolling.event_ms.append(result.event_latency_ms)
            acquired_at = packet.received_at if packet.received_at is not None else packet.captured_at
            rolling.frame_age_ms.append(max(0.0, processed_at - acquired_at) * 1000)
            if ready_at is not None:
                rolling.detection_queue_wait_ms.append(max(0.0, tracking_start - ready_at) * 1000)
        if not announced:
            announced = True
            if on_stage is not None:
                on_stage("first_frame", "passed")
                on_stage("detector_inference", "passed")
            if status_callback is not None:
                status_callback("RUNNING")
        for outcome in result.outcomes:
            handoff(outcome)
        observer.offer(frame=(packet.image, result, last_timestamp), snapshot=metrics_snapshot())
        return max_processed_frames is None or processed_sequence < max_processed_frames

    def stopped():
        return pipeline_stop.is_set() or (stop_event is not None and stop_event.is_set())

    def put_detection(item):
        while not stopped():
            try:
                detect_queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def detect_worker():
        sequence = 0
        warmed = False
        try:
            while not stopped():
                packet = capture.read(timeout=0.1)
                if packet is None:
                    if capture.exhausted:
                        break
                    continue
                if config.runtime.warmup and not warmed:
                    detector.warmup(np.zeros_like(packet.image))
                    warmed = True
                detection_ran = sequence % config.runtime.detect_every_n_frames == 0
                if detection_ran:
                    began = time.perf_counter()
                    found = tuple(detector.detect(packet.image))
                    elapsed = (time.perf_counter() - began) * 1000
                    reader = getattr(detector, "tracking_partials", None)
                    partials = () if reader is None else tuple(reader())
                    diagnostics = getattr(detector, "last_diagnostics", None)
                else:
                    found, elapsed, partials, diagnostics = (), 0.0, (), None
                sequence += 1
                put_detection(_DetectedFrame(packet, found, partials, elapsed,
                                             time.monotonic(), diagnostics, detection_ran))
        except Exception as exc:  # noqa: BLE001 - propagated to the session owner
            detect_error.append(exc)
            fail_pipeline()
        finally:
            put_detection(None)

    classify_thread = threading.Thread(target=classify_worker, name="layer3-classifier-output", daemon=True)
    detect_thread = None
    observer.thread.start()
    classify_thread.start()
    try:
        if config.runtime.stage_pipeline:
            detect_thread = threading.Thread(target=detect_worker, name="layer1-segmentation", daemon=True)
            detect_thread.start()
            # This session owner is the sole ordered Layer 2 state/crop worker.
            while not stopped():
                check_errors()
                try:
                    item = detect_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    metrics.source_exhausted = True
                    break
                if not consume(item.packet, item.detections, item.detector_ms,
                               item.tracking_only, item.diagnostics, item.ready_at, item.detection_ran):
                    break
        else:
            warmed = False
            while not stopped():
                check_errors()
                packet = capture.read(timeout=0.1)
                if packet is None:
                    if capture.exhausted:
                        metrics.source_exhausted = True
                        break
                    continue
                if config.runtime.warmup and not warmed:
                    detector.warmup(np.zeros_like(packet.image))
                    warmed = True
                if not consume(packet):
                    break
        check_errors()
        if stop_event is not None:
            stop_event.set()
        for outcome in processor.close(last_timestamp):
            handoff(outcome)
    except KeyboardInterrupt:
        fail_pipeline()
        for outcome in processor.close(last_timestamp):
            handoff(outcome)
    except BaseException:
        fail_pipeline()
        raise
    finally:
        pipeline_stop.set()
        if stop_event is not None:
            stop_event.set()
        capture.stop()
        try:
            if detect_thread is not None:
                detect_thread.join(timeout=30)
                if detect_thread.is_alive():
                    fail_pipeline()
                    raise RuntimeError("Layer 1 segmentation worker did not stop")
            drain_deadline = time.monotonic() + 30
            while classify_thread.is_alive():
                if time.monotonic() >= drain_deadline:
                    fail_pipeline()
                    raise RuntimeError("Layer 3 passage queue did not drain")
                try:
                    classify_queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    if classify_error:
                        break
            classify_thread.join(timeout=max(0.0, drain_deadline - time.monotonic()))
            if classify_thread.is_alive():
                fail_pipeline()
                raise RuntimeError("Layer 3 classifier/output worker did not stop")
            observer.offer(snapshot=metrics_snapshot())
        finally:
            try:
                observer.close()
            finally:
                if sink is not None:
                    sink.close()
        check_errors()
    metrics.captured_frames = capture.captured_frames
    metrics.dropped_frames = capture.dropped_frames
    return metrics

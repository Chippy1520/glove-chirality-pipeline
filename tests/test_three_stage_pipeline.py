"""Three-stage ownership, callback isolation and explicit failure regressions."""
import threading
import time
from dataclasses import replace

import numpy as np
import pytest

from glove_chirality.config import DetectorConfig, EventConfig, ExtractionConfig, RuntimeConfig
from glove_chirality.detection.yolo import YoloDetectionDiagnostics
from glove_chirality.events import FrameResult, PassageOutcome, PassageProcessor
from glove_chirality.factory_live import FactoryLiveSession
from glove_chirality.inference import TorchClassifier
from glove_chirality.live import CapturedFrame, run_live_inference
from glove_chirality.types import Detection


class Capture:
    def __init__(self, count=12):
        self.count = count
        self.index = 0
        self.captured_frames = count
        self.dropped_frames = 0
        self.time_origin = time.monotonic()
        self.stopped = False

    def start(self):
        return self

    def read(self, timeout=.1):
        del timeout
        if self.index >= self.count:
            return None
        index = self.index
        self.index += 1
        return CapturedFrame(index, self.time_origin + index * .04,
                             np.full((40, 40, 3), index, dtype=np.uint8), time.monotonic())

    @property
    def exhausted(self):
        return self.index >= self.count

    def stop(self):
        self.stopped = True


class Detector:
    name = "test detector"

    def __init__(self, count=12):
        self.calls = 0
        self.count = count
        self.finished = threading.Event()
        self.last_diagnostics = None
        self.threads = []

    def warmup(self, frame):
        assert frame.shape == (40, 40, 3)

    def detect(self, frame):
        self.threads.append(threading.current_thread().name)
        self.calls += 1
        self.last_diagnostics = YoloDetectionDiagnostics(raw_yolo_count=int(frame[0, 0, 0]))
        if self.calls == self.count:
            self.finished.set()
        return []


class Classifier:
    def __init__(self):
        self.calls = 0
        self.warm_thread = None

    def warmup(self):
        self.warm_thread = threading.current_thread().name

    def predict_array(self, crop):
        self.calls += 1
        return "right", .9


def config(**runtime):
    options = {"warmup": False, "report_interval_seconds": 60, **runtime}
    return ExtractionConfig(
        detector=DetectorConfig(roi=(0, 0, 1, 1), trigger_zone=(0, 0, 1, 1)),
        event=EventConfig(output_size=32),
        runtime=RuntimeConfig(**options),
    )


@pytest.mark.parametrize("callback_kind", ["frame", "metrics"])
def test_slow_observer_does_not_block_segmentation_or_ordered_tracking(monkeypatch, callback_kind):
    entered, release, tracked = threading.Event(), threading.Event(), threading.Event()
    processed = []
    original = PassageProcessor.process


    def callback(*_args):
        entered.set()
        assert release.wait(3)

    # The first callback follows process(0), so let it enter before gating process(1).
    def process_gated(self, frame, index, *args, **kwargs):
        if index == 1:
            assert entered.wait(2)
        processed.append(index)
        result = original(self, frame, index, *args, **kwargs)
        if index == 11:
            tracked.set()
        return result

    monkeypatch.setattr(PassageProcessor, "process", process_gated)
    detector = Detector()
    capture = Capture()
    errors = []
    results = []

    def run():
        try:
            results.append(run_live_inference(
                0, "unused", config(), detector=detector, classifier=Classifier(), capture=capture,
                event_callback=lambda _: None,
                frame_callback=callback if callback_kind == "frame" else None,
                metrics_callback=callback if callback_kind == "metrics" else None,
            ))
        except Exception as exc:  # noqa: BLE001 - collect failures from the test worker
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert entered.wait(2)
        assert tracked.wait(2), "A presentation callback blocked Layer 2"
        assert detector.finished.is_set()
    finally:
        release.set()
        thread.join(4)
    assert not thread.is_alive() and not errors
    assert processed == list(range(12))
    assert results[0].processed_frames == 12
    assert set(detector.threads) == {"layer1-segmentation"}
    assert capture.stopped


def test_latest_frame_callback_keeps_frame_matched_diagnostics(monkeypatch):
    seen = []
    original = PassageProcessor.process

    def slower(self, *args, **kwargs):
        time.sleep(.002)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PassageProcessor, "process", slower)
    detector = Detector()

    def callback(frame, result, _stamp):
        seen.append((int(frame[0, 0, 0]), result.detector_diagnostics.raw_yolo_count))

    run_live_inference(0, "unused", config(), detector=detector, classifier=Classifier(),
                       capture=Capture(), frame_callback=callback, event_callback=lambda _: None)
    assert seen and all(frame_id == diagnostic_id for frame_id, diagnostic_id in seen)
    assert seen[-1] == (11, 11)


def test_classifier_warmup_and_output_have_same_owner(monkeypatch):
    box = Detection(5, 5, 30, 30, .9)
    threads = []

    def process(self, frame, index, _stamp, **_kwargs):
        outcome = PassageOutcome("event", "test", "unknown", "accepted", "", index, .1,
                                 1, box, crop=frame.copy())
        return FrameResult((outcome,) if index == 0 else (), (), 1, .2)

    monkeypatch.setattr(PassageProcessor, "process", process)
    monkeypatch.setattr(PassageProcessor, "close", lambda *_: [])
    classifier = Classifier()
    run_live_inference(0, "unused", config(warmup=True), detector=Detector(2), classifier=classifier,
                       capture=Capture(2), event_callback=lambda _: threads.append(threading.current_thread().name))
    assert classifier.warm_thread == "layer3-classifier-output"
    assert threads == [classifier.warm_thread] and classifier.calls == 1


def test_full_prediction_path_is_used_for_warmup():
    classifier = object.__new__(TorchClassifier)
    seen = []
    classifier.predict_array = lambda crop: seen.append(crop)
    classifier.warmup()
    assert len(seen) == 1 and seen[0].shape == (256, 256, 3)
    assert np.all(seen[0] == 114)


def test_bounded_passage_queue_fault_is_explicit_and_sets_stop(monkeypatch):
    began, release = threading.Event(), threading.Event()
    stop = threading.Event()
    capture = Capture(4)
    box = Detection(5, 5, 30, 30, .9)

    class SlowClassifier(Classifier):
        def predict_array(self, crop):
            began.set()
            assert release.wait(3)
            return "right", .9

    def process(self, frame, index, stamp, **_kwargs):
        if index == 1:
            assert began.wait(2)
        if index == 2:
            threading.Thread(target=lambda: (stop.wait(2), release.set()), daemon=True).start()
        event = PassageOutcome(f"event-{index}", "test", "unknown", "accepted", "", index,
                               stamp, 1, box, crop=frame.copy())
        return FrameResult((event,), (), 1, .2)

    monkeypatch.setattr(PassageProcessor, "process", process)
    monkeypatch.setattr(PassageProcessor, "close", lambda *_: [])
    try:
        with pytest.raises(RuntimeError, match="Layer 3 passage queue is full for event"):
            run_live_inference(0, "unused", config(classifier_queue_size=1), detector=Detector(4),
                               classifier=SlowClassifier(), capture=capture,
                               event_callback=lambda _: None, stop_event=stop)
    finally:
        release.set()
    assert stop.is_set() and capture.stopped


@pytest.mark.parametrize("owner", ["detector", "classifier", "observer"])
def test_worker_failures_stop_capture_and_propagate(owner, monkeypatch):
    capture = Capture(2)
    detector = Detector(2)
    classifier = Classifier()
    stop = threading.Event()

    def fail(*_args):
        raise ValueError(f"{owner} failed")

    if owner == "detector":
        detector.detect = fail
    if owner == "classifier":
        classifier.warmup = fail
    with pytest.raises(ValueError, match=f"{owner} failed"):
        run_live_inference(0, "unused", config(warmup=True), detector=detector,
                           classifier=classifier, capture=capture, event_callback=lambda _: None,
                           frame_callback=fail if owner == "observer" else None, stop_event=stop)
    assert stop.is_set() and capture.stopped


def test_factory_observer_uses_packet_diagnostics_not_detector_globals(tmp_path):
    session = FactoryLiveSession(tmp_path)
    session._config = ExtractionConfig()
    session._active_detector = type("LaterDetector", (), {"last_diagnostics": YoloDetectionDiagnostics(99)})()
    result = FrameResult((), (), 1, .2, YoloDetectionDiagnostics(3, 1, 2))
    session._on_frame(np.zeros((32, 32, 3), np.uint8), result, .1)
    assert session._yolo_counts == {"raw": 3, "size_rejected": 1, "kept": 2, "eligible": 0}


def test_old_observation_cannot_replace_a_new_classified_decision(tmp_path):
    session = FactoryLiveSession(tmp_path)
    session._config = ExtractionConfig()
    result = FrameResult((), (Detection(1, 1, 8, 8, .9),), 1, .2,
                         processed_at_monotonic=time.perf_counter())
    session._on_event({"event_id": "latest", "status": "accepted", "prediction": "right"})
    session._on_frame(np.zeros((32, 32, 3), np.uint8), result, .1)
    assert session.decision()["phase"] == "REJECT"
    session._on_frame(np.zeros((32, 32, 3), np.uint8),
                      replace(result, processed_at_monotonic=time.perf_counter()), .2)
    assert session.decision()["phase"] == "INSPECTING"


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_classifier_queue_size_requires_positive_integer(size):
    with pytest.raises(ValueError, match="classifier_queue_size"):
        RuntimeConfig(classifier_queue_size=size)


def test_three_stage_is_default_but_fused_mode_remains_available():
    assert RuntimeConfig().stage_pipeline
    assert not RuntimeConfig(stage_pipeline=False).stage_pipeline


def test_skipped_detection_frames_preserve_fused_tracking_semantics():
    class ConstantDetector(Detector):
        def detect(self, frame):
            super().detect(frame)
            return [Detection(5, 5, 30, 30, .9)]

    records = []
    for staged in (False, True):
        cfg = config(stage_pipeline=staged, detect_every_n_frames=3)
        cfg.event.exit_missing_frames = 1
        cfg.event.min_detected_frames = 1
        events = []
        detector = ConstantDetector()
        run_live_inference(0, "unused", cfg, detector=detector, classifier=Classifier(),
                           capture=Capture(), event_callback=events.append)
        assert detector.calls == 4
        records.append([(e["status"], e["reject_reason"], e["timestamp_s"], e["bbox"]) for e in events])
    assert records[0] and records[0] == records[1]


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("max_frames", [None, 1])
def test_normal_completion_inhibits_actuation_before_event_drain(monkeypatch, staged, max_frames):
    stop = threading.Event()
    capture = Capture(2)
    box = Detection(5, 5, 30, 30, .9)

    def close(self, stamp):
        assert stop.is_set(), "Hardware must be inhibited before final events are handed off"
        return [PassageOutcome("final", "test", "unknown", "accepted", "", 0, stamp,
                               1, box, crop=np.zeros((32, 32, 3), np.uint8))]

    monkeypatch.setattr(PassageProcessor, "close", close)
    records = []
    metrics = run_live_inference(
        0, "unused", config(stage_pipeline=staged), detector=Detector(2), classifier=Classifier(),
        capture=capture, stop_event=stop, max_processed_frames=max_frames,
        event_callback=lambda record: records.append((record["event_id"], stop.is_set())),
    )
    assert records == [("final", True)] and capture.stopped
    assert metrics.source_exhausted == (max_frames is None)


def test_stopping_pipeline_suppresses_pending_and_manual_actuation(tmp_path):
    class Serial:
        def __init__(self):
            self.commands = []

        def snapshot(self):
            return {"connected": True}

        def submit(self, command):
            self.commands.append(command)

    session = FactoryLiveSession(tmp_path)
    session._serial = Serial()
    session.running = True
    session.status = "running"
    session.mode = "armed"
    session._stop.set()
    session._on_event({"event_id": "pending", "status": "accepted", "prediction": "right"})
    assert not session._serial.commands
    assert session.snapshot(reveal_paths=True)["latest"]["operating_mode"] == "shadow"
    with pytest.raises(ValueError, match="Manual trigger"):
        session.manual_trigger(confirm=True)

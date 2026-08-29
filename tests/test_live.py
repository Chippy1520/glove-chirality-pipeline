import time

import numpy as np

from glove_chirality.config import DetectorConfig, EventConfig, ExtractionConfig, RuntimeConfig
from glove_chirality.live import CapturedFrame, LatestFrameCapture, run_live_inference
from glove_chirality.types import Detection


class _OpenCvCapture:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None

    def release(self):
        self.released = True

    def get(self, _property):
        return 30.0


class _ImmediateCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.captured_frames = len(frames)
        self.dropped_frames = 0

    def start(self):
        return self

    def read(self, timeout=0.1):
        del timeout
        if not self.frames:
            return None
        index, frame = self.frames.pop(0)
        return CapturedFrame(index, time.monotonic(), frame)

    @property
    def exhausted(self):
        return not self.frames

    def stop(self):
        pass


class _Detector:
    name = "mock-yolo"

    def __init__(self, sequence):
        self.sequence = iter(sequence)
        self.warmups = 0

    def detect(self, _frame):
        return next(self.sequence, [])

    def warmup(self, _frame):
        self.warmups += 1


class _Classifier:
    def __init__(self):
        self.calls = 0
        self.warmups = 0

    def warmup(self):
        self.warmups += 1

    def predict_array(self, _crop):
        self.calls += 1
        return "right", 0.94


def test_latest_frame_capture_queue_stays_bounded_and_keeps_newest(monkeypatch):
    frames = [np.full((4, 4, 3), index, dtype=np.uint8) for index in range(10)]
    fake = _OpenCvCapture(frames)
    monkeypatch.setattr("glove_chirality.live.cv2.VideoCapture", lambda *_args: fake)
    capture = LatestFrameCapture(0, queue_size=2).start()
    deadline = time.monotonic() + 1.0
    while not capture.finished.is_set() and time.monotonic() < deadline:
        time.sleep(0.005)

    packets = []
    while not capture.queue.empty():
        packets.append(capture.queue.get_nowait())
    capture.stop()

    assert len(packets) <= 2
    assert packets[-1].index == 9
    assert capture.dropped_frames >= 8
    assert fake.released is True


def test_live_classifier_runs_once_per_accepted_physical_event():
    frame = np.full((40, 40, 3), 100, dtype=np.uint8)
    detection = Detection(10, 10, 30, 30, 0.9, 0, ((10, 10), (30, 10), (30, 30), (10, 30)))
    detector = _Detector([[detection], [detection], [], []])
    classifier = _Classifier()
    capture = _ImmediateCapture([(index, frame.copy()) for index in range(4)])
    config = ExtractionConfig(
        detector=DetectorConfig(roi=(0, 0, 1, 1), trigger_zone=(0, 0, 1, 1)),
        event=EventConfig(
            min_detected_frames=2,
            exit_missing_frames=1,
            cooldown_frames=0,
            output_size=32,
        ),
        runtime=RuntimeConfig(report_interval_seconds=60.0),
    )
    emitted = []

    metrics = run_live_inference(
        0,
        "unused.pt",
        config,
        event_callback=emitted.append,
        detector=detector,
        classifier=classifier,
        capture=capture,
    )

    assert classifier.calls == 1
    assert classifier.warmups == 1
    assert detector.warmups == 1
    assert metrics.accepted_passages == 1
    assert metrics.rejected_passages == 0
    assert len(emitted) == 1
    assert emitted[0]["status"] == "accepted"
    assert emitted[0]["prediction"] == "right"

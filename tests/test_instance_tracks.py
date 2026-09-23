import threading
import time

import numpy as np

from glove_chirality.config import ExtractionConfig
from glove_chirality.events import PassageProcessor
from glove_chirality.live import CapturedFrame, run_live_inference
from glove_chirality.types import Detection


def _config() -> ExtractionConfig:
    config = ExtractionConfig()
    config.event.trigger_line_enabled = True
    config.event.belt_direction = "bottom_to_top"
    config.event.timing_mode = "time"
    config.event.reject_multiple_detections = True
    return config


def _box(x: int, y: int, width: int = 30, height: int = 40, confidence: float = 0.9) -> Detection:
    return Detection(x, y, x + width, y + height, confidence)


class _ScriptDetector:
    def __init__(self, frames):
        self.frames = frames
        self.calls = 0

    def detect(self, frame):
        del frame
        items = self.frames[self.calls] if self.calls < len(self.frames) else []
        self.calls += 1
        return items

    def warmup(self, frame):
        del frame


def test_each_glove_emits_once_at_its_own_crossing():
    detector = _ScriptDetector([
        [_box(50, 90), _box(120, 90)],
        [_box(50, 50), _box(120, 50)],
        [_box(50, 45), _box(120, 45)],
    ])
    processor = PassageProcessor(detector, _config(), "belt.mp4", "live")
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    first = processor.process(frame, 0, 0.0)
    second = processor.process(frame, 1, 0.05)
    third = processor.process(frame, 2, 0.10)
    accepted = [item for item in (*first.outcomes, *second.outcomes, *third.outcomes) if item.accepted]
    assert first.outcomes == ()
    assert len(accepted) == 2
    assert len({item.event_id for item in accepted}) == 2
    assert all(item.crop is not None and item.crop.shape == (256, 256, 3) for item in accepted)
    assert all(item.trigger_crossing_s is not None for item in accepted)
    assert third.outcomes == ()


def test_touching_gloves_do_not_produce_two_crops():
    detector = _ScriptDetector([
        [_box(70, 90), _box(80, 92)],
        [_box(70, 50), _box(80, 52)],
    ])
    processor = PassageProcessor(detector, _config(), "belt.mp4", "live")
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    first = processor.process(frame, 0, 0.0)
    second = processor.process(frame, 1, 0.05)
    assert any(item.status == "multiple_candidates" for item in first.outcomes)
    assert not any(item.accepted for item in (*first.outcomes, *second.outcomes))


class _SlowClassifier:
    def __init__(self):
        self.started = threading.Event()
        self.detects_during = 0

    def predict_array(self, image):
        del image
        self.started.set()
        time.sleep(0.15)
        return "left", 0.8

    def warmup(self):
        return None


class _CountingDetector:
    def __init__(self, classifier, script):
        self.classifier = classifier
        self.script = script
        self.calls = 0

    def detect(self, frame):
        del frame
        if self.classifier.started.is_set():
            self.classifier.detects_during += 1
        items = self.script[self.calls] if self.calls < len(self.script) else []
        self.calls += 1
        return items

    def warmup(self, frame):
        del frame


class _Frames:
    def __init__(self, count, classifier):
        self.count = count
        self.classifier = classifier
        self.index = 0
        self.exhausted = False
        self.dropped_frames = 0
        self.captured_frames = count

    def start(self):
        return self

    def read(self, timeout=0.1):
        del timeout
        if self.index >= 2:
            self.classifier.started.wait(1)
        if self.index >= self.count:
            self.exhausted = True
            return None
        packet = CapturedFrame(self.index, time.monotonic(), np.zeros((200, 200, 3), dtype=np.uint8))
        self.index += 1
        return packet

    def stop(self):
        return None


def test_layer2_runs_while_layer1_continues():
    classifier = _SlowClassifier()
    detector = _CountingDetector(classifier, [
        [_box(50, 90)],
        [_box(50, 50)],
        [],
        [],
        [],
    ])
    events = []
    run_live_inference(
        0,
        "unused.pt",
        _config(),
        detector=detector,
        classifier=classifier,
        capture=_Frames(5, classifier),
        event_callback=events.append,
        max_processed_frames=5,
    )
    assert classifier.detects_during > 0
    assert any(item.get("prediction") == "left" for item in events)

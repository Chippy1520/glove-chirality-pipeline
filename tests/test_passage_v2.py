import numpy as np

from glove_chirality.config import ExtractionConfig
from glove_chirality.events import PassageProcessor
from glove_chirality.types import Detection


def _config() -> ExtractionConfig:
    config = ExtractionConfig()
    config.detector.roi = (0.0, 0.0, 1.0, 1.0)
    config.detector.trigger_zone = (0.0, 0.0, 1.0, 1.0)
    config.event.trigger_line_enabled = True
    config.event.belt_direction = "bottom_to_top"
    config.event.tracker_mode = "passage_v2"
    config.event.crop_selector = "best_frame"
    config.event.low_conf_recovery = True
    config.event.trigger_hysteresis_ratio = 0.0
    config.event.max_track_distance_ratio = 0.35
    config.event.validate()
    return config


def _box(cx: int, cy: int, confidence: float = 0.8, size: int = 20) -> Detection:
    half = size // 2
    return Detection(cx - half, cy - half, cx + half, cy + half, confidence)


class _Script:
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


def _run(frames, config=None):
    config = config or _config()
    processor = PassageProcessor(_Script(frames), config, "belt.mp4", "live")
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    outcomes = []
    for index, _items in enumerate(frames):
        outcomes.extend(processor.process(image, index, index * 0.1).outcomes)
    return [item for item in outcomes if item.accepted]


def test_one_glove_emits_once_despite_a_confidence_dip():
    accepted = _run([
        [_box(100, 140, 0.8)],
        [_box(100, 120, 0.2)],
        [_box(100, 70, 0.9)],
        [_box(100, 60, 0.85)],
    ])
    assert len(accepted) == 1
    assert accepted[0].crop is not None
    assert accepted[0].crop.shape == (256, 256, 3)


def test_weak_detection_cannot_birth_a_passage():
    accepted = _run([[_box(100, 140, 0.2)], [_box(100, 70, 0.2)]])
    assert accepted == []


def test_two_separated_gloves_emit_two_crops():
    accepted = _run([
        [_box(60, 140, 0.8), _box(140, 140, 0.8)],
        [_box(60, 70, 0.8), _box(140, 70, 0.8)],
    ])
    assert len(accepted) == 2
    assert len({item.event_id for item in accepted}) == 2


def test_line_jitter_does_not_emit_twice():
    config = _config()
    config.event.trigger_hysteresis_ratio = 0.08
    config.event.validate()
    accepted = _run([
        [_box(100, 130, 0.8)],
        [_box(100, 70, 0.8)],
        [_box(100, 125, 0.8)],
        [_box(100, 65, 0.8)],
    ], config)
    assert len(accepted) == 1


def test_best_pre_cross_frame_is_selected():
    accepted = _run([
        [_box(100, 150, 0.50)],
        [_box(100, 130, 0.95)],
        [_box(100, 40, 0.55)],
    ])
    assert len(accepted) == 1
    assert accepted[0].detection is not None
    assert accepted[0].detection.confidence == 0.95

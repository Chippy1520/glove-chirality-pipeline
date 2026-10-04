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
    config.event.tracker_mode = "count"
    config.event.output_size = 32
    config.event.validate()
    return config


def _box(cx: int, cy: int, size: int = 40) -> Detection:
    half = size // 2
    return Detection(cx - half, cy - half, cx + half, cy + half, 0.9)


class _Script:
    def __init__(self, frames):
        self.frames = frames
        self.calls = 0

    def detect(self, frame):
        del frame
        items = self.frames[self.calls] if self.calls < len(self.frames) else []
        self.calls += 1
        return items


def _accepted(frames):
    processor = PassageProcessor(_Script(frames), _config(), "belt.mp4", "live")
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    outcomes = []
    for index, _items in enumerate(frames):
        outcomes.extend(processor.process(image, index, index * 0.1).outcomes)
    return [item for item in outcomes if item.accepted]


def test_separated_gloves_in_one_lane_are_counted_apart():
    accepted = _accepted([
        [_box(100, 165, size=20), _box(100, 120, size=20)],
        [_box(100, 145, size=20), _box(100, 105, size=20)],
        [_box(100, 115, size=20), _box(100, 85, size=20)],
        [_box(100, 90, size=20), _box(100, 65, size=20)],
    ])
    assert len(accepted) == 2
    assert len({item.event_id for item in accepted}) == 2


def test_one_glove_is_one_count():
    accepted = _accepted([
        [_box(100, 150)],
        [_box(100, 130)],
        [_box(100, 80)],
    ])
    assert len(accepted) == 1

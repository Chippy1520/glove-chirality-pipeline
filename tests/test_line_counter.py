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
    config.event.tracker_mode = "line"
    config.event.output_size = 32
    config.event.reentry_time_s = 0.5
    config.event.validate()
    return config


def _box(cx: int, cy: int, size: int = 30) -> Detection:
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


def _run(frames):
    processor = PassageProcessor(_Script(frames), _config(), "belt.mp4", "live")
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    outcomes = []
    for index in range(len(frames)):
        outcomes.extend(processor.process(image, index, index * 0.1).outcomes)
    outcomes.extend(processor.close((len(frames) - 1) * 0.1))
    return outcomes


def test_one_crossing_is_one_crop():
    accepted = [
        item
        for item in _run([[_box(100, 150)], [_box(100, 130)], [_box(100, 110)], [_box(100, 90)]])
        if item.accepted
    ]
    assert len(accepted) == 1


def test_two_separated_crossings_stay_two():
    frames = [
        [_box(100, 170, size=24), _box(100, 110, size=24)],
        [_box(100, 155, size=24), _box(100, 95, size=24)],
        [_box(100, 140, size=24), _box(100, 80, size=24)],
        [_box(100, 125, size=24), _box(100, 65, size=24)],
        [_box(100, 110, size=24), _box(100, 50, size=24)],
        [_box(100, 95, size=24), _box(100, 35, size=24)],
    ]
    accepted = [item for item in _run(frames) if item.accepted]
    assert len(accepted) == 2


def test_trailing_boxes_do_not_count_again():
    frames = [[_box(100, 140)], [_box(100, 120)], [_box(100, 100)], [_box(100, 80)], [_box(100, 60)]]
    outcomes = _run(frames)
    assert len([item for item in outcomes if item.accepted]) == 1
    assert not [item for item in outcomes if item.reject_reason == "born_past_line"]


def test_close_gloves_with_separate_masks_stay_two():
    def glove(cx, cy, top, bottom):
        half = 30
        return Detection(
            cx - half,
            cy - half,
            cx + half,
            cy + half,
            0.9,
            polygon=((cx - 20, top), (cx + 20, top), (cx + 20, bottom), (cx - 20, bottom)),
        )

    frames = []
    for step in range(6):
        frames.append([
            glove(100, 160 - step * 12, 148 - step * 12, 172 - step * 12),
            glove(100, 112 - step * 12, 96 - step * 12, 124 - step * 12),
        ])
    accepted = [item for item in _run(frames) if item.accepted]
    assert len(accepted) == 2


def test_a_short_gap_with_the_same_mask_stays_one_track():
    frames = [
        [_box(100, 150)],
        [_box(100, 135)],
        [],
        [],
        [],
        [],
        [_box(100, 110)],
        [_box(100, 90)],
    ]
    accepted = [item for item in _run(frames) if item.accepted]
    assert len(accepted) == 1


def test_a_bottom_led_frame_is_not_the_crop():
    frames = [
        [_box(100, 170, size=80)],
        [_box(100, 140, size=40)],
        [_box(100, 110, size=40)],
        [_box(100, 90, size=40)],
    ]
    accepted = [item for item in _run(frames) if item.accepted]
    assert len(accepted) == 1
    assert accepted[0].frame_index >= 2


def test_crop_comes_from_the_line_not_the_entry():
    frames = [[_box(100, 170)], [_box(100, 150)], [_box(100, 130)], [_box(100, 110)], [_box(100, 90)]]
    accepted = [item for item in _run(frames) if item.accepted]
    assert len(accepted) == 1
    assert accepted[0].frame_index >= 3

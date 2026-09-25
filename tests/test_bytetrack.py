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
    config.event.tracker_mode = "bytetrack"
    config.event.output_size = 32
    config.event.validate()
    return config


def _box(cx: int, cy: int, size: int = 40, confidence: float = 0.9) -> Detection:
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


def _accepted(frames):
    processor = PassageProcessor(_Script(frames), _config(), "belt.mp4", "live")
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    outcomes = []
    for index, _items in enumerate(frames):
        outcomes.extend(processor.process(image, index, index * 0.1).outcomes)
    return [item for item in outcomes if item.accepted]


def test_hungarian_keeps_the_globally_better_pairing():
    from glove_chirality.bytetrack import _hungarian

    # Greedy highest-score takes (0, 0). The minimum total cost is the swap.
    cost = np.array([[0.1, 0.2], [0.15, 4.0]])
    assert set(_hungarian(cost)) == {(0, 1), (1, 0)}


def test_fixed_gap_gloves_stay_two_tracks_without_box_overlap():
    accepted = _accepted([
        [_box(100, 170, size=24), _box(100, 120, size=24)],
        [_box(100, 150, size=24), _box(100, 100, size=24)],
        [_box(100, 130, size=24), _box(100, 80, size=24)],
        [_box(100, 110, size=24), _box(100, 60, size=24)],
        [_box(100, 90, size=24), _box(100, 40, size=24)],
    ])
    assert len(accepted) == 2
    assert len({item.event_id for item in accepted}) == 2


def test_separated_gloves_in_one_lane_stay_two_tracks():
    accepted = _accepted([
        [_box(100, 170, size=30), _box(100, 120, size=30)],
        [_box(100, 155, size=30), _box(100, 105, size=30)],
        [_box(100, 140, size=30), _box(100, 90, size=30)],
        [_box(100, 125, size=30), _box(100, 75, size=30)],
        [_box(100, 100, size=30), _box(100, 60, size=30)],
        [_box(100, 80, size=30), _box(100, 45, size=30)],
    ])
    assert len(accepted) == 2
    assert len({item.event_id for item in accepted}) == 2


def test_mask_overlap_matches_when_centers_are_far():
    from glove_chirality.bytetrack import _assignment_cost

    previous = Detection(0, 0, 20, 20, 0.9, polygon=((0, 0), (40, 0), (40, 20), (0, 20)))
    detection = Detection(80, 0, 100, 20, 0.9, polygon=((20, 0), (40, 0), (40, 20), (20, 20)))
    assert _assignment_cost((10, 10), previous, detection, _config()) < 1e5


def test_upstream_track_that_never_crosses_is_lost_before_trigger():
    processor = PassageProcessor(
        _Script([[_box(100, 160)], [_box(100, 145)], []]),
        _config(),
        "belt.mp4",
        "live",
    )
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    processor.process(image, 0, 0.0)
    processor.process(image, 1, 0.1)
    closed = processor.close(1.0)
    assert [item.reject_reason for item in closed] == ["lost_before_trigger"]


def test_one_glove_is_one_crop():
    accepted = _accepted([
        [_box(100, 150)],
        [_box(100, 135)],
        [_box(100, 115)],
        [_box(100, 80)],
    ])
    assert len(accepted) == 1

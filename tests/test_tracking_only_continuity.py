"""Continuity checks for size-rejected boxes and trailing sightings."""

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
    config.event.reentry_time_s = 0.50
    config.event.validate()
    return config


def _box(cx: int, cy: int, size: int = 40, confidence: float = 0.9) -> Detection:
    half = size // 2
    return Detection(cx - half, cy - half, cx + half, cy + half, confidence)


class _ScriptWithPartials:
    def __init__(self, kept_frames, partial_frames):
        self.kept_frames = kept_frames
        self.partial_frames = partial_frames
        self.calls = 0

    def detect(self, frame):
        del frame
        items = self.kept_frames[self.calls] if self.calls < len(self.kept_frames) else []
        self.calls += 1
        return items

    def tracking_partials(self):
        index = self.calls - 1
        return self.partial_frames[index] if 0 <= index < len(self.partial_frames) else []


def test_occlusion_gap_over_the_line_stays_one_track_with_a_weak_box_upstream():
    kept = [
        [_box(100, 170)],
        [_box(100, 150)],
        [],
        [],
        [],
        [],
        [],
        [],
        [_box(100, 60)],
        [_box(100, 40)],
        [_box(100, 20)],
    ]
    partials = [
        [],
        [],
        [_box(100, 132, size=10)],
        [_box(100, 116, size=10)],
        [_box(100, 100, size=10)],
        [_box(100, 84, size=10)],
        [_box(100, 68, size=10)],
        [],
        [],
        [],
        [],
    ]
    processor = PassageProcessor(_ScriptWithPartials(kept, partials), _config(), "belt.mp4", "live")
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    outcomes = []
    for index in range(len(kept)):
        outcomes.extend(processor.process(image, index, index * 0.1).outcomes)
    outcomes.extend(processor.close((len(kept) - 1) * 0.1))
    accepted = [item for item in outcomes if item.accepted]
    assert len(accepted) == 1
    assert not [item for item in outcomes if item.reject_reason == "born_past_line"]


def test_trailing_sightings_after_emission_do_not_spawn_a_phantom_track():
    kept = [
        [_box(100, 150)],
        [_box(100, 130)],
        [_box(100, 110)],
        [_box(100, 90)],
        [_box(100, 70)],
        [_box(100, 50)],
        [_box(100, 30)],
    ]
    processor = PassageProcessor(
        _ScriptWithPartials(kept, [[] for _ in kept]), _config(), "belt.mp4", "live"
    )
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    outcomes = []
    for index in range(len(kept)):
        outcomes.extend(processor.process(image, index, index * 0.1).outcomes)
    outcomes.extend(processor.close((len(kept) - 1) * 0.1))
    assert len([item for item in outcomes if item.accepted]) == 1
    assert not [item for item in outcomes if item.reject_reason == "born_past_line"]


def test_enforce_order_repairs_a_belt_order_inversion():
    from glove_chirality.bytetrack import _enforce_order, _Kalman, _Track

    def _track(track_id: int, y: float):
        return _Track(track_id=track_id, detection=_box(100, int(y)), kalman=_Kalman(100, y, 0, -150))

    tracks = [_track(1, 60), _track(2, 120)]
    inverted_pairs = [(0, _box(100, 100)), (1, _box(100, 40))]
    repaired = _enforce_order(inverted_pairs, tracks, _config())
    by_track = dict(repaired)
    assert by_track[0].center[1] < by_track[1].center[1]

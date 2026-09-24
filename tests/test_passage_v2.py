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
    def __init__(self, frames, partials=None):
        self.frames = frames
        self.partials = partials or [[] for _ in frames]
        self.calls = 0
        self._current: list = []

    def detect(self, frame):
        del frame
        self._current = self.partials[self.calls] if self.calls < len(self.partials) else []
        items = self.frames[self.calls] if self.calls < len(self.frames) else []
        self.calls += 1
        return items

    def tracking_partials(self):
        return list(self._current)

    def warmup(self, frame):
        del frame


def _run(frames, config=None, tracking=None):
    config = config or _config()
    processor = PassageProcessor(_Script(frames, tracking), config, "belt.mp4", "live")
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


def test_reentry_gap_still_emits_one_crop():
    accepted = _run([
        [_box(100, 140, 0.8)],
        [],
        [],
        [],
        [_box(100, 120, 0.8)],
        [_box(100, 70, 0.8)],
    ])
    assert len(accepted) == 1


def test_fragmented_reentry_near_the_same_crossing_does_not_emit_again():
    config = _config()
    config.event.max_track_distance_ratio = 0.40
    config.event.reentry_time_s = 0.05
    config.event.validate()
    accepted = _run([
        [_box(100, 140, 0.8)],
        [_box(100, 60, 0.8)],
        [],
        [],
        [_box(100, 75, 0.8)],
        [_box(100, 55, 0.8)],
    ], config)
    assert len(accepted) == 1


def test_merge_recovery_off_still_skips_the_second_glove():
    accepted = _run([
        [_box(50, 160), _box(150, 160)],
        [_box(50, 140), _box(150, 140)],
        [Detection(20, 70, 180, 120, 0.8)],
        [Detection(20, 40, 180, 90, 0.8)],
    ])
    assert len(accepted) == 1


def test_merged_blob_keeps_two_already_separated_gloves():
    config = _config()
    config.event.merge_recovery = True
    config.event.validate()
    accepted = _run([
        [_box(50, 160), _box(150, 160)],
        [_box(50, 140), _box(150, 140)],
        [Detection(20, 70, 180, 120, 0.8)],
        [Detection(20, 40, 180, 90, 0.8)],
    ], config)
    assert len(accepted) == 2
    assert len({item.event_id for item in accepted}) == 2
    assert all(item.detection is not None and item.detection.width < 160 for item in accepted)


def test_wrinkle_split_still_emits_one_crop():
    config = _config()
    config.event.merge_recovery = True
    config.event.validate()
    accepted = _run([
        [Detection(70, 145, 110, 175, 0.8), Detection(90, 145, 130, 175, 0.8)],
        [Detection(70, 125, 110, 155, 0.8), Detection(90, 125, 130, 155, 0.8)],
        [Detection(70, 40, 130, 80, 0.8)],
    ], config)
    assert len(accepted) == 1


def test_same_lane_glove_half_a_second_later_is_not_a_duplicate():
    config = _config()
    config.event.reentry_time_s = 0.15
    config.event.max_track_distance_ratio = 0.40
    config.event.validate()
    accepted = _run([
        [_box(100, 140, 0.8)],
        [_box(100, 60, 0.8)],
        [],
        [],
        [],
        [_box(100, 150, 0.8)],
        [_box(100, 70, 0.8)],
    ], config)
    assert len(accepted) == 2


def test_weak_crossing_emits_the_strong_frame_once():
    accepted = _run([
        [_box(100, 150, 0.80)],
        [_box(100, 130, 0.90)],
        [_box(100, 70, 0.20)],
        [_box(100, 40, 0.85)],
    ])
    assert len(accepted) == 1
    assert accepted[0].detection is not None
    assert accepted[0].detection.confidence == 0.90


def test_partial_box_keeps_the_track_and_is_not_the_crop():
    accepted = _run(
        [
            [_box(100, 150, 0.80)],
            [_box(100, 130, 0.90)],
            [],
            [_box(100, 50, 0.85)],
        ],
        tracking=[[], [], [_box(100, 90, 0.80, size=8)], []],
    )
    assert len(accepted) == 1
    assert accepted[0].detection is not None
    assert accepted[0].detection.confidence == 0.90


def test_lost_track_writes_a_terminal_reason():
    config = _config()
    config.event.reentry_time_s = 0.15
    config.event.validate()
    processor = PassageProcessor(_Script([[_box(100, 150, 0.8)], [], [], []]), config, "belt.mp4", "live")
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    outcomes = []
    for index in range(4):
        outcomes.extend(processor.process(image, index, index * 0.1).outcomes)
    outcomes.extend(processor.close(0.4))
    reasons = [item.reject_reason for item in outcomes if not item.accepted]
    assert "lost_before_trigger" in reasons or "insufficient_confirmation" in reasons


def test_best_pre_cross_frame_is_selected():
    accepted = _run([
        [_box(100, 150, 0.50)],
        [_box(100, 130, 0.95)],
        [_box(100, 40, 0.55)],
    ])
    assert len(accepted) == 1
    assert accepted[0].detection is not None
    assert accepted[0].detection.confidence == 0.95

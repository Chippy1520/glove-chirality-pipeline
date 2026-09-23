import cv2
import numpy as np
import pytest

from glove_chirality.config import DetectorConfig, EventConfig, ExtractionConfig
from glove_chirality.events import (
    PassageProcessor,
    create_event_crop,
    directional_crossing_alpha,
    trigger_line_axis,
    trigger_line_position,
)
from glove_chirality.extraction import _letterbox
from glove_chirality.types import Detection


class _SequenceDetector:
    name = "sequence"

    def __init__(self, sequence):
        self.sequence = iter(sequence)

    def detect(self, _frame):
        return next(self.sequence, [])


def _box(cx: float, cy: float, size: int = 10) -> Detection:
    half = size / 2
    return Detection(int(cx - half), int(cy - half), int(cx + half), int(cy + half), 0.9)


def _wide_image() -> np.ndarray:
    image = np.zeros((40, 80, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 10), (59, 29), (255, 255, 255), -1)
    return image


def _white_aspect(image: np.ndarray) -> float:
    white = np.where(np.all(image == 255, axis=2), 255, 0).astype(np.uint8)
    _x, _y, width, height = cv2.boundingRect(cv2.findNonZero(white))
    return width / height


def test_letterbox_fill_is_constant_and_preserves_geometry():
    assert EventConfig().letterbox_fill == 114
    image = _wide_image()
    result = _letterbox(image, 100)
    assert result.shape == (100, 100, 3)
    assert _white_aspect(result) == pytest.approx(2.0, rel=0.06)
    assert np.all(result[0, 0] == 114)
    assert np.all(result[0, -1] == 114)
    assert np.all(result[-1, 0] == 114)
    assert np.all(result[-1, -1] == 114)

    custom = _letterbox(image, 100, 7)
    assert custom.shape == (100, 100, 3)
    assert np.all(custom[0, 0] == 7)
    assert np.all(custom[-1, -1] == 7)
    assert _white_aspect(custom) == pytest.approx(2.0, rel=0.06)


def test_create_event_crop_uses_configured_letterbox_fill():
    frame = np.full((40, 80, 3), 20, dtype=np.uint8)
    frame[10:30, 10:70] = 200
    detection = Detection(10, 10, 70, 30, 0.95)
    custom = ExtractionConfig(
        event=EventConfig(
            crop_padding=0.0,
            make_square=False,
            output_size=90,
            letterbox_fill=7,
        )
    )
    crop = create_event_crop(frame, detection, custom)
    assert crop.shape == (90, 90, 3)
    assert np.all(crop[0, 0] == 7)
    assert np.all(crop[-1, -1] == 7)
    assert not np.any(crop == 114)

    default = ExtractionConfig(
        event=EventConfig(crop_padding=0.0, make_square=False, output_size=90)
    )
    default_crop = create_event_crop(frame, detection, default)
    assert default_crop.shape == (90, 90, 3)
    assert np.all(default_crop[0, 0] == 114)
    assert np.all(default_crop[-1, -1] == 114)


@pytest.mark.parametrize("fill", [-1, 256, 1.5, True])
def test_invalid_letterbox_fill_rejected(fill):
    with pytest.raises(ValueError, match="letterbox_fill"):
        EventConfig(letterbox_fill=fill)


@pytest.mark.parametrize(
    ("direction", "increasing", "axis"),
    [
        ("left_to_right", True, "x"),
        ("right_to_left", False, "x"),
        ("top_to_bottom", True, "y"),
        ("bottom_to_top", False, "y"),
    ],
)
def test_crossing_alpha_follows_each_belt_direction(direction, increasing, axis):
    assert trigger_line_axis(direction) == axis
    forward_prev, forward_curr = (10.0, 30.0) if increasing else (30.0, 10.0)
    alpha = directional_crossing_alpha(forward_prev, forward_curr, 20.0, increasing)
    assert alpha == pytest.approx(0.5)
    assert directional_crossing_alpha(forward_curr, forward_prev, 20.0, increasing) is None
    assert directional_crossing_alpha(1.0, 2.0, 5.0, increasing) is None
    assert directional_crossing_alpha(4.0, 4.0, 4.0, increasing) is None


def test_crossing_alpha_interpolates_and_rejects_a_miss():
    assert directional_crossing_alpha(0.0, 10.0, 2.5, True) == pytest.approx(0.25)
    assert directional_crossing_alpha(10.0, 0.0, 2.5, False) == pytest.approx(0.75)
    assert directional_crossing_alpha(1.0, 5.0, 5.0, True) == pytest.approx(1.0)
    assert directional_crossing_alpha(5.0, 9.0, 5.0, True) is None
    assert directional_crossing_alpha(3.0, 8.0, 3.0, False) is None
    clamped = directional_crossing_alpha(0.0, 1.0 + 1e-12, 1.0, True)
    assert clamped is not None
    assert 0.0 <= clamped <= 1.0


def test_trigger_line_position_uses_zone_fraction():
    zone = (0.2, 0.1, 0.8, 0.7)
    width, height = 200, 100
    fraction = 0.25
    expected = {
        "left_to_right": ("x", (0.2 + fraction * (0.8 - 0.2)) * width),
        "right_to_left": ("x", (0.2 + fraction * (0.8 - 0.2)) * width),
        "top_to_bottom": ("y", (0.1 + fraction * (0.7 - 0.1)) * height),
        "bottom_to_top": ("y", (0.1 + fraction * (0.7 - 0.1)) * height),
    }
    for direction, (axis, position) in expected.items():
        config = ExtractionConfig(
            detector=DetectorConfig(trigger_zone=zone),
            event=EventConfig(belt_direction=direction, trigger_line_fraction=fraction),
        )
        got_axis, got_position = trigger_line_position(config, width, height)
        assert got_axis == axis
        assert got_position == pytest.approx(position)


def test_trigger_line_axis_rejects_unknown_direction():
    with pytest.raises(ValueError):
        trigger_line_axis("diagonal")


def _passage_config(**event_values) -> ExtractionConfig:
    values = {
        "min_detected_frames": 2,
        "exit_missing_frames": 1,
        "cooldown_frames": 0,
        "max_track_distance_ratio": 1.0,
        "trigger_line_enabled": True,
        "belt_direction": "left_to_right",
        "trigger_line_fraction": 0.5,
    }
    values.update(event_values)
    return ExtractionConfig(
        detector=DetectorConfig(roi=(0, 0, 1, 1), trigger_zone=(0, 0, 1, 1)),
        event=EventConfig(**values),
    )


def _run_passage(direction: str, start, end, enabled: bool = True):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    processor = PassageProcessor(
        _SequenceDetector([[_box(*start)], [_box(*end)], []]),
        _passage_config(belt_direction=direction, trigger_line_enabled=enabled),
        "belt.avi",
    )
    outcomes = []
    for index, timestamp in enumerate((0.0, 2.0, 2.2)):
        outcomes.extend(processor.process(frame, index, timestamp).outcomes)
    accepted = [outcome for outcome in outcomes if outcome.accepted]
    assert len(accepted) == 1
    return accepted[0]


@pytest.mark.parametrize(
    ("direction", "start", "end", "position", "norm"),
    [
        ("left_to_right", (30, 20), (70, 40), (50.0, 30.0), (0.5, 0.3)),
        ("right_to_left", (70, 20), (30, 40), (50.0, 30.0), (0.5, 0.3)),
        ("top_to_bottom", (20, 30), (40, 70), (30.0, 50.0), (0.3, 0.5)),
        ("bottom_to_top", (20, 70), (40, 30), (30.0, 50.0), (0.3, 0.5)),
    ],
)
def test_passage_stores_interpolated_directional_crossing(direction, start, end, position, norm):
    outcome = _run_passage(direction, start, end)
    assert outcome.detection is not None
    assert outcome.trigger_crossing_s == pytest.approx(1.0)
    assert outcome.trigger_crossing_position_px == pytest.approx(position)
    assert outcome.trigger_crossing_position_norm == pytest.approx(norm)
    assert outcome.center_px == pytest.approx(outcome.detection.center)
    assert outcome.center_norm == pytest.approx(
        (outcome.detection.center[0] / 100, outcome.detection.center[1] / 100)
    )
    assert outcome.center_px != pytest.approx(position)
    assert outcome.trigger_crossing_s not in {0.0, 2.0}


def test_disabled_trigger_line_does_not_invent_a_crossing():
    outcome = _run_passage("left_to_right", (30, 20), (70, 40), enabled=False)
    assert outcome.trigger_crossing_s is None
    assert outcome.trigger_crossing_position_px is None
    assert outcome.trigger_crossing_position_norm is None
    assert outcome.detection is not None
    assert outcome.center_px == pytest.approx(outcome.detection.center)


def test_reset_clears_crossing_so_the_next_passage_interpolates_again():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    sequence = [
        [_box(30, 20)],
        [_box(70, 40)],
        [],
        [],
        [_box(10, 20)],
        [_box(90, 60)],
        [],
    ]
    processor = PassageProcessor(
        _SequenceDetector(sequence),
        _passage_config(),
        "belt.avi",
    )
    outcomes = []
    timestamps = (0.0, 2.0, 2.2, 2.3, 4.0, 6.0, 6.2)
    for index, timestamp in enumerate(timestamps):
        outcomes.extend(processor.process(frame, index, timestamp).outcomes)
    accepted = [outcome for outcome in outcomes if outcome.accepted]
    assert len(accepted) == 2
    assert accepted[0].trigger_crossing_s == pytest.approx(1.0)
    assert accepted[1].trigger_crossing_s == pytest.approx(5.0)
    assert accepted[1].trigger_crossing_position_px == pytest.approx((50.0, 40.0))
    assert processor._previous_center is None
    assert processor._trigger_crossing_s is None


def test_crossing_accepts_without_waiting_for_the_old_frame_count():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    processor = PassageProcessor(
        _SequenceDetector([[_box(30, 20)], [_box(70, 40)], []]),
        _passage_config(min_detected_frames=3),
        "belt.avi",
    )
    outcomes = []
    for index, timestamp in enumerate((0.0, 2.0, 2.2)):
        outcomes.extend(processor.process(frame, index, timestamp).outcomes)
    assert [outcome.status for outcome in outcomes] == ["accepted"]
    outcome = outcomes[0]
    assert outcome.detection is not None
    assert outcome.trigger_crossing_s == pytest.approx(1.0)
    assert outcome.trigger_crossing_position_px == pytest.approx((50.0, 30.0))
    assert outcome.center_px == pytest.approx(outcome.detection.center)
    assert outcome.center_px != pytest.approx((50.0, 30.0))


def test_separated_gloves_are_not_an_immediate_multiple_reject():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    processor = PassageProcessor(
        _SequenceDetector([[_box(20, 50), _box(80, 50)]]),
        _passage_config(),
        "belt.avi",
    )
    assert processor.process(frame, 0, 0.0).outcomes == ()


def test_overlapping_gloves_stay_ambiguous():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    processor = PassageProcessor(
        _SequenceDetector([[_box(50, 50, size=30), _box(55, 50, size=30)]]),
        _passage_config(),
        "belt.avi",
    )
    outcome = processor.process(frame, 0, 0.0).outcomes[0]
    assert outcome.status == "multiple_candidates"
    assert outcome.detection is None
    assert outcome.center_px is None
    assert outcome.trigger_crossing_s is None


def test_extraction_and_live_share_passage_processor():
    from glove_chirality import extraction, live

    assert extraction.PassageProcessor is PassageProcessor
    assert live.PassageProcessor is PassageProcessor

import numpy as np
import pytest

from glove_chirality.config import DetectorConfig, EventConfig, ExtractionConfig
from glove_chirality.detection.base import inside_trigger
from glove_chirality.events import PassageProcessor, create_event_crop
from glove_chirality.types import Detection


class _SequenceDetector:
    name = "sequence"

    def __init__(self, sequence):
        self.sequence = iter(sequence)

    def detect(self, _frame):
        return next(self.sequence, [])

    def warmup(self, _frame):
        pass


def _segmentation_detection(x1=5, y1=5, x2=15, y2=15):
    return Detection(
        x1,
        y1,
        x2,
        y2,
        0.9,
        0,
        ((x1, y1), (x2, y1), (x2, y2), (x1, y2)),
    )


def _config(**event_values):
    values = {
        "min_detected_frames": 2,
        "exit_missing_frames": 1,
        "cooldown_frames": 0,
        "output_size": 20,
    }
    values.update(event_values)
    return ExtractionConfig(
        detector=DetectorConfig(
            roi=(0, 0, 1, 1),
            trigger_zone=(0, 0, 1, 1),
        ),
        event=EventConfig(**values),
    )


def _crop_source():
    frame = np.full((20, 20, 3), 10, dtype=np.uint8)
    frame[5:16, 5:16] = 200
    return frame


def test_bbox_crop_mode_preserves_rectangular_background():
    config = _config(crop_padding=0.5, make_square=False, crop_mode="bbox")
    crop = create_event_crop(_crop_source(), _segmentation_detection(), config)
    assert crop.shape == (20, 20, 3)
    assert np.all(crop[0, 0] == 10)
    assert np.all(crop[10, 10] == 200)

def test_tight_bbox_crop_excludes_adjacent_glove_without_changing_output_contract():
    frame = np.full((20, 30, 3), 10, dtype=np.uint8)

    # Selected glove.
    frame[5:10, 5:15] = 200

    # Adjacent glove that must not enter the selected bbox crop.
    frame[5:10, 15:23] = 100

    detection = _segmentation_detection(
        5, 5, 15, 10
    )

    config = _config(
        crop_padding=0.0,
        make_square=False,
        crop_mode="bbox",
        output_size=20,
    )

    crop = create_event_crop(
        frame,
        detection,
        config,
    )

    assert crop.shape == (20, 20, 3)

    # Selected glove is preserved.
    assert np.all(crop == 200)

    # Adjacent glove must not appear.
    assert not np.any(crop == 100)

def test_masked_crop_suppresses_non_glove_pixels():
    config = _config(crop_padding=0.5, make_square=False, crop_mode="masked")
    crop = create_event_crop(_crop_source(), _segmentation_detection(), config)
    assert crop.shape == (20, 20, 3)
    assert np.all(crop[0, 0] == 0)
    assert np.all(crop[10, 10] == 200)


def test_masked_fill_uses_deterministic_background_median():
    config = _config(crop_padding=0.5, make_square=False, crop_mode="masked_fill")
    first = create_event_crop(_crop_source(), _segmentation_detection(), config)
    second = create_event_crop(_crop_source(), _segmentation_detection(), config)
    assert np.array_equal(first, second)
    assert np.all(first[0, 0] == 10)
    assert np.all(first[10, 10] == 200)


def test_segmentation_box_obeys_full_containment_gate():
    config = DetectorConfig(trigger_zone=(0.2, 0.2, 0.8, 0.8))
    partial = _segmentation_detection(2, 5, 15, 15)
    complete = _segmentation_detection(5, 5, 15, 15)
    assert inside_trigger(partial, config, 20, 20) is False
    assert inside_trigger(complete, config, 20, 20) is True


def test_one_mask_produces_one_accepted_passage():
    detection = _segmentation_detection()
    processor = PassageProcessor(
        _SequenceDetector([[detection], [detection], []]),
        _config(),
        "source.avi",
    )
    frame = _crop_source()
    outcomes = []
    for index in range(3):
        outcomes.extend(processor.process(frame, index, index / 25).outcomes)
    accepted = [outcome for outcome in outcomes if outcome.accepted]
    assert len(accepted) == 1
    assert accepted[0].crop.shape == (20, 20, 3)


def test_two_masks_are_ambiguous_by_default():
    detections = [_segmentation_detection(2, 4, 8, 12), _segmentation_detection(11, 4, 18, 12)]
    processor = PassageProcessor(
        _SequenceDetector([detections]),
        _config(),
        "source.avi",
    )
    result = processor.process(_crop_source(), 0, 0.0)
    assert [outcome.status for outcome in result.outcomes] == ["multiple_candidates"]
    assert not any(outcome.accepted for outcome in result.outcomes)


def test_empty_frame_produces_no_crop_or_outcome():
    processor = PassageProcessor(_SequenceDetector([[]]), _config(), "source.avi")
    result = processor.process(_crop_source(), 0, 0.0)
    assert result.outcomes == ()


def test_partial_glove_never_becomes_accepted_event():
    partial = _segmentation_detection(0, 5, 10, 15)
    config = _config()
    config.detector.trigger_zone = (0.2, 0.2, 0.8, 0.8)
    processor = PassageProcessor(
        _SequenceDetector([[partial], []]),
        config,
        "source.avi",
    )
    first = processor.process(_crop_source(), 0, 0.0)
    second = processor.process(_crop_source(), 1, 0.04)
    outcomes = first.outcomes + second.outcomes
    assert [outcome.status for outcome in outcomes] == ["partial"]
    assert not any(outcome.accepted for outcome in outcomes)


def test_time_mode_remains_stable_across_dropped_frame_intervals():
    detection = _segmentation_detection()
    config = _config(
        timing_mode="time",
        min_detected_seconds=0.08,
        exit_missing_seconds=0.20,
    )
    processor = PassageProcessor(
        _SequenceDetector([[detection], [detection], [], []]),
        config,
        "source.avi",
    )
    frame = _crop_source()
    outcomes = []
    for index, timestamp in enumerate((0.0, 0.10, 0.40, 0.65)):
        outcomes.extend(processor.process(frame, index * 4, timestamp).outcomes)
    assert len([outcome for outcome in outcomes if outcome.accepted]) == 1


def test_canonical_crop_is_identical_for_offline_and_deployment_callers():
    config = _config(crop_padding=0.25, crop_mode="masked")
    frame = _crop_source()
    detection = _segmentation_detection()
    offline_crop = create_event_crop(frame, detection, config)
    deployment_crop = create_event_crop(frame, detection, config)
    assert np.array_equal(offline_crop, deployment_crop)


def test_rearming_requires_continuous_empty_frames_and_prevents_duplicate_acceptance():
    detection = _segmentation_detection()
    config = _config(
        min_detected_frames=1,
        exit_missing_frames=1,
        cooldown_frames=2,
    )
    processor = PassageProcessor(
        _SequenceDetector([[detection], [], [detection], [detection], [detection], [], [], []]),
        config,
        "source.avi",
    )
    outcomes = []
    frame = _crop_source()
    for index in range(8):
        outcomes.extend(processor.process(frame, index, index / 25).outcomes)
    assert len([outcome for outcome in outcomes if outcome.accepted]) == 1


def test_timestamp_regression_fails_clearly():
    processor = PassageProcessor(_SequenceDetector([[], []]), _config(), "source.avi")
    processor.process(_crop_source(), 0, 1.0)
    with pytest.raises(ValueError, match="nondecreasing"):
        processor.process(_crop_source(), 1, 0.9)


def test_first_candidate_selection_is_independent_of_detector_order():
    left = _segmentation_detection(2, 5, 8, 11)
    right = _segmentation_detection(12, 5, 18, 11)
    config = _config(
        min_detected_frames=1,
        reject_multiple_detections=False,
    )
    selected = []
    for ordered in ([right, left], [left, right]):
        processor = PassageProcessor(
            _SequenceDetector([ordered, []]),
            config,
            "source.avi",
        )
        processor.process(_crop_source(), 0, 0.0)
        outcome = processor.process(_crop_source(), 1, 0.04).outcomes[0]
        selected.append(outcome.detection.x1)
    assert selected == [2, 2]

def test_second_detection_outside_trigger_does_not_create_ambiguity():
    inside = _segmentation_detection(
        6, 6, 12, 12
    )

    outside = _segmentation_detection(
        0, 6, 3, 12
    )

    config = _config(
        min_detected_frames=1,
        exit_missing_frames=1,
        cooldown_frames=0,
        reject_multiple_detections=True,
    )

    config.detector.trigger_zone = (
        0.2,
        0.2,
        0.8,
        0.8,
    )

    processor = PassageProcessor(
        _SequenceDetector([
            [inside, outside],
            [],
        ]),
        config,
        "source.avi",
    )

    frame = _crop_source()

    outcomes = []

    outcomes.extend(
        processor.process(
            frame,
            0,
            0.0,
        ).outcomes
    )

    outcomes.extend(
        processor.process(
            frame,
            1,
            0.04,
        ).outcomes
    )

    assert not any(
        outcome.status == "multiple_candidates"
        for outcome in outcomes
    )

    assert len([
        outcome
        for outcome in outcomes
        if outcome.accepted
    ]) == 1

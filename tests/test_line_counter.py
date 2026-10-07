import numpy as np
import pytest

from glove_chirality.config import ExtractionConfig
from glove_chirality.events import (
    PassageProcessor,
    keep_selected_mask,
    suppress_other_gloves,
)
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


def test_pixels_outside_the_selected_mask_are_removed_for_any_color():
    crop = np.zeros((20, 20, 3), dtype=np.uint8)
    crop[:, :] = (40, 40, 200)
    target = Detection(0, 0, 20, 20, 0.9, polygon=((2, 2), (12, 2), (12, 18), (2, 18)))
    cleaned = keep_selected_mask(crop, (0, 0), target, 114)
    assert tuple(cleaned[8, 6]) == (40, 40, 200)
    assert tuple(cleaned[8, 16]) == (114, 114, 114)


def test_other_glove_is_painted_out_of_the_crop():
    crop = np.zeros((20, 20, 3), dtype=np.uint8)
    crop[:, :] = (0, 180, 0)
    target = Detection(0, 0, 20, 20, 0.9, polygon=((0, 0), (12, 0), (12, 20), (0, 20)))
    other = Detection(10, 0, 20, 20, 0.9, polygon=((14, 0), (20, 0), (20, 20), (14, 20)))
    cleaned = suppress_other_gloves(crop, (0, 0), target, [other], 114)
    assert cleaned[10, 16, 1] == 114
    assert cleaned[10, 4, 1] == 180


def test_one_crossing_is_one_crop():
    accepted = [
        item
        for item in _run([[_box(100, 150)], [_box(100, 130)], [_box(100, 110)], [_box(100, 90)]])
        if item.accepted
    ]
    assert len(accepted) == 1


@pytest.mark.parametrize("centers, reason", [
    ((90, 80), "born_past_line"),
    ((120, 110), "lost_before_trigger"),
])
@pytest.mark.parametrize("expire", [False, True])
def test_unobserved_crossings_remain_explicit_rejections(centers, reason, expire):
    frames = [[_box(100, cy)] for cy in centers]
    if expire:
        frames.extend([[] for _ in range(12)])
    outcomes = _run(frames)
    assert len(outcomes) == 1
    assert not outcomes[0].accepted
    assert outcomes[0].reject_reason == reason
    assert outcomes[0].crop is None


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


def test_overlapping_mask_jitter_does_not_count_the_same_glove_twice():
    # A shape/center change must not let velocity extrapolation throw away
    # an overlapping, already-counted glove and arm a replacement identity.
    frames = [[_box(100, cy, size=60)] for cy in (130, 90, 115, 90)]
    accepted = [item for item in _run(frames) if item.accepted]
    assert len(accepted) == 1


def test_overlap_across_a_blink_preserves_the_original_identity():
    frames = [[_box(100, 150)], [_box(100, 135)], [], [], [], [], [_box(100, 110)]]
    processor = PassageProcessor(_Script(frames), _config(), "belt.mp4", "live")
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    for index in range(len(frames)):
        processor.process(image, index, index * 0.1)
    births = [event for event in processor._line_crossings().debug_events if event["kind"] == "birth"]
    assert len(births) == 1


def test_near_identical_query_masks_do_not_create_two_crossings():
    frames = []
    for cy in (140, 120, 100, 80):
        box = _box(100, cy, size=50)
        polygon = ((box.x1, box.y1), (box.x2, box.y1), (box.x2, box.y2), (box.x1, box.y2))
        frames.append([
            Detection(box.x1, box.y1, box.x2, box.y2, 0.9, polygon=polygon),
            Detection(box.x1, box.y1, box.x2, box.y2, 0.8, polygon=polygon),
        ])
    assert len([event for event in _run(frames) if event.accepted]) == 1


def test_line_candidate_is_selected_without_sharpness_or_confidence_ranking(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("sharpness must not run on the line extraction path")

    monkeypatch.setattr("glove_chirality.line_counter.cv2.Laplacian", unexpected)
    frames = []
    for cy in (124, 120, 116, 112, 108, 104, 99):
        box = _box(100, cy, size=40)
        frames.append([Detection(box.x1, box.y1, box.x2, box.y2, 0.7 if cy == 99 else 0.95)])
    accepted = [event for event in _run(frames) if event.accepted]
    assert len(accepted) == 1
    assert accepted[0].frame_index == 6
    assert accepted[0].quality_score == 0.7


def test_only_emitted_candidate_is_masked_and_its_buffer_is_released(monkeypatch):
    import glove_chirality.line_counter as module

    calls = []
    original = module.keep_selected_mask

    def isolate(*args):
        calls.append(args[2])
        return original(*args)

    monkeypatch.setattr(module, "keep_selected_mask", isolate)
    frames = [[_box(100, cy, size=40)] for cy in (124, 120, 116, 112, 108, 104, 99, 96, 92, 88)]
    processor = PassageProcessor(_Script(frames), _config(), "belt.mp4", "live")
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    outcomes = []
    for index in range(len(frames)):
        outcomes.extend(processor.process(image, index, index * 0.1).outcomes)
    assert len([event for event in outcomes if event.accepted]) == 1
    assert len(calls) == 1
    sighting = processor._line_crossings()._sightings[0]
    assert sighting.emitted and sighting.shot is None


def test_saved_candidate_owns_its_pixels_and_preserves_gray_mask_background():
    polygon = ((90, 100), (110, 100), (110, 120), (90, 120))
    first = Detection(80, 90, 120, 130, 0.9, polygon=polygon)
    second = Detection(80, 70, 120, 110, 0.9, polygon=((90, 80), (110, 80), (110, 100), (90, 100)))
    processor = PassageProcessor(_Script([[first], [second]]), _config(), "belt.mp4", "live")
    frame = np.full((200, 200, 3), 23, dtype=np.uint8)
    assert not processor.process(frame, 0, 0.0).outcomes
    frame[:] = 201  # Camera buffers may be overwritten before the crossing.
    accepted = [event for event in processor.process(frame, 1, 0.1).outcomes if event.accepted]
    assert len(accepted) == 1
    assert accepted[0].frame_index == 0  # Equal line distance keeps the first candidate.
    assert tuple(accepted[0].crop[16, 16]) == (23, 23, 23)
    assert tuple(accepted[0].crop[0, 0]) == (114, 114, 114)


def test_compiled_mask_fill_matches_boolean_fill_including_strided_views():
    import cv2

    target = Detection(20, 30, 90, 80, 0.9, polygon=((15, 25), (61, 32), (83, 69), (38, 94)))
    rng = np.random.default_rng(17)
    for stride in (1, 2):
        for fill in (0, 114, 255):
            parent = rng.integers(0, 256, (80, 120, 3), dtype=np.uint8)
            crop = parent[::stride, ::stride]
            expected = crop.copy()
            mask = np.zeros(crop.shape[:2], dtype=np.uint8)
            points = np.rint(np.asarray(target.polygon) - (20, 30)).astype(np.int32)
            cv2.fillPoly(mask, [points], 255)
            expected[mask == 0] = fill
            assert keep_selected_mask(crop, (20, 30), target, fill) is crop
            assert np.array_equal(crop, expected)


def test_candidate_foreign_masks_exclude_self_but_retain_neighbors():
    first = [_box(80, 120), _box(150, 120)]
    second = [_box(80, 110), _box(150, 110)]
    processor = PassageProcessor(_Script([first, second]), _config(), "belt.mp4", "live")
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    processor.process(frame, 0, 0.0)
    processor.process(frame, 1, 0.1)
    sightings = processor._line_crossings()._sightings
    assert sightings[0].shot.others == (second[1],)
    assert sightings[1].shot.others == (second[0],)


def test_instance_mask_alone_matches_foreign_suppression_then_mask():
    from glove_chirality.events import suppress_other_gloves

    target = Detection(80, 90, 120, 130, 0.9, polygon=((85, 95), (115, 100), (110, 125), (90, 120)))
    neighbors = [
        target,
        Detection(95, 85, 125, 135, 0.8, polygon=((95, 85), (125, 95), (120, 135))),
        Detection(70, 100, 100, 140, 0.8),
    ]
    raw = np.random.default_rng(23).integers(0, 256, (40, 40, 3), dtype=np.uint8)
    for fill in (0, 114, 255):
        previous = suppress_other_gloves(raw.copy(), (80, 90), target, neighbors, fill)
        previous = keep_selected_mask(previous, (80, 90), target, fill)
        simplified = keep_selected_mask(raw.copy(), (80, 90), target, fill)
        assert np.array_equal(previous, simplified)


def test_segmented_passage_does_not_prepare_redundant_foreign_masks(monkeypatch):
    import glove_chirality.line_counter as counter_module

    def unexpected(*args, **kwargs):
        raise AssertionError("The selected instance mask already removes all outside pixels")

    monkeypatch.setattr(counter_module, "suppress_other_gloves", unexpected)
    frames = []
    for cy in (150, 110, 90):
        box = _box(100, cy)
        frames.append([Detection(box.x1, box.y1, box.x2, box.y2, 0.9,
                                 polygon=((70, cy - 30), (130, cy - 30), (130, cy + 30), (70, cy + 30)))])
    accepted = [event for event in _run(frames) if event.accepted]
    assert len(accepted) == 1

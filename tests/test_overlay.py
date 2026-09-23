import numpy as np

from glove_chirality.config import DetectorConfig, EventConfig, ExtractionConfig
from glove_chirality.events import trigger_line_position
from glove_chirality.overlay import detection_states, draw_live_overlay
from glove_chirality.types import Detection

WIDTH = 200
HEIGHT = 160


def _config(**event_overrides) -> ExtractionConfig:
    return ExtractionConfig(
        detector=DetectorConfig(
            roi=(0.10, 0.10, 0.90, 0.90),
            trigger_zone=(0.25, 0.25, 0.75, 0.75),
            require_full_containment=True,
        ),
        event=EventConfig(**event_overrides),
    )


def _eligible(confidence: float = 0.91, shift: int = 0) -> Detection:
    return Detection(60 + shift, 50, 110 + shift, 100, confidence)


def _partial() -> Detection:
    return Detection(0, 0, 30, 30, 0.42)


def _line_pixel(config: ExtractionConfig) -> tuple[int, int]:
    axis, position = trigger_line_position(config, WIDTH, HEIGHT)
    coord = round(position)
    if axis == "y":
        return 70, coord
    return coord, 50


def test_overlay_does_not_mutate_frame_and_draws_roi_and_trigger():
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    before = frame.copy()
    config = _config()
    display = draw_live_overlay(frame, config, [_eligible(), _partial()])

    assert display is not frame
    assert np.array_equal(frame, before)
    assert np.array_equal(frame, np.zeros_like(frame))
    roi_x, roi_y = round(0.10 * WIDTH), round(0.90 * HEIGHT)
    trigger_x, trigger_y = round(0.25 * WIDTH), round(0.75 * HEIGHT)
    assert np.any(display[roi_y, roi_x])
    assert np.any(display[trigger_y, trigger_x])
    assert not np.any(before[roi_y, roi_x])
    assert not np.any(before[trigger_y, trigger_x])


def test_detection_states_eligible_partial_and_multiple():
    config = _config(reject_multiple_detections=True)
    eligible = _eligible()
    partial = _partial()
    assert detection_states(config, [eligible], WIDTH, HEIGHT) == ["ELIGIBLE"]
    assert detection_states(config, [partial], WIDTH, HEIGHT) == ["PARTIAL"]
    assert detection_states(config, [partial, eligible], WIDTH, HEIGHT) == [
        "PARTIAL",
        "ELIGIBLE",
    ]

    multiple = detection_states(
        config,
        [_eligible(0.91), _eligible(0.80, shift=20)],
        WIDTH,
        HEIGHT,
    )
    assert multiple == ["MULTIPLE", "MULTIPLE"]

    allowing = _config(reject_multiple_detections=False)
    assert detection_states(
        allowing,
        [_eligible(0.91), _eligible(0.80, shift=20)],
        WIDTH,
        HEIGHT,
    ) == ["ELIGIBLE", "ELIGIBLE"]


def test_trigger_line_absent_when_disabled_and_present_when_enabled():
    blank = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    disabled_config = _config(trigger_line_enabled=False, belt_direction="left_to_right")
    enabled_config = _config(
        trigger_line_enabled=True,
        belt_direction="left_to_right",
        trigger_line_fraction=0.5,
    )
    horizontal = _config(
        trigger_line_enabled=True,
        belt_direction="top_to_bottom",
        trigger_line_fraction=0.5,
    )
    disabled = draw_live_overlay(blank, disabled_config, [])
    enabled = draw_live_overlay(blank, enabled_config, [])
    sideways = draw_live_overlay(blank, horizontal, [])

    vertical = _line_pixel(enabled_config)
    horizontal_pixel = _line_pixel(horizontal)
    assert vertical == (100, 50)
    assert horizontal_pixel == (70, 80)
    assert np.array_equal(disabled[vertical[1], vertical[0]], blank[vertical[1], vertical[0]])
    assert np.array_equal(
        disabled[horizontal_pixel[1], horizontal_pixel[0]],
        blank[horizontal_pixel[1], horizontal_pixel[0]],
    )
    assert not np.array_equal(enabled[vertical[1], vertical[0]], disabled[vertical[1], vertical[0]])
    assert np.array_equal(
        enabled[horizontal_pixel[1], horizontal_pixel[0]],
        disabled[horizontal_pixel[1], horizontal_pixel[0]],
    )
    assert not np.array_equal(
        sideways[horizontal_pixel[1], horizontal_pixel[0]],
        disabled[horizontal_pixel[1], horizontal_pixel[0]],
    )
    assert np.array_equal(
        sideways[vertical[1], vertical[0]],
        disabled[vertical[1], vertical[0]],
    )


def test_polygon_is_drawn_when_provided_and_omitted_otherwise():
    blank = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    config = _config(trigger_line_enabled=False)
    polygon = ((70, 65), (100, 65), (85, 88))
    with_polygon = Detection(60, 50, 120, 110, 0.93, polygon=polygon)
    without_polygon = Detection(60, 50, 120, 110, 0.93)
    drawn = draw_live_overlay(blank, config, [with_polygon])
    plain = draw_live_overlay(blank, config, [without_polygon])

    assert np.array_equal(blank, np.zeros_like(blank))
    assert not np.array_equal(drawn, plain)
    assert np.any(drawn[64:67, 70:100])
    assert not np.any(plain[64:67, 70:100])
    center = with_polygon.center
    cx, cy = round(center[0]), round(center[1])
    assert np.any(drawn[cy, cx])
    assert np.any(plain[cy, cx])

"""Display-only live overlay.

Drawing never mutates the input frame, the config, or detections, and it never
calls a detector. Passage decisions stay in the event processor.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from glove_chirality.config import ExtractionConfig
from glove_chirality.detection.base import inside_trigger
from glove_chirality.types import Detection

_ROI_COLOR = (255, 180, 0)
_TRIGGER_COLOR = (0, 255, 255)
_LINE_COLOR = (255, 0, 255)
_ELIGIBLE_COLOR = (0, 200, 0)
_PARTIAL_COLOR = (0, 0, 255)
_MULTIPLE_COLOR = (0, 140, 255)
_TEXT_COLOR = (255, 255, 255)
_BANNER_COLOR = (24, 24, 24)
_STATE_COLORS = {
    "ELIGIBLE": _ELIGIBLE_COLOR,
    "PARTIAL": _PARTIAL_COLOR,
    "MULTIPLE": _MULTIPLE_COLOR,
}


def detection_states(
    config: ExtractionConfig,
    detections: Sequence[Detection],
    width: int,
    height: int,
) -> list[str]:
    """Return ELIGIBLE, PARTIAL, or MULTIPLE for each detection, in order."""
    inside = [
        inside_trigger(detection, config.detector, width, height) for detection in detections
    ]
    reject_multiple = config.event.reject_multiple_detections and sum(inside) > 1
    states: list[str] = []
    for is_inside in inside:
        if not is_inside:
            states.append("PARTIAL")
        elif reject_multiple:
            states.append("MULTIPLE")
        else:
            states.append("ELIGIBLE")
    return states


def draw_live_overlay(
    frame: np.ndarray,
    config: ExtractionConfig,
    detections: Sequence[Detection],
) -> np.ndarray:
    """Return an annotated copy. The input frame is not modified."""
    image = frame.copy()
    height, width = image.shape[:2]
    items = tuple(detections)
    states = detection_states(config, items, width, height)
    eligible = sum(state != "PARTIAL" for state in states)

    _draw_zone(image, config.detector.roi, width, height, _ROI_COLOR, "ROI")
    _draw_zone(image, config.detector.trigger_zone, width, height, _TRIGGER_COLOR, "TRIGGER")
    _draw_trigger_line(image, config, width, height)
    for index, (detection, state) in enumerate(zip(items, states), start=1):
        _draw_detection(image, detection, state, index)
    _draw_banner(
        image,
        detection_count=len(items),
        eligible_count=eligible,
        reject_passage=config.event.reject_multiple_detections and eligible > 1,
    )
    return image


def _pixel_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        round(x1 * width),
        round(y1 * height),
        round(x2 * width),
        round(y2 * height),
    )


def _local_trigger_line_position(
    config: ExtractionConfig,
    width: int,
    height: int,
) -> tuple[str, float]:
    """Same geometry as glove_chirality.events.trigger_line_position."""
    direction = config.event.belt_direction
    tx1, ty1, tx2, ty2 = config.detector.trigger_zone
    fraction = config.event.trigger_line_fraction
    if direction in {"top_to_bottom", "bottom_to_top"}:
        return "y", float((ty1 + fraction * (ty2 - ty1)) * height)
    return "x", float((tx1 + fraction * (tx2 - tx1)) * width)


def _trigger_line_segment(
    config: ExtractionConfig,
    width: int,
    height: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if not config.event.trigger_line_enabled:
        return None
    try:
        from glove_chirality.events import trigger_line_position
    except ImportError:
        axis, position = _local_trigger_line_position(config, width, height)
    else:
        axis, position = trigger_line_position(config, width, height)
    tx1, ty1, tx2, ty2 = _pixel_box(config.detector.trigger_zone, width, height)
    coord = round(position)
    if axis == "y":
        return (tx1, coord), (tx2, coord)
    return (coord, ty1), (coord, ty2)


def _draw_zone(image, box, width: int, height: int, color, name: str) -> None:
    x1, y1, x2, y2 = _pixel_box(box, width, height)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        name,
        (x1 + 4, min(y2 - 4, y1 + 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_trigger_line(image, config: ExtractionConfig, width: int, height: int) -> None:
    segment = _trigger_line_segment(config, width, height)
    if segment is None:
        return
    cv2.line(image, segment[0], segment[1], _LINE_COLOR, 2, cv2.LINE_8)


def _draw_detection(image, detection: Detection, state: str, index: int) -> None:
    color = _STATE_COLORS[state]
    if detection.polygon:
        points = np.rint(np.asarray(detection.polygon, dtype=np.float32)).astype(np.int32)
        cv2.polylines(image, [points], True, color, 2, cv2.LINE_8)
    cv2.rectangle(
        image,
        (detection.x1, detection.y1),
        (detection.x2, detection.y2),
        color,
        2,
    )
    center = (round(detection.center[0]), round(detection.center[1]))
    cv2.circle(image, center, 4, color, -1, cv2.LINE_8)
    label = f"GLOVE {index} | {detection.confidence:.2f} | {state}"
    text_y = detection.y1 - 8
    if text_y < 52:
        text_y = min(image.shape[0] - 6, detection.y2 + 16)
    cv2.putText(
        image,
        label,
        (detection.x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_banner(image, *, detection_count: int, eligible_count: int, reject_passage: bool) -> None:
    width = image.shape[1]
    bar_height = 48 if reject_passage else 26
    cv2.rectangle(image, (0, 0), (width - 1, bar_height), _BANNER_COLOR, -1)
    cv2.putText(
        image,
        f"DETECTIONS: {detection_count}  ELIGIBLE: {eligible_count}",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        _TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    if reject_passage:
        cv2.putText(
            image,
            "MULTIPLE GLOVES - PASSAGE REJECTED",
            (8, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _MULTIPLE_COLOR,
            1,
            cv2.LINE_AA,
        )

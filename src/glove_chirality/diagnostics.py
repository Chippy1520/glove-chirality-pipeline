from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from glove_chirality.config import ExtractionConfig
from glove_chirality.detection import build_detector
from glove_chirality.detection.base import inside_trigger


def _pixel_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height)


def save_calibration_preview(
    video: str | Path,
    output: str | Path,
    config: ExtractionConfig,
    seconds: float = 0.0,
    warmup_seconds: float = 2.0,
):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    target_frame = max(0, round(seconds * fps))
    warmup_frames = max(0, round(warmup_seconds * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_frame - warmup_frames))
    detector = build_detector(config.detector)
    clean_frame = None
    detections = []
    while capture.get(cv2.CAP_PROP_POS_FRAMES) <= target_frame:
        ok, current = capture.read()
        if not ok:
            break
        clean_frame = current
        detections = detector.detect(current)
    capture.release()
    if clean_frame is None:
        raise RuntimeError("Could not decode preview frame")

    frame = clean_frame.copy()
    height, width = frame.shape[:2]
    if config.diagnostics.show_masks:
        overlay = frame.copy()
        for detection in detections:
            if detection.polygon is None:
                continue
            points = np.rint(np.asarray(detection.polygon)).astype(np.int32)
            cv2.fillPoly(overlay, [points], (80, 220, 80))
        cv2.addWeighted(
            overlay,
            config.diagnostics.mask_alpha,
            frame,
            1.0 - config.diagnostics.mask_alpha,
            0,
            frame,
        )

    for box, color, name in [
        (config.detector.roi, (255, 180, 0), "ROI"),
        (config.detector.trigger_zone, (0, 255, 255), "TRIGGER"),
    ]:
        x1, y1, x2, y2 = _pixel_box(box, width, height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            frame,
            name,
            (x1 + 6, y1 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

    ambiguous = config.event.reject_multiple_detections and len(detections) > 1
    cv2.putText(
        frame,
        f"CANDIDATES: {len(detections)}",
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    if ambiguous:
        cv2.putText(
            frame,
            "AMBIGUOUS",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 165, 255),
            3,
        )

    roi_px = _pixel_box(config.detector.roi, width, height)
    trigger_px = _pixel_box(config.detector.trigger_zone, width, height)
    for detection in detections:
        eligible = inside_trigger(detection, config.detector, width, height)
        color = (0, 165, 255) if ambiguous else ((0, 200, 0) if eligible else (0, 0, 255))
        if config.diagnostics.show_masks and detection.polygon is not None:
            points = np.rint(np.asarray(detection.polygon)).astype(np.int32)
            cv2.polylines(frame, [points], True, color, 2)
        cv2.rectangle(
            frame,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            color,
            3,
        )
        clearance = min(
            detection.x1 - trigger_px[0],
            detection.y1 - trigger_px[1],
            trigger_px[2] - detection.x2,
            trigger_px[3] - detection.y2,
        )
        statuses = ["ELIGIBLE" if eligible else "PARTIAL"]
        if (
            detection.x1 <= roi_px[0]
            or detection.y1 <= roi_px[1]
            or detection.x2 >= roi_px[2]
            or detection.y2 >= roi_px[3]
        ):
            statuses.append("ROI EDGE")
        if (
            detection.x1 <= 0
            or detection.y1 <= 0
            or detection.x2 >= width
            or detection.y2 >= height
        ):
            statuses.append("FRAME EDGE")
        cv2.putText(
            frame,
            f"{detection.confidence:.2f} {'/'.join(statuses)} clr={clearance}px",
            (detection.x1, max(18, detection.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), frame):
        raise RuntimeError(f"Could not write preview: {output}")
    return output

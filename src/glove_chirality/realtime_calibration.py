from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from glove_chirality.config import ExtractionConfig
from glove_chirality.detection.yolo import YoloDetectionDiagnostics
from glove_chirality.types import Detection


def validate_calibration_config(config: ExtractionConfig) -> None:
    if config.detector.backend != "yolo":
        raise ValueError("Real-time detector calibration requires detector.backend: yolo")


def normalized_box_pixels(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height)


def _draw_detection(
    image: np.ndarray,
    detection: Detection,
    color: tuple[int, int, int],
    label: str,
    frame_area: int,
) -> None:
    if detection.polygon:
        points = np.asarray(detection.polygon, dtype=np.int32)
        overlay = image.copy()
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, 0.22, image, 0.78, 0.0, image)
        cv2.polylines(image, [points], True, color, 2)
    cv2.rectangle(image, (detection.x1, detection.y1), (detection.x2, detection.y2), color, 2)
    ratio = detection.area / max(1, frame_area)
    cv2.putText(
        image,
        f"{label} conf={detection.confidence:.3f} area={ratio:.4f}",
        (detection.x1, max(18, detection.y1 - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_calibration_overlay(
    frame: np.ndarray,
    config: ExtractionConfig,
    detections: list[Detection],
    diagnostics: YoloDetectionDiagnostics,
    *,
    fps: float,
    size_filter_enabled: bool,
    show_size_rejected: bool,
) -> np.ndarray:
    image = frame.copy()
    height, width = image.shape[:2]
    frame_area = max(1, width * height)
    roi = normalized_box_pixels(config.detector.roi, width, height)
    trigger = normalized_box_pixels(config.detector.trigger_zone, width, height)
    cv2.rectangle(image, roi[:2], roi[2:], (255, 180, 0), 2)
    cv2.rectangle(image, trigger[:2], trigger[2:], (0, 255, 255), 2)
    cv2.putText(image, "ROI", (roi[0] + 4, roi[1] + 18), 0, 0.55, (255, 180, 0), 2)
    cv2.putText(
        image,
        "TRIGGER",
        (trigger[0] + 4, trigger[1] + 18),
        0,
        0.55,
        (0, 255, 255),
        2,
    )

    rejected = {item.detection for item in diagnostics.size_rejected}
    for detection in detections:
        if detection not in rejected:
            _draw_detection(image, detection, (0, 210, 0), "GLOVE", frame_area)
    if show_size_rejected:
        for item in diagnostics.size_rejected:
            _draw_detection(image, item.detection, (0, 0, 255), "REJECT SIZE", frame_area)

    status = (
        f"FPS {fps:.1f} | raw {diagnostics.raw_yolo_count} | "
        f"size reject {diagnostics.size_rejected_count} | "
        f"returned {diagnostics.returned_detection_count} | "
        f"filter {'ON' if size_filter_enabled else 'OFF'}"
    )
    cv2.rectangle(image, (0, 0), (width, 30), (0, 0, 0), -1)
    cv2.putText(image, status, (8, 21), 0, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def screenshot_path(directory: str | Path, now: datetime | None = None) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S_%f")
    return directory / f"calibration_{timestamp}.png"


def save_screenshot(image: np.ndarray, directory: str | Path) -> Path:
    path = screenshot_path(directory)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not save calibration screenshot: {path}")
    return path

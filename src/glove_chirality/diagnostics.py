from __future__ import annotations

from pathlib import Path

import cv2

from glove_chirality.config import ExtractionConfig
from glove_chirality.detection import build_detector


def save_calibration_preview(video: str | Path, output: str | Path, config: ExtractionConfig, seconds: float = 0.0):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    if seconds > 0:
        capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("Could not decode preview frame")
    detections = build_detector(config.detector).detect(frame)
    height, width = frame.shape[:2]
    for box, color, name in [
        (config.detector.roi, (255, 180, 0), "ROI"),
        (config.detector.trigger_zone, (0, 255, 255), "TRIGGER"),
    ]:
        x1, y1, x2, y2 = box
        p1, p2 = (round(x1 * width), round(y1 * height)), (round(x2 * width), round(y2 * height))
        cv2.rectangle(frame, p1, p2, color, 3)
        cv2.putText(frame, name, (p1[0] + 6, p1[1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    ambiguous = config.event.reject_multiple_detections and len(detections) > 1
    detection_color = (0, 165, 255) if ambiguous else (0, 0, 255)
    if ambiguous:
        cv2.putText(frame, f"AMBIGUOUS: {len(detections)} candidates", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, detection_color, 2)
    for detection in detections:
        cv2.rectangle(frame, (detection.x1, detection.y1), (detection.x2, detection.y2), detection_color, 3)
        cv2.putText(frame, f"{detection.confidence:.2f}", (detection.x1, detection.y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, detection_color, 2)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), frame):
        raise RuntimeError(f"Could not write preview: {output}")
    return output

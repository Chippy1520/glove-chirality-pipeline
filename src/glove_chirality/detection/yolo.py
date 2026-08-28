from __future__ import annotations

import numpy as np

from glove_chirality.config import DetectorConfig
from glove_chirality.detection.base import GloveDetector, inside_trigger
from glove_chirality.types import Detection


def _polygon_box(polygon: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(points) < 3:
        return None
    x1 = max(0, int(np.floor(points[:, 0].min())))
    y1 = max(0, int(np.floor(points[:, 1].min())))
    x2 = min(width, int(np.ceil(points[:, 0].max())))
    y2 = min(height, int(np.ceil(points[:, 1].max())))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


class YoloDetector(GloveDetector):
    """Optional Ultralytics adapter for a custom glove detector."""

    name = "yolo"

    def __init__(self, config: DetectorConfig):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install the YOLO option with: pip install -e .[yolo]") from exc
        self.config = config
        self.model = YOLO(config.yolo_model)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        options: dict[str, object] = {
            "conf": self.config.yolo_confidence,
            "verbose": False,
            "half": self.config.yolo_half,
        }
        if self.config.yolo_device != "auto":
            options["device"] = self.config.yolo_device
        result = self.model.predict(frame, **options)[0]
        detections: list[Detection] = []
        polygons = result.masks.xy if self.config.yolo_use_masks and result.masks is not None else []
        for index, box in enumerate(result.boxes):
            class_id = int(box.cls.item())
            if self.config.yolo_class_id is not None and class_id != self.config.yolo_class_id:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            if index < len(polygons):
                x1, y1, x2, y2 = _polygon_box(polygons[index], width, height) or (x1, y1, x2, y2)
            detection = Detection(x1, y1, x2, y2, float(box.conf.item()), class_id)
            if inside_trigger(detection, self.config, width, height):
                detections.append(detection)
        return detections

from __future__ import annotations

import numpy as np

from glove_chirality.config import DetectorConfig
from glove_chirality.detection.base import GloveDetector, inside_trigger
from glove_chirality.types import Detection


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
        for box in result.boxes:
            class_id = int(box.cls.item())
            if self.config.yolo_class_id is not None and class_id != self.config.yolo_class_id:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            detection = Detection(x1, y1, x2, y2, float(box.conf.item()), class_id)
            if inside_trigger(detection, self.config, width, height):
                detections.append(detection)
        return detections

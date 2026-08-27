from glove_chirality.config import DetectorConfig
from glove_chirality.detection.base import GloveDetector
from glove_chirality.detection.classical import DarkContourDetector


def build_detector(config: DetectorConfig) -> GloveDetector:
    if config.backend == "dark_contour":
        return DarkContourDetector(config)
    if config.backend == "yolo":
        from glove_chirality.detection.yolo import YoloDetector

        return YoloDetector(config)
    raise ValueError(f"Unknown detector backend: {config.backend!r}")

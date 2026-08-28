from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from glove_chirality.config import DetectorConfig
from glove_chirality.types import Detection


class GloveDetector(ABC):
    """Backend contract shared by dataset extraction and inference."""

    name: str

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return candidates in full-frame pixel coordinates."""

    def warmup(self, frame: np.ndarray) -> None:
        """Optionally initialize model kernels without creating an event."""


def inside_trigger(
    detection: Detection,
    config: DetectorConfig,
    width: int,
    height: int,
) -> bool:
    """Apply one trigger policy consistently across detector backends."""
    x1, y1, x2, y2 = config.trigger_zone
    margin = config.trigger_inner_margin_ratio
    inner_x1 = (x1 + margin * (x2 - x1)) * width
    inner_x2 = (x2 - margin * (x2 - x1)) * width
    inner_y1 = (y1 + margin * (y2 - y1)) * height
    inner_y2 = (y2 - margin * (y2 - y1)) * height
    if config.require_full_containment:
        return (
            detection.x1 >= inner_x1
            and detection.y1 >= inner_y1
            and detection.x2 <= inner_x2
            and detection.y2 <= inner_y2
        )
    center_x, center_y = detection.center
    return inner_x1 <= center_x <= inner_x2 and inner_y1 <= center_y <= inner_y2

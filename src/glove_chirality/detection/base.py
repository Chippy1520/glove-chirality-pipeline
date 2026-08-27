from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from glove_chirality.types import Detection


class GloveDetector(ABC):
    """Backend contract shared by dataset extraction and inference."""

    name: str

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return candidates in full-frame pixel coordinates."""

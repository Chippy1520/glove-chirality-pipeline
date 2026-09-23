import numpy as np

from glove_chirality.config import ExtractionConfig
from glove_chirality.factory_live import FactoryLiveSession
from glove_chirality.types import Detection


class _Result:
    def __init__(self, detections):
        self.detections = detections


def test_preview_is_not_built_until_the_browser_asks(tmp_path):
    session = FactoryLiveSession(tmp_path)
    session._config = ExtractionConfig()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    detection = Detection(100, 100, 400, 500, 0.9)
    session._on_frame(frame, _Result([detection]), 0.0)
    assert session._preview_source is not None
    assert session._jpeg_cache is None
    encoded = session.frame_jpeg()
    assert encoded
    assert session.frame_jpeg() is encoded
    assert session._preview_source.shape == frame.shape

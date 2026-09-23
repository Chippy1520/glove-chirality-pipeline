import numpy as np

from glove_chirality.config import ExtractionConfig
from glove_chirality.factory_live import FactoryLiveSession, camera_request
from glove_chirality.types import Detection


class _Result:
    def __init__(self, detections):
        self.detections = detections


def test_grip_fps_choice_reaches_the_camera_request():
    request = camera_request({"camera_preset": "grip_1080p", "camera_fps": "30"})
    assert (request["width"], request["height"], request["fps"]) == (1920, 1080, 30.0)
    assert camera_request({"camera_preset": "auto", "camera_fps": "auto"}) == {}
    assert camera_request({"camera_preset": "auto", "camera_fps": "15"}) == {"fps": 15.0}


def test_new_glove_clears_the_previous_decision(tmp_path):
    session = FactoryLiveSession(tmp_path)
    session._config = ExtractionConfig()
    session._settled_phase = "PASS"
    session._decision = {"phase": "PASS"}
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    session._on_frame(frame, _Result([Detection(1, 1, 8, 8, 0.9)]), 0.0)
    assert session.decision() == {"phase": "INSPECTING", "running": False, "fault": None}
    session._on_event({
        "status": "accepted",
        "prediction": "right",
        "event_id": "g2",
        "confidence": 0.91,
    })
    assert session.decision()["phase"] == "REJECT"


def test_preview_encode_stays_off_the_detector_callback(tmp_path):
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

import numpy as np

from glove_chirality.config import ExtractionConfig
from glove_chirality.debug_birth import debug_video
from glove_chirality.detection.yolo import SizeRejectedDetection, YoloDetectionDiagnostics
from glove_chirality.types import Detection


def _config() -> ExtractionConfig:
    config = ExtractionConfig()
    config.detector.backend = "yolo"
    config.detector.roi = (0.0, 0.0, 1.0, 1.0)
    config.detector.trigger_zone = (0.0, 0.0, 1.0, 1.0)
    config.detector.yolo_model = "unused.pt"
    config.event.trigger_line_enabled = True
    config.event.belt_direction = "bottom_to_top"
    config.event.tracker_mode = "bytetrack"
    config.event.output_size = 32
    config.event.validate()
    return config


def _box(cx: int, cy: int, size: int = 30) -> Detection:
    half = size // 2
    return Detection(cx - half, cy - half, cx + half, cy + half, 0.9)


class _Detector:
    name = "yolo"

    def __init__(self):
        self.calls = 0

    def detect_with_diagnostics(self, frame):
        del frame
        self.calls += 1
        if self.calls == 1:
            partial = _box(100, 160, size=16)
            diagnostics = YoloDetectionDiagnostics(
                1,
                1,
                0,
                (SizeRejectedDetection(partial, 0.01),),
            )
            return [], diagnostics
        kept = [_box(100, 70)]
        return kept, YoloDetectionDiagnostics(1, 0, 1, ())

    def tracking_partials(self):
        return []


def test_debugger_names_a_discarded_upstream_box(monkeypatch, tmp_path):
    from glove_chirality import debug_birth

    monkeypatch.setattr(debug_birth, "build_detector", lambda config: _Detector())
    monkeypatch.setattr(debug_birth.cv2, "VideoCapture", _Capture)
    report = debug_video(tmp_path / "clip.mkv", _config())
    assert "size_rejected_upstream 1" in report
    assert "born_past_line_with_upstream_partial 1" in report
    assert "born_past_line_with_no_upstream_box 0" in report


class _Capture:
    def __init__(self, path):
        del path
        self.left = 2

    def isOpened(self):
        return True

    def get(self, prop):
        del prop
        return 10.0

    def read(self):
        if self.left == 0:
            return False, None
        self.left -= 1
        return True, np.zeros((200, 200, 3), dtype=np.uint8)

    def release(self):
        return None

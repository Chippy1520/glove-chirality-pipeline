from pathlib import Path

import cv2
import numpy as np

from glove_chirality.config import DetectorConfig, ExtractionConfig
from glove_chirality.diagnostics import save_calibration_preview


class _RecordingDetector:
    name = "recording"

    def __init__(self):
        self.frame = None
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        self.frame = frame.copy()
        return []


def _write_frame(path: Path, frame: np.ndarray):
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (width, height))
    assert writer.isOpened()
    writer.write(frame)
    writer.release()


def _write_frames(path: Path, frames: list[np.ndarray], fps: int):
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()


def _read_frame(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    capture.release()
    assert ok
    return frame


def test_preview_detects_clean_frame_before_drawing_overlays(tmp_path, monkeypatch):
    video = tmp_path / "frame.avi"
    source = np.full((180, 240, 3), (65, 175, 65), dtype=np.uint8)
    _write_frame(video, source)
    decoded = _read_frame(video)
    detector = _RecordingDetector()
    monkeypatch.setattr("glove_chirality.diagnostics.build_detector", lambda _config: detector)

    output = save_calibration_preview(
        video,
        tmp_path / "preview.jpg",
        ExtractionConfig(
            detector=DetectorConfig(
                roi=(0.1, 0.1, 0.9, 0.9),
                trigger_zone=(0.2, 0.2, 0.8, 0.8),
            )
        ),
    )

    assert detector.frame is not None
    assert np.array_equal(detector.frame, decoded)
    preview = cv2.imread(str(output))
    assert preview is not None
    assert not np.array_equal(preview, decoded)


def test_preview_warms_temporal_detector_before_target_frame(tmp_path, monkeypatch):
    video = tmp_path / "sequence.avi"
    frames = [np.full((80, 120, 3), index, dtype=np.uint8) for index in range(20)]
    _write_frames(video, frames, fps=10)
    detector = _RecordingDetector()
    monkeypatch.setattr("glove_chirality.diagnostics.build_detector", lambda _config: detector)

    save_calibration_preview(
        video,
        tmp_path / "preview.jpg",
        ExtractionConfig(),
        seconds=1.0,
        warmup_seconds=0.4,
    )

    assert detector.calls == 5

from pathlib import Path

import cv2
import numpy as np
import pytest

from glove_chirality.config import DetectorConfig, EventConfig, ExtractionConfig
from glove_chirality.extraction import (
    _letterbox,
    config_hash,
    event_rows,
    extract_video,
    write_manifest,
)
from glove_chirality.types import Detection


def _synthetic_video(path: Path, glove_color: tuple[int, int, int]):
    width, height, fps = 320, 240, 25
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    assert writer.isOpened()
    for index in range(100):
        frame = np.full((height, width, 3), (65, 175, 65), dtype=np.uint8)
        cv2.rectangle(frame, (20, 15), (300, 225), (90, 205, 90), 3)
        if 10 <= index < 75:
            y = -50 + (index - 10) * 4
            cv2.rectangle(frame, (120, y), (200, y + 85), glove_color, -1)
            for finger in range(4):
                x = 121 + finger * 20
                cv2.rectangle(frame, (x, y - 22), (x + 14, y + 8), glove_color, -1)
        writer.write(frame)
    writer.release()


@pytest.mark.parametrize("glove_color", [(35, 35, 35), (235, 235, 235), (30, 30, 220)])
def test_extracts_exactly_one_event_and_manifest(tmp_path, glove_color):
    video = tmp_path / "left_sample.avi"
    _synthetic_video(video, glove_color)
    config = ExtractionConfig(
        detector=DetectorConfig(
            roi=(0, 0, 1, 1),
            trigger_zone=(0.15, 0.15, 0.85, 0.85),
            color_distance_threshold=22,
            min_area_ratio=0.008,
            max_area_ratio=0.5,
            morph_kernel=5,
        ),
        event=EventConfig(min_detected_frames=2, exit_missing_frames=4, cooldown_frames=5, crop_padding=0.1),
    )
    output = tmp_path / "dataset"
    events = extract_video(video, output, "left", config)
    assert len(events) == 1
    detection = events[0].detection
    assert detection.x1 >= 0.15 * 320
    assert detection.y1 >= 0.15 * 240
    assert detection.x2 <= 0.85 * 320
    assert detection.y2 <= 0.85 * 240
    assert events[0].image_path.exists()
    crop = cv2.imread(str(events[0].image_path))
    assert crop is not None and crop.shape[:2] == (256, 256)
    rows = event_rows(events, output, config_hash(config))
    manifest = write_manifest(rows, output / "manifest.csv")
    assert manifest.exists()
    assert rows[0]["label"] == "left"
    assert rows[0]["label_provenance"] == "known_stream"
    assert rows[0]["status"] == "accepted"
    assert rows[0]["candidate_count"] == 1
    assert rows[0]["detector_confidence"]
    assert rows[0]["used_segmentation"] is False


def test_config_hash_excludes_nonsemantic_live_and_diagnostic_settings():
    baseline = ExtractionConfig()
    changed = ExtractionConfig()
    changed.diagnostics.show_masks = False
    changed.runtime.capture_queue_size = 1
    changed.runtime.report_interval_seconds = 20.0
    changed.runtime.warmup = False
    assert config_hash(changed) == config_hash(baseline)


def test_config_hash_includes_detection_frequency_and_crop_mode():
    baseline = ExtractionConfig()
    skipped = ExtractionConfig()
    skipped.runtime.detect_every_n_frames = 2
    masked = ExtractionConfig()
    masked.event.crop_mode = "masked"
    assert config_hash(skipped) != config_hash(baseline)
    assert config_hash(masked) != config_hash(baseline)


def test_multiple_candidates_do_not_create_an_arbitrary_crop(tmp_path, monkeypatch):
    video = tmp_path / "ambiguous.avi"
    _synthetic_video(video, (35, 35, 35))

    class AmbiguousDetector:
        name = "ambiguous"
        frames = 0

        def detect(self, _frame):
            self.frames += 1
            if self.frames <= 3:
                return [Detection(70, 60, 140, 160, 0.9)]
            return [
                Detection(70, 60, 140, 160, 0.9),
                Detection(180, 60, 250, 160, 0.8),
            ]

    monkeypatch.setattr("glove_chirality.extraction.build_detector", lambda _config: AmbiguousDetector())
    events = extract_video(video, tmp_path / "output", "unknown", ExtractionConfig())
    assert events == []


def test_letterbox_preserves_content_aspect_ratio():
    image = np.zeros((40, 80, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 10), (59, 29), (255, 255, 255), -1)
    result = _letterbox(image, 100)
    foreground = cv2.findNonZero(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))
    _x, _y, width, height = cv2.boundingRect(foreground)
    assert result.shape == (100, 100, 3)
    assert width / height == pytest.approx(2.0, rel=0.06)

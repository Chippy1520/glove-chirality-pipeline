from pathlib import Path

import cv2
import numpy as np
import pytest

from glove_chirality.config import DetectorConfig, EventConfig, ExtractionConfig
from glove_chirality.extraction import config_hash, event_rows, extract_video, write_manifest


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
    assert events[0].image_path.exists()
    crop = cv2.imread(str(events[0].image_path))
    assert crop is not None and crop.shape[0] == crop.shape[1]
    rows = event_rows(events, output, config_hash(config))
    manifest = write_manifest(rows, output / "manifest.csv")
    assert manifest.exists()
    assert rows[0]["label"] == "left"
    assert rows[0]["label_provenance"] == "known_stream"

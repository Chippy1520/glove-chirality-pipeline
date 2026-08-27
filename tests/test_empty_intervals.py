from pathlib import Path

import cv2
import numpy as np

from glove_chirality.config import DetectorConfig, EventConfig, ExtractionConfig
from glove_chirality.extraction import extract_video


def _video_with_passages(path: Path, starts: list[int], frames: int = 260):
    width, height = 320, 240
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (width, height))
    assert writer.isOpened()
    for frame_index in range(frames):
        frame = np.full((height, width, 3), (65, 175, 65), dtype=np.uint8)
        for start in starts:
            step = frame_index - start
            if 0 <= step < 35:
                y = -95 + step * 10
                cv2.rectangle(frame, (122, y), (198, y + 92), (220, 35, 35), -1)
                for finger in range(4):
                    x = 124 + finger * 18
                    cv2.rectangle(frame, (x, y - 25), (x + 12, y + 8), (220, 35, 35), -1)
        writer.write(frame)
    writer.release()


def _config():
    return ExtractionConfig(
        detector=DetectorConfig(
            roi=(0, 0, 1, 1),
            trigger_zone=(0.15, 0.15, 0.85, 0.85),
            color_distance_threshold=22,
            adaptive_background=True,
            mog_empty_learning_rate=0.05,
            mog_foreground_learning_rate=0.0,
            min_area_ratio=0.008,
            max_area_ratio=0.5,
            blur_kernel=3,
            morph_kernel=5,
        ),
        event=EventConfig(
            min_detected_frames=2,
            exit_missing_frames=4,
            cooldown_frames=5,
        ),
    )


def test_entirely_empty_conveyor_emits_no_events(tmp_path):
    video = tmp_path / "empty.avi"
    _video_with_passages(video, [], frames=150)
    events = extract_video(video, tmp_path / "empty_output", "unknown", _config())
    assert events == []


def test_long_empty_gaps_emit_only_physical_passages(tmp_path):
    video = tmp_path / "spaced.avi"
    _video_with_passages(video, [35, 155], frames=240)
    events = extract_video(video, tmp_path / "spaced_output", "left", _config())
    assert len(events) == 2
    assert events[0].frame_index < events[1].frame_index
    assert events[1].frame_index - events[0].frame_index > 80

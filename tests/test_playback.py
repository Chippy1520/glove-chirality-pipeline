import threading
import time

import cv2
import numpy as np
import pytest

from glove_chirality.config import DetectorConfig, EventConfig, ExtractionConfig, RuntimeConfig
from glove_chirality.events import PassageProcessor
from glove_chirality.live import SequentialVideoCapture, run_live_inference
from glove_chirality.types import Detection


class FileCaptureMock:
    def __init__(self, frames, fps=25):
        self.frames = iter(frames)
        self.fps = fps
        self.released = False

    def isOpened(self):
        return True

    def get(self, property_id):
        return self.fps if property_id == cv2.CAP_PROP_FPS else 0

    def read(self):
        return next(self.frames, (False, None))

    def release(self):
        self.released = True


def test_sequential_video_preserves_every_frame_and_media_timestamps_at_all_speeds():
    for speed in (0.25, 0.5, 1, 2):
        frames = [(True, np.full((8, 8, 3), i, np.uint8)) for i in range(4)]
        raw = FileCaptureMock(frames, fps=1000)
        capture = SequentialVideoCapture("unused", capture_factory=lambda _, raw=raw: raw).start()
        capture.playback(speed=speed)
        packets = []
        deadline = time.monotonic() + 2
        while not capture.exhausted and time.monotonic() < deadline:
            packet = capture.read()
            if packet is not None:
                packets.append(packet)
        assert [p.index for p in packets] == list(range(4))
        assert [p.image[0, 0, 0] for p in packets] == list(range(4))
        assert [p.captured_at - capture.time_origin for p in packets] == pytest.approx(
            [0, 0.001, 0.002, 0.003])
        assert capture.dropped_frames == 0 and raw.released


def test_pause_resume_stop_are_interruptible_and_do_not_advance_video():
    raw = FileCaptureMock([(True, np.zeros((8, 8, 3), np.uint8))])
    capture = SequentialVideoCapture("unused", capture_factory=lambda _, raw=raw: raw).start()
    assert capture.playback(pause=True) == {"paused": True, "speed": 1}
    assert capture.read(timeout=0.001) is None and capture.captured_frames == 0
    capture.playback(pause=False, speed=0.5)
    assert capture.read().index == 0
    capture.playback(pause=True)
    waiting = threading.Thread(target=lambda: capture.read(timeout=10))
    waiting.start()
    capture.stop()
    waiting.join(1)
    assert not waiting.is_alive() and capture.exhausted and raw.released
    for speed in (0, 3, float("nan"), True):
        with pytest.raises(ValueError):
            capture.playback(speed=speed)
    with pytest.raises(ValueError):
        capture.playback(pause="yes")


def test_file_video_reuses_passage_processor_crop_and_accurate_frame_times(tmp_path):
    path = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (40, 40))
    assert writer.isOpened()
    for _ in range(5):
        writer.write(np.full((40, 40, 3), 80, np.uint8))
    writer.release()
    box = Detection(8, 8, 30, 30, 0.9)
    detections = [[box], [box], [], [], []]

    class Detector:
        def __init__(self):
            self.sequence = iter(detections)

        def detect(self, _frame):
            return next(self.sequence, [])

        def warmup(self, _frame):
            pass

    class Classifier:
        def warmup(self):
            pass

        def predict_array(self, crop):
            assert crop.shape == (32, 32, 3)
            return "right", 0.9

    config = ExtractionConfig(
        detector=DetectorConfig(roi=(0, 0, 1, 1), trigger_zone=(0, 0, 1, 1)),
        event=EventConfig(min_detected_frames=2, exit_missing_frames=1,
                          cooldown_frames=0, output_size=32),
        runtime=RuntimeConfig(warmup=False, stage_pipeline=True, report_interval_seconds=60),
    )
    raw = cv2.VideoCapture(str(path))
    decoded = []
    while True:
        ok, frame = raw.read()
        if not ok:
            break
        decoded.append(frame)
    raw.release()
    processor = PassageProcessor(Detector(), config, "input", "live")
    baseline = []
    for i, frame in enumerate(decoded):
        baseline.extend(processor.process(frame, i, i / 25).outcomes)
    baseline.extend(processor.close((len(decoded) - 1) / 25))
    accepted = [outcome for outcome in baseline if outcome.accepted]
    capture = SequentialVideoCapture(path)
    capture.playback(speed=2)
    seen = []
    emitted = []
    metrics = run_live_inference(
        str(path), "injected", config, capture=capture, detector=Detector(), classifier=Classifier(),
        event_callback=emitted.append,
        frame_callback=lambda _frame, _result, stamp: seen.append(stamp),
    )
    assert seen == pytest.approx([i / 25 for i in range(5)])
    assert metrics.processed_frames == 5 and metrics.dropped_frames == 0
    assert len(emitted) == len(accepted) == 1
    assert emitted[0]["timestamp_s"] == pytest.approx(accepted[0].timestamp_s)
    np.testing.assert_array_equal(emitted[0]["crop"], accepted[0].crop)


def test_bad_video_fps_is_rejected_and_handle_released():
    for fps in (0, float("nan"), float("inf")):
        raw = FileCaptureMock([], fps=fps)
        with pytest.raises(ValueError, match="frame rate"):
            SequentialVideoCapture("unused", capture_factory=lambda _, raw=raw: raw)
        assert raw.released


def test_variable_frame_rate_uses_decoder_presentation_timestamps():
    times = [0, 50, 140]

    class VariableCapture(FileCaptureMock):
        def __init__(self):
            super().__init__([(True, np.zeros((8, 8, 3), np.uint8)) for _ in times])
            self.index = -1

        def read(self):
            self.index += 1
            return super().read()

        def get(self, property_id):
            return self.fps if property_id == cv2.CAP_PROP_FPS else times[self.index]

    raw = VariableCapture()
    capture = SequentialVideoCapture("unused", capture_factory=lambda _: raw).start()
    seen = []
    for _ in times:
        capture._due = None
        seen.append(capture.read().captured_at - capture.time_origin)
    capture.stop()
    assert seen == pytest.approx([0, 0.05, 0.14])

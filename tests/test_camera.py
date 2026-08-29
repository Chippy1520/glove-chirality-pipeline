import cv2
import numpy as np
import pytest

from glove_chirality.camera import CameraBackend, camera_backends, open_camera


class _Capture:
    def __init__(self, *, opened=True, frame=None, fps=0.0):
        self.opened = opened
        self.frame = frame
        self.fps = fps
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        frame, self.frame = self.frame, None
        return frame is not None, frame

    def get(self, property_id):
        return self.fps if property_id == cv2.CAP_PROP_FPS else 0.0

    def release(self):
        self.released = True


def test_windows_backend_order_is_explicit():
    assert [item.name for item in camera_backends("nt")] == [
        "DirectShow",
        "Media Foundation",
        "OpenCV default",
    ]


def test_camera_falls_back_when_opened_backend_cannot_read():
    first = _Capture(frame=None)
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    second = _Capture(frame=frame, fps=29.97)
    calls = []

    def factory(_index, api=None):
        calls.append(api)
        return first if len(calls) == 1 else second

    opened = open_camera(
        2,
        backends=(CameraBackend("first", 1), CameraBackend("second", 2)),
        capture_factory=factory,
    )

    assert calls == [1, 2]
    assert first.released is True
    assert opened.capture is second
    assert opened.backend == "second"
    assert (opened.width, opened.height) == (32, 24)
    assert opened.fps == pytest.approx(29.97)


def test_camera_preferred_backend_is_attempted_first():
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    calls = []

    def factory(_index, api=None):
        calls.append(api)
        return _Capture(frame=frame.copy())

    open_camera(
        0,
        preferred_backend="second",
        backends=(CameraBackend("first", 1), CameraBackend("second", 2)),
        capture_factory=factory,
    )
    assert calls == [2]


def test_camera_failure_reports_open_and_first_read_failures():
    closed = _Capture(opened=False)
    unreadable = _Capture(frame=None)
    captures = iter((closed, unreadable))

    with pytest.raises(RuntimeError, match="open failed.*first frame read failed"):
        open_camera(
            0,
            backends=(CameraBackend("closed", 1), CameraBackend("unreadable", 2)),
            capture_factory=lambda *_args: next(captures),
        )

    assert closed.released is True
    assert unreadable.released is True

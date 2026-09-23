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


class _ModeCapture(_Capture):
    def __init__(self, frame):
        super().__init__(frame=frame, fps=30)
        self.sets = []
        self.sets_before_read = None

    def set(self, prop, value):
        self.sets.append((prop, value))
        return True

    def read(self):
        self.sets_before_read = list(self.sets)
        return super().read()

    def get(self, property_id):
        if property_id == cv2.CAP_PROP_FOURCC:
            return float(cv2.VideoWriter_fourcc(*"MJPG"))
        return super().get(property_id)


def test_requested_mode_is_applied_before_first_read():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    capture = _ModeCapture(frame)

    opened = open_camera(
        2,
        backends=(CameraBackend("DirectShow", 700),),
        capture_factory=lambda *_args: capture,
        requested_width=1920,
        requested_height=1080,
        preferred_fourcc="MJPG",
    )

    props = [item[0] for item in capture.sets_before_read]
    assert props.index(cv2.CAP_PROP_FOURCC) < props.index(cv2.CAP_PROP_FRAME_WIDTH)
    assert props.index(cv2.CAP_PROP_FRAME_WIDTH) < props.index(cv2.CAP_PROP_FRAME_HEIGHT)
    assert (opened.width, opened.height) == (1920, 1080)
    assert opened.geometry_matches is True
    assert opened.actual_fourcc == "MJPG"


def test_negotiated_size_comes_from_frame_not_requested_size():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    opened = open_camera(
        2,
        backends=(CameraBackend("DirectShow", 700),),
        capture_factory=lambda *_args: _ModeCapture(frame),
        requested_width=1920,
        requested_height=1080,
    )
    assert (opened.width, opened.height) == (640, 480)
    assert opened.geometry_matches is False


def test_omitted_resolution_does_not_call_set():
    capture = _ModeCapture(np.zeros((8, 8, 3), dtype=np.uint8))
    opened = open_camera(
        0,
        backends=(CameraBackend("DirectShow", 1),),
        capture_factory=lambda *_args: capture,
    )
    assert capture.sets == []
    assert opened.geometry_matches is None

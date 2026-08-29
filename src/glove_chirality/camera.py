from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraBackend:
    name: str
    api: int | None


@dataclass
class OpenedCamera:
    capture: object
    first_frame: np.ndarray
    backend: str
    width: int
    height: int
    fps: float


CaptureFactory = Callable[..., object]


def camera_backends(platform: str | None = None) -> tuple[CameraBackend, ...]:
    platform = platform or os.name
    if platform == "nt":
        return (
            CameraBackend("DirectShow", cv2.CAP_DSHOW),
            CameraBackend("Media Foundation", cv2.CAP_MSMF),
            CameraBackend("OpenCV default", None),
        )
    return (CameraBackend("OpenCV default", None),)


def _make_capture(factory: CaptureFactory, source: int, api: int | None):
    return factory(source) if api is None else factory(source, api)


def open_camera(
    index: int,
    *,
    preferred_backend: str | None = None,
    backends: Iterable[CameraBackend] | None = None,
    capture_factory: CaptureFactory = cv2.VideoCapture,
) -> OpenedCamera:
    """Open a camera only after a backend returns an actual frame."""
    candidates = list(backends or camera_backends())
    if preferred_backend:
        preferred = preferred_backend.casefold()
        candidates.sort(key=lambda item: item.name.casefold() != preferred)

    failures: list[str] = []
    for backend in candidates:
        capture = _make_capture(capture_factory, index, backend.api)
        if not capture.isOpened():
            failures.append(f"{backend.name}: open failed")
            capture.release()
            continue
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            failures.append(f"{backend.name}: first frame read failed")
            capture.release()
            continue
        height, width = frame.shape[:2]
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        return OpenedCamera(capture, frame, backend.name, width, height, fps)

    details = "; ".join(failures) if failures else "no capture backends configured"
    raise RuntimeError(f"Could not stream from camera index {index}: {details}")

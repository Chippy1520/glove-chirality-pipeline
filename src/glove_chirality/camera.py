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
    requested_width: int | None = None
    requested_height: int | None = None
    requested_fps: float | None = None
    requested_fourcc: str | None = None
    actual_fourcc: str | None = None

    @property
    def geometry_matches(self) -> bool | None:
        if self.requested_width is None or self.requested_height is None:
            return None
        return (self.width, self.height) == (self.requested_width, self.requested_height)


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


def camera_geometry_status(
    requested_width: int | None,
    requested_height: int | None,
    actual_width: int,
    actual_height: int,
) -> dict[str, object]:
    """Report negotiated capture size. Never treats a resize as a successful mode."""
    requested = requested_width is not None and requested_height is not None
    matches = (
        None
        if not requested
        else (actual_width, actual_height) == (requested_width, requested_height)
    )
    warning = None
    if requested and not matches:
        warning = (
            "CAMERA GEOMETRY MISMATCH\n"
            f"Expected GRIP capture: {requested_width}x{requested_height}\n"
            f"Actual capture: {actual_width}x{actual_height}\n"
            "The field of view/aspect ratio may differ from the training dataset."
        )
    return {
        "requested": requested,
        "matches": matches,
        "warning": warning,
        "requested_width": requested_width,
        "requested_height": requested_height,
        "actual_width": actual_width,
        "actual_height": actual_height,
    }


def _fourcc_code(name: str) -> int:
    token = name.strip().upper()
    if len(token) != 4:
        raise ValueError("camera fourcc must be four characters")
    return cv2.VideoWriter_fourcc(*token)


def _fourcc_name(value: float) -> str | None:
    code = int(value)
    if code <= 0:
        return None
    text = bytes(
        (
            code & 0xFF,
            (code >> 8) & 0xFF,
            (code >> 16) & 0xFF,
            (code >> 24) & 0xFF,
        )
    ).decode("ascii", errors="ignore").strip("\x00 ")
    return text or None


def _apply_capture_mode(
    capture,
    *,
    requested_width: int | None,
    requested_height: int | None,
    requested_fps: float | None,
    preferred_fourcc: str | None,
) -> None:
    if not any(value is not None for value in (requested_width, requested_height, requested_fps, preferred_fourcc)):
        return
    setter = getattr(capture, "set", None)
    if setter is None:
        raise RuntimeError("Camera backend cannot apply a requested capture mode")
    if preferred_fourcc:
        setter(cv2.CAP_PROP_FOURCC, _fourcc_code(preferred_fourcc))
    if requested_width is not None:
        setter(cv2.CAP_PROP_FRAME_WIDTH, int(requested_width))
    if requested_height is not None:
        setter(cv2.CAP_PROP_FRAME_HEIGHT, int(requested_height))
    if requested_fps is not None:
        setter(cv2.CAP_PROP_FPS, float(requested_fps))


def _make_capture(factory: CaptureFactory, source: int, api: int | None):
    return factory(source) if api is None else factory(source, api)


def open_camera(
    index: int,
    *,
    preferred_backend: str | None = None,
    backends: Iterable[CameraBackend] | None = None,
    capture_factory: CaptureFactory = cv2.VideoCapture,
    requested_width: int | None = None,
    requested_height: int | None = None,
    requested_fps: float | None = None,
    preferred_fourcc: str | None = None,
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
        try:
            _apply_capture_mode(
                capture,
                requested_width=requested_width,
                requested_height=requested_height,
                requested_fps=requested_fps,
                preferred_fourcc=preferred_fourcc,
            )
        except RuntimeError:
            capture.release()
            raise
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            failures.append(f"{backend.name}: first frame read failed")
            capture.release()
            continue
        height, width = frame.shape[:2]
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        fourcc = None
        getter = getattr(capture, "get", None)
        if getter is not None:
            fourcc = _fourcc_name(float(getter(cv2.CAP_PROP_FOURCC) or 0.0))
        return OpenedCamera(
            capture,
            frame,
            backend.name,
            width,
            height,
            fps,
            requested_width,
            requested_height,
            requested_fps,
            preferred_fourcc.upper() if preferred_fourcc else None,
            fourcc,
        )

    details = "; ".join(failures) if failures else "no capture backends configured"
    raise RuntimeError(f"Could not stream from camera index {index}: {details}")

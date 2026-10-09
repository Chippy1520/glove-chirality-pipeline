from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from glove_chirality.camera import camera_geometry_status, open_camera
from glove_chirality.config import ExtractionConfig
from glove_chirality.models import CLASSIFIER_CHOICES

MODES = ("preview", "shadow", "armed")
BELT_DIRECTIONS = ("left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top")
GRIP_CAMERA = {
    "backend": "DirectShow",
    "width": 1920,
    "height": 1080,
    "fourcc": "MJPG",
    "fps": None,
}
GRIP_GEOMETRY = {
    "roi": (0.12, 0.03, 0.98, 0.995),
    "trigger_zone": (0.15, 0.10, 0.97, 0.96),
    "belt_direction": "bottom_to_top",
    "trigger_line_enabled": True,
    "trigger_line_fraction": 0.5,
}
DISPLAY_NAMES = {
    "tiny_cnn": "TinyCNN",
    "resnet18": "ResNet18",
    "mobilenet_v3_small": "MobileNetV3 Small",
    "vit_b_16": "ViT-B/16",
    "swin_t": "Swin-T",
    "convnextv2_pico": "ConvNeXtV2 Pico",
    "dinov3_convnext_tiny": "DINOv3 ConvNeXt Tiny",
    "dinov3_vit_small": "DINOv3 ViT-Small",
}


def requested_fps(value) -> float | None:
    text = str(value or "").strip().lower()
    if text in {"", "auto", "none"}:
        return None
    fps = float(text)
    if fps not in {15.0, 20.0, 25.0, 30.0, 60.0}:
        raise ValueError("camera FPS must be auto, 15, 20, 25, 30, or 60")
    return fps


def camera_request(options: dict[str, Any]) -> dict[str, Any]:
    preset = str(options.get("camera_preset") or "auto")
    fps = requested_fps(options.get("camera_fps"))
    if preset == "grip_1080p":
        request = dict(GRIP_CAMERA)
        request["fps"] = fps
        return request
    if preset != "custom":
        return {"fps": fps} if fps is not None else {}

    def optional_number(key: str, caster):
        value = options.get(key)
        if value in {None, ""}:
            return None
        return caster(value)

    fourcc = str(options.get("camera_fourcc") or "").strip().upper() or None
    backend = str(options.get("camera_backend") or "").strip() or None
    return {
        "backend": backend,
        "width": optional_number("camera_width", int),
        "height": optional_number("camera_height", int),
        "fps": fps,
        "fourcc": fourcc,
    }


def apply_grip_geometry(config: ExtractionConfig) -> ExtractionConfig:
    """Apply the current-camera geometry in memory. Does not write YAML."""
    config.detector.roi = GRIP_GEOMETRY["roi"]
    config.detector.trigger_zone = GRIP_GEOMETRY["trigger_zone"]
    config.detector.validate()
    config.event.belt_direction = GRIP_GEOMETRY["belt_direction"]
    config.event.trigger_line_enabled = True
    config.event.trigger_line_fraction = GRIP_GEOMETRY["trigger_line_fraction"]
    config.event.validate()
    return config


def default_models_root(workdir: str | Path) -> Path:
    configured = os.environ.get("GRIP_DATA_ROOT", "").strip()
    if configured:
        return Path(configured) / "runs" / "layer2_classifier"
    return Path(workdir) / "runs" / "layer2_classifier"


def _settings_path(workdir: Path) -> Path:
    return workdir / "outputs" / "factory_settings.json"


def load_settings(workdir: str | Path) -> dict[str, Any]:
    path = _settings_path(Path(workdir))
    if not path.is_file():
        return {"models_root": str(default_models_root(workdir))}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    raw.pop("mode", None)
    raw.setdefault("models_root", str(default_models_root(workdir)))
    return raw


def save_settings(workdir: str | Path, updates: dict[str, Any]) -> dict[str, Any]:
    current = load_settings(workdir)
    current.update(updates)
    current.pop("mode", None)
    path = _settings_path(Path(workdir))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return current


def scan_cameras(
    indices: range = range(8),
    *,
    capture_factory=None,
) -> list[dict[str, Any]]:
    """Open each index only long enough to read one frame, then release it."""
    found: list[dict[str, Any]] = []
    for index in indices:
        try:
            opened = _open_scan_camera(index, capture_factory)
        except RuntimeError:
            continue
        try:
            default_size = f"{opened.width}x{opened.height}"
            hd_available = (opened.width, opened.height) == (1920, 1080)
        finally:
            opened.capture.release()
        if not hd_available:
            try:
                probed = _open_scan_camera(
                    index,
                    capture_factory,
                    preferred_backend="DirectShow",
                    requested_width=1920,
                    requested_height=1080,
                    preferred_fourcc="MJPG",
                )
            except RuntimeError:
                probed = None
            if probed is not None:
                try:
                    hd_available = (probed.width, probed.height) == (1920, 1080)
                finally:
                    probed.capture.release()
        hd_text = "1920x1080 GRIP mode" if default_size == "1920x1080" else (
            "1080p available" if hd_available else "1080p unavailable"
        )
        found.append(
            {
                "index": index,
                "backend": opened.backend,
                "width": opened.width,
                "height": opened.height,
                "fps": opened.fps,
                "hd_available": hd_available,
                "label": (
                    f"Camera {index} | {opened.backend} | default {default_size} | {hd_text}"
                ),
            }
        )
    return found


def _open_scan_camera(index: int, capture_factory, **kwargs):
    if capture_factory is None:
        return open_camera(index, **kwargs)
    return open_camera(index, capture_factory=capture_factory, **kwargs)


def discover_configs(workdir: str | Path) -> list[dict[str, Any]]:
    root = Path(workdir) / "configs"
    items = []
    if not root.is_dir():
        return items
    paths = sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml"))
    for path in paths:
        error = ""
        try:
            ExtractionConfig.from_yaml(path)
            valid = True
        except (OSError, ValueError) as exc:
            valid = False
            error = str(exc)
        items.append({"name": path.name, "path": str(path), "valid": valid, "error": error})
    return items


def _infer_model_name(path: Path, saved: dict[str, Any]) -> str | None:
    name = saved.get("model_name")
    if isinstance(name, str) and name:
        return name
    stem = path.stem.lower().replace("-", "_")
    for choice in CLASSIFIER_CHOICES:
        if choice in stem:
            return choice
    return None


def _anti_spurious(path: Path, saved: dict[str, Any]) -> bool:
    if saved.get("augmentation") == "anti_spurious":
        return True
    blob = path.as_posix().lower().replace("-", "_")
    return "anti_spurious" in blob


def friendly_checkpoint_label(path: Path, saved: dict[str, Any]) -> str:
    name = _infer_model_name(path, saved)
    pretty = DISPLAY_NAMES.get(name or "", name or path.stem)
    if _anti_spurious(path, saved):
        return f"{pretty} - Anti-Spurious"
    return f"{pretty} - {path.name}"


def summarize_checkpoint(path: str | Path, *, compute_hash: bool = False) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Classifier checkpoint not found: {path}")
    stat = path.stat()
    saved: dict[str, Any] = {}
    error = ""
    try:
        import torch

        loaded = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict):
            saved = loaded
    except Exception as exc:  # noqa: BLE001 - unreadable checkpoints stay listed with an error
        error = str(exc)
    summary = {
        "path": str(path),
        "label": friendly_checkpoint_label(path, saved),
        "model_name": _infer_model_name(path, saved),
        "image_size": saved.get("image_size"),
        "classes": saved.get("classes"),
        "augmentation": saved.get("augmentation"),
        "modified": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
        "error": error,
    }
    if compute_hash:
        summary["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return summary


def discover_checkpoints(root: str | Path) -> list[dict[str, Any]]:
    root = Path(root)
    if not root.exists():
        return []
    return [summarize_checkpoint(path) for path in sorted(root.rglob("*.pt"))]


def device_status() -> dict[str, Any]:
    choices = ["auto", "cuda", "cuda:0", "cpu"]
    try:
        import torch
    except ImportError:
        return {
            "cuda_available": False,
            "device_name": None,
            "choices": choices,
            "label": "CUDA unavailable: PyTorch is not installed",
        }
    available = bool(torch.cuda.is_available())
    name = torch.cuda.get_device_name(0) if available else None
    label = f"CUDA available: {name}" if available else "CUDA unavailable"
    return {
        "cuda_available": available,
        "device_name": name,
        "choices": choices,
        "label": label,
    }


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jpeg(image: np.ndarray, quality: int = 80) -> bytes | None:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return encoded.tobytes() if ok else None


def _scale_detection(detection, scale: float):
    from glove_chirality.types import Detection

    if scale == 1.0:
        return detection
    polygon = None
    if detection.polygon:
        polygon = tuple((x * scale, y * scale) for x, y in detection.polygon)
    x1 = int(detection.x1 * scale)
    y1 = int(detection.y1 * scale)
    x2 = max(x1 + 1, int(detection.x2 * scale))
    y2 = max(y1 + 1, int(detection.y2 * scale))
    return Detection(x1, y1, x2, y2, detection.confidence, detection.class_id, polygon)


class _PreviewRejected:
    def __init__(self, detection, box_area_ratio: float) -> None:
        self.detection = detection
        self.box_area_ratio = box_area_ratio


def _display_preview(frame, config, detections, rejected, show_rejected) -> np.ndarray:
    from glove_chirality.overlay import draw_live_overlay

    height, width = frame.shape[:2]
    scale = 1.0
    image = frame
    if width > 1280:
        scale = 1280 / width
        image = cv2.resize(
            frame,
            (1280, max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    scaled = [_scale_detection(item, scale) for item in detections]
    scaled_rejected = [
        _PreviewRejected(_scale_detection(item.detection, scale), item.box_area_ratio)
        for item in rejected
    ]
    return draw_live_overlay(
        image,
        config,
        scaled,
        rejected=scaled_rejected,
        show_rejected=show_rejected,
    )


class FactoryLiveSession:
    """Host-side factory inspection session. Inference stays on this server."""

    def __init__(self, workdir: str | Path, *, hardware_allowed: bool = True):
        self.workdir = Path(workdir)
        self._lock = threading.RLock()
        self._controls_lock = threading.Lock()
        self._control_revision = 0
        self._arm_revision = 0
        self.hardware_allowed = bool(hardware_allowed)
        self.source_type = "camera"
        self.job_id = None
        self.current_stage = "idle"
        self.checks = []
        self._ready_stages = set()
        self._ended_wall = None
        self._started_monotonic = None
        self._ended_monotonic = None
        self._capture = None
        self._playback = {"paused": False, "speed": 1.0}
        self._controls = {"decision_policy": "argmax", "decision_class": "argmax", "decision_threshold": 0.5, "reject_class": "right", "actuator_delay_ms": 850}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.running = False
        self.status = "idle"
        self.mode = "shadow"
        self.fault: str | None = None
        self.session_id: str | None = None
        self.session_dir: Path | None = None
        self.reject_class = "right"
        self.actual_device: str | None = None
        self.amp_active = False
        self.model_name: str | None = None
        self.checkpoint: str | None = None
        self.config_path: str | None = None
        self._config: ExtractionConfig | None = None
        self._latest_frame: np.ndarray | None = None
        self._latest_crop: bytes | None = None
        self._classified: tuple[dict[str, Any], bytes] | None = None
        self._latest: dict[str, Any] | None = None
        self._positions: list[dict[str, Any]] = []
        self._events: deque[dict[str, Any]] = deque(maxlen=500)
        self._metrics: dict[str, Any] = {}
        self._counters = self._empty_counters()
        self._commanded: set[str] = set()
        self._mode_history: list[dict[str, str]] = []
        self._event_file = None
        self._actuator_file = None
        self._serial = None
        self._delay_ms = 850
        self._started_wall: str | None = None
        self._checkpoint_sha256: str | None = None
        self._camera_request: dict[str, Any] = {}
        self._camera_actual: dict[str, Any] = {}
        self._geometry_override = "yaml"
        self._show_rejected = False
        self._yolo_counts: dict[str, Any] = {}
        self._active_detector = None
        self._preview_source: np.ndarray | None = None
        self._preview_detections: tuple = ()
        self._preview_rejected: tuple = ()
        self._preview_stamp = 0.0
        self._preview_wanted_until = 0.0
        self._jpeg_cache: bytes | None = None
        self._jpeg_at = 0.0
        self._preview_encode_ms: float | None = None
        self._device_status = {"label": "Not probed", "cuda_available": None}
        self._decision = {"phase": "WAITING"}
        self._settled_phase = "WAITING"
        self._inspecting_until = 0.0
        self._options: dict[str, Any] = {}

    @staticmethod
    def _empty_counters() -> dict[str, int]:
        return {
            "total_accepted": 0,
            "left_pass": 0,
            "right_reject": 0,
            "pipeline_rejected": 0,
            "multiple_candidates": 0,
            "partial": 0,
            "commands_sent": 0,
        }

    def _set_status(self, status: str) -> None:
        stages = {
            "Loading Layer-1 detector...": "detector",
            "Loading Layer-2 classifier...": "classifier",
            "Opening camera...": "source_open",
            "Warming models...": "warmup",
        }
        with self._lock:
            if status == "RUNNING":
                return  # A runner's announcement is not proof of readiness.
            if status in stages:
                self.status = "starting"
                self.current_stage = stages[status]
            else:
                self.status = status

    def _on_stage(self, stage: str, status: str) -> None:
        with self._lock:
            self.current_stage = stage
            self.checks = [item for item in self.checks if item["field"] != stage]
            self.checks.append({"field": stage, "label": stage.replace("_", " ").title(),
                                "status": status, "message": status})
            if status == "passed":
                self._ready_stages.add(stage)
            if self.running and not self._stop.is_set() and {"first_frame", "detector_inference", "classifier_warmup"} <= self._ready_stages:
                self.status = "running"
                self.current_stage = "running"

    def _cancel_pending(self) -> None:
        if self._serial is not None:
            cancel = getattr(self._serial, "cancel_pending", None)
            if cancel is not None:
                cancel()

    def _hardware_guard(self) -> None:
        if not self.hardware_allowed or self.source_type == "video":
            raise ValueError("Hardware is disabled for generic inference and video playback")

    @staticmethod
    def _validate_controls(payload, current):
        values = dict(current)
        aliases = {"threshold": "decision_threshold", "policy": "decision_policy", "delay_ms": "actuator_delay_ms"}
        for key, value in payload.items():
            key = aliases.get(key, key)
            if key in values:
                values[key] = value
        policy = str(values["decision_policy"])
        if "decision_class" in payload and "decision_policy" not in payload and "policy" not in payload:
            policy = "argmax" if values["decision_class"] == "argmax" else "threshold"
        if policy not in {"argmax", "threshold"}:
            raise ValueError("decision_policy must be argmax or threshold")
        decision_class = str(values["decision_class"])
        if policy == "argmax":
            decision_class = "argmax"
        elif decision_class not in {"left", "right"}:
            raise ValueError("threshold policy requires decision_class left or right")
        threshold = float(values["decision_threshold"])
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("decision_threshold must be finite and in [0, 1]")
        reject = str(values["reject_class"])
        if reject not in {"left", "right"}:
            raise ValueError("reject_class must be left or right")
        delay = float(values["actuator_delay_ms"])
        if not math.isfinite(delay) or not 0 <= delay <= 60000 or not delay.is_integer():
            raise ValueError("actuator delay must be an integer in [0, 60000] ms")
        return {"decision_policy": policy, "decision_class": decision_class,
                "decision_threshold": threshold, "reject_class": reject, "actuator_delay_ms": int(delay)}

    def update_controls(self, payload: dict) -> dict:
        with self._controls_lock:
            values = self._validate_controls(payload, self._controls)
            with self._lock:
                if values != self._controls:
                    self._control_revision += 1
                    self.mode = "shadow"
                    self._cancel_pending()
                self._controls = values
                self.reject_class = values["reject_class"]
                self._delay_ms = values["actuator_delay_ms"]
            return dict(values)

    def _classify(self, crop, classifier):
        # Only Layer 2 takes this lock; Layer 1 never waits for model inference.
        with self._controls_lock:
            values = dict(self._controls)
            classifier.decision_class = values["decision_class"]
            classifier.decision_threshold = values["decision_threshold"]
            values["control_revision"] = self._control_revision
            values["arm_revision"] = self._arm_revision
            prediction, confidence = classifier.predict_array(crop)
        return prediction, confidence, values

    def playback(self, pause: bool | None = None, speed: float | None = None) -> dict:
        if self.source_type != "video":
            raise ValueError("Playback controls require a video source")
        if pause is not None and not isinstance(pause, bool):
            raise ValueError("pause must be a boolean")
        if speed is not None and (isinstance(speed, bool) or speed not in {0.25, 0.5, 1, 2}):
            raise ValueError("speed must be 0.25, 0.5, 1, or 2")
        with self._lock:
            if pause is not None:
                self._playback["paused"] = pause
            if speed is not None:
                self._playback["speed"] = float(speed)
            if self._capture is not None:
                self._capture.playback(pause=pause, speed=speed)
            return dict(self._playback)

    def manual_trigger(self, *, confirm: bool = False) -> dict:
        from glove_chirality.actuator import reject_command

        with self._lock:
            self._hardware_guard()
            if confirm is not True or self._stop.is_set() or self.mode != "armed" or self.status != "running" or self.fault:
                raise ValueError("Manual trigger requires confirmed, running ARMED mode")
            if not self._serial_snapshot().get("connected"):
                raise ValueError("Actuator is not connected")
            event_id = "manual_" + uuid.uuid4().hex
            self._serial_actor().submit(reject_command(event_id, self._delay_ms))
            self._counters["commands_sent"] += 1
            self._append_actuator({"kind": "manual_trigger", "event_id": event_id})
            return {"status": "queued", "event_id": event_id}

    def simulate_trigger(self) -> dict:
        record = {"kind": "simulated_trigger", "actuated": False}
        self._append_actuator(record)
        with self._lock:
            self._events.append(record)
        return record

    @staticmethod
    def _line_to_actuator_ms(options: dict) -> int | None:
        value = options.get("line_to_actuator_ms")
        if value is None or str(value).strip() == "":
            return None
        travel = float(value)
        if not math.isfinite(travel) or not 1 <= travel <= 60000 or not travel.is_integer():
            raise ValueError("Line-to-actuator travel time must be an integer in [1, 60000] ms")
        return int(travel)

    def preflight(self, options: dict) -> dict:
        """Accumulate plain validation without opening a camera, loading models, or serial."""
        checks = []
        def check(field, action):
            try:
                action()
            except Exception:  # noqa: BLE001 - independent YAML/config field diagnostics
                checks.append({"field": field, "label": field.replace("_", " ").title(),
                               "status": "failed", "message": f"Invalid {field.replace('_', ' ')}"})
            else:
                checks.append({"field": field, "label": field.replace("_", " ").title(),
                               "status": "passed", "message": "Validated"})
        def require(condition):
            if not condition:
                raise ValueError("Invalid option")
        source = str(options.get("source", options.get("source_path", ""))).strip()
        kind = str(options.get("source_type", "camera"))
        def source_check():
            from urllib.parse import urlsplit
            require(kind in {"camera", "video", "stream"})
            if kind == "camera":
                require(source.isdigit())  # A file cannot spoof a camera to actuate.
            elif kind == "video":
                require(Path(source).is_file())
            else:
                require(urlsplit(source).scheme.lower() in {"rtsp", "rtsps", "http", "https", "rtmp"})
                require(bool(urlsplit(source).hostname))
        check("source", source_check)
        check("mode", lambda: require(options.get("mode", "shadow") in MODES))
        if options.get("mode") == "armed" and kind != "video":
            check("confirm_armed", lambda: require(options.get("confirm_armed") is True))
            check("hardware", lambda: require(self.hardware_allowed))
        def config_check():
            path = Path(str(options.get("config", "")))
            require(path.is_file())
            config = ExtractionConfig.from_yaml(path)
            if options.get("geometry") == "grip":
                apply_grip_geometry(config)
            else:
                self._apply_trigger_overrides(config, options)
        check("config", config_check)
        check("checkpoint", lambda: require(Path(str(options.get("checkpoint", ""))).is_file()))
        def camera_check():
            require(options.get("camera_preset", "auto") in {"auto", "custom", "grip_1080p"})
            camera = camera_request(options)
            for field in ("width", "height"):
                require(camera.get(field) is None or 0 < camera[field] <= 16384)
            require(camera.get("fourcc") is None or len(camera["fourcc"]) == 4)
        check("camera", camera_check)
        check("line_to_actuator_ms", lambda: self._line_to_actuator_ms(options))
        check("device", lambda: require(bool(re.fullmatch(r"auto|cpu|cuda(?::[0-9]+)?", str(options.get("device", "auto"))))))
        # Independently validate fields so the UI reports all invalid inputs together.
        for field in ("decision_policy", "decision_class", "decision_threshold", "reject_class", "actuator_delay_ms"):
            subset = {field: options[field]} if field in options else {}
            if field == "decision_class" and "decision_class" in options:
                subset["decision_policy"] = "argmax" if options[field] == "argmax" else "threshold"
            if field == "decision_policy" and options.get(field) == "threshold":
                subset["decision_class"] = options.get("decision_class", "right")
            if field == "actuator_delay_ms" and "delay_ms" in options:
                subset[field] = options["delay_ms"]
            if field == "decision_threshold" and "threshold" in options:
                subset[field] = options["threshold"]
            check(field, lambda subset=subset: self._validate_controls(subset, self._controls))
        return {"status": "ready" if all(c["status"] == "passed" for c in checks) else "failed", "checks": checks}

    def start(self, options: dict, *, runner=None, detector=None, classifier=None, capture=None) -> dict:
        with self._lock:
            if self.running or (self._thread is not None and self._thread.is_alive()):
                raise ValueError("Stop the current session before starting another")
            self.job_id = uuid.uuid4().hex
            self.session_dir = None
            self.session_id = None
            self.running = True
            self.status = "preflight"
            self.current_stage = "validation"
            self.checks = []
            self.fault = None
            self.mode = "shadow"
            self._stop.clear()
            self._ready_stages.clear()
            self._started_wall = datetime.now().astimezone().isoformat(timespec="milliseconds")
            self._started_monotonic = time.monotonic()
            self._ended_wall = None
            self._ended_monotonic = None
            self._capture = None
            self._latest = None
            self._latest_crop = None
            self._classified = None
            self._preview_source = None
            self._preview_wanted_until = 0.0
            self._jpeg_cache = None
            self._events.clear()
            self._metrics = {}
            self._positions = []
            self._camera_actual = {}
            self._yolo_counts = {}
            self.actual_device = None
            self.model_name = None
            self._checkpoint_sha256 = None
            self._cancel_pending()
            self._options = dict(options)
            self._injected = {"runner": runner, "detector": detector, "classifier": classifier, "capture": capture}
            response = {"status": "preflight", "job_id": self.job_id, "checks": []}
            self._thread = threading.Thread(target=self._startup, args=(dict(options),), daemon=True)
            self._thread.start()
            return response

    def _startup(self, options):
        try:
            result = self.preflight(options)
            with self._lock:
                self.checks = result["checks"]
            if result["status"] == "failed":
                raise ValueError("Preflight validation failed; review checks")
            if self._stop.is_set():
                self._set_status("stopped")
                return
            options["source"] = str(options.get("source", options.get("source_path", ""))).strip()
            self.source_type = str(options.get("source_type", "camera"))
            self.update_controls(options)
            # Startup is always shadow. Arming is a separate confirmed action after readiness.
            options["mode"] = "preview" if options.get("mode") == "preview" and self.source_type != "video" else "shadow"
            self._prepare_start(options)
            self._run()
        except Exception as exc:  # noqa: BLE001 - always expose startup failures as diagnostics
            with self._lock:
                self.fault = self._safe_error(exc)
                self.status = "fault"
                self.checks.append({"field": self.current_stage, "label": "Startup", "status": "failed", "message": self.fault})
        finally:
            if self._capture is not None and hasattr(self._capture, "stop"):
                self._capture.stop()
            with self._lock:
                self._cancel_pending()
                self.running = False
                self.mode = "shadow"
                self._ended_wall = datetime.now().astimezone().isoformat(timespec="milliseconds")
                self._ended_monotonic = time.monotonic()
                if self._stop.is_set() and self.status != "fault":
                    self.status = "stopped"
                self._cancel_pending()
            self._write_session()

    def _safe_error(self, exc):
        message = str(exc)
        source = str(self._options.get("source", ""))
        if source and "://" in source:
            message = message.replace(source, "[stream]")
        return re.sub(r"(\w+://)[^\s/@]+:[^\s/@]+@", r"\1[redacted]@", message)

    def _remember_mode(self, mode: str) -> None:
        self._mode_history.append(
            {
                "mode": mode,
                "wall_time_iso": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            }
        )

    def set_mode(self, mode: str, *, confirm: bool = False) -> str:
        if mode not in MODES:
            raise ValueError("mode must be preview, shadow, or armed")
        with self._lock:
            if mode == "armed":
                self._hardware_guard()
                if self.status != "running" or not self.running or self._stop.is_set():
                    raise ValueError("ARMED mode requires a successfully running shadow session")
                if confirm is not True:
                    raise ValueError("ARMED mode requires explicit confirmation")
                if self.fault:
                    raise ValueError(f"Cannot arm while a fault is active: {self.fault}")
                if not self._serial_snapshot().get("connected"):
                    raise ValueError("Connect the actuator before ARMED mode")
            self._cancel_pending()
            self._arm_revision += 1
            self.mode = mode
            self._remember_mode(mode)
        self._write_session()
        return mode

    def _serial_actor(self):
        self._hardware_guard()
        if self._serial is None:
            from glove_chirality.actuator import SerialActuator

            self._serial = SerialActuator()
        return self._serial

    def serial_ports(self) -> list[dict[str, Any]]:
        from glove_chirality.actuator import list_serial_ports

        return list_serial_ports()

    def connect_serial(self, port: str, baud: int = 115200) -> dict[str, Any]:
        self._hardware_guard()
        self.set_mode("shadow")
        if not port.strip():
            raise ValueError("Serial port is required")
        if baud <= 0:
            raise ValueError("Baud rate must be positive")
        actor = self._serial_actor()
        actor.connect(
            port.strip(),
            baud,
            on_fault=self._on_serial_fault,
            on_ack=self._on_ack,
            on_command=self._on_command,
        )
        return actor.snapshot()

    def disconnect_serial(self) -> dict[str, Any]:
        with self._lock:
            self.mode = "shadow"
            self._cancel_pending()
        if self._serial is None:
            return {"connected": False, "last_command": None, "last_ack": None, "fault": None}
        self._serial.disconnect()
        return self._serial.snapshot()

    def _on_serial_fault(self, message: str) -> None:
        with self._lock:
            self._cancel_pending()
            self.fault = f"Serial fault: {message}"
            if self.mode == "armed":
                self.mode = "shadow"
                self._remember_mode("shadow")
        self._write_session()

    def _on_ack(self, line: str) -> None:
        self._append_actuator({"kind": "ack", "line": line})

    def _on_command(self, line: str) -> None:
        self._append_actuator({"kind": "command", "line": line})

    def _serial_snapshot(self) -> dict[str, Any]:
        if self._serial is None:
            return {"connected": False, "port": None, "baud": None, "last_command": None, "last_ack": None, "fault": None}
        return self._serial.snapshot()

    def reset_counters(self, *, confirm: bool = False) -> dict[str, int]:
        if self.running and not confirm:
            raise ValueError("Confirm counter reset while a session is running")
        with self._lock:
            self._counters = self._empty_counters()
            counters = dict(self._counters)
        return counters

    def _prepare_start(
        self,
        options: dict[str, Any],
        *,
        runner=None,
        detector=None,
        classifier=None,
        capture=None,
    ) -> dict[str, Any]:
        self._line_to_actuator_ms(options)
        mode = str(options.get("mode", "shadow"))
        if mode not in MODES:
            raise ValueError("mode must be preview, shadow, or armed")
        if mode == "armed" and not options.get("confirm_armed"):
            raise ValueError("ARMED mode requires explicit confirmation")
        source = str(options.get("source", "")).strip()
        if not source:
            raise ValueError("Camera source is required")
        config_path = Path(str(options.get("config", "")))
        if not config_path.is_file():
            raise ValueError(f"Extraction config not found: {config_path}")
        config = ExtractionConfig.from_yaml(config_path)
        if str(options.get("geometry") or "yaml") == "grip":
            apply_grip_geometry(config)
        else:
            self._apply_trigger_overrides(config, options)
        self._camera_request = camera_request(options)
        self._geometry_override = "grip" if str(options.get("geometry") or "yaml") == "grip" else "yaml"
        checkpoint = str(options.get("checkpoint", "")).strip()
        if not checkpoint or not Path(checkpoint).is_file():
            raise ValueError(f"Classifier checkpoint not found: {checkpoint}")
        device = str(options.get("device", "auto"))
        if device.startswith("cuda"):
            status = device_status()
            if not status["cuda_available"]:
                raise ValueError("CUDA was requested but PyTorch reports CUDA unavailable")
        reject_class = self._controls["reject_class"]
        if reject_class not in {"left", "right"}:
            raise ValueError("reject_class must be left or right")
        delay_ms = self._controls["actuator_delay_ms"]
        if delay_ms < 0:
            raise ValueError("Actuator delay must be non-negative")
        if mode == "armed" and not self._serial_snapshot().get("connected"):
            raise ValueError("Connect the actuator before ARMED mode")

        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        workflow = "factory" if self.hardware_allowed else "inference"
        directory = self.workdir / "outputs" / f"{workflow}_live" / f"{stamp}_{self.job_id}"
        directory.mkdir(parents=True, exist_ok=False)
        with self._lock:
            self.running = True
            self.fault = None
            self.status = "starting"
            self.current_stage = "detector"
            self.mode = "shadow"
            self.session_id = f"{workflow}_{stamp}_{self.job_id}"
            self.session_dir = directory
            self.reject_class = reject_class
            self.checkpoint = checkpoint
            self.config_path = str(config_path)
            self._config = config
            self._delay_ms = delay_ms
            self._options = dict(options)
            self._latest = None
            self._latest_crop = None
            self._classified = None
            self._latest_frame = None
            self._positions = []
            self._events.clear()
            self._commanded.clear()
            self._mode_history = []
            self._remember_mode(mode)
            self.mode = mode
            if not options.get("continue_counters"):
                self._counters = self._empty_counters()
            self._started_wall = datetime.now().astimezone().isoformat(timespec="milliseconds")
        self._open_logs(directory)
        self._write_session()
    @staticmethod
    def _apply_trigger_overrides(config: ExtractionConfig, options: dict[str, Any]) -> None:
        if options.get("belt_direction"):
            config.event.belt_direction = str(options["belt_direction"])
        if "trigger_line_enabled" in options:
            config.event.trigger_line_enabled = bool(options["trigger_line_enabled"])
        if options.get("trigger_line_fraction") not in {None, ""}:
            config.event.trigger_line_fraction = float(options["trigger_line_fraction"])
        config.event.validate()

    def _open_logs(self, directory: Path) -> None:
        self._event_file = (directory / "events.jsonl").open("a", encoding="utf-8")
        self._actuator_file = (directory / "actuator.jsonl").open("a", encoding="utf-8")

    def _close_logs(self) -> None:
        for handle in (self._event_file, self._actuator_file):
            if handle is not None and not handle.closed:
                handle.close()
        self._event_file = None
        self._actuator_file = None

    def _append_jsonl(self, handle, payload: dict[str, Any]) -> None:
        if handle is None or handle.closed:
            return
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
        handle.flush()

    def _append_actuator(self, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        payload["wall_time_iso"] = datetime.now().astimezone().isoformat(timespec="milliseconds")
        self._append_jsonl(self._actuator_file, payload)

    def _write_session(self) -> None:
        if self.session_dir is None:
            return
        serial = self._serial_snapshot()
        config_path = Path(self.config_path) if self.config_path else None
        payload = {
            "session_id": self.session_id,
            "start_wall_time_iso": self._started_wall,
            "end_wall_time_iso": None if self.running else datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "camera": "[stream]" if self.source_type == "stream" else self._options.get("source"),
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self._checkpoint_sha256,
            "model_name": self.model_name,
            "extraction_config": self.config_path,
            "config_sha256": _sha256(config_path) if config_path else None,
            "device": self._options.get("device"),
            "actual_device": self.actual_device,
            "amp": bool(self._options.get("amp")),
            "amp_active": self.amp_active,
            "mode_history": list(self._mode_history),
            "reject_class": self.reject_class,
            "serial_port": serial.get("port"),
            "baud": serial.get("baud") or self._options.get("baud"),
            "actuator_delay_ms": self._delay_ms,
            "line_to_actuator_ms": self._line_to_actuator_ms(self._options),
            "remaining_time_compensation": False,
            "fault": self.fault,
            "status": self.status,
            "geometry_override": self._geometry_override,
            "effective_roi": None if self._config is None else list(self._config.detector.roi),
            "effective_trigger_zone": None if self._config is None else list(self._config.detector.trigger_zone),
            "effective_belt_direction": None if self._config is None else self._config.event.belt_direction,
            "yolo_imgsz": None if self._config is None else self._config.detector.yolo_imgsz,
            "yolo_min_box_area_ratio": None if self._config is None else self._config.detector.yolo_min_box_area_ratio,
            "yolo_max_box_area_ratio": None if self._config is None else self._config.detector.yolo_max_box_area_ratio,
            "requested_camera_width": self._camera_request.get("width"),
            "requested_camera_height": self._camera_request.get("height"),
            "requested_camera_fps": self._camera_request.get("fps"),
            "requested_fourcc": self._camera_request.get("fourcc"),
            "preferred_backend": self._camera_request.get("backend"),
            "actual_camera_width": self._camera_actual.get("width"),
            "actual_camera_height": self._camera_actual.get("height"),
            "actual_camera_fps": self._camera_actual.get("fps"),
            "actual_backend": self._camera_actual.get("backend"),
            "actual_fourcc": self._camera_actual.get("fourcc"),
            "camera_geometry_warning": self._camera_actual.get("warning"),
        }
        (self.session_dir / "session.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _run(self) -> None:
        from glove_chirality.detection import build_detector
        from glove_chirality.inference import TorchClassifier
        from glove_chirality.live import (
            LatestFrameCapture,
            SequentialVideoCapture,
            parse_capture_source,
            run_live_inference,
        )

        injected = getattr(self, "_injected", {})
        runner = injected.get("runner") or run_live_inference
        try:
            config = self._config
            if config is None:
                raise RuntimeError("Extraction config was not loaded")
            self._set_status("Loading Layer-1 detector...")
            if self._stop.is_set():
                return
            detector = injected.get("detector") or build_detector(config.detector)
            self._active_detector = detector
            self._on_stage("detector", "passed")
            self._set_status("Loading Layer-3 classifier...")
            if self._stop.is_set():
                return
            classifier = injected.get("classifier") or TorchClassifier(
                self.checkpoint,
                device=str(self._options.get("device", "auto")),
                amp=bool(self._options.get("amp")),
                decision_class=str(self._options.get("decision_class", "argmax")),
                decision_threshold=float(self._options.get("decision_threshold", 0.5)),
            )
            self._on_stage("classifier", "passed")
            device_obj = getattr(classifier, "device", "cpu")
            device_type = getattr(device_obj, "type", str(device_obj))
            with self._lock:
                self.model_name = getattr(classifier, "model_name", None)
                self.actual_device = str(device_obj)
                self.amp_active = bool(getattr(classifier, "use_amp", False))
                requested = str(self._options.get("device", "auto"))
                if requested.startswith("cuda") and device_type != "cuda":
                    raise RuntimeError("CUDA was requested but PyTorch selected CPU")
            if self.checkpoint:
                self._checkpoint_sha256 = _sha256(Path(self.checkpoint))
            self._write_session()
            self._set_status("Opening camera...")
            if self._stop.is_set():
                return
            capture = injected.get("capture")
            if capture is None:
                if self.source_type == "video":
                    capture = SequentialVideoCapture(self._options["source"])
                else:
                    capture = LatestFrameCapture(
                        parse_capture_source(self._options["source"]),
                        config.runtime.capture_queue_size, camera_mode=self._camera_request,
                    )
            self._capture = capture
            if self.source_type == "video":
                capture.playback(pause=self._playback["paused"], speed=self._playback["speed"])
            self._on_stage("source_open", "passed")
            self._record_camera(capture)
            self._set_status("Warming models...")
            run_metrics = runner(
                self._options.get("source"),
                self.checkpoint,
                config,
                device=str(self._options.get("device", "auto")),
                amp=bool(self._options.get("amp")),
                decision_class=str(self._options.get("decision_class", "argmax")),
                decision_threshold=float(self._options.get("decision_threshold", 0.5)),
                detector=detector,
                classifier=classifier,
                capture=capture,
                event_callback=self._on_event,
                frame_callback=self._on_frame,
                metrics_callback=self._on_metrics,
                status_callback=self._set_status,
                before_classify=self.note_classifying,
                stop_event=self._stop,
                session_id=self.session_id,
                config_path=self.config_path,
                should_classify=lambda: self.mode != "preview",
                on_stage=self._on_stage,
                on_firstinference=lambda: self._on_stage("classifier_inference", "passed"),
                classify_callback=self._classify,
            )
            if self.source_type != "video" and (getattr(run_metrics, "source_exhausted", False) or not self._stop.is_set()):
                raise RuntimeError("Camera stream ended")
            self._set_status("stopped")
        except Exception as exc:  # noqa: BLE001 - camera, model, and runtime faults must disarm
            with self._lock:
                self.fault = self._safe_error(exc)
                self.checks.append({"field": self.current_stage, "label": "Runtime", "status": "failed", "message": self.fault})
                self.mode = "shadow"
                self.status = "fault"
                self._remember_mode("shadow")
        finally:
            with self._lock:
                self.running = False
                if self.mode == "armed":
                    self.mode = "shadow"
                    self._remember_mode("shadow")
            self._write_session()
            self._close_logs()

    def note_classifying(self) -> None:
        with self._lock:
            self._decision = {"phase": "CLASSIFYING"}
            self._inspecting_until = time.monotonic() + 2.0

    def decision(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            if self._decision["phase"] == "INSPECTING" and now > self._inspecting_until:
                self._decision = {"phase": self._settled_phase}
            return {
                "phase": self._decision["phase"],
                "running": self.running,
                "fault": "Session fault; see host diagnostics" if self.fault else None,
            }

    def set_display(self, *, show_size_rejected: bool) -> None:
        self._show_rejected = bool(show_size_rejected)

    def _rejected_boxes(self):
        diagnostics = getattr(self._active_detector, "last_diagnostics", None)
        return tuple(getattr(diagnostics, "size_rejected", ()) or ())

    def _record_camera(self, capture) -> None:
        opened = getattr(capture, "opened", None)
        if opened is None:
            return
        status = camera_geometry_status(
            opened.requested_width,
            opened.requested_height,
            opened.width,
            opened.height,
        )
        with self._lock:
            self._camera_actual = {
                "backend": opened.backend,
                "width": opened.width,
                "height": opened.height,
                "fps": opened.fps,
                "fourcc": opened.actual_fourcc,
                "requested_width": opened.requested_width,
                "requested_height": opened.requested_height,
                "requested_fps": opened.requested_fps,
                "requested_fourcc": opened.requested_fourcc,
                "matches": status["matches"],
                "warning": status["warning"],
            }
        self._write_session()

    def _on_frame(self, frame: np.ndarray, result, timestamp_s: float) -> None:
        del timestamp_s
        config = self._config
        if config is None:
            return
        from glove_chirality.overlay import detection_states

        height, width = frame.shape[:2]
        states = detection_states(config, result.detections, width, height)
        frame_area = max(1, width * height)
        positions = []
        for detection, state in zip(result.detections, states):
            cx, cy = detection.center
            positions.append(
                {
                    "state": state,
                    "confidence": detection.confidence,
                    "area_ratio": detection.area / frame_area,
                    "bbox": [detection.x1, detection.y1, detection.x2, detection.y2],
                    "center_px": [cx, cy],
                    "center_norm": [cx / max(width, 1), cy / max(height, 1)],
                }
            )
        diagnostics = getattr(result, "detector_diagnostics", None)
        counts = {
            "raw": getattr(diagnostics, "raw_yolo_count", None),
            "size_rejected": getattr(diagnostics, "size_rejected_count", None),
            "kept": getattr(diagnostics, "returned_detection_count", len(result.detections)),
            "eligible": sum(state == "ELIGIBLE" for state in states),
        }
        now = time.monotonic()
        inspecting = bool(result.detections)
        source = None
        if now <= self._preview_wanted_until and now - self._preview_stamp >= 0.1:
            source = frame.copy()
        with self._lock:
            self._positions = positions
            self._yolo_counts = counts
            phase = self._decision["phase"]
            processed_at = getattr(result, "processed_at_monotonic", None)
            current_observation = processed_at is None or processed_at >= getattr(self, "_settled_at_monotonic", 0.0)
            if current_observation and inspecting and phase != "CLASSIFYING":
                self._decision = {"phase": "INSPECTING"}
                self._inspecting_until = now + 0.35
            elif current_observation and phase == "INSPECTING" and now > self._inspecting_until:
                self._decision = {"phase": self._settled_phase}
            if source is not None:
                self._preview_source = source
                self._preview_detections = tuple(result.detections)
                self._preview_rejected = tuple(getattr(diagnostics, "size_rejected", ()) or ()) if self._show_rejected else ()
                self._preview_stamp = now
                self._jpeg_cache = None
            elif now > self._preview_wanted_until:
                self._preview_source = None
                self._preview_detections = ()
                self._preview_rejected = ()
                self._jpeg_cache = None

    def frame_jpeg(self) -> bytes | None:
        now = time.monotonic()
        self._preview_wanted_until = now + 1.0
        with self._lock:
            if self._jpeg_cache is not None and now - self._jpeg_at < 0.08:
                return self._jpeg_cache
            frame = self._preview_source
            detections = self._preview_detections
            rejected = self._preview_rejected
            config = self._config
            show = self._show_rejected
        if frame is None or config is None:
            return None
        started = time.perf_counter()
        preview = _display_preview(frame, config, detections, rejected, show)
        encoded = _jpeg(preview, 60)
        elapsed = (time.perf_counter() - started) * 1000.0
        with self._lock:
            self._jpeg_cache = encoded
            self._jpeg_at = time.monotonic()
            self._preview_encode_ms = elapsed
        return encoded

    def _on_metrics(self, metrics: dict[str, Any]) -> None:
        with self._lock:
            self._metrics = dict(metrics)

    def _on_event(self, payload: dict[str, Any]) -> None:
        from glove_chirality.actuator import command_for_event

        record = dict(payload)
        with self._lock:
            mode = self.mode if self.hardware_allowed and self.source_type != "video" else "shadow"
            if mode == "armed" and (record.get("control_revision", self._control_revision) != self._control_revision or record.get("arm_revision", self._arm_revision) != self._arm_revision):
                mode = "shadow"
            if mode == "armed" and (self._stop.is_set() or not self.running or self.status != "running" or self.fault or not self._serial_snapshot().get("connected")):
                mode = "shadow"
            reject_class = record.get("reject_class", self.reject_class)
            delay_ms = record.get("actuator_delay_ms", self._delay_ms)
            for key, value in self._controls.items():
                record.setdefault(key, value)
            commanded = set(self._commanded)
            arm_revision = self._arm_revision
            control_revision = self._control_revision
        prediction = record.get("prediction")
        status = str(record.get("status"))
        event_id = str(record.get("event_id"))
        command = command_for_event(
            mode=mode,
            status=status,
            prediction=prediction,
            reject_class=reject_class,
            event_id=event_id,
            delay_ms=delay_ms,
            commanded=commanded,
        )
        record["policy"] = record["decision_policy"]
        record["threshold"] = record["decision_threshold"]
        record["operating_mode"] = mode
        record["result"] = _result_label(status, prediction, reject_class)
        record["would_reject"] = mode == "shadow" and status == "accepted" and prediction == reject_class
        record["actuator_command"] = command.strip() if command else None
        record["classifier_ran"] = status == "accepted" and prediction is not None
        if status == "accepted" and prediction == reject_class:
            phase = "REJECT"
        elif status == "accepted" and prediction:
            phase = "PASS"
        else:
            phase = "NO DECISION"
        with self._lock:
            self._settled_phase = phase
            self._decision = {"phase": phase}
            self._settled_at_monotonic = time.perf_counter()
        if command:
            with self._lock:
                self._commanded.add(event_id)
                self._counters["commands_sent"] += 1
            try:
                with self._lock:
                    if not self._stop.is_set() and self.mode == "armed" and self.running and self.status == "running" and not self.fault and self._arm_revision == arm_revision and self._control_revision == control_revision:
                        self._serial_actor().submit(command)
            except Exception as exc:  # noqa: BLE001 - a failed enqueue must not block inference
                self._on_serial_fault(str(exc))
        self._count(record)
        crop = record.pop("crop", None)
        encoded = None
        if isinstance(crop, np.ndarray):
            encoded = _jpeg(crop, 95)
        with self._lock:
            if encoded is not None:
                self._latest_crop = encoded
            self._latest = {key: value for key, value in record.items() if key != "crop"}
            if record["classifier_ran"] and encoded is not None:
                self._classified = (dict(self._latest), encoded)
            self._events.append(dict(self._latest))
        self._append_jsonl(self._event_file, self._latest)

    def _count(self, record: dict[str, Any]) -> None:
        status = record.get("status")
        reason = record.get("reject_reason") or status
        with self._lock:
            if status == "accepted":
                self._counters["total_accepted"] += 1
                if record.get("prediction") == "left":
                    self._counters["left_pass"] += 1
                elif record.get("prediction") == "right":
                    self._counters["right_reject"] += 1
            else:
                self._counters["pipeline_rejected"] += 1
                if reason == "multiple_candidates":
                    self._counters["multiple_candidates"] += 1
                if reason == "partial":
                    self._counters["partial"] += 1

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            self.mode = "shadow"
            self._cancel_pending()
        if self._capture is not None and hasattr(self._capture, "stop"):
            self._capture.stop()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        with self._lock:
            self.running = False
            if self.mode == "armed":
                self.mode = "shadow"
            if self.status not in {"fault", "stopped"}:
                self.status = "stopped"
        self._write_session()

    def crop_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_crop

    def classified_result(self) -> dict[str, Any] | None:
        """One atomic crop/result pair, retained until the next classified passage."""
        with self._lock:
            pair = self._classified
            session_id, job_id = self.session_id, self.job_id
        if pair is None:
            return None
        event, encoded = pair
        return {"session_id": session_id, "job_id": job_id, "event": dict(event),
                "crop_base64": base64.b64encode(encoded).decode("ascii")}

    def snapshot(self, *, reveal_paths: bool) -> dict[str, Any]:
        with self._lock:
            serial = self._serial_snapshot()
            payload = {
                "running": self.running,
                "status": self.status,
                "startup_status": self.status,
                "job_id": self.job_id,
                "workflow": "factory" if self.hardware_allowed else "inference",
                "hardware_allowed": self.hardware_allowed and self.source_type != "video",
                "pending_cancellation": "Unsent host commands only; firmware-delayed commands cannot be recalled",
                "actuator_state": "DISABLED" if not self.hardware_allowed or self.source_type == "video" else self.mode.upper(),
                "source_type": self.source_type,
                "current_stage": self.current_stage,
                "checks": [dict(c) for c in self.checks],
                "error": self.fault,
                "start_time": self._started_wall,
                "end_time": self._ended_wall,
                "elapsed_s": None if self._started_monotonic is None else (self._ended_monotonic or time.monotonic()) - self._started_monotonic,
                "playback": dict(self._playback),
                "policy": self._controls["decision_policy"],
                "threshold": self._controls["decision_threshold"],
                "decision_class": self._controls["decision_class"],
                "actuator_delay_ms": self._delay_ms,
                "mode": self.mode,
                "fault": self.fault,
                "session_id": self.session_id,
                "reject_class": self.reject_class,
                "model_name": self.model_name,
                "actual_device": self.actual_device,
                "amp_active": self.amp_active,
                "counters": dict(self._counters),
                "metrics": dict(self._metrics),
                "latest": None if self._latest is None else dict(self._latest),
                "latest_classified": None if self._classified is None else dict(self._classified[0]),
                "positions": list(self._positions),
                "events": [dict(item) for item in self._events],
                "serial": serial,
                "cuda": self._device_status,
                "preview_encode_ms": self._preview_encode_ms,
                "camera_actual": dict(self._camera_actual),
                "camera_request": dict(self._camera_request),
                "geometry_override": self._geometry_override,
                "geometry_warning": self._camera_actual.get("warning"),
                "yolo_counts": dict(self._yolo_counts),
                "yolo_imgsz": None if self._config is None else self._config.detector.yolo_imgsz,
            }
            if reveal_paths:
                payload.update(
                    {
                        "session_dir": None if self.session_dir is None else str(self.session_dir),
                        "checkpoint": self.checkpoint,
                        "config_path": self.config_path,
                        "camera": "[stream]" if self.source_type == "stream" else self._options.get("source"),
                    }
                )
            else:
                payload["fault"] = "Session fault; see host diagnostics" if self.fault else None
                payload["error"] = payload["fault"]
                payload["checks"] = [{**c, "message": "See host diagnostics" if c["status"] == "failed" else c["message"]} for c in self.checks]
                payload["serial"] = {
                    "connected": serial.get("connected", False),
                    "last_command": serial.get("last_command"),
                    "last_ack": serial.get("last_ack"),
                    "fault": "Serial fault" if serial.get("fault") else None,
                }
                payload["events"] = [_public_event(item) for item in payload["events"]]
                if payload["latest"] is not None:
                    payload["latest"] = _public_event(payload["latest"])
                if payload["latest_classified"] is not None:
                    payload["latest_classified"] = _public_event(payload["latest_classified"])
        payload["preflight"] = {"checks": payload["checks"],
                                "errors": [c["message"] for c in payload["checks"] if c["status"] == "failed"]}
        payload["elapsed_time"] = payload["elapsed_s"]
        return payload

    def export_csv(self) -> str:
        if self.session_dir is None or not (self.session_dir / "events.jsonl").is_file():
            raise ValueError("No Factory Live session log is available")
        rows = [
            json.loads(line)
            for line in (self.session_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        fieldnames = [
            "wall_time_iso",
            "event_id",
            "status",
            "prediction",
            "confidence",
            "detector_confidence",
            "trigger_crossing_wall_time_iso",
            "result",
            "actuator_command",
        ]
        from io import StringIO

        buffer = StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue()


def _result_label(status: str, prediction: str | None, reject_class: str) -> str | None:
    if status != "accepted" or not prediction:
        return None
    return "REJECT" if prediction == reject_class else "PASS"


def _public_event(event: dict[str, Any]) -> dict[str, Any]:
    hidden = {"checkpoint", "extraction_config", "session_dir"}
    return {key: value for key, value in event.items() if key not in hidden}

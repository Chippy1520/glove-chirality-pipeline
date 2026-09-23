from __future__ import annotations

import csv
import hashlib
import json
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from glove_chirality.camera import open_camera
from glove_chirality.config import ExtractionConfig
from glove_chirality.models import CLASSIFIER_CHOICES

MODES = ("preview", "shadow", "armed")
BELT_DIRECTIONS = ("left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top")
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
            opened = open_camera(index) if capture_factory is None else open_camera(
                index, capture_factory=capture_factory
            )
        except RuntimeError:
            continue
        try:
            found.append(
                {
                    "index": index,
                    "backend": opened.backend,
                    "width": opened.width,
                    "height": opened.height,
                    "fps": opened.fps,
                    "label": (
                        f"Camera {index} | {opened.backend} | "
                        f"{opened.width}x{opened.height} | {opened.fps:.0f} FPS"
                    ),
                }
            )
        finally:
            opened.capture.release()
    return found


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


class FactoryLiveSession:
    """Host-side factory inspection session. Inference stays on this server."""

    def __init__(self, workdir: str | Path):
        self.workdir = Path(workdir)
        self._lock = threading.Lock()
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
        with self._lock:
            self.status = status

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
        if mode == "armed":
            if not confirm:
                raise ValueError("ARMED mode requires explicit confirmation")
            if self.fault:
                raise ValueError(f"Cannot arm while a fault is active: {self.fault}")
            serial = self._serial_snapshot()
            if not serial.get("connected"):
                raise ValueError("Connect the actuator before ARMED mode")
        with self._lock:
            self.mode = mode
            self._remember_mode(mode)
        self._write_session()
        return mode

    def _serial_actor(self):
        if self._serial is None:
            from glove_chirality.actuator import SerialActuator

            self._serial = SerialActuator()
        return self._serial

    def serial_ports(self) -> list[dict[str, Any]]:
        from glove_chirality.actuator import list_serial_ports

        return list_serial_ports()

    def connect_serial(self, port: str, baud: int = 115200) -> dict[str, Any]:
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
        if self._serial is None:
            return {"connected": False, "last_command": None, "last_ack": None, "fault": None}
        self._serial.disconnect()
        return self._serial.snapshot()

    def _on_serial_fault(self, message: str) -> None:
        with self._lock:
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

    def start(
        self,
        options: dict[str, Any],
        *,
        runner=None,
        detector=None,
        classifier=None,
        capture=None,
    ) -> dict[str, Any]:
        if self.running:
            raise ValueError("Stop the current Factory Live session before starting another")
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
        self._apply_trigger_overrides(config, options)
        checkpoint = str(options.get("checkpoint", "")).strip()
        if not checkpoint or not Path(checkpoint).is_file():
            raise ValueError(f"Classifier checkpoint not found: {checkpoint}")
        device = str(options.get("device", "auto"))
        if device.startswith("cuda"):
            status = device_status()
            if not status["cuda_available"]:
                raise ValueError("CUDA was requested but PyTorch reports CUDA unavailable")
        reject_class = str(options.get("reject_class", "right"))
        if reject_class not in {"left", "right"}:
            raise ValueError("reject_class must be left or right")
        delay_ms = int(options.get("delay_ms", 850))
        if delay_ms < 0:
            raise ValueError("Actuator delay must be non-negative")
        if mode == "armed" and not self._serial_snapshot().get("connected"):
            raise ValueError("Connect the actuator before ARMED mode")

        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        directory = self.workdir / "outputs" / "factory_live" / stamp
        if directory.exists():
            directory = self.workdir / "outputs" / "factory_live" / f"{stamp}_{time.time_ns()}"
        directory.mkdir(parents=True, exist_ok=False)
        with self._lock:
            self._stop.clear()
            self.running = True
            self.fault = None
            self.status = "Loading Layer-1 detector..."
            self.mode = "shadow"
            self.session_id = f"factory_{stamp}"
            self.session_dir = directory
            self.reject_class = reject_class
            self.checkpoint = checkpoint
            self.config_path = str(config_path)
            self._config = config
            self._delay_ms = delay_ms
            self._options = dict(options)
            self._latest = None
            self._latest_crop = None
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
        self._injected = {
            "runner": runner,
            "detector": detector,
            "classifier": classifier,
            "capture": capture,
        }
        self._thread = threading.Thread(
            target=self._run,
            name="grip-factory-live",
            daemon=True,
        )
        self._thread.start()
        return {"status": "starting", "session_id": self.session_id, "session_dir": str(directory)}

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
            "camera": self._options.get("source"),
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
            "fault": self.fault,
            "status": self.status,
        }
        (self.session_dir / "session.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _run(self) -> None:
        from glove_chirality.detection import build_detector
        from glove_chirality.inference import TorchClassifier
        from glove_chirality.live import (
            LatestFrameCapture,
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
            self._set_status("Loading Layer-2 classifier...")
            if self._stop.is_set():
                return
            classifier = injected.get("classifier") or TorchClassifier(
                self.checkpoint,
                device=str(self._options.get("device", "auto")),
                amp=bool(self._options.get("amp")),
                decision_class=str(self._options.get("decision_class", "argmax")),
                decision_threshold=float(self._options.get("decision_threshold", 0.5)),
            )
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
            capture = injected.get("capture") or LatestFrameCapture(
                parse_capture_source(str(self._options.get("source"))),
                config.runtime.capture_queue_size,
            )
            self._set_status("Warming models...")
            runner(
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
                stop_event=self._stop,
                session_id=self.session_id,
                config_path=self.config_path,
                should_classify=lambda: self.mode != "preview",
            )
            if not self._stop.is_set():
                raise RuntimeError("Camera stream ended")
            self._set_status("stopped")
        except Exception as exc:  # noqa: BLE001 - camera, model, and runtime faults must disarm
            with self._lock:
                self.fault = str(exc)
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

    def _on_frame(self, frame: np.ndarray, result, timestamp_s: float) -> None:
        del timestamp_s
        from glove_chirality.overlay import detection_states, draw_live_overlay

        config = self._config
        if config is None:
            return
        annotated = draw_live_overlay(frame, config, result.detections)
        height, width = frame.shape[:2]
        states = detection_states(config, result.detections, width, height)
        positions = []
        for detection, state in zip(result.detections, states):
            cx, cy = detection.center
            positions.append(
                {
                    "state": state,
                    "confidence": detection.confidence,
                    "bbox": [detection.x1, detection.y1, detection.x2, detection.y2],
                    "center_px": [cx, cy],
                    "center_norm": [cx / max(width, 1), cy / max(height, 1)],
                }
            )
        with self._lock:
            self._latest_frame = annotated
            self._positions = positions

    def _on_metrics(self, metrics: dict[str, Any]) -> None:
        with self._lock:
            self._metrics = dict(metrics)

    def _on_event(self, payload: dict[str, Any]) -> None:
        from glove_chirality.actuator import command_for_event

        record = dict(payload)
        with self._lock:
            mode = self.mode
            reject_class = self.reject_class
            delay_ms = self._delay_ms
            commanded = set(self._commanded)
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
        record["operating_mode"] = mode
        record["result"] = _result_label(status, prediction, reject_class)
        record["would_reject"] = mode == "shadow" and status == "accepted" and prediction == reject_class
        record["actuator_command"] = command.strip() if command else None
        record["classifier_ran"] = status == "accepted" and prediction is not None
        if command:
            with self._lock:
                self._commanded.add(event_id)
                self._counters["commands_sent"] += 1
            try:
                self._serial_actor().submit(command)
            except Exception as exc:  # noqa: BLE001 - a failed enqueue must not block inference
                self._on_serial_fault(str(exc))
        self._count(record)
        crop = record.pop("crop", None)
        if isinstance(crop, np.ndarray):
            encoded = _jpeg(crop, 95)
            with self._lock:
                self._latest_crop = encoded
        with self._lock:
            self._latest = {key: value for key, value in record.items() if key != "crop"}
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

    def frame_jpeg(self) -> bytes | None:
        with self._lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
        if frame is None:
            return None
        return _jpeg(frame, 70)

    def crop_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_crop

    def snapshot(self, *, reveal_paths: bool) -> dict[str, Any]:
        with self._lock:
            serial = self._serial_snapshot()
            payload = {
                "running": self.running,
                "status": self.status,
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
                "positions": list(self._positions),
                "events": [dict(item) for item in self._events],
                "serial": serial,
                "cuda": device_status(),
            }
            if reveal_paths:
                payload.update(
                    {
                        "session_dir": None if self.session_dir is None else str(self.session_dir),
                        "checkpoint": self.checkpoint,
                        "config_path": self.config_path,
                        "camera": self._options.get("source"),
                    }
                )
            else:
                payload["serial"] = {
                    "connected": serial.get("connected", False),
                    "last_command": serial.get("last_command"),
                    "last_ack": serial.get("last_ack"),
                    "fault": serial.get("fault"),
                }
                payload["events"] = [_public_event(item) for item in payload["events"]]
                if payload["latest"] is not None:
                    payload["latest"] = _public_event(payload["latest"])
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

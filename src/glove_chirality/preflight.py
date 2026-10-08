"""Non-actuating checks shared by host validation and asynchronous jobs."""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any

_REQUIRED = {
    "extract_dataset": ("left", "right", "output", "config"),
    "extract_single": ("input", "output", "config"),
    "preview": ("video", "output", "config"),
    "train": ("manifest", "output"),
    "audit_dataset": ("manifest", "output"),
    "infer_video": ("video", "checkpoint", "output", "config"),
    "infer_images": ("input", "checkpoint", "output"),
    "infer_live": ("source", "checkpoint", "output", "config"),
    "explain": ("image", "checkpoint", "output"),
    "tensorboard": ("logdir",),
    "factory": ("source", "config"),
}


def probe_video(path: Path):
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
        if not capture.isOpened() or not ok or frame is None or not frame.size:
            raise ValueError(f"Video has no readable frames: {path}")
        return frame
    finally:
        capture.release()


def probe_camera(source: str):
    from glove_chirality.camera import open_camera

    opened = open_camera(int(source))
    try:
        return opened.first_frame
    finally:
        opened.capture.release()


def probe_classifier(checkpoint: Path, device: str, payload: dict):
    import numpy as np
    import torch

    from glove_chirality.inference import TorchClassifier

    # Decode the external file safely first. The legacy shared classifier currently
    # uses unrestricted torch.load; hand it only a reconstructed, trusted snapshot.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("classes") != ["left", "right"]:
        raise ValueError("Checkpoint classes must be ordered left, right")
    trusted = {key: saved[key] for key in
               ("model_name", "classes", "state_dict", "image_size")}
    if "model_backend" in saved:
        trusted["model_backend"] = saved["model_backend"]
    with tempfile.TemporaryDirectory(prefix="grip-preflight-") as directory:
        snapshot = Path(directory) / "checkpoint.pt"
        torch.save(trusted, snapshot)
        classifier = TorchClassifier(snapshot, device=device,
                                     decision_class=payload.get("decision_class", "argmax"),
                                     decision_threshold=float(payload.get("decision_threshold", 0.5)))
        classifier.warmup()
        label, confidence = classifier.predict_array(
            np.zeros((classifier.image_size, classifier.image_size, 3), dtype=np.uint8))
        if label not in {"left", "right"} or not math.isfinite(confidence):
            raise ValueError("Classifier warm-up returned invalid prediction")


def check_workflow(action: str, payload: dict[str, Any], workdir: str | Path) -> dict[str, Any]:
    root = Path(workdir).resolve()
    checks: list[dict] = []
    errors: list[dict] = []
    warnings: list[dict] = []
    dataset = None

    def record(field, status, message):
        checks.append({"field": field, "label": field.replace("_", " ").title(),
                       "status": status, "message": message})
        if status == "failed":
            errors.append({"field": field, "message": message})
        elif status == "warning":
            warnings.append({"field": field, "message": message})

    def run(field, function):
        try:
            value = function()
            record(field, "passed", "Check passed")
            return value
        except Exception as error:  # noqa: BLE001 - aggregate optional backend failures as checks
            record(field, "failed", str(error))
            return None

    def path(field):
        candidate = Path(str(payload[field])).expanduser()
        return (root / candidate).resolve()

    if action not in _REQUIRED:
        record("action", "failed", "Unknown workflow")
    required = _REQUIRED.get(action, ())
    for field in required:
        if not str(payload.get(field, "")).strip():
            record(field, "failed", "Required field is missing")
    if errors:
        return {"ok": False, "checks": checks, "errors": errors, "warnings": warnings,
                "summary": "Required fields are missing"}

    def output_check():
        destination = path("output")
        ancestor = destination if destination.is_dir() else destination.parent
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        if not ancestor.is_dir() or not os.access(ancestor, os.W_OK):
            raise ValueError("Output parent is not writable")
        with tempfile.TemporaryFile(dir=ancestor) as probe:
            probe.write(b"grip-preflight")
            probe.flush()
        file_output = action in {"train", "preview", "infer_images", "audit_dataset", "explain"}
        if destination.exists() and (destination.is_dir() if file_output else destination.is_file()):
            raise ValueError("Output resource has the wrong type for this workflow")
        inputs = ("manifest", "checkpoint", "config", "video", "image", "input", "left", "right")
        if any(payload.get(field) and path(field) == destination for field in inputs):
            raise ValueError("Output must not overwrite an input resource")
        return destination

    if payload.get("output"):
        run("output", output_check)
    for field, default, minimum, maximum, integer in (
        ("epochs", 20, 1, None, True), ("batch_size", 32, 1, None, True),
        ("head_only_epochs", 0, 0, None, True), ("seed", 42, 0, None, True),
        ("image_size", 224, 1, None, True), ("workers", 0, 0, None, True),
        ("learning_rate", .001, 0, None, False),
        ("validation_fraction", .2, 0, 1, False), ("recall_weight", 1, 0, None, False),
        ("decision_threshold", .5, 0, 1, False), ("seconds", 0, 0, None, False),
        ("warmup_seconds", 2, 0, None, False),
    ):
        if field not in payload and action != "train":
            continue

        def numeric(field=field, default=default, minimum=minimum,
                    maximum=maximum, integer=integer):
            value = float(payload.get(field, default))
            strict = field in {"learning_rate", "validation_fraction"}
            if (not math.isfinite(value) or value < minimum or (strict and value == minimum)
                    or (maximum is not None and (value > maximum or (strict and value == maximum)))
                    or (integer and (value != int(value) or isinstance(payload.get(field), bool)))):
                raise ValueError("Numeric value is outside the allowed range")
        run(field, numeric)
    device = str(payload.get("device", "auto"))
    if action in {"train", "infer_video", "infer_images", "infer_live", "explain", "factory"}:
        def device_check():
            import torch

            if device not in {"auto", "cpu", "cuda"} and not (
                    device.startswith("cuda:") and device[5:].isdigit()):
                raise ValueError("Device must be auto, cpu, cuda or cuda:N")
            if device.startswith("cuda"):
                index = int(device.split(":")[1]) if ":" in device else 0
                if not torch.cuda.is_available() or index >= torch.cuda.device_count():
                    raise ValueError("Requested CUDA device is unavailable")
                torch.zeros(1, device=device)
        run("device", device_check)
    config = None
    if payload.get("config"):
        from glove_chirality.config import ExtractionConfig

        config = run("config", lambda: ExtractionConfig.from_yaml(path("config")))
    frame = None
    for field in ("left", "right", "video") + (("input",) if action == "extract_single" else ()):
        if not payload.get(field):
            continue

        def videos(field=field):
            from glove_chirality.extraction import discover_videos

            files = discover_videos(path(field))
            if not files:
                raise ValueError("No source videos found")
            frames = [probe_video(file) for file in files]
            return frames[0]
        sample = run(field, videos)
        if sample is not None:
            frame = sample
    if action in {"factory", "infer_live"}:
        source = str(payload.get("source", "0"))
        frame = run("source", lambda: probe_camera(source) if source.isdigit()
                    else probe_video((root / source).resolve()))
    if config is not None and frame is not None:
        def detector_check():
            from glove_chirality.detection import build_detector

            if config.detector.backend == "yolo":
                model_path = root / config.detector.yolo_model
                if not model_path.is_file():
                    raise ValueError("Detector checkpoint does not exist")
                config.detector.yolo_model = str(model_path.resolve())
            detector = build_detector(config.detector)
            detector.warmup(frame)
            detector.detect(frame)
        run("detector", detector_check)
    if action in {"train", "audit_dataset"}:
        from glove_chirality.dataset_audit import audit_dataset

        def dataset_check():
            nonlocal dataset
            report = audit_dataset(path("manifest"),
                                   validation_fraction=float(payload.get("validation_fraction", .2)),
                                   seed=int(payload.get("seed", 42)))
            dataset = report
            for message in report["warnings"]:
                record("manifest", "warning", message)
            if action == "train" and not report["dataset_ready"]:
                raise ValueError("; ".join(report["errors"] +
                                 ([report["grouped_split"].get("error", "")])))
            if not path("manifest").is_file():
                raise ValueError("Manifest does not exist")
            return report
        run("manifest", dataset_check)
    if action == "train":
        def training_check():
            from glove_chirality.fine_tuning import fine_tuning_config
            from glove_chirality.models import model_backend

            model_backend(str(payload.get("model", "resnet18")))
            fine_tuning_config(int(payload.get("epochs", 20)),
                               float(payload.get("learning_rate", .001)),
                               int(payload.get("head_only_epochs", 0)),
                               float(payload["backbone_learning_rate"])
                               if payload.get("backbone_learning_rate") else None)
        run("training", training_check)
    if action in {"infer_images", "explain"}:
        def images_check():
            import cv2

            from glove_chirality.inference import IMAGE_EXTENSIONS

            source = path("image" if action == "explain" else "input")
            files = [source] if source.is_file() else [p for p in source.rglob("*")
                                                     if p.suffix.lower() in IMAGE_EXTENSIONS]
            if not files:
                raise ValueError("No images found")
            if any(cv2.imread(str(p)) is None for p in files):
                raise ValueError("One or more input images cannot be decoded")
        run("images", images_check)
    if payload.get("checkpoint"):
        run("checkpoint", lambda: probe_classifier(path("checkpoint"), device, payload))
    if action == "tensorboard":
        def tensorboard_check():
            from glove_chirality.web_service import _validate_tensorboard, build_web_command

            _, command = build_web_command(action, payload)
            _validate_tensorboard(command, root)
        run("tensorboard", tensorboard_check)
    if action != "factory":
        def command_check():
            import argparse

            from glove_chirality.cli import build_parser
            from glove_chirality.gui_commands import _base
            from glove_chirality.web_service import build_web_command

            _, command = build_web_command(action, payload)
            if action != "tensorboard":
                parser = build_parser()
                subcommands = next(item for item in parser._actions
                                   if isinstance(item, argparse._SubParsersAction))
                selected = subcommands.choices.get(command[len(_base())])
                if selected is None:
                    raise ValueError("Workflow CLI command is not installed")
                # Validate enumerations against the real parser, without calling
                # parse_args (which writes usage and terminates the host thread).
                for option in selected._actions:
                    if option.choices is None:
                        continue
                    for flag in option.option_strings:
                        if flag in command:
                            value = command[command.index(flag) + 1]
                            if value not in option.choices:
                                raise ValueError(f"{flag} must be one of: {', '.join(option.choices)}")
        run("command", command_check)
    return {"ok": not errors, "checks": checks, "errors": errors, "warnings": warnings,
            "summary": "Ready" if not errors else f"{len(errors)} checks failed", "dataset": dataset}

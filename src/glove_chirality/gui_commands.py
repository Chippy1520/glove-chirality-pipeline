from __future__ import annotations

import sys


def _required(**values: str) -> None:
    missing = [name.replace("_", " ") for name, value in values.items() if not str(value).strip()]
    if missing:
        raise ValueError(f"Required: {', '.join(missing)}")


def _base() -> list[str]:
    return [sys.executable, "-m", "glove_chirality.cli"]


def extract_dataset(left: str, right: str, output: str, config: str) -> list[str]:
    _required(left=left, right=right, output=output, config=config)
    return _base() + [
        "extract-dataset", "--left", left, "--right", right,
        "--output", output, "--config", config,
    ]


def extract_single(input_path: str, output: str, label: str, config: str) -> list[str]:
    _required(input_path=input_path, output=output, config=config)
    return _base() + [
        "extract", "--input", input_path, "--output", output,
        "--label", label, "--config", config,
    ]


def preview(video: str, output: str, seconds: float, config: str, warmup_seconds: float = 2.0) -> list[str]:
    _required(video=video, output=output, config=config)
    return _base() + [
        "preview", "--video", video, "--output", output,
        "--seconds", str(seconds), "--warmup-seconds", str(warmup_seconds),
        "--config", config,
    ]


def train(
    manifest: str,
    output: str,
    model: str,
    epochs: int,
    batch_size: int,
    image_size: int,
    learning_rate: float,
    validation_fraction: float,
    seed: int,
    device: str,
    workers: int,
    amp: bool,
    loss: str = "weighted_cross_entropy",
    recall_target: str = "right",
    recall_weight: float = 1.0,
    selection_metric: str = "macro_recall",
) -> list[str]:
    _required(manifest=manifest, output=output)
    command = _base() + [
        "train", "--manifest", manifest, "--output", output,
        "--model", model, "--epochs", str(epochs),
        "--batch-size", str(batch_size), "--image-size", str(image_size),
        "--learning-rate", str(learning_rate),
        "--validation-fraction", str(validation_fraction),
        "--seed", str(seed), "--device", device, "--workers", str(workers),
        "--loss", loss, "--recall-target", recall_target,
        "--recall-weight", str(recall_weight),
        "--selection-metric", selection_metric,
    ]
    if amp:
        command.append("--amp")
    return command


def infer_video(
    video: str,
    checkpoint: str,
    output: str,
    config: str,
    device: str,
    decision_class: str = "argmax",
    decision_threshold: float = 0.5,
) -> list[str]:
    _required(video=video, checkpoint=checkpoint, output=output, config=config)
    return _base() + [
        "infer-video", "--video", video, "--checkpoint", checkpoint,
        "--output", output, "--config", config, "--device", device,
        "--decision-class", decision_class,
        "--decision-threshold", str(decision_threshold),
    ]


def infer_images(
    input_path: str,
    checkpoint: str,
    output: str,
    device: str,
    decision_class: str = "argmax",
    decision_threshold: float = 0.5,
) -> list[str]:
    _required(input_path=input_path, checkpoint=checkpoint, output=output)
    return _base() + [
        "infer-images", "--input", input_path, "--checkpoint", checkpoint,
        "--output", output, "--device", device,
        "--decision-class", decision_class,
        "--decision-threshold", str(decision_threshold),
    ]


def infer_live(
    source: str,
    checkpoint: str,
    output: str,
    config: str,
    device: str,
    amp: bool = False,
    decision_class: str = "argmax",
    decision_threshold: float = 0.5,
) -> list[str]:
    _required(source=source, checkpoint=checkpoint, output=output, config=config)
    command = _base() + [
        "infer-live",
        "--source", source,
        "--checkpoint", checkpoint,
        "--output", output,
        "--config", config,
        "--device", device,
        "--decision-class", decision_class,
        "--decision-threshold", str(decision_threshold),
    ]
    if amp:
        command.append("--amp")
    return command

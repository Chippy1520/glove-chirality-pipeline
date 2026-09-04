from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from pathlib import Path

COMPARISON_METRICS = (
    "recall_right",
    "macro_recall",
    "macro_f1",
    "accuracy",
    "recall_left",
)


@dataclass(frozen=True)
class ModelRun:
    source: str
    model: str
    augmentation: str
    selection_metric: str
    split_id: str
    accuracy: float
    macro_recall: float
    macro_f1: float
    recall_left: float
    recall_right: float
    train_samples: int | None = None
    validation_samples: int | None = None

    def metric(self, name: str) -> float:
        if name not in COMPARISON_METRICS:
            raise ValueError(f"comparison metric must be one of: {', '.join(COMPARISON_METRICS)}")
        return float(getattr(self, name))


def _number(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _optional_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _count_total(value) -> int | None:
    if not isinstance(value, dict):
        return None
    counts = [_optional_int(count) for count in value.values()]
    if any(count is None for count in counts):
        return None
    return sum(count for count in counts if count is not None)


def _run_from_metrics(path: Path, payload: dict[str, object]) -> ModelRun | None:
    metrics = payload.get("best_validation")
    if not isinstance(metrics, dict):
        return None
    recalls = metrics.get("recall_per_class", [0.0, 0.0])
    if not isinstance(recalls, list) or len(recalls) < 2:
        recalls = [0.0, 0.0]
    return ModelRun(
        source=str(path),
        model=str(payload.get("model", "unknown")),
        augmentation=str(payload.get("augmentation", "unknown")),
        selection_metric=str(payload.get("selection_metric", "unknown")),
        split_id=str(payload.get("split_id", "unknown")),
        accuracy=_number(metrics.get("accuracy")),
        macro_recall=_number(metrics.get("macro_recall", metrics.get("balanced_accuracy"))),
        macro_f1=_number(metrics.get("macro_f1")),
        recall_left=_number(recalls[0]),
        recall_right=_number(recalls[1]),
        train_samples=_optional_int(payload.get("train_samples")),
        validation_samples=_optional_int(payload.get("validation_samples")),
    )


def _run_from_history(path: Path, payload: dict[str, object]) -> ModelRun | None:
    history = payload.get("history")
    if not isinstance(history, list):
        return None
    candidates = []
    for row in history:
        if not isinstance(row, dict) or not isinstance(row.get("validation"), dict):
            continue
        metrics = row["validation"]
        candidates.append((
            _number(metrics.get("macro_recall", metrics.get("balanced_accuracy"))),
            _number(metrics.get("macro_f1")),
            metrics,
        ))
    if not candidates:
        return None
    _, _, metrics = max(candidates, key=lambda item: (item[0], item[1]))
    recalls = metrics.get("recall_per_class", [0.0, 0.0])
    if not isinstance(recalls, list) or len(recalls) < 2:
        recalls = [0.0, 0.0]
    train_counts = payload.get("train_counts")
    validation_counts = payload.get("validation_counts")
    return ModelRun(
        source=str(path),
        model=str(payload.get("model", "unknown")),
        augmentation=str(payload.get("augmentation", "standard")),
        selection_metric=str(payload.get("selection_metric", "macro_recall")),
        split_id=str(payload.get("split_id", "unknown")),
        accuracy=_number(metrics.get("accuracy")),
        macro_recall=_number(metrics.get("macro_recall", metrics.get("balanced_accuracy"))),
        macro_f1=_number(metrics.get("macro_f1")),
        recall_left=_number(recalls[0]),
        recall_right=_number(recalls[1]),
        train_samples=_count_total(train_counts),
        validation_samples=_count_total(validation_counts),
    )


def load_model_run(path: str | Path) -> ModelRun | None:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return _run_from_metrics(path, payload) or _run_from_history(path, payload)


def discover_model_runs(paths: Iterable[str | Path]) -> list[ModelRun]:
    files: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(path.rglob("*.metrics.json"))
            files.update(path.rglob("*.history.json"))
    runs = [run for path in sorted(files) if (run := load_model_run(path)) is not None]
    return sort_model_runs(runs)


def sort_model_runs(runs: Iterable[ModelRun], metric: str = "recall_right") -> list[ModelRun]:
    if metric not in COMPARISON_METRICS:
        raise ValueError(f"comparison metric must be one of: {', '.join(COMPARISON_METRICS)}")
    return sorted(
        runs,
        key=lambda run: (run.metric(metric), run.macro_recall, run.macro_f1),
        reverse=True,
    )


def write_comparison_csv(runs: Iterable[ModelRun], output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(run) for run in runs]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[field.name for field in fields(ModelRun)])
        writer.writeheader()
        writer.writerows(rows)
    return output

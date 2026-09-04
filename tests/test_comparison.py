import csv
import json

from glove_chirality.comparison import (
    discover_model_runs,
    load_model_run,
    sort_model_runs,
    write_comparison_csv,
)


def _metrics(accuracy, macro_recall, macro_f1, left, right):
    return {
        "accuracy": accuracy,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "recall_per_class": [left, right],
    }


def test_discovers_current_training_summary_and_sorts_for_right_recall(tmp_path):
    first = tmp_path / "resnet.pt.metrics.json"
    second = tmp_path / "mobile.pt.metrics.json"
    first.write_text(
        json.dumps(
            {
                "model": "resnet18",
                "augmentation": "standard",
                "selection_metric": "macro_recall",
                "train_samples": 80,
                "validation_samples": 20,
                "best_validation": _metrics(0.9, 0.88, 0.87, 0.96, 0.80),
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "model": "mobilenet_v3_small",
                "augmentation": "anti_spurious",
                "selection_metric": "recall_right",
                "split_id": "abc123",
                "best_validation": _metrics(0.87, 0.89, 0.86, 0.84, 0.94),
            }
        ),
        encoding="utf-8",
    )

    runs = discover_model_runs([tmp_path])

    assert [run.model for run in runs] == ["mobilenet_v3_small", "resnet18"]
    assert runs[0].augmentation == "anti_spurious"
    assert runs[0].split_id == "abc123"
    assert runs[1].train_samples == 80


def test_reads_explicit_split_history_and_chooses_best_epoch(tmp_path):
    path = tmp_path / "experiment.history.json"
    path.write_text(
        json.dumps(
            {
                "model": "convnextv2_pico",
                "train_counts": {"left": 40, "right": 35},
                "validation_counts": {"left": 10, "right": 9},
                "history": [
                    {"epoch": 1, "validation": _metrics(0.7, 0.72, 0.71, 0.8, 0.64)},
                    {"epoch": 2, "validation": _metrics(0.8, 0.82, 0.81, 0.78, 0.86)},
                ],
            }
        ),
        encoding="utf-8",
    )

    run = load_model_run(path)

    assert run is not None
    assert run.model == "convnextv2_pico"
    assert run.recall_right == 0.86
    assert run.train_samples == 75
    assert run.validation_samples == 19


def test_export_and_alternate_sort_metric(tmp_path):
    paths = []
    for index, (model, left, right) in enumerate(
        [("right_first", 0.5, 0.95), ("left_first", 0.99, 0.7)]
    ):
        path = tmp_path / f"{index}.metrics.json"
        path.write_text(
            json.dumps(
                {
                    "model": model,
                    "best_validation": _metrics(0.8, (left + right) / 2, 0.75, left, right),
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    runs = discover_model_runs(paths)
    assert sort_model_runs(runs, "recall_left")[0].model == "left_first"

    output = write_comparison_csv(runs, tmp_path / "comparison.csv")
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert "recall_right" in rows[0]


def test_ignores_invalid_or_unrelated_json(tmp_path):
    invalid = tmp_path / "broken.metrics.json"
    invalid.write_text("not json", encoding="utf-8")
    unrelated = tmp_path / "other.history.json"
    unrelated.write_text(json.dumps({"history": []}), encoding="utf-8")
    assert discover_model_runs([tmp_path]) == []

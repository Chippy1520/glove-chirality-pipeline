"""Regression coverage for existing history/Compare consumers and Python 3.10."""
import hashlib
import json

import pytest

from glove_chirality import explicit_training
from glove_chirality.comparison import load_model_run


def test_explicit_digest_without_python311_file_digest(tmp_path, monkeypatch):
    path = tmp_path / "bytes"
    path.write_bytes(b"synthetic fixture")
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    assert explicit_training.digest(path) == hashlib.sha256(b"synthetic fixture").hexdigest()


@pytest.mark.parametrize("tensorboard", [False, True])
def test_explicit_history_and_metrics_remain_comparison_compatible(tmp_path, monkeypatch, tensorboard):
    torch = pytest.importorskip("torch")
    if tensorboard:
        pytest.importorskip("tensorboard")
    root = tmp_path / "data"
    for split in ("train", "val"):
        for label in ("left", "right"):
            folder = root / split / label
            folder.mkdir(parents=True)
            for index in range(2):
                (folder / f"{index}.jpg").write_bytes(f"{split}-{label}-{index}".encode())
    monkeypatch.setattr(explicit_training, "ManifestDataset", lambda rows, *args:
                        torch.utils.data.TensorDataset(
                            torch.tensor([[1., 0., 0., 0.] if r["label"] == "left"
                                          else [0., 1., 0., 0.] for r in rows]),
                            torch.tensor([int(r["label"] == "right") for r in rows]),
                        ))
    monkeypatch.setattr(explicit_training, "build_model", lambda *args, **kwargs:
                        torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2)))
    output = tmp_path / "run" / "best.pt"
    args = explicit_training.parser().parse_args([
        "--root", str(root), "--output", str(output), "--model", "tiny_cnn",
        "--epochs", "2", "--head-only-epochs", "1", "--no-amp", "--device", "cpu",
        "--fine-tune-head-learning-rate", "0.0001",
    ])
    if tensorboard:
        args.tensorboard_logdir = tmp_path / "tensorboard"
    summary = explicit_training.run(args)
    assert summary["history"][1]["head_learning_rate"] == 0.0001
    assert summary["runtime_versions"]["torch"]
    checkpoint = torch.load(output, weights_only=True)
    assert checkpoint["class_weights"] == [1.0, 1.0]
    assert "head_learning_rate" in checkpoint and "runtime_versions" in checkpoint
    if tensorboard:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events = EventAccumulator(str(args.tensorboard_logdir)).Reload()
        tags = events.Tags()["scalars"]
        assert "training/gradient_norm_max" in tags
        assert "training/optimizer_steps" in tags
        assert "validation/predicted_fraction_right" in tags
        assert "validation/recall_right" in tags
        assert len(events.Scalars("training/head_learning_rate")) == 2
    for suffix in (".history.json", ".metrics.json"):
        path = output.with_suffix(suffix)
        payload = json.loads(path.read_text())
        assert isinstance(payload, dict), "Existing consumers expect a metadata/history object"
        assert payload["classes"] == ["left", "right"]
        assert len(payload["history"]) == 2
        run = load_model_run(path)
        assert run is not None
        assert run.model == "tiny_cnn"
        assert run.augmentation == "standard"
        assert run.selection_metric == "macro_recall"
        assert run.split_id != "unknown"
        assert run.train_samples == run.validation_samples == 4
        assert run.macro_recall == summary["best_validation"]["macro_recall"]


def test_unfreeze_head_lr_must_exceed_backbone_lr():
    args = explicit_training.parser().parse_args([
        "--root", "unused", "--output", "unused.pt", "--staged-finetune",
        "--fine-tune-head-learning-rate", "0.00001",
    ])
    with pytest.raises(ValueError, match="must exceed"):
        explicit_training.recipe(args)

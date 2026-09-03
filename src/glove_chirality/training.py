from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from glove_chirality.dataset import CLASSES, ManifestDataset, grouped_split, read_manifest
from glove_chirality.models import build_model, model_backend

LOSS_CHOICES = ("cross_entropy", "weighted_cross_entropy", "recall_hybrid")
SELECTION_METRICS = (
    "accuracy",
    "macro_recall",
    "macro_f1",
    "recall_left",
    "recall_right",
)


def classification_metrics(targets: list[int], predictions: list[int], classes: int = 2):
    confusion = [[0 for _ in range(classes)] for _ in range(classes)]
    for target, prediction in zip(targets, predictions, strict=True):
        confusion[target][prediction] += 1
    total = sum(sum(row) for row in confusion)
    accuracy = sum(confusion[index][index] for index in range(classes)) / max(total, 1)
    recalls, precisions, f1_scores = [], [], []
    for index in range(classes):
        true_positive = confusion[index][index]
        false_negative = sum(confusion[index]) - true_positive
        false_positive = sum(row[index] for row in confusion) - true_positive
        recall = true_positive / max(true_positive + false_negative, 1)
        precision = true_positive / max(true_positive + false_positive, 1)
        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(2 * precision * recall / max(precision + recall, 1e-12))
    return {
        "accuracy": accuracy,
        "balanced_accuracy": sum(recalls) / classes,
        "macro_recall": sum(recalls) / classes,
        "macro_f1": sum(f1_scores) / classes,
        "recall_per_class": recalls,
        "precision_per_class": precisions,
        "confusion_matrix": confusion,
    }


def metric_score(metrics: dict[str, object], metric: str, class_names=CLASSES) -> float:
    if metric in {"accuracy", "macro_recall", "macro_f1"}:
        return float(metrics[metric])
    if metric.startswith("recall_"):
        label = metric.removeprefix("recall_")
        if label not in class_names:
            raise ValueError(f"Unknown recall target: {label}")
        recalls = metrics["recall_per_class"]
        return float(recalls[class_names.index(label)])
    raise ValueError(f"Unknown selection metric: {metric}")


def build_training_loss(
    torch,
    name: str,
    class_weights,
    recall_target_index: int,
    recall_weight: float,
):
    if name not in LOSS_CHOICES:
        raise ValueError(f"Unknown training loss: {name}")
    if recall_weight < 0.0:
        raise ValueError("recall_weight must be non-negative")
    weights = class_weights if name != "cross_entropy" else None
    cross_entropy = torch.nn.CrossEntropyLoss(weight=weights)
    if name != "recall_hybrid":
        return cross_entropy

    class RecallHybridLoss(torch.nn.Module):
        def forward(self, logits, targets):
            ce_loss = cross_entropy(logits, targets)
            target_mask = targets == recall_target_index
            target_count = target_mask.sum()
            target_probability = torch.softmax(logits, dim=1)[:, recall_target_index]
            soft_recall = (target_probability * target_mask).sum() / target_count.clamp_min(1)
            recall_penalty = torch.where(
                target_count > 0,
                1.0 - soft_recall,
                torch.zeros_like(soft_recall),
            )
            return ce_loss + recall_weight * recall_penalty

    return RecallHybridLoss()


def train_classifier(
    manifest: str | Path,
    output: str | Path,
    model_name: str = "tiny_cnn",
    epochs: int = 10,
    batch_size: int = 32,
    image_size: int = 224,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    seed: int = 42,
    device_name: str = "auto",
    amp: bool = False,
    workers: int = 0,
    loss_name: str = "weighted_cross_entropy",
    recall_target: str = "right",
    recall_weight: float = 1.0,
    selection_metric: str = "macro_recall",
) -> dict[str, object]:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("Training requires: pip install -e .[ml]") from exc

    torch.manual_seed(seed)
    if recall_target not in CLASSES:
        raise ValueError(f"recall_target must be one of: {', '.join(CLASSES)}")
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of: {', '.join(SELECTION_METRICS)}")
    rows = read_manifest(manifest)
    train_rows, validation_rows = grouped_split(rows, validation_fraction, seed)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch installation cannot access a GPU")
    device = torch.device(device_name)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        ManifestDataset(train_rows, image_size, True), shuffle=True, **loader_options
    )
    validation_loader = DataLoader(
        ManifestDataset(validation_rows, image_size, False), **loader_options
    )
    model = build_model(model_name, len(CLASSES), pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    counts = Counter(row["label"] for row in train_rows)
    class_weights = torch.tensor(
        [len(train_rows) / (len(CLASSES) * counts[label]) for label in CLASSES],
        dtype=torch.float32,
        device=device,
    )
    loss_fn = build_training_loss(
        torch,
        loss_name,
        class_weights,
        CLASSES.index(recall_target),
        recall_weight,
    )
    use_amp = amp and device.type == "cuda"
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:  # PyTorch 2.2 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_score = -1.0
    best_tiebreak = -1.0
    best_metrics: dict[str, object] = {}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(images), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        targets_all: list[int] = []
        predictions_all: list[int] = []
        with torch.no_grad():
            for images, targets in validation_loader:
                predictions = model(images.to(device)).argmax(1).cpu()
                predictions_all.extend(predictions.tolist())
                targets_all.extend(targets.tolist())
        metrics = classification_metrics(targets_all, predictions_all, len(CLASSES))
        print(
            f"epoch={epoch + 1}/{epochs} "
            f"val_accuracy={metrics['accuracy']:.4f} "
            f"val_macro_recall={metrics['macro_recall']:.4f} "
            f"val_recall_right={metrics['recall_per_class'][CLASSES.index('right')]:.4f} "
            f"val_macro_f1={metrics['macro_f1']:.4f}"
        )
        score = metric_score(metrics, selection_metric)
        tiebreak = float(metrics["macro_f1"])
        if score > best_score or (score == best_score and tiebreak > best_tiebreak):
            best_score, best_tiebreak, best_metrics = score, tiebreak, metrics
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_name": model_name,
                    "model_backend": model_backend(model_name),
                    "classes": CLASSES,
                    "image_size": image_size,
                    "preprocessing": "imagenet_rgb_normalized_no_reflection",
                    "training_loss": loss_name,
                    "recall_target": recall_target,
                    "recall_weight": recall_weight,
                    "selection_metric": selection_metric,
                    "validation_metrics": metrics,
                },
                output,
            )

    summary = {
        "model": model_name,
        "device": str(device),
        "mixed_precision": use_amp,
        "workers": workers,
        "training_loss": loss_name,
        "recall_target": recall_target,
        "recall_weight": recall_weight,
        "selection_metric": selection_metric,
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "best_validation": best_metrics,
    }
    output.with_suffix(output.suffix + ".metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary

import json
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter


CHECKPOINT = Path(
    r"checkpoints\chirality_mobilenet_v3_small_aug27_v1.pt"
)

HISTORY = Path(
    r"checkpoints\chirality_mobilenet_v3_small_aug27_v1.history.json"
)

LOGDIR = Path(
    r"runs\tensorboard\chirality_mobilenet_v3_small_aug27_v1"
)

LOGDIR.mkdir(parents=True, exist_ok=True)

history_data = json.loads(
    HISTORY.read_text(encoding="utf-8")
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

writer = SummaryWriter(str(LOGDIR))


for row in history_data["history"]:

    epoch = int(row["epoch"])
    val = row["validation"]

    # Loss
    writer.add_scalar(
        "Loss/train",
        row["train_loss"],
        epoch,
    )

    writer.add_scalar(
        "Loss/validation",
        val["loss"],
        epoch,
    )

    # Main metrics
    writer.add_scalar(
        "Metrics/accuracy",
        val["accuracy"],
        epoch,
    )

    writer.add_scalar(
        "Metrics/macro_recall",
        val["macro_recall"],
        epoch,
    )

    writer.add_scalar(
        "Metrics/macro_f1",
        val["macro_f1"],
        epoch,
    )

    writer.add_scalar(
        "Metrics/balanced_accuracy",
        val["balanced_accuracy"],
        epoch,
    )

    # Per-class recall
    writer.add_scalar(
        "Recall/left",
        val["recall_per_class"][0],
        epoch,
    )

    writer.add_scalar(
        "Recall/right",
        val["recall_per_class"][1],
        epoch,
    )

    # Per-class precision
    writer.add_scalar(
        "Precision/left",
        val["precision_per_class"][0],
        epoch,
    )

    writer.add_scalar(
        "Precision/right",
        val["precision_per_class"][1],
        epoch,
    )

    # Learning rate
    writer.add_scalar(
        "Training/learning_rate",
        row["learning_rate"],
        epoch,
    )

    # Confusion matrix as text
    cm = val["confusion_matrix"]

    cm_text = (
        "| | Pred Left | Pred Right |\n"
        "|---|---:|---:|\n"
        f"| True Left | {cm[0][0]} | {cm[0][1]} |\n"
        f"| True Right | {cm[1][0]} | {cm[1][1]} |"
    )

    writer.add_text(
        "Validation/confusion_matrix",
        cm_text,
        epoch,
    )


# Experiment summary
best_metrics = checkpoint["validation_metrics"]

summary = (
    f"**Model:** {checkpoint['model_name']}  \n"
    f"**Best epoch:** {checkpoint['best_epoch']}  \n"
    f"**Image size:** {checkpoint['image_size']}  \n"
    f"**Training loss:** {checkpoint['training_loss']}  \n"
    f"**Accuracy:** {best_metrics['accuracy']:.6f}  \n"
    f"**Macro recall:** {best_metrics['macro_recall']:.6f}  \n"
    f"**Macro F1:** {best_metrics['macro_f1']:.6f}  \n"
    f"**Left recall:** {best_metrics['recall_per_class'][0]:.6f}  \n"
    f"**Right recall:** {best_metrics['recall_per_class'][1]:.6f}"
)

writer.add_text(
    "Experiment/summary",
    summary,
    0,
)

writer.flush()
writer.close()

print("TensorBoard logs written to:")
print(LOGDIR)

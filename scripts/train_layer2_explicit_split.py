from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from glove_chirality.dataset import CLASSES, ManifestDataset
from glove_chirality.models import build_model
from glove_chirality.training import classification_metrics


# ============================================================
# COMMAND-LINE PATHS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Train a Layer-2 chirality classifier using an explicit "
        "train/validation directory split."
    )
)

parser.add_argument(
    "--root",
    type=Path,
    required=True,
    help="Dataset root containing train/ and val/ directories.",
)

parser.add_argument(
    "--output",
    type=Path,
    required=True,
    help="Output classifier checkpoint path.",
)

args = parser.parse_args()

ROOT = args.root
OUTPUT = args.output


# ============================================================
# TRAINING CONFIG
# ============================================================

MODEL_NAME = "mobilenet_v3_small"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

PATIENCE = 8
SEED = 42
WORKERS = 0

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

USE_AMP = DEVICE.type == "cuda"


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DATA
# ============================================================

def collect_rows(split: str):

    rows = []

    for label in CLASSES:

        base = ROOT / split / label

        images = sorted(
            p
            for p in base.rglob("*.jpg")
            if "images" in p.parts
        )

        for path in images:

            rows.append(
                {
                    "absolute_path": str(path),
                    "label": label,
                }
            )

    return rows


train_rows = collect_rows("train")
val_rows = collect_rows("val")

train_counts = Counter(
    row["label"]
    for row in train_rows
)

val_counts = Counter(
    row["label"]
    for row in val_rows
)


print()
print("==============================")
print("DATASET")
print("==============================")

print("TRAIN:", dict(train_counts))
print("VAL:", dict(val_counts))

print("TRAIN TOTAL:", len(train_rows))
print("VAL TOTAL:", len(val_rows))


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = ManifestDataset(
    train_rows,
    IMAGE_SIZE,
    training=True,
)

val_dataset = ManifestDataset(
    val_rows,
    IMAGE_SIZE,
    training=False,
)

generator = torch.Generator()
generator.manual_seed(SEED)

loader_options = {
    "batch_size": BATCH_SIZE,
    "num_workers": WORKERS,
    "pin_memory": DEVICE.type == "cuda",
    "persistent_workers": WORKERS > 0,
}

train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    generator=generator,
    **loader_options,
)

val_loader = DataLoader(
    val_dataset,
    shuffle=False,
    **loader_options,
)


# ============================================================
# MODEL
# ============================================================

model = build_model(
    MODEL_NAME,
    num_classes=len(CLASSES),
    pretrained=True,
).to(DEVICE)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_weights = torch.tensor(
    [
        len(train_rows)
        /
        (
            len(CLASSES)
            * train_counts[label]
        )
        for label in CLASSES
    ],
    dtype=torch.float32,
    device=DEVICE,
)

print()
print("CLASS WEIGHTS:")

for label, weight in zip(
    CLASSES,
    class_weights.tolist(),
):
    print(
        label,
        "=",
        round(weight, 4),
    )


loss_fn = torch.nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3,
)


# ============================================================
# AMP
# ============================================================

if hasattr(torch.amp, "GradScaler"):
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP,
    )
else:
    scaler = torch.cuda.amp.GradScaler(
        enabled=USE_AMP,
    )


# ============================================================
# VALIDATION
# ============================================================

def validate():

    model.eval()

    all_targets = []
    all_predictions = []

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for images, targets in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            targets = targets.to(
                DEVICE,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP,
            ):

                logits = model(images)

                loss = loss_fn(
                    logits,
                    targets,
                )

            batch_size = targets.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

            predictions = (
                logits.argmax(dim=1)
                .detach()
                .cpu()
                .tolist()
            )

            all_predictions.extend(
                predictions
            )

            all_targets.extend(
                targets.detach()
                .cpu()
                .tolist()
            )

    metrics = classification_metrics(
        all_targets,
        all_predictions,
        classes=len(CLASSES),
    )

    metrics["loss"] = (
        total_loss
        / max(total_samples, 1)
    )

    return metrics


# ============================================================
# TRAIN
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

history = []

best_score = -1.0
best_f1 = -1.0

epochs_without_improvement = 0


print()
print("==============================")
print("TRAINING")
print("==============================")

print("MODEL:", MODEL_NAME)
print("DEVICE:", DEVICE)
print("AMP:", USE_AMP)
print()


for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0
    seen = 0

    for images, targets in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=USE_AMP,
        ):

            logits = model(images)

            loss = loss_fn(
                logits,
                targets,
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        batch = targets.size(0)

        running_loss += (
            loss.item()
            * batch
        )

        seen += batch


    train_loss = (
        running_loss
        / max(seen, 1)
    )

    metrics = validate()

    score = float(
        metrics["macro_recall"]
    )

    macro_f1 = float(
        metrics["macro_f1"]
    )

    left_recall = float(
        metrics["recall_per_class"][0]
    )

    right_recall = float(
        metrics["recall_per_class"][1]
    )

    lr = optimizer.param_groups[0]["lr"]

    print(
        f"epoch={epoch:02d}/{EPOCHS} "
        f"train_loss={train_loss:.4f} "
        f"val_loss={metrics['loss']:.4f} "
        f"acc={metrics['accuracy']:.4f} "
        f"macro_recall={score:.4f} "
        f"macro_f1={macro_f1:.4f} "
        f"R_left={left_recall:.4f} "
        f"R_right={right_recall:.4f} "
        f"lr={lr:.2e}"
    )

    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "validation": metrics,
        "learning_rate": lr,
    }

    history.append(row)

    improved = (
        score > best_score
        or (
            score == best_score
            and macro_f1 > best_f1
        )
    )

    if improved:

        best_score = score
        best_f1 = macro_f1

        epochs_without_improvement = 0

        torch.save(
            {
                "state_dict":
                    model.state_dict(),

                "model_name":
                    MODEL_NAME,

                "classes":
                    CLASSES,

                "image_size":
                    IMAGE_SIZE,

                "preprocessing":
                    "imagenet_rgb_normalized_no_reflection",

                "training_loss":
                    "weighted_cross_entropy",

                "class_weights":
                    class_weights
                    .detach()
                    .cpu()
                    .tolist(),

                "selection_metric":
                    "macro_recall",

                "best_epoch":
                    epoch,

                "validation_metrics":
                    metrics,

                "train_counts":
                    dict(train_counts),

                "validation_counts":
                    dict(val_counts),
            },
            OUTPUT,
        )

        print(
            "  -> SAVED BEST CHECKPOINT"
        )

    else:

        epochs_without_improvement += 1


    scheduler.step(score)


    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print()
        print(
            "EARLY STOPPING:",
            PATIENCE,
            "epochs without improvement",
        )

        break


# ============================================================
# SAVE HISTORY
# ============================================================

history_path = OUTPUT.with_suffix(
    ".history.json"
)

history_path.write_text(
    json.dumps(
        {
            "model": MODEL_NAME,
            "classes": CLASSES,
            "image_size": IMAGE_SIZE,
            "train_counts":
                dict(train_counts),
            "validation_counts":
                dict(val_counts),
            "class_weights":
                class_weights
                .detach()
                .cpu()
                .tolist(),
            "best_macro_recall":
                best_score,
            "best_macro_f1":
                best_f1,
            "history":
                history,
        },
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("==============================")
print("DONE")
print("==============================")

print("BEST CHECKPOINT:", OUTPUT)
print(
    "BEST MACRO RECALL:",
    round(best_score, 4),
)
print(
    "BEST MACRO F1:",
    round(best_f1, 4),
)
print(
    "HISTORY:",
    history_path,
)


from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

CLASSES = ["left", "right"]


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    valid = [row for row in rows if row["label"] in CLASSES]
    if not valid:
        raise ValueError("Manifest has no left/right labeled rows")
    for row in valid:
        row["absolute_path"] = str(path.parent / row["image_path"])
    return valid


def grouped_split(rows: list[dict[str, str]], validation_fraction: float, seed: int):
    """Split entire source videos, never adjacent crops, between train and validation."""
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[row["label"]].append(row["source_video"])
    rng = random.Random(seed)
    validation_groups: set[str] = set()
    for label, names in groups.items():
        unique = sorted(set(names))
        if len(unique) < 2:
            raise ValueError(f"Need at least two source videos for class {label!r} to make a grouped split")
        rng.shuffle(unique)
        count = max(1, min(len(unique) - 1, round(len(unique) * validation_fraction)))
        validation_groups.update(unique[:count])
    train = [row for row in rows if row["source_video"] not in validation_groups]
    validation = [row for row in rows if row["source_video"] in validation_groups]
    return train, validation


class ManifestDataset:
    def __init__(self, rows, image_size: int, training: bool):
        import cv2
        from PIL import Image
        from torchvision import transforms

        self.cv2, self.Image, self.rows = cv2, Image, rows
        augment = [transforms.ColorJitter(brightness=0.15, contrast=0.15), transforms.RandomRotation(7)] if training else []
        # Intentionally no horizontal flip: reflection can change chirality semantics.
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)), *augment,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = self.cv2.imread(row["absolute_path"])
        if image is None:
            raise FileNotFoundError(row["absolute_path"])
        image = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)
        return self.transform(self.Image.fromarray(image)), CLASSES.index(row["label"])

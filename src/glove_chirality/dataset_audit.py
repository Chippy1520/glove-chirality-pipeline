"""Read-only, decoded-image audit using the trainer's source-grouped split."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from glove_chirality.dataset import CLASSES, grouped_split


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_dataset(manifest: str | Path, output: str | Path | None = None,
                  *, validation_fraction: float = 0.2, seed: int = 42) -> dict[str, Any]:
    import cv2

    path = Path(manifest).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    rows: list[dict] = []
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            missing = {"image_path", "label", "source_video"} - set(reader.fieldnames or [])
            if missing:
                errors.append("Missing manifest columns: " + ", ".join(sorted(missing)))
            raw_rows = list(reader)
            if any(None in row or any(value is None for value in row.values()) for row in raw_rows):
                errors.append("Manifest rows do not match the CSV column schema")
            rows = [{key: value or "" for key, value in row.items() if key is not None}
                    for row in raw_rows]
    except (OSError, csv.Error, UnicodeError) as error:
        errors.append(f"Cannot read manifest: {error}")
    counts = Counter(row.get("label", "") for row in rows)
    sources: dict[str, Counter] = defaultdict(Counter)
    sizes: Counter = Counter()
    hashes: dict[str, list[int]] = defaultdict(list)
    paths: dict[str, list[int]] = defaultdict(list)
    events: dict[str, list[int]] = defaultdict(list)
    invalid: list[dict] = []
    valid_images = 0
    for index, row in enumerate(rows, 2):
        label = row.get("label", "")
        source = row.get("source_video", "")
        sources[source][label] += 1
        problems = []
        if label not in CLASSES:
            problems.append("label must be left or right")
        if not source.strip():
            problems.append("missing source_video")
        raw = row.get("image_path", "")
        image_path = (path.parent / raw).resolve()
        paths[str(image_path)].append(index)
        if row.get("event_id"):
            events[row["event_id"]].append(index)
        try:
            if not raw.strip() or not image_path.is_file():
                raise ValueError("missing image")
            image = cv2.imread(str(image_path))
            if image is None or not image.size:
                raise ValueError("image cannot be decoded")
            height, width = image.shape[:2]
            sizes[f"{width}x{height}"] += 1
            hashes[file_sha256(image_path)].append(index)
            valid_images += 1
        except (OSError, ValueError) as error:
            problems.append(str(error))
        if problems:
            invalid.append({"row": index, "image_path": raw, "errors": problems})
    duplicates = {
        "paths": [indices for indices in paths.values() if len(indices) > 1],
        "content": [indices for indices in hashes.values() if len(indices) > 1],
        "event_ids": [indices for indices in events.values() if len(indices) > 1],
    }
    if not rows:
        errors.append("Dataset has zero crops")
    if invalid:
        errors.append(f"{len(invalid)} invalid manifest rows")
    if any(not counts[label] for label in CLASSES):
        errors.append("Dataset requires both left and right classes")
    if duplicates["paths"] or duplicates["event_ids"]:
        errors.append("Duplicate image paths or event IDs")
    if duplicates["content"]:
        warnings.append("Identical file content detected; review duplicates before training")
        if any(len({rows[index - 2].get("source_video", "") for index in indices}) > 1
               for indices in duplicates["content"]):
            errors.append("Duplicate image content crosses source groups (split leakage risk)")
    if len(sizes) > 1:
        warnings.append("Mixed image dimensions; verify canonical extraction configuration")
    mixed = [source for source, labels in sources.items() if len(labels) > 1]
    if mixed:
        errors.append("Source groups contain conflicting labels")
    split: dict[str, Any] = {"ok": False, "validation_fraction": validation_fraction, "seed": seed}
    try:
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one")
        if errors:
            raise ValueError("Dataset validity checks failed")
        train, validation = grouped_split(rows, validation_fraction, seed)
        train_sources = sorted({row["source_video"] for row in train})
        val_sources = sorted({row["source_video"] for row in validation})
        if set(train_sources) & set(val_sources):
            raise ValueError("Source leakage across split")
        if any(not any(row["label"] == label for row in part)
               for part in (train, validation) for label in CLASSES):
            raise ValueError("Both splits must contain both classes")
        split.update(ok=True, train_count=len(train), validation_count=len(validation),
                     train_sources=train_sources, validation_sources=val_sources)
    except (ValueError, KeyError, TypeError) as error:
        split["error"] = str(error)
    ready = not errors and split["ok"]
    report = {
        "manifest": str(path), "manifest_sha256": file_sha256(path) if path.is_file() else None,
        "counts": {"total": len(rows), "valid_images": valid_images,
                   "classes": dict(counts), "sources": len(sources)},
        "validity": {"ok": not errors, "invalid_rows": invalid},
        "duplicates": duplicates, "dimensions": dict(sizes),
        "sources": {source: dict(labels) for source, labels in sources.items()},
        "grouped_split": split, "errors": errors, "warnings": warnings,
        "dataset_ready": ready,
        "summary": {"ok": ready, "crops": len(rows), "errors": len(errors),
                    "warnings": len(warnings), "dataset_ready": ready},
    }
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report

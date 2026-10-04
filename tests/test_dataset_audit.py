import csv
import json

import cv2
import numpy as np

from glove_chirality.dataset_audit import audit_dataset


def fixture_manifest(tmp_path):
    rows = []
    for index, label in enumerate(("left", "left", "right", "right")):
        image = tmp_path / f"{index}.png"
        cv2.imwrite(str(image), np.full((20, 24, 3), index * 40, dtype=np.uint8))
        rows.append({"event_id": str(index), "image_path": image.name,
                     "label": label, "source_video": f"{label}{index}.mkv"})
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest, rows


def rewrite(manifest, rows):
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_audit_decodes_images_and_uses_source_groups(tmp_path):
    manifest, _ = fixture_manifest(tmp_path)
    report = audit_dataset(manifest, tmp_path / "report.json")
    assert report["dataset_ready"]
    assert report["counts"]["classes"] == {"left": 2, "right": 2}
    assert report["dimensions"] == {"24x20": 4}
    split = report["grouped_split"]
    assert not set(split["train_sources"]) & set(split["validation_sources"])
    assert json.loads((tmp_path / "report.json").read_text())["manifest_sha256"]


def test_missing_corrupt_and_duplicate_rows_are_reported(tmp_path):
    manifest, rows = fixture_manifest(tmp_path)
    (tmp_path / "0.png").unlink()
    (tmp_path / "1.png").write_bytes(b"not an image")
    rows.append(rows[-1].copy())
    rewrite(manifest, rows)
    report = audit_dataset(manifest)
    assert not report["dataset_ready"]
    assert len(report["validity"]["invalid_rows"]) == 2
    assert report["duplicates"]["paths"]
    assert report["duplicates"]["content"]
    assert report["duplicates"]["event_ids"]


def test_empty_missing_schema_and_single_source_fail_readiness(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_path,label,source_video\n", encoding="utf-8")
    assert not audit_dataset(manifest)["dataset_ready"]
    manifest.write_text("wrong\nvalue\n", encoding="utf-8")
    assert any("columns" in error for error in audit_dataset(manifest)["errors"])
    manifest, rows = fixture_manifest(tmp_path)
    for row in rows:
        row["source_video"] = row["label"]
    rewrite(manifest, rows)
    report = audit_dataset(manifest)
    assert report["validity"]["ok"]
    assert not report["grouped_split"]["ok"]
    assert not report["dataset_ready"]


def test_missing_manifest_is_json_ready(tmp_path):
    report = audit_dataset(tmp_path / "missing.csv")
    assert not report["dataset_ready"]
    json.dumps(report, allow_nan=False)


def test_training_preflight_returns_decoded_dataset_summary(tmp_path):
    from glove_chirality.preflight import check_workflow

    manifest, _ = fixture_manifest(tmp_path)
    report = check_workflow("train", {"manifest": str(manifest), "output": "candidate.pt",
                                     "device": "cpu"}, tmp_path)
    assert report["ok"], report
    assert report["dataset"]["counts"]["classes"] == {"left": 2, "right": 2}
    assert report["dataset"]["grouped_split"]["train_count"] == 2


def test_failed_training_preflight_keeps_audit_for_actionable_display(tmp_path):
    from glove_chirality.preflight import check_workflow

    manifest, _ = fixture_manifest(tmp_path)
    (tmp_path / "0.png").unlink()
    report = check_workflow("train", {"manifest": str(manifest), "output": "candidate.pt",
                                     "device": "cpu"}, tmp_path)
    assert not report["ok"]
    assert len(report["dataset"]["validity"]["invalid_rows"]) == 1

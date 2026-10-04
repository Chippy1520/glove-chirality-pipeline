import csv
import json

from glove_chirality.cli import main


def test_audit_cli_saves_report_and_uses_requested_split(tmp_path, capsys):
    from PIL import Image

    rows = []
    for label in ("left", "right"):
        for index in range(4):
            image = tmp_path / f"{label}_{index}.png"
            Image.new("RGB", (24 + index, 24), ((20 if label == "left" else 220), index, 10)).save(image)
            rows.append({"image_path": image.name, "label": label,
                         "source_video": f"{label}_{index}.avi"})
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_path", "label", "source_video"])
        writer.writeheader()
        writer.writerows(rows)
    main(["audit-dataset", "--manifest", str(manifest), "--validation-fraction", "0.5", "--seed", "19"])
    report = json.loads((tmp_path / "dataset_report.json").read_text())
    assert report["grouped_split"]["seed"] == 19
    assert report["grouped_split"]["validation_fraction"] == 0.5
    assert report["grouped_split"]["train_count"] == report["grouped_split"]["validation_count"] == 4
    assert "GRIP_PROGRESS" in capsys.readouterr().out

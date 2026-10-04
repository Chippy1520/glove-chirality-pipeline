"""Regression tests for instrumentation without a second training/extraction algorithm."""
import json
import math

import cv2
import numpy as np
import pytest

from glove_chirality.config import DetectorConfig, EventConfig, ExtractionConfig
from glove_chirality.extraction import extract_video_with_report
from glove_chirality.inference import decision_index
from glove_chirality.progress import emit_progress


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_classifier_output_never_becomes_a_pass(value):
    with pytest.raises(ValueError, match="finite"):
        decision_index(["left", "right"], [value, 0.5])


def test_cli_telemetry_is_one_unbuffered_json_record(capsys):
    emit_progress({"stage": "Training", "progress": {"epoch": 1, "epochs": 2}})
    line = capsys.readouterr().out.strip()
    assert line.startswith("GRIP_PROGRESS ")
    assert json.loads(line.removeprefix("GRIP_PROGRESS "))["progress"]["epoch"] == 1


def test_extraction_reports_actual_counts_without_changing_outcomes(tmp_path):
    video = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (160, 120))
    assert writer.isOpened()
    for index in range(40):
        frame = np.full((120, 160, 3), (65, 175, 65), np.uint8)
        if 8 <= index < 25:
            cv2.rectangle(frame, (60, 35), (100, 85), (35, 35, 35), -1)
        writer.write(frame)
    writer.release()
    config = ExtractionConfig(
        detector=DetectorConfig(roi=(0, 0, 1, 1), trigger_zone=(0.1, 0.1, 0.9, 0.9),
                                min_area_ratio=0.008, morph_kernel=3),
        event=EventConfig(min_detected_frames=2, exit_missing_frames=3, cooldown_frames=3),
    )
    progress = []
    original = extract_video_with_report(video, tmp_path / "original", "left", config)
    observed = extract_video_with_report(video, tmp_path / "observed", "left", config,
                                         progress_callback=progress.append)
    assert len(original.events) == len(observed.events)
    assert [(r.status, r.frame_index) for r in original.records] == [
        (r.status, r.frame_index) for r in observed.records
    ]
    assert progress[-1]["frames"] == 40
    assert progress[-1]["accepted"] == len(observed.events)
    assert progress[-1]["rejected"] == len(observed.records) - len(observed.events)
    assert progress[-1]["done"] is True
    assert progress[-1]["candidates"] > 0


def test_real_training_exposes_loss_metrics_config_and_compatible_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from glove_chirality.extraction import write_manifest
    from glove_chirality.inference import TorchClassifier
    from glove_chirality.training import train_classifier

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        rows = []
        random = np.random.default_rng(44)
        for label in ("left", "right"):
            for index in range(4):
                image = tmp_path / f"{label}_{index}.png"
                assert cv2.imwrite(str(image), random.integers(0, 255, (32, 32, 3), dtype=np.uint8))
                rows.append({"event_id": image.stem, "image_path": image.name,
                             "label": label, "source_video": f"{label}_{index}.avi"})
        manifest = write_manifest(rows, tmp_path / "manifest.csv")
        output = tmp_path / "tiny.pt"
        updates = []
        result = train_classifier(manifest, output, epochs=1, batch_size=2, image_size=32,
                                  device_name="cpu", augmentation="none",
                                  progress_callback=updates.append)
        assert output.is_file()
        assert result["checkpoint"] == str(output.resolve())
        config = json.loads(output.with_suffix(".pt.training_config.json").read_text())
        assert config["epochs"] == 1 and config["amp"] is False
        assert config["architecture"] == "tiny_cnn"
        assert config["manifest_sha256"]
        completed = next(u for u in updates if u["stage"] == "Epoch completed")["progress"]
        assert math.isfinite(completed["train_loss"])
        assert math.isfinite(completed["validation_loss"])
        assert len(completed["validation"]["recall_per_class"]) == 2
        classifier = TorchClassifier(output, device="cpu")
        label, confidence = classifier.predict(tmp_path / rows[0]["image_path"])
        assert label in ("left", "right") and 0 <= confidence <= 1
        assert updates[-1]["stage"] == "Training complete"
        assert json.loads(output.with_suffix(".pt.progress.json").read_text())["status"] == "succeeded"
    finally:
        torch.set_num_threads(previous_threads)

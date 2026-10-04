import numpy as np

from glove_chirality.preflight import check_workflow


def test_all_required_errors_accumulate(tmp_path):
    report = check_workflow("infer_video", {}, tmp_path)
    assert not report["ok"]
    assert {error["field"] for error in report["errors"]} == {
        "video", "checkpoint", "output", "config"}
    assert all(check["status"] == "failed" for check in report["checks"])


def test_train_numbers_missing_dataset_and_bad_device_are_accumulated(tmp_path):
    report = check_workflow("train", {"manifest": "missing.csv", "output": "model.pt",
                                     "epochs": 0, "workers": -1, "batch_size": 1.5,
                                     "learning_rate": "nan", "validation_fraction": 1,
                                     "device": "cuda:-1"}, tmp_path)
    fields = {error["field"] for error in report["errors"]}
    assert {"epochs", "workers", "batch_size", "learning_rate", "validation_fraction",
            "device", "manifest"} <= fields


def test_preview_checks_actual_video_and_yaml(tmp_path):
    (tmp_path / "broken.mkv").write_bytes(b"invalid")
    (tmp_path / "config.yaml").write_text("unknown: true\n", encoding="utf-8")
    report = check_workflow("preview", {"video": "broken.mkv", "config": "config.yaml",
                                       "output": "preview.png"}, tmp_path)
    assert {"video", "config"} <= {error["field"] for error in report["errors"]}


def test_factory_preflight_injectable_camera_detector_no_actuators(tmp_path, monkeypatch):
    from glove_chirality.config import ExtractionConfig

    ExtractionConfig().to_yaml(tmp_path / "config.yaml")
    calls = []
    monkeypatch.setattr("glove_chirality.preflight.probe_camera",
                        lambda source: np.zeros((20, 20, 3), dtype=np.uint8))

    class Detector:
        def warmup(self, frame):
            calls.append("warmup")

        def detect(self, frame):
            calls.append("detect")
            return []
    monkeypatch.setattr("glove_chirality.detection.build_detector", lambda config: Detector())
    report = check_workflow("factory", {"source": "0", "config": "config.yaml", "device": "cpu"},
                            tmp_path)
    assert report["ok"], report
    assert calls == ["warmup", "detect"]


def test_checkpoint_probe_injected_and_all_images_decoded(tmp_path, monkeypatch):
    (tmp_path / "bad.png").write_bytes(b"corrupt")
    calls = []
    monkeypatch.setattr("glove_chirality.preflight.probe_classifier",
                        lambda *args: calls.append("classifier_warmup"))
    report = check_workflow("infer_images", {"input": "bad.png", "checkpoint": "model.pt",
                                            "output": "result.csv", "device": "cpu"}, tmp_path)
    assert not report["ok"]
    assert calls == ["classifier_warmup"]
    assert any(error["field"] == "images" for error in report["errors"])


def test_output_cannot_overwrite_input(tmp_path):
    (tmp_path / "manifest.csv").write_text("image_path,label,source_video\n", encoding="utf-8")
    report = check_workflow("audit_dataset", {"manifest": "manifest.csv", "output": "manifest.csv"},
                            tmp_path)
    assert not report["ok"]
    assert any(error["field"] == "output" for error in report["errors"])

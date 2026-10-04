"""Real CPU CLI jobs and shared video processing; no physical I/O."""
import runpy
import time
from pathlib import Path

import pytest

from glove_chirality.web_app import create_app, create_viewer_app
from glove_chirality.web_service import CommandService

make_video = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/generate_synthetic_videos.py"))["make_video"]


def wait_job(service, job_id, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = service.get_job(job_id)
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(0.03)
    service.stop_job(job_id)
    pytest.fail(f"Timed out: {service.get_job(job_id)}")


def test_missing_fields_are_retained_as_failed_job(tmp_path):
    service = CommandService(tmp_path)
    client = create_app(service).test_client()
    response = client.post("/api/run", json={"action": "train"})
    assert response.status_code == 202
    job = wait_job(service, response.json["job_id"])
    assert job["status"] == "failed"
    assert job["exit_code"] is None
    assert {item["field"] for item in job["preflight"]["errors"]} >= {"manifest", "output"}
    assert client.get(f"/api/jobs/{job['job_id']}").status_code == 200
    assert client.get(f"/api/jobs/{job['job_id']}/logs").status_code == 200


def test_real_extraction_audit_training_and_safe_video(tmp_path):
    pytest.importorskip("torch")
    for label in ("left", "right"):
        for variant in (1, 2):
            make_video(tmp_path / label / f"{label}_{variant}.avi", label, variant, events=2)
    config = tmp_path / "extract.yaml"
    config.write_text("detector:\n  backend: belt_foreground\n  roi: [0, 0, 1, 1]\n  trigger_zone: [0, 0, 1, 1]\nevent:\n  min_detected_frames: 1\n  exit_missing_frames: 2\n  crop_padding: 0\n  output_size: 32\n", encoding="utf-8")
    service = CommandService(tmp_path)
    app = create_app(service)
    client = app.test_client()
    extract = client.post("/api/run", json={"action": "extract_dataset", "left": "left", "right": "right", "output": "dataset", "config": str(config)})
    assert extract.status_code == 202
    extraction = wait_job(service, extract.json["job_id"])
    assert extraction["status"] == "succeeded", extraction
    assert extraction["result"]["dataset_ready"]
    assert (tmp_path / "dataset" / "manifest.csv").is_file()
    checkpoint = tmp_path / "candidate.pt"
    train = client.post("/api/run", json={"action": "train", "manifest": "dataset/manifest.csv", "output": str(checkpoint), "model": "tiny_cnn", "epochs": 2, "image_size": 32, "batch_size": 4, "workers": 0, "device": "cpu", "augmentation": "none", "validation_fraction": .5})
    assert train.status_code == 202
    training = wait_job(service, train.json["job_id"])
    assert training["status"] == "succeeded", training
    assert training["result"]["checkpoint_valid"]
    assert checkpoint.with_suffix(".pt.training_config.json").is_file()
    logs = service.job_logs(training["job_id"])
    assert any("GRIP_PROGRESS" in line["text"] for line in logs)
    artifact = next(index for index, item in enumerate(training["artifacts"]) if item["path"] == str(checkpoint))
    assert client.get(f"/api/jobs/{training['job_id']}/artifacts/{artifact}").status_code == 200
    inference = client.post("/api/inference/start", json={"source_type": "video", "source": str(tmp_path / "right/right_1.avi"), "config": str(config), "checkpoint": str(checkpoint), "device": "cpu", "mode": "armed", "confirm_armed": True, "decision_class": "right", "decision_threshold": 0})
    assert inference.status_code == 202
    session = app.config["INFERENCE"]
    deadline = time.monotonic() + 45
    saw_frame = False
    while time.monotonic() < deadline:
        status = session.snapshot(reveal_paths=True)
        if session.frame_jpeg():
            saw_frame = True
        if not status["running"]:
            break
        time.sleep(.05)
    else:
        session.stop()
        pytest.fail("Video did not finish")
    assert not status.get("fault"), status
    assert saw_frame
    assert status["counters"]["right_reject"] > 0
    assert status["counters"]["commands_sent"] == 0
    assert status["mode"] == "shadow"
    assert client.post("/api/inference/serial/connect", json={"port": "COM_FAKE"}).status_code == 404
    assert client.post("/api/inference/manual-trigger", json={"confirm": True}).status_code == 404
    assert client.post("/api/inference/mode", json={"mode": "armed", "confirm": True}).status_code == 404
    viewer = create_viewer_app(service, lan_token="test-viewer", factory_session=app.config["FACTORY"], inference_session=session).test_client()
    assert viewer.get("/api/state").status_code == 401
    headers = {"X-GRIP-Token": "test-viewer"}
    public = viewer.get("/api/state", headers=headers)
    assert public.status_code == 200
    assert str(tmp_path) not in public.get_data(as_text=True)
    assert "test-viewer" not in public.get_data(as_text=True)
    assert viewer.post("/api/inference/start", json={}, headers=headers).status_code == 404
    assert viewer.get(f"/api/jobs/{training['job_id']}/artifacts", headers=headers).status_code == 404

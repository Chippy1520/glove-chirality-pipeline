from __future__ import annotations

import sys

import pytest

from glove_chirality import gui_commands
from glove_chirality.desktop_worker import create_desktop_app
from glove_chirality.factory_live import FactoryLiveSession
from glove_chirality.web_service import CommandService, build_web_command


@pytest.fixture
def desktop(tmp_path):
    service = CommandService(tmp_path)
    factory = FactoryLiveSession(tmp_path)
    inference = FactoryLiveSession(tmp_path, hardware_allowed=False)
    stopped = []
    app = create_desktop_app(service, factory, inference, "native-test-capability", lambda: stopped.append(True))
    app.config["TESTING"] = True
    yield app.test_client(), app, service, factory, inference, stopped
    service.shutdown()
    factory.stop()
    inference.stop()


HEADERS = {"X-GRIP-Desktop": "native-test-capability"}


def test_native_capability_required_before_admission(desktop):
    client, *_ = desktop
    assert client.get("/api/health").status_code == 403
    assert client.post("/api/run", json={"action": "train"}).status_code == 403
    assert client.get("/api/health", headers={"X-GRIP-Desktop": "wrong"}).status_code == 403
    assert client.get("/api/health", headers=HEADERS).status_code == 200
    assert client.get("/api/health", headers=HEADERS, environ_base={"REMOTE_ADDR": "192.168.1.2"}).status_code == 403


def test_native_schema_retains_all_existing_workflow_fields(desktop):
    client, *_ = desktop
    response = client.get("/api/desktop/schema", headers=HEADERS)
    assert response.status_code == 200
    schema = response.get_json()
    forms = {form["action"]: form for form in schema["forms"]}
    assert set(forms) == {
        "tensorboard", "extract_dataset", "extract_single", "preview", "train",
        "infer_video", "infer_images", "infer_live", "explain", "audit_dataset",
    }
    train = {field["name"]: field for field in forms["train"]["fields"]}
    assert set(train) == {
        "manifest", "output", "model", "device", "epochs", "head_only_epochs",
        "backbone_learning_rate", "batch_size", "image_size", "learning_rate",
        "validation_fraction", "seed", "workers", "loss", "recall_target",
        "recall_weight", "selection_metric", "augmentation", "tensorboard_logdir", "amp",
    }
    assert not train["manifest"]["advanced"]
    assert train["workers"]["advanced"]
    assert train["amp"]["value"] is False
    assert len(train["model"]["choices"]) >= 7
    explain = {field["name"]: field for field in forms["explain"]["fields"]}
    assert {choice["value"] for choice in explain["method"]["choices"]} == {"smoothgrad", "occlusion"}
    factory = {field["name"]: field for field in schema["factory"]}
    assert factory["mode"]["value"] == "shadow"
    assert factory["amp"]["value"] is False
    assert factory["source"]["value"] == "0"
    assert factory["geometry"]["value"] == "yaml"
    assert factory["delay_ms"]["advanced"]
    assert factory["source"]["advanced"] is False


def test_native_state_keeps_shared_live_sessions_and_no_raw_logs(desktop):
    client, app, _, _, inference, _ = desktop
    response = client.get("/api/desktop/state", headers=HEADERS)
    assert response.status_code == 200
    value = response.get_json()
    assert not value.get("logs")
    assert "factory" in value and "inference" in value
    assert not inference.hardware_allowed
    before = app.config["DESKTOP_LAST_SEEN"]
    assert client.post("/api/desktop/heartbeat", headers=HEADERS).status_code == 200
    assert app.config["DESKTOP_LAST_SEEN"] >= before


def test_native_inference_never_registers_hardware_routes(desktop):
    client, *_ = desktop
    for path in ("ports", "mode", "manual-trigger", "serial/connect"):
        response = client.post(f"/api/inference/{path}", json={}, headers=HEADERS)
        assert response.status_code in {404, 405}


def test_native_polling_retains_host_job_details_without_exposing_viewer_paths(desktop):
    client, _, service, *_ = desktop
    job = service._reserve("pipeline", "audit_dataset", "succeeded", {"output": "private-output.json"})
    job.result = {"metrics": {"count": 3}}
    job.artifacts = [{"path": "private-output.json"}]
    job.preflight = {"checks": [{"field": "output", "status": "passed"}]}
    host = client.get("/api/desktop/state", headers=HEADERS).get_json()
    assert host["jobs"]["pipeline"]["output"] == "private-output.json"
    assert host["jobs"]["pipeline"]["result"] == job.result
    assert host["jobs"]["pipeline"]["preflight"] == job.preflight
    assert not host["logs"]
    viewer = service.snapshot(include_logs=False)["jobs"]["pipeline"]
    assert "output" not in viewer and not viewer["result"] and not viewer["artifacts"]


def test_native_worker_uses_existing_preflight(desktop):
    client, *_ = desktop
    response = client.post("/api/jobs/preflight", json={"action": "train", "manifest": "missing.csv", "output": "out.pt"}, headers=HEADERS)
    assert response.status_code in {200, 400}
    assert response.get_json().get("ok") is False or response.get_json().get("error")


def test_frozen_cli_and_tensorboard_are_not_python_interpreter_commands(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert gui_commands._base() == [sys.executable, "cli"]
    command = gui_commands.preview("video.mkv", "preview.png", 0, "config.yaml")
    assert command[:3] == [sys.executable, "cli", "preview"]
    _, command = build_web_command("audit_dataset", {"manifest": "data.csv", "output": "audit.json"})
    assert command[:3] == [sys.executable, "cli", "audit-dataset"]
    assert gui_commands.tensorboard("logs")[:2] == [sys.executable, "tensorboard"]


def test_unfrozen_commands_are_unchanged():
    assert gui_commands._base() == [sys.executable, "-m", "glove_chirality.cli"]
    assert gui_commands.tensorboard("logs")[:3] == [sys.executable, "-m", "tensorboard.main"]


@pytest.mark.parametrize("frozen", [False, True])
def test_preflight_resolves_cli_subcommand_for_both_launchers(tmp_path, monkeypatch, frozen):
    from glove_chirality.preflight import check_workflow

    monkeypatch.setattr(sys, "frozen", frozen, raising=False)
    (tmp_path / "manifest.csv").write_text("image_path,label,source_video\n", encoding="utf-8")
    report = check_workflow("audit_dataset", {"manifest": "manifest.csv", "output": "audit.json"}, tmp_path)
    command_check = next(check for check in report["checks"] if check["field"] == "command")
    assert command_check["status"] == "passed", report

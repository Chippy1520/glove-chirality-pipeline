import subprocess

from glove_chirality import monitoring


def reset(monkeypatch):
    monkeypatch.setattr(monitoring, "_checked", float("-inf"))
    monkeypatch.setattr(monitoring, "_cached", {"status": "unavailable", "gpus": []})


def test_no_gpu_tool_returns_unavailable(monkeypatch):
    reset(monkeypatch)
    monkeypatch.setattr(monitoring.shutil, "which", lambda _: None)
    assert monitoring.gpu_telemetry() == {"status": "unavailable", "gpus": []}


def test_gpu_counters_are_real_and_cached(monkeypatch):
    reset(monkeypatch)
    calls = []
    monkeypatch.setattr(monitoring.shutil, "which", lambda _: "nvidia-smi")

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "0, Fixture GPU, 71, 5120, 8192\n", "")

    monkeypatch.setattr(monitoring.subprocess, "run", run)
    result = monitoring.gpu_telemetry()
    assert result["gpus"][0]["utilization_percent"] == 71
    assert result["gpus"][0]["memory_used_mb"] == 5120
    assert monitoring.gpu_telemetry() == result
    assert len(calls) == 1


def test_gpu_tool_timeout_does_not_fail_workstation(monkeypatch):
    reset(monkeypatch)
    monkeypatch.setattr(monitoring.shutil, "which", lambda _: "nvidia-smi")

    def run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(monitoring.subprocess, "run", run)
    assert monitoring.gpu_telemetry()["status"] == "unavailable"


def test_host_and_authenticated_viewer_expose_only_gpu_counters(tmp_path, monkeypatch):
    from glove_chirality.web_app import create_app, create_viewer_app
    from glove_chirality.web_service import CommandService

    counters = {"status": "available", "gpus": [{"index": 0, "name": "Fixture GPU",
                "utilization_percent": 25, "memory_used_mb": 100, "memory_total_mb": 1000}]}
    monkeypatch.setattr("glove_chirality.web_app.gpu_telemetry", lambda: counters)
    service = CommandService(tmp_path)
    host = create_app(service).test_client()
    assert host.get("/api/state").get_json()["gpu"] == counters
    assert b"workstation.js" in host.get("/").data
    viewer = create_viewer_app(service, lan_token="test-only").test_client()
    assert viewer.get("/api/state").status_code == 401
    response = viewer.get("/api/state", headers={"X-GRIP-Token": "test-only"})
    assert response.get_json()["gpu"] == counters
    assert response.get_json()["logs"] == []

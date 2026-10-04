import json
import sys
import threading
import time

import pytest

from glove_chirality.web_service import CommandService, build_web_command


def wait_job(service, job_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = service.get_job(job_id)
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(.01)
    pytest.fail("job did not finish")


def test_preflight_failure_retained_and_sanitized(tmp_path):
    service = CommandService(tmp_path)
    job_id = service.start_workflow("train", {"manifest": "secret/path.csv"})
    job = wait_job(service, job_id)
    assert job["status"] == "failed"
    assert job["preflight"]["errors"]
    public = json.dumps(service.snapshot(False))
    assert "secret" not in public
    assert "command" not in public
    assert "output" not in public
    assert (tmp_path / "outputs/jobs" / job_id / "job.json").is_file()


def test_preflight_reservation_cancellation_and_independent_slots(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def check(*args):
        entered.set()
        assert release.wait(5)
        return {"ok": True, "checks": [], "errors": [], "warnings": [], "summary": "ready"}
    monkeypatch.setattr("glove_chirality.preflight.check_workflow", check)
    monkeypatch.setattr("glove_chirality.web_service._validate_tensorboard", lambda *args: None)
    service = CommandService(tmp_path)
    job_id = service.start_workflow("preview", {})
    assert entered.wait(5)
    assert service.get_job(job_id)["status"] == "preflight"
    with pytest.raises(RuntimeError):
        service.start("pipeline", [sys.executable, "-c", "print('must not run')"])
    tensorboard = service.start("tensorboard", [sys.executable, "-c", "print('independent')"])
    assert wait_job(service, tensorboard)["status"] == "succeeded"
    assert service.stop_job(job_id)
    release.set()
    assert wait_job(service, job_id)["status"] == "cancelled"
    assert not service.job_logs(job_id) or all("must not run" not in log["text"]
                                             for log in service.job_logs(job_id))


def test_complete_disk_logs_split_stderr_progress_and_history(tmp_path):
    service = CommandService(tmp_path, max_log_entries=3)
    code = ("import sys; print('GRIP_PROGRESS '+"
            "'{\"stage\":\"epoch\",\"progress\":0.5,\"result\":{\"score\":0.8}}'); "
            "print('GRIP_PROGRESS notjson'); print('warning',file=sys.stderr); "
            "[print(i) for i in range(20)]")
    job_id = service.start("pipeline", [sys.executable, "-c", code])
    job = wait_job(service, job_id)
    assert job["status"] == "succeeded"
    assert job["result"]["score"] == .8
    logs = service.job_logs(job_id)
    assert len(logs) > 20
    stderr = next(log for log in logs if log["text"] == "warning")
    assert stderr["stream"] == "stderr"
    assert stderr["level"] == "error"
    assert stderr["timestamp"] and stderr["job_id"] == job_id
    assert len(service.snapshot(True)["logs"]) == 3
    next_id = service.start("pipeline", [sys.executable, "-c", "pass"])
    wait_job(service, next_id)
    assert service.get_job(job_id)["result"]["score"] == .8
    config = json.loads((tmp_path / "outputs/jobs" / job_id / "config.json").read_text())
    assert config["command"] == [sys.executable, "-c", code]


def test_spawn_failure_is_visible_job(tmp_path, monkeypatch):
    monkeypatch.setattr("glove_chirality.preflight.check_workflow", lambda *args: {
        "ok": True, "checks": [], "errors": [], "warnings": [], "summary": "ready"})
    monkeypatch.setattr("glove_chirality.web_service.build_web_command",
                        lambda *args: ("pipeline", [str(tmp_path / "missing_executable")]))
    service = CommandService(tmp_path)
    job = wait_job(service, service.start_workflow("preview", {}))
    assert job["status"] == "failed" and job["error"]["stage"] == "starting"
    assert not service.snapshot()["running"]["pipeline"]


def test_bounded_malformed_progress_is_nonfatal(tmp_path):
    service = CommandService(tmp_path)
    with service._lock:
        job = service._reserve("pipeline", "preview", "preflight", {})
        for payload in ({"stage": "x", "progress": 2}, {"stage": "x", "progress": float("nan")},
                        {"stage": "x", "progress": True}, [], {"stage": "x", "progress": .2}):
            service._progress(job, "GRIP_PROGRESS " + json.dumps(payload))
        assert job.progress == .2
        service._finish(job, "cancelled")


def test_audit_command_is_typed():
    slot, command = build_web_command("audit_dataset", {"manifest": "a.csv", "output": "audit.json"})
    assert slot == "pipeline"
    assert command == [sys.executable, "-m", "glove_chirality.cli", "audit-dataset",
                       "--manifest", "a.csv", "--output", "audit.json"]


def test_actual_stop_has_terminal_cancellation(tmp_path):
    service = CommandService(tmp_path)
    job_id = service.start("pipeline", [sys.executable, "-c", "import time; time.sleep(30)"])
    assert service.stop_job(job_id)
    assert wait_job(service, job_id)["status"] == "cancelled"
    assert not service.stop_job(job_id)


def test_extraction_zero_crops_fails_post_run_audit(tmp_path, monkeypatch):
    monkeypatch.setattr("glove_chirality.preflight.check_workflow", lambda *args: {
        "ok": True, "checks": [], "errors": [], "warnings": [], "summary": "ready"})
    code = ("from pathlib import Path; p=Path('dataset'); p.mkdir(); "
            "(p/'manifest.csv').write_text('image_path,label,source_video\\n')")
    monkeypatch.setattr("glove_chirality.web_service.build_web_command",
                        lambda *args: ("pipeline", [sys.executable, "-c", code]))
    service = CommandService(tmp_path)
    job = wait_job(service, service.start_workflow("extract_dataset", {"output": "dataset"}))
    assert job["exit_code"] == 0
    assert job["status"] == "failed"
    assert job["result"]["dataset_ready"] is False
    assert job["result"]["dataset_audit"]["counts"]["total"] == 0
    assert (tmp_path / "outputs/jobs" / job["id"] / "dataset_audit.json").is_file()


def test_progress_dictionary_telemetry_preserved_host_only(tmp_path):
    service = CommandService(tmp_path)
    with service._lock:
        job = service._reserve("pipeline", "train", "preflight", {})
        service._progress(job, 'GRIP_PROGRESS {"stage":"Training", "progress":'
                          '{"epoch":1,"epochs":3,"train_loss":0.3,"source":"private/path"}}')
        assert job.progress["train_loss"] == .3
        assert "private" not in json.dumps(service.snapshot(False))
        service._progress(job, 'GRIP_PROGRESS {"stage":"Training", "progress":'
                          '{"epoch":4,"epochs":3}}')
        assert job.progress["epoch"] == 1
        service._progress(job, 'GRIP_PROGRESS {"stage":"Training", "progress":0.4,'
                          '"result":{"score":NaN}}')
        assert "score" not in job.result
        service._finish(job, "cancelled")


def test_stale_checkpoint_not_advertised_as_new_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr("glove_chirality.preflight.check_workflow", lambda *args: {
        "ok": True, "checks": [], "errors": [], "warnings": [], "summary": "ready"})
    (tmp_path / "old.pt").write_bytes(b"old checkpoint")
    monkeypatch.setattr("glove_chirality.web_service.build_web_command",
                        lambda *args: ("pipeline", [sys.executable, "-c", "pass"]))
    service = CommandService(tmp_path)
    job = wait_job(service, service.start_workflow("train", {"output": "old.pt"}))
    assert job["status"] == "failed"
    assert job["result"]["checkpoint_valid"] is False
    assert not service.job_artifacts(job["id"])

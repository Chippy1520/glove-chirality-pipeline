import socket
import sys
import threading
import time

import pytest

from glove_chirality.web_service import CommandService, build_web_command


def test_training_web_command_preserves_typed_options():
    slot, command = build_web_command(
        "train",
        {
            "manifest": "manifest.csv",
            "output": "model.pt",
            "model": "mobilenet_v3_small",
            "epochs": "4",
            "batch_size": "8",
            "image_size": "192",
            "learning_rate": "0.0005",
            "validation_fraction": "0.25",
            "seed": "7",
            "device": "cuda:0",
            "workers": "2",
            "amp": True,
            "loss": "recall_hybrid",
            "recall_target": "right",
            "recall_weight": "1.5",
            "selection_metric": "recall_right",
            "augmentation": "anti_spurious",
            "tensorboard_logdir": "tb",
        },
    )

    assert slot == "pipeline"
    assert command[command.index("--model") + 1] == "mobilenet_v3_small"
    assert command[command.index("--selection-metric") + 1] == "recall_right"
    assert "--amp" in command
    assert command[command.index("--tensorboard-logdir") + 1] == "tb"


def test_tensorboard_uses_independent_process_slot():
    slot, command = build_web_command("tensorboard", {"logdir": "tb", "port": 6010})

    assert slot == "tensorboard"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "6010"


def test_tensorboard_start_rejects_missing_log_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "glove_chirality.web_service.importlib.util.find_spec",
        lambda _name: object(),
    )
    service = CommandService(tmp_path)
    slot, command = build_web_command(
        "tensorboard",
        {"logdir": "missing", "port": 6010},
    )

    with pytest.raises(ValueError, match="log directory not found"):
        service.start(slot, command, action="tensorboard")


def test_tensorboard_start_rejects_occupied_port(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "glove_chirality.web_service.importlib.util.find_spec",
        lambda _name: object(),
    )
    (tmp_path / "tb").mkdir()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    service = CommandService(tmp_path)
    slot, command = build_web_command(
        "tensorboard",
        {"logdir": "tb", "port": port},
    )
    try:
        with pytest.raises(ValueError, match="already in use"):
            service.start(slot, command, action="tensorboard")
    finally:
        listener.close()


def test_tensorboard_start_rejects_missing_package(tmp_path, monkeypatch):
    (tmp_path / "tb").mkdir()
    monkeypatch.setattr(
        "glove_chirality.web_service.importlib.util.find_spec",
        lambda _name: None,
    )
    service = CommandService(tmp_path)
    slot, command = build_web_command(
        "tensorboard",
        {"logdir": "tb", "port": 6010},
    )

    with pytest.raises(ValueError, match="not installed"):
        service.start(slot, command, action="tensorboard")


def test_unknown_action_cannot_become_an_arbitrary_command():
    with pytest.raises(ValueError, match="Unknown web action"):
        build_web_command("command", {"value": "anything"})


def test_command_service_collects_output_and_releases_slot(tmp_path):
    service = CommandService(tmp_path)
    job_id = service.start(
        "pipeline",
        [
            sys.executable,
            "-c",
            "import os; print(os.environ['PYTHONUNBUFFERED']); print('completed')",
        ],
        action="preview",
    )

    deadline = time.monotonic() + 5
    while service.snapshot(include_logs=True)["running"]["pipeline"]:
        if time.monotonic() >= deadline:
            pytest.fail("test process did not finish")
        time.sleep(0.02)

    logs = service.snapshot(include_logs=True)["logs"]
    job = service.snapshot(include_logs=False)["jobs"]["pipeline"]
    assert len(job_id) == 32
    assert any(entry["text"] == "1" for entry in logs)
    assert any(entry["text"] == "completed" for entry in logs)
    assert logs[-1]["text"] == "Process finished with exit code 0."
    assert job["job_id"] == job_id
    assert job["action"] == "preview"
    assert job["status"] == "succeeded"
    assert job["exit_code"] == 0


def test_command_service_atomically_rejects_concurrent_slot_start(tmp_path):
    service = CommandService(tmp_path)
    barrier = threading.Barrier(3)
    outcomes = []

    def start() -> None:
        barrier.wait()
        try:
            service.start(
                "pipeline",
                [sys.executable, "-c", "import time; time.sleep(2)"],
            )
        except RuntimeError:
            outcomes.append("rejected")
        else:
            outcomes.append("started")

    threads = [threading.Thread(target=start) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)

    assert sorted(outcomes) == ["rejected", "started"]
    assert service.stop("pipeline") is True


def test_command_service_bounds_retained_logs(tmp_path):
    service = CommandService(tmp_path, max_log_entries=3)

    for index in range(5):
        service._append("pipeline", str(index))

    logs = service.snapshot(include_logs=True)["logs"]
    assert [entry["text"] for entry in logs] == ["2", "3", "4"]

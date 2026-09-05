import sys
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


def test_unknown_action_cannot_become_an_arbitrary_command():
    with pytest.raises(ValueError, match="Unknown web action"):
        build_web_command("command", {"value": "anything"})


def test_command_service_collects_output_and_releases_slot(tmp_path):
    service = CommandService(tmp_path)
    service.start("pipeline", [sys.executable, "-c", "print('completed')"])

    deadline = time.monotonic() + 5
    while service.snapshot(include_logs=True)["running"]["pipeline"]:
        if time.monotonic() >= deadline:
            pytest.fail("test process did not finish")
        time.sleep(0.02)

    logs = service.snapshot(include_logs=True)["logs"]
    assert any(entry["text"] == "completed" for entry in logs)
    assert logs[-1]["text"] == "Process finished with exit code 0."

import pytest

from glove_chirality import gui_commands


def test_extract_dataset_command_preserves_paths():
    command = gui_commands.extract_dataset(
        "D:/data/left gloves",
        "D:/data/right gloves",
        "D:/output dataset",
        "configs/custom.yaml",
    )
    assert command[3:] == [
        "extract-dataset", "--left", "D:/data/left gloves",
        "--right", "D:/data/right gloves", "--output", "D:/output dataset",
        "--config", "configs/custom.yaml",
    ]


def test_train_command_includes_gpu_controls():
    command = gui_commands.train(
        "manifest.csv", "model.pt", "resnet18", 20, 64, 224,
        0.001, 0.2, 42, "cuda:1", 4, True,
    )
    assert command[-1] == "--amp"
    assert command[command.index("--device") + 1] == "cuda:1"
    assert command[command.index("--workers") + 1] == "4"


def test_preview_command_includes_detector_warmup():
    command = gui_commands.preview("video.mkv", "preview.jpg", 30.0, "config.yaml", 3.0)
    assert command[command.index("--warmup-seconds") + 1] == "3.0"


def test_required_gui_fields_fail_early():
    with pytest.raises(ValueError, match="Required"):
        gui_commands.infer_video("", "model.pt", "output", "config.yaml", "auto")

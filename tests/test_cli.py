from glove_chirality.cli import build_parser


def test_training_gpu_options_parse():
    args = build_parser().parse_args([
        "train",
        "--manifest", "manifest.csv",
        "--output", "model.pt",
        "--device", "cuda:1",
        "--amp",
        "--workers", "4",
    ])
    assert args.device == "cuda:1"
    assert args.amp is True
    assert args.workers == 4


def test_inference_device_option_parse():
    args = build_parser().parse_args([
        "infer-video",
        "--video", "input.mkv",
        "--checkpoint", "model.pt",
        "--output", "predictions",
        "--device", "cuda",
    ])
    assert args.device == "cuda"

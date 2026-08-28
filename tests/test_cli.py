from glove_chirality.cli import build_parser


def test_training_gpu_and_recall_options_parse():
    args = build_parser().parse_args([
        "train",
        "--manifest", "manifest.csv",
        "--output", "model.pt",
        "--device", "cuda:1",
        "--amp",
        "--workers", "4",
        "--loss", "recall_hybrid",
        "--recall-target", "right",
        "--recall-weight", "2.5",
        "--selection-metric", "recall_right",
    ])
    assert args.device == "cuda:1"
    assert args.amp is True
    assert args.workers == 4
    assert args.loss == "recall_hybrid"
    assert args.recall_target == "right"
    assert args.recall_weight == 2.5
    assert args.selection_metric == "recall_right"


def test_inference_recall_threshold_options_parse():
    args = build_parser().parse_args([
        "infer-video",
        "--video", "input.mkv",
        "--checkpoint", "model.pt",
        "--output", "predictions",
        "--device", "cuda",
        "--decision-class", "right",
        "--decision-threshold", "0.25",
    ])
    assert args.device == "cuda"
    assert args.decision_class == "right"
    assert args.decision_threshold == 0.25


def test_live_inference_options_parse():
    args = build_parser().parse_args([
        "infer-live",
        "--source", "0",
        "--checkpoint", "model.pt",
        "--config", "configs/production.yaml",
        "--device", "cuda",
        "--amp",
        "--output", "events.jsonl",
        "--decision-class", "right",
        "--decision-threshold", "0.20",
    ])
    assert args.source == "0"
    assert args.device == "cuda"
    assert args.amp is True
    assert args.output == "events.jsonl"
    assert args.decision_class == "right"
    assert args.decision_threshold == 0.20

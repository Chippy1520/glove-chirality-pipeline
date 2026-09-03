from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

from glove_chirality.config import ExtractionConfig
from glove_chirality.extraction import (
    config_hash,
    discover_videos,
    event_report_rows,
    event_rows,
    extract_video_with_report,
    write_event_report,
    write_manifest,
)
from glove_chirality.models import CLASSIFIER_CHOICES


def _extract_sources(sources, output: Path, config: ExtractionConfig):
    all_events = []
    all_records = []
    for path, label in sources:
        videos = discover_videos(path)
        if not videos:
            raise ValueError(f"No supported videos found under {path}")
        for index, video in enumerate(videos, 1):
            print(f"[{label} {index}/{len(videos)}] {video}")
            run = extract_video_with_report(video, output, label, config)
            print(
                f"  accepted passages: {len(run.events)}; "
                f"audited outcomes: {len(run.records)}"
            )
            all_events.extend(run.events)
            all_records.extend(run.records)
    rows = event_rows(all_events, output, config_hash(config))
    manifest = write_manifest(rows, output / "manifest.csv")
    report = write_event_report(event_report_rows(all_records), output / "event_report.csv")
    print(f"Wrote {len(rows)} accepted rows to {manifest}")
    print(f"Wrote {len(all_records)} accepted/rejected outcomes to {report}")
    return all_events


def _add_decision_arguments(parser) -> None:
    parser.add_argument(
        "--decision-class",
        choices=["argmax", "left", "right"],
        default="argmax",
        help="Target class for thresholding; use right for right-glove recall",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Predict the decision class at or above this probability",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="glove-pipeline",
        description="Modular glove chirality pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="Extract event crops from one video or directory")
    extract.add_argument("--input", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument(
        "--label",
        choices=["left", "right", "unknown"],
        default="unknown",
    )
    extract.add_argument("--config", default="configs/default.yaml")

    dataset = sub.add_parser(
        "extract-dataset",
        help="Build a labeled crop dataset from left/right streams",
    )
    dataset.add_argument("--left", required=True, help="Left-only video or directory")
    dataset.add_argument("--right", required=True, help="Right-only video or directory")
    dataset.add_argument("--output", required=True)
    dataset.add_argument("--config", default="configs/default.yaml")

    preview = sub.add_parser("preview", help="Render detector/ROI overlay on one video frame")
    preview.add_argument("--video", required=True)
    preview.add_argument("--output", required=True)
    preview.add_argument("--seconds", type=float, default=0.0)
    preview.add_argument("--warmup-seconds", type=float, default=2.0)
    preview.add_argument("--config", default="configs/default.yaml")

    train = sub.add_parser("train", help="Train a swappable chirality classifier")
    train.add_argument("--manifest", required=True)
    train.add_argument("--output", required=True)
    train.add_argument(
        "--model",
        choices=CLASSIFIER_CHOICES,
        default="tiny_cnn",
    )
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--image-size", type=int, default=224)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--validation-fraction", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    train.add_argument("--amp", action="store_true", help="Use CUDA mixed precision")
    train.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")
    train.add_argument(
        "--loss",
        choices=["cross_entropy", "weighted_cross_entropy", "recall_hybrid"],
        default="weighted_cross_entropy",
    )
    train.add_argument("--recall-target", choices=["left", "right"], default="right")
    train.add_argument(
        "--recall-weight",
        type=float,
        default=1.0,
        help="Soft-recall penalty strength for recall_hybrid loss",
    )
    train.add_argument(
        "--selection-metric",
        choices=["accuracy", "macro_recall", "macro_f1", "recall_left", "recall_right"],
        default="macro_recall",
        help="Validation metric used to retain the best checkpoint",
    )

    images = sub.add_parser("infer-images", help="Classify a crop or crop directory")
    images.add_argument("--input", required=True)
    images.add_argument("--checkpoint", required=True)
    images.add_argument("--output", required=True)
    images.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    _add_decision_arguments(images)

    video = sub.add_parser("infer-video", help="Extract passages and classify each one")
    video.add_argument("--video", required=True)
    video.add_argument("--checkpoint", required=True)
    video.add_argument("--output", required=True)
    video.add_argument("--config", default="configs/default.yaml")
    video.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    video.add_argument("--amp", action="store_true", help="Use classifier CUDA AMP")
    _add_decision_arguments(video)

    live = sub.add_parser(
        "infer-live",
        help="Event-driven inference from a camera, video, or OpenCV stream",
    )
    live.add_argument("--source", default="0", help="Camera index or OpenCV-compatible source")
    live.add_argument("--checkpoint", required=True)
    live.add_argument("--config", default="configs/production.yaml")
    live.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    live.add_argument("--amp", action="store_true", help="Use classifier CUDA AMP")
    _add_decision_arguments(live)
    live.add_argument(
        "--output",
        default="-",
        help="JSONL event path, or - for stdout",
    )
    return parser


def _infer_video(args) -> None:
    from glove_chirality.inference import TorchClassifier

    output = Path(args.output)
    config = ExtractionConfig.from_yaml(args.config)
    run = extract_video_with_report(args.video, output, "unknown", config)
    write_manifest(event_rows(run.events, output, config_hash(config)), output / "manifest.csv")
    write_event_report(event_report_rows(run.records), output / "event_report.csv")
    classifier = TorchClassifier(
        args.checkpoint,
        device=args.device,
        amp=args.amp,
        decision_class=args.decision_class,
        decision_threshold=args.decision_threshold,
    )
    prediction_path = output / "predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "event_id",
            "source_video",
            "frame_index",
            "timestamp_s",
            "image_path",
            "prediction",
            "confidence",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for event in run.events:
            prediction, confidence = classifier.predict(event.image_path)
            writer.writerow(
                {
                    "event_id": event.event_id,
                    "source_video": event.source_video,
                    "frame_index": event.frame_index,
                    "timestamp_s": f"{event.timestamp_s:.6f}",
                    "image_path": event.image_path.relative_to(output).as_posix(),
                    "prediction": prediction,
                    "confidence": f"{confidence:.6f}",
                }
            )
    print(f"Wrote {len(run.events)} per-passage predictions to {prediction_path}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "extract":
        _extract_sources(
            [(args.input, args.label)],
            Path(args.output),
            ExtractionConfig.from_yaml(args.config),
        )
    elif args.command == "extract-dataset":
        _extract_sources(
            [(args.left, "left"), (args.right, "right")],
            Path(args.output),
            ExtractionConfig.from_yaml(args.config),
        )
    elif args.command == "preview":
        from glove_chirality.diagnostics import save_calibration_preview

        print(
            save_calibration_preview(
                args.video,
                args.output,
                ExtractionConfig.from_yaml(args.config),
                args.seconds,
                args.warmup_seconds,
            )
        )
    elif args.command == "train":
        from glove_chirality.training import train_classifier

        metrics = train_classifier(
            manifest=args.manifest,
            output=args.output,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            learning_rate=args.learning_rate,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            device_name=args.device,
            amp=args.amp,
            workers=args.workers,
            loss_name=args.loss,
            recall_target=args.recall_target,
            recall_weight=args.recall_weight,
            selection_metric=args.selection_metric,
        )
        print(json.dumps(metrics, indent=2))
    elif args.command == "infer-images":
        from glove_chirality.inference import infer_images

        rows = infer_images(
            args.input,
            args.checkpoint,
            args.output,
            device=args.device,
            decision_class=args.decision_class,
            decision_threshold=args.decision_threshold,
        )
        print(f"Wrote {len(rows)} predictions to {args.output}")
    elif args.command == "infer-video":
        _infer_video(args)
    elif args.command == "infer-live":
        from glove_chirality.live import run_live_inference

        metrics = run_live_inference(
            args.source,
            args.checkpoint,
            ExtractionConfig.from_yaml(args.config),
            device=args.device,
            amp=args.amp,
            output=args.output,
            decision_class=args.decision_class,
            decision_threshold=args.decision_threshold,
        )
        print(json.dumps(asdict(metrics)), file=sys.stderr)


if __name__ == "__main__":
    main()

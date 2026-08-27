from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from glove_chirality.config import ExtractionConfig
from glove_chirality.extraction import (
    config_hash,
    discover_videos,
    event_rows,
    extract_video,
    write_manifest,
)


def _extract_sources(sources, output: Path, config: ExtractionConfig):
    all_events = []
    for path, label in sources:
        videos = discover_videos(path)
        if not videos:
            raise ValueError(f"No supported videos found under {path}")
        for index, video in enumerate(videos, 1):
            print(f"[{label} {index}/{len(videos)}] {video}")
            events = extract_video(video, output, label, config)
            print(f"  accepted passages: {len(events)}")
            all_events.extend(events)
    rows = event_rows(all_events, output, config_hash(config))
    manifest = write_manifest(rows, output / "manifest.csv")
    print(f"Wrote {len(rows)} rows to {manifest}")
    return all_events


def build_parser():
    parser = argparse.ArgumentParser(prog="glove-pipeline", description="Modular glove chirality pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="Extract event crops from one video or directory")
    extract.add_argument("--input", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--label", choices=["left", "right", "unknown"], default="unknown")
    extract.add_argument("--config", default="configs/default.yaml")

    dataset = sub.add_parser("extract-dataset", help="Build a labeled crop dataset from left/right streams")
    dataset.add_argument("--left", required=True, help="Left-only video or directory")
    dataset.add_argument("--right", required=True, help="Right-only video or directory")
    dataset.add_argument("--output", required=True)
    dataset.add_argument("--config", default="configs/default.yaml")

    preview = sub.add_parser("preview", help="Render detector/ROI overlay on one video frame")
    preview.add_argument("--video", required=True)
    preview.add_argument("--output", required=True)
    preview.add_argument("--seconds", type=float, default=0.0)
    preview.add_argument("--config", default="configs/default.yaml")

    train = sub.add_parser("train", help="Train a swappable chirality classifier")
    train.add_argument("--manifest", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--model", choices=["tiny_cnn", "resnet18", "mobilenet_v3_small", "vit_b_16"], default="tiny_cnn")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--image-size", type=int, default=224)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--validation-fraction", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    train.add_argument("--amp", action="store_true", help="Use CUDA mixed precision")
    train.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")

    images = sub.add_parser("infer-images", help="Classify a crop or crop directory")
    images.add_argument("--input", required=True)
    images.add_argument("--checkpoint", required=True)
    images.add_argument("--output", required=True)
    images.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")

    video = sub.add_parser("infer-video", help="Extract passages and classify each one")
    video.add_argument("--video", required=True)
    video.add_argument("--checkpoint", required=True)
    video.add_argument("--output", required=True)
    video.add_argument("--config", default="configs/default.yaml")
    video.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "extract":
        _extract_sources([(args.input, args.label)], Path(args.output), ExtractionConfig.from_yaml(args.config))
    elif args.command == "extract-dataset":
        _extract_sources([(args.left, "left"), (args.right, "right")], Path(args.output), ExtractionConfig.from_yaml(args.config))
    elif args.command == "preview":
        from glove_chirality.diagnostics import save_calibration_preview
        print(save_calibration_preview(args.video, args.output, ExtractionConfig.from_yaml(args.config), args.seconds))
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
        )
        print(json.dumps(metrics, indent=2))
    elif args.command == "infer-images":
        from glove_chirality.inference import infer_images
        rows = infer_images(args.input, args.checkpoint, args.output, device=args.device)
        print(f"Wrote {len(rows)} predictions to {args.output}")
    elif args.command == "infer-video":
        from glove_chirality.inference import TorchClassifier
        output = Path(args.output)
        config = ExtractionConfig.from_yaml(args.config)
        events = extract_video(args.video, output, "unknown", config)
        classifier = TorchClassifier(args.checkpoint, device=args.device)
        prediction_path = output / "predictions.csv"
        with prediction_path.open("w", newline="", encoding="utf-8") as stream:
            fields = ["event_id", "source_video", "frame_index", "timestamp_s", "image_path", "prediction", "confidence"]
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for event in events:
                prediction, confidence = classifier.predict(event.image_path)
                writer.writerow({"event_id": event.event_id, "source_video": event.source_video, "frame_index": event.frame_index, "timestamp_s": f"{event.timestamp_s:.6f}", "image_path": event.image_path.relative_to(output).as_posix(), "prediction": prediction, "confidence": f"{confidence:.6f}"})
        print(f"Wrote {len(events)} per-passage predictions to {prediction_path}")


if __name__ == "__main__":
    main()

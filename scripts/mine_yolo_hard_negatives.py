from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from glove_chirality.config import ExtractionConfig
from glove_chirality.detection.yolo import YoloDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mine exact raw video frames containing YOLO predictions outside a "
            "chosen physical bbox-area range. The pipeline size filter is bypassed "
            "for mining only; raw and annotated preview frames are saved separately."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input video")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--config",
        default=Path("configs/grip_aug27_seed.yaml"),
        type=Path,
        help="Extraction config using the seed YOLO checkpoint",
    )
    parser.add_argument("--min-ratio", type=float, default=0.03)
    parser.add_argument("--max-ratio", type=float, default=0.40)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Run YOLO every Nth decoded frame",
    )
    parser.add_argument(
        "--min-gap-seconds",
        type=float,
        default=1.0,
        help="Minimum time between saved candidate frames",
    )
    parser.add_argument(
        "--max-saves",
        type=int,
        default=0,
        help="Maximum candidate frames to save; 0 means unlimited",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if not 0.0 <= args.min_ratio < args.max_ratio <= 1.0:
        raise ValueError("Require 0 <= min-ratio < max-ratio <= 1")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.min_gap_seconds < 0:
        raise ValueError("--min-gap-seconds must be non-negative")
    if args.max_saves < 0:
        raise ValueError("--max-saves must be >= 0")

    raw_dir = args.output / "raw"
    preview_dir = args.output / "preview"
    raw_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "hard_negative_candidates.csv"

    config = ExtractionConfig.from_yaml(args.config)
    if config.detector.backend != "yolo":
        raise ValueError(f"Expected YOLO backend, got {config.detector.backend!r}")

    detector = YoloDetector(config.detector)

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not fps or fps <= 0:
        fps = 25.0

    frame_count_reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    min_gap_frames = max(0, int(round(args.min_gap_seconds * fps)))

    print(f"Input: {args.input}")
    print(f"FPS: {fps:.3f}")
    print(f"Reported frames: {frame_count_reported}")
    print(f"Mining ratio outside [{args.min_ratio:.4f}, {args.max_ratio:.4f}]")
    print(f"Stride: {args.stride}")
    print(f"Minimum save gap: {min_gap_frames} frames ({args.min_gap_seconds:.2f}s)")
    print("Physical size filtering is bypassed for mining only.")

    fields = [
        "source_video",
        "frame_index",
        "time_seconds",
        "raw_path",
        "preview_path",
        "candidate_index",
        "x1",
        "y1",
        "x2",
        "y2",
        "confidence",
        "box_area_ratio",
        "reason",
    ]

    decoded = 0
    inferred = 0
    saved_frames = 0
    saved_predictions = 0
    last_saved_frame = -10**12

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_index = decoded
            decoded += 1

            if frame_index % args.stride != 0:
                continue

            inferred += 1
            detections, diagnostics = detector.detect_with_diagnostics(
                frame,
                apply_size_filter=False,
            )

            height, width = frame.shape[:2]
            frame_area = max(1, width * height)

            suspicious = []
            for det_index, detection in enumerate(detections):
                x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2
                area_ratio = ((x2 - x1) * (y2 - y1)) / frame_area
                if area_ratio < args.min_ratio:
                    reason = "below_min_ratio"
                elif area_ratio > args.max_ratio:
                    reason = "above_max_ratio"
                else:
                    continue
                suspicious.append((det_index, detection, area_ratio, reason))

            if not suspicious:
                continue

            if frame_index - last_saved_frame < min_gap_frames:
                continue

            if args.max_saves and saved_frames >= args.max_saves:
                break

            time_seconds = frame_index / fps
            stem = (
                f"{args.input.stem}"
                f"_f{frame_index:07d}"
                f"_t{time_seconds:010.2f}"
            )
            raw_path = raw_dir / f"{stem}.jpg"
            preview_path = preview_dir / f"{stem}.jpg"

            if not cv2.imwrite(str(raw_path), frame):
                raise RuntimeError(f"Failed to save {raw_path}")

            preview = frame.copy()
            for det_index, detection, area_ratio, reason in suspicious:
                x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 0, 255), 3)
                label = (
                    f"{reason} ratio={area_ratio:.4f} "
                    f"conf={detection.confidence:.3f}"
                )
                text_y = max(25, y1 - 8)
                cv2.putText(
                    preview,
                    label,
                    (max(0, x1), text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                writer.writerow(
                    {
                        "source_video": args.input.name,
                        "frame_index": frame_index,
                        "time_seconds": f"{time_seconds:.3f}",
                        "raw_path": str(raw_path),
                        "preview_path": str(preview_path),
                        "candidate_index": det_index,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": f"{detection.confidence:.6f}",
                        "box_area_ratio": f"{area_ratio:.8f}",
                        "reason": reason,
                    }
                )
                saved_predictions += 1

            cv2.putText(
                preview,
                (
                    f"frame={frame_index} t={time_seconds:.2f}s "
                    f"raw_yolo={diagnostics.raw_yolo_count} "
                    f"suspicious={len(suspicious)}"
                ),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if not cv2.imwrite(str(preview_path), preview):
                raise RuntimeError(f"Failed to save {preview_path}")

            saved_frames += 1
            last_saved_frame = frame_index
            print(
                f"saved #{saved_frames}: frame={frame_index} "
                f"t={time_seconds:.2f}s suspicious={len(suspicious)}"
            )

    capture.release()

    print()
    print("DONE")
    print(f"Decoded frames: {decoded}")
    print(f"Frames sent to YOLO: {inferred}")
    print(f"Candidate frames saved: {saved_frames}")
    print(f"Suspicious predictions saved: {saved_predictions}")
    print(f"Raw frames: {raw_dir}")
    print(f"Preview frames: {preview_dir}")
    print(f"CSV: {csv_path}")
    print()
    print(
        "Review PREVIEW images first. Copy/use only the corresponding RAW images "
        "for annotation/training. A nuisance-only raw frame gets zero glove labels; "
        "if a real glove is visible, annotate every real glove."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


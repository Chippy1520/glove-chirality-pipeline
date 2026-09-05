from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

from glove_chirality.config import ExtractionConfig
from glove_chirality.detection import build_detector
from glove_chirality.extraction import (
    config_hash,
    event_report_rows,
    event_rows,
    extract_video_with_report,
    write_event_report,
    write_manifest,
)

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".mov", ".m4v"}


def rotate_image(image, degrees: int):
    if degrees == 0:
        return image

    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)

    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    raise ValueError("Rotation must be 0, 90, 180, or 270.")


def discover_selected_videos(source: Path, extensions: set[str]):
    if source.is_file():
        if source.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(
                f"Unsupported video extension: {source.suffix}"
            )
        return [source]

    if not source.is_dir():
        raise FileNotFoundError(source)

    return sorted(
        p
        for p in source.rglob("*")
        if p.is_file()
        and p.suffix.lower() in extensions
    )


def ensure_fresh_output(output: Path):
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty:\n{output}\n\n"
            "Use a new output directory so old and new crops "
            "cannot be mixed."
        )

    output.mkdir(parents=True, exist_ok=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Extract RIGHT glove passages using the normal GRIP "
            "event pipeline, then rotate accepted classifier crops "
            "into the canonical left-lane orientation."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Single video or directory containing videos.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Fresh output dataset directory.",
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Extraction YAML calibrated for the source right lane.",
    )

    parser.add_argument(
        "--rotation",
        type=int,
        choices=[0, 90, 180, 270],
        default=180,
        help="Clockwise rotation applied to accepted crops.",
    )

    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".mkv"],
        help=(
            "Extensions to process when input is a directory. "
            "Default: .mkv"
        ),
    )

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    source = Path(args.input)
    output = Path(args.output)
    config_path = Path(args.config)

    extensions = {
        ext.lower()
        if ext.startswith(".")
        else "." + ext.lower()
        for ext in args.extensions
    }

    ensure_fresh_output(output)

    config = ExtractionConfig.from_yaml(config_path)

    videos = discover_selected_videos(
        source,
        extensions,
    )

    if not videos:
        raise RuntimeError(
            f"No matching videos found under:\n{source}"
        )

    print("Videos selected:", len(videos))

    for video in videos:
        print(" ", video)

    detector = build_detector(config.detector)

    all_events = []
    all_records = []
    rotation_rows = []

    for index, video in enumerate(videos, start=1):

        print()
        print(
            f"[right {index}/{len(videos)}] "
            f"{video}"
        )

        run = extract_video_with_report(
            video,
            output,
            "right",
            config,
            detector=detector,
        )

        print(
            f"  accepted passages: {len(run.events)}; "
            f"audited outcomes: {len(run.records)}"
        )

        for event in run.events:

            image = cv2.imread(
                str(event.image_path)
            )

            if image is None:
                raise RuntimeError(
                    "Could not read extracted crop:\n"
                    f"{event.image_path}"
                )

            rotated = rotate_image(
                image,
                args.rotation,
            )

            if not cv2.imwrite(
                str(event.image_path),
                rotated,
            ):
                raise RuntimeError(
                    "Could not write rotated crop:\n"
                    f"{event.image_path}"
                )

            rotation_rows.append(
                {
                    "event_id": event.event_id,
                    "image_path":
                        event.image_path
                        .relative_to(output)
                        .as_posix(),
                    "source_video":
                        event.source_video,
                    "timestamp_s":
                        f"{event.timestamp_s:.6f}",
                    "rotation_degrees_clockwise":
                        args.rotation,
                    "canonical_lane": "left",
                }
            )

        all_events.extend(run.events)
        all_records.extend(run.records)

    cfg_hash = config_hash(config)

    manifest_path = write_manifest(
        event_rows(
            all_events,
            output,
            cfg_hash,
        ),
        output / "manifest.csv",
    )

    report_path = write_event_report(
        event_report_rows(all_records),
        output / "event_report.csv",
    )

    rotation_log = (
        output / "rotation_log.csv"
    )

    with rotation_log.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "event_id",
                "image_path",
                "source_video",
                "timestamp_s",
                "rotation_degrees_clockwise",
                "canonical_lane",
            ],
        )

        writer.writeheader()
        writer.writerows(rotation_rows)

    metadata = {
        "purpose":
            "right-lane to left-lane canonical crop normalization",

        "input":
            str(source),

        "output":
            str(output),

        "config":
            str(config_path),

        "config_hash":
            cfg_hash,

        "label":
            "right",

        "rotation_degrees_clockwise":
            args.rotation,

        "canonical_lane":
            "left",

        "extensions":
            sorted(extensions),

        "videos":
            [str(video) for video in videos],

        "accepted_events":
            len(all_events),

        "audited_outcomes":
            len(all_records),

        "note": (
            "Only accepted classifier crops are rotated. "
            "Detection, ROI, trigger-zone checks and event tracking "
            "operate on the original source-video orientation. "
            "Therefore the YAML must be calibrated for the "
            "right-lane recording."
        ),
    }

    metadata_path = (
        output / "run_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print()
    print(
        "Accepted events:",
        len(all_events),
    )

    print(
        "Manifest:",
        manifest_path,
    )

    print(
        "Event report:",
        report_path,
    )

    print(
        "Rotation audit:",
        rotation_log,
    )

    print(
        "Metadata:",
        metadata_path,
    )


if __name__ == "__main__":
    main()

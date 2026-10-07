"""Measure synchronized detection, passage processing, and canonical crop latency.

Requires the ml/yolo extras. Downloads, decoding, crop writes, classification,
and overlay encoding are outside the core Layer 1 timer. Actual source PTS
are used for passage processing; reported container FPS is not trusted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict, fields
from pathlib import Path

import cv2
import numpy as np

from glove_chirality.config import ExtractionConfig
from glove_chirality.detection.yolo import YoloDetector
from glove_chirality.events import PassageProcessor


def distribution_ms(values: list[float]) -> dict[str, float | int] | None:
    """Summarize wall-clock milliseconds without silently inventing empty samples."""
    if not values:
        return None
    samples = np.asarray(values, dtype=np.float64)
    return {
        "samples": len(values),
        "mean_ms": float(samples.mean()),
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
        "p99_ms": float(np.percentile(samples, 99)),
        "max_ms": float(samples.max()),
    }


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def benchmark(
    video: Path,
    model: Path,
    config_path: Path,
    output: Path,
    *,
    device: str = "cuda",
    half: bool = False,
    warmup: int = 16,
    save_detections: bool = False,
) -> dict:
    import torch

    import glove_chirality.events as events_module
    import glove_chirality.line_counter as counter_module

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no GPU timings produced")
    if warmup < 1:
        raise ValueError("At least one untimed warmup inference is required")
    output.mkdir(parents=True, exist_ok=True)
    config = ExtractionConfig.from_yaml(config_path)
    config.detector.yolo_model = str(model)
    config.detector.yolo_device = device
    config.detector.yolo_half = half
    config.event.tracker_mode = "line"
    detector = YoloDetector(config.detector)
    processor = PassageProcessor(None, config, str(video), "unlabeled")

    def synchronize() -> None:
        if target.type == "cuda":
            torch.cuda.synchronize(target)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    reported_fps = capture.get(cv2.CAP_PROP_FPS)
    ok, first_frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("Source contains no decodable frames")
    with torch.inference_mode():
        for _ in range(warmup):
            detector.detect_with_diagnostics(first_frame)
        synchronize()
        capture = cv2.VideoCapture(str(video))
        rows = []
        outcomes = []
        accepted = 0
        first_pts = None
        last_pts = None
        loop_start = time.perf_counter()
        while True:
            read_start = time.perf_counter_ns()
            ok, frame = capture.read()
            decode_ms = (time.perf_counter_ns() - read_start) / 1e6
            if not ok:
                break
            pts = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if first_pts is None:
                first_pts = pts
            timestamp = pts - first_pts
            if last_pts is not None and timestamp <= last_pts:
                raise RuntimeError("Source PTS are not strictly increasing")
            synchronize()
            start = time.perf_counter_ns()
            detections, diagnostics = detector.detect_with_diagnostics(frame)
            synchronize()
            detected = time.perf_counter_ns()
            result = processor.process(frame, len(rows), timestamp, detections=detections)
            synchronize()
            finished = time.perf_counter_ns()
            emitted = 0
            for event in result.outcomes:
                record = {
                    field.name: getattr(event, field.name)
                    for field in fields(event)
                    if field.name != "crop"
                }
                if event.detection is not None:
                    record["detection"] = asdict(event.detection)
                outcomes.append(record)
                if event.accepted:
                    accepted += 1
                    emitted += 1
                    if not cv2.imwrite(str(output / f"glove_{accepted:03d}.png"), event.crop):
                        raise RuntimeError("Failed to write accepted crop")
            rows.append({
                "frame": len(rows),
                "timestamp_s": timestamp,
                "decode_ms": decode_ms,
                "detector_ms": (detected - start) / 1e6,
                "counter_and_crop_ms": (finished - detected) / 1e6,
                "layer1_ms": (finished - start) / 1e6,
                "detector_masks": len(detections),
                "raw_yolo_count": diagnostics.raw_yolo_count,
                "emitted_crops": emitted,
                "running_count": accepted,
            })
            if save_detections:
                rows[-1]["detections"] = [asdict(detection) for detection in detections]
            last_pts = timestamp
            if len(rows) % 200 == 0:
                print(device, "half", half, "frames", len(rows), "count", accepted, flush=True)
        elapsed = time.perf_counter() - loop_start
        capture.release()
        closing = processor.close(last_pts or 0.0)
        if any(event.accepted for event in closing):
            raise RuntimeError("Untimed accepted crop emitted at EOF")
        for event in closing:
            record = {field.name: getattr(event, field.name) for field in fields(event) if field.name != "crop"}
            if event.detection is not None:
                record["detection"] = asdict(event.detection)
            outcomes.append(record)
        (output / "counter_debug.json").write_text(json.dumps(processor._line_crossings().debug_events, indent=2), encoding="utf-8")
    if len(rows) < 2:
        raise RuntimeError("Need multiple decoded frames to measure source cadence")
    stages = {}
    for name, subset in {
        "all_frames": rows,
        "mask_present_frames": [row for row in rows if row["detector_masks"]],
        "no_mask_frames": [row for row in rows if not row["detector_masks"]],
        "crop_emission_frames": [row for row in rows if row["emitted_crops"]],
    }.items():
        stages[name] = {
            stage: distribution_ms([row[stage] for row in subset])
            for stage in ("detector_ms", "counter_and_crop_ms", "layer1_ms", "decode_ms")
        }
    period_ms = float(np.mean(np.diff([row["timestamp_s"] for row in rows])) * 1000)
    layer1 = stages["all_frames"]["layer1_ms"]
    summary = {
        "hardware": {
            "gpu": torch.cuda.get_device_name(target) if target.type == "cuda" else None,
            "device": device,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cpu_threads": torch.get_num_threads(),
        },
        "half_precision": half,
        "detections_saved": save_detections,
        "warmup_inferences_excluded": warmup,
        "frames_decoded": len(rows),
        "frames_reported": reported_frames,
        "container_frame_count_matches": len(rows) == reported_frames,
        "fps_reported": reported_fps,
        "last_source_timestamp_s": last_pts,
        "source_cadence_fps": 1000 / period_ms,
        "source_frame_budget_ms": period_ms,
        "accepted_crops": accepted,
        "crossing_times_s": [e["trigger_crossing_s"] for e in outcomes if e["status"] == "accepted"],
        "core_fps_inverse_mean": 1000 / layer1["mean_ms"],
        "core_frames_over_source_budget": sum(row["layer1_ms"] > period_ms for row in rows),
        "loop_seconds_including_decode_crop_writes": elapsed,
        "stages": stages,
        "measurement": "Synchronized sequential wall time: YOLO adapter plus passage processor, including canonical crop creation. No classifier, downloads, decoding, disk writes or overlay inside core timer. Emission-frame statistics have only as many samples as passages.",
        "video_sha256": digest(video),
        "weights_sha256": digest(model),
        "original_config_sha256": digest(config_path),
        "counter_source_sha256": digest(Path(counter_module.__file__)),
        "events_source_sha256": digest(Path(events_module.__file__)),
        "config": asdict(config),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "outcomes.json").write_text(json.dumps(outcomes, indent=2), encoding="utf-8")
    with (output / "timings.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    print("RESULT", json.dumps({k: v for k, v in summary.items() if k != "config"}), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--half", action="store_true", help="Explicit FP16 comparison, not default")
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--save-detections", action="store_true", help="Save actual detections outside the core timer for replay audits")
    args = parser.parse_args()
    benchmark(args.video, args.model, args.config, args.output,
              device=args.device, half=args.half, warmup=args.warmup, save_detections=args.save_detections)


if __name__ == "__main__":
    main()

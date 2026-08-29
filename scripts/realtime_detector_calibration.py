"""Live GRIP camera/YOLO calibration using the deployment detector configuration."""

from __future__ import annotations

import argparse
import time

import cv2

from glove_chirality.camera import open_camera
from glove_chirality.config import ExtractionConfig
from glove_chirality.detection.yolo import YoloDetector
from glove_chirality.realtime_calibration import (
    draw_calibration_overlay,
    save_screenshot,
    validate_calibration_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Extraction/deployment YAML configuration")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument(
        "--backend",
        choices=("DirectShow", "Media Foundation", "OpenCV default"),
        help="Preferred backend; other available backends remain fallbacks",
    )
    parser.add_argument(
        "--screenshots",
        default="outputs/calibration",
        help="Directory used when S is pressed",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = ExtractionConfig.from_yaml(args.config)
    validate_calibration_config(config)

    opened = open_camera(args.camera, preferred_backend=args.backend)
    print(
        f"Camera {args.camera}: backend={opened.backend}, "
        f"resolution={opened.width}x{opened.height}, reported_fps={opened.fps:.2f}"
    )
    print("Keys: Q/ESC quit | S screenshot | F toggle size filter | R show/hide rejects")

    detector = YoloDetector(config.detector)
    frame = opened.first_frame
    size_filter_enabled = True
    show_size_rejected = True
    frame_count = 0
    measured_fps = 0.0
    interval_start = time.perf_counter()
    window = "GRIP detector calibration"

    try:
        while True:
            detections, diagnostics = detector.detect_with_diagnostics(
                frame,
                apply_size_filter=size_filter_enabled,
            )
            frame_count += 1
            elapsed = time.perf_counter() - interval_start
            if elapsed >= 1.0:
                measured_fps = frame_count / elapsed
                frame_count = 0
                interval_start = time.perf_counter()
            display = draw_calibration_overlay(
                frame,
                config,
                detections,
                diagnostics,
                fps=measured_fps,
                size_filter_enabled=size_filter_enabled,
                show_size_rejected=show_size_rejected,
            )
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q"), ord("Q")}:
                break
            if key in {ord("f"), ord("F")}:
                size_filter_enabled = not size_filter_enabled
            elif key in {ord("r"), ord("R")}:
                show_size_rejected = not show_size_rejected
            elif key in {ord("s"), ord("S")}:
                path = save_screenshot(display, args.screenshots)
                print(f"Saved screenshot: {path}")

            ok, next_frame = opened.capture.read()
            if not ok or next_frame is None or next_frame.size == 0:
                raise RuntimeError(
                    f"Camera {args.camera} ({opened.backend}) opened but frame reading failed"
                )
            frame = next_frame
    finally:
        opened.capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

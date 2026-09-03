from pathlib import Path
import argparse
import cv2
import numpy as np

from glove_chirality.config import ExtractionConfig
from glove_chirality.detection.yolo import YoloDetector


def norm_box(box, width, height):
    x1, y1, x2, y2 = box
    return (
        int(round(x1 * width)),
        int(round(y1 * height)),
        int(round(x2 * width)),
        int(round(y2 * height)),
    )


def draw_box(img, det, color, label_prefix=""):
    x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    if det.polygon is not None and len(det.polygon) >= 3:
        pts = np.array(det.polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, color, 2)

    text = f"{label_prefix}{det.confidence:.2f}"
    cv2.putText(
        img,
        text,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Render YOLO glove detections to a saved video."
    )
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output mp4 path")
    parser.add_argument(
        "--config",
        default=r"configs\grip_aug27_final_v2.yaml",
        help="Extraction config YAML"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional limit; 0 means full video"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Override output FPS; 0 means use source FPS"
    )
    args = parser.parse_args()

    cfg = ExtractionConfig.from_yaml(args.config)
    detector = YoloDetector(cfg.detector)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0:
        src_fps = 25.0

    out_fps = args.fps if args.fps > 0 else src_fps

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame")

    height, width = frame.shape[:2]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (width, height),
    )

    roi_px = norm_box(cfg.detector.roi, width, height)
    trigger_px = norm_box(cfg.detector.trigger_zone, width, height)

    frames = 0
    total_green = 0
    total_red = 0

    while True:
        if frames == 0:
            current = frame
        else:
            ok, current = cap.read()
            if not ok:
                break

        detections, diag = detector.detect_with_diagnostics(
            current,
            apply_size_filter=True
        )

        vis = current.copy()

        # ROI = blue
        cv2.rectangle(
            vis,
            (roi_px[0], roi_px[1]),
            (roi_px[2], roi_px[3]),
            (255, 0, 0),
            2,
        )

        # Trigger = yellow
        cv2.rectangle(
            vis,
            (trigger_px[0], trigger_px[1]),
            (trigger_px[2], trigger_px[3]),
            (0, 255, 255),
            2,
        )

        # Green = kept detections
        for det in detections:
            draw_box(vis, det, (0, 255, 0), "G ")
            total_green += 1

        # Red = size-rejected detections
        for item in diag.size_rejected:
            draw_box(vis, item.detection, (0, 0, 255), "R ")
            total_red += 1

        info1 = (
            f"frame={frames}  raw={diag.raw_yolo_count}  "
            f"kept={len(detections)}  rejected={diag.size_rejected_count}"
        )
        info2 = (
            f"model={Path(cfg.detector.yolo_model).name}"
        )

        cv2.putText(
            vis, info1, (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (255, 255, 255), 2
        )
        cv2.putText(
            vis, info2, (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (255, 255, 255), 2
        )

        writer.write(vis)

        frames += 1

        if frames % 100 == 0:
            print(f"{frames} frames processed")

        if args.max_frames > 0 and frames >= args.max_frames:
            break

    cap.release()
    writer.release()

    print()
    print("DONE")
    print("Frames written:", frames)
    print("Green detections drawn:", total_green)
    print("Red rejected detections drawn:", total_red)
    print("Output:", output_path)


if __name__ == "__main__":
    main()

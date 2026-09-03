from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import cv2

from glove_chirality.config import ExtractionConfig


def normalized_box(selection, width, height):
    x, y, w, h = selection

    return (
        round(x / width, 4),
        round(y / height, 4),
        round((x + w) / width, 4),
        round((y + h) / height, 4),
    )


def pixel_box(normalized, width, height):
    x1, y1, x2, y2 = normalized

    return (
        round(x1 * width),
        round(y1 * height),
        round(x2 * width),
        round(y2 * height),
    )


def draw_box(frame, box, color, text, thickness=3):
    x1, y1, x2, y2 = box

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
    )

    cv2.putText(
        frame,
        text,
        (x1 + 6, max(25, y1 + 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


def fit_for_display(frame, max_width, max_height):
    """
    Resize only for the calibration UI.
    Original frame geometry remains unchanged.
    """

    h, w = frame.shape[:2]

    scale = min(
        max_width / w,
        max_height / h,
        1.0,
    )

    display_w = int(round(w * scale))
    display_h = int(round(h * scale))

    if scale < 1.0:
        display = cv2.resize(
            frame,
            (display_w, display_h),
            interpolation=cv2.INTER_AREA,
        )
    else:
        display = frame.copy()

    return display, scale


def select_box(window_name, image):
    """
    Resizable ROI-selection window.
    """

    h, w = image.shape[:2]

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        window_name,
        w,
        h,
    )

    selection = cv2.selectROI(
        window_name,
        image,
        fromCenter=False,
        showCrosshair=True,
    )

    cv2.destroyWindow(window_name)

    return selection


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-config", required=True)

    parser.add_argument(
        "--time",
        type=float,
        default=5.0,
        help="Video timestamp in seconds",
    )

    parser.add_argument(
        "--preview",
        default="outputs/video_calibration_preview.jpg",
    )

    parser.add_argument(
        "--display-width",
        type=int,
        default=1100,
        help="Maximum calibration UI width",
    )

    parser.add_argument(
        "--display-height",
        type=int,
        default=650,
        help="Maximum calibration UI height",
    )

    args = parser.parse_args()

    config = ExtractionConfig.from_yaml(args.config)

    cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {args.input}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    frame_index = int(round(args.time * fps))

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_index,
    )

    ok, frame = cap.read()

    cap.release()

    if not ok or frame is None:
        raise RuntimeError(
            f"Could not read frame {frame_index}"
        )

    original_height, original_width = frame.shape[:2]

    display, scale = fit_for_display(
        frame,
        args.display_width,
        args.display_height,
    )

    display_height, display_width = display.shape[:2]

    print()
    print("VIDEO:", args.input)
    print("FRAME:", frame_index)
    print("TIME:", args.time)
    print(
        "ORIGINAL RESOLUTION:",
        f"{original_width}x{original_height}",
    )
    print(
        "DISPLAY RESOLUTION:",
        f"{display_width}x{display_height}",
    )
    print(
        "DISPLAY SCALE:",
        round(scale, 4),
    )

    # -------------------------------------------------
    # ROI
    # -------------------------------------------------

    print()
    print("STEP 1:")
    print("Draw the useful detection ROI.")
    print()
    print("Mouse:")
    print("  drag = select")
    print("  ENTER/SPACE = accept")
    print("  C = cancel")

    roi_selection = select_box(
        "1 - SELECT DETECTION ROI",
        display,
    )

    if (
        roi_selection[2] <= 0
        or roi_selection[3] <= 0
    ):
        raise RuntimeError(
            "ROI selection cancelled"
        )

    # Since the display is uniformly scaled,
    # normalized coordinates are identical to
    # normalized coordinates in the original frame.
    roi = normalized_box(
        roi_selection,
        display_width,
        display_height,
    )

    # -------------------------------------------------
    # Trigger preview
    # -------------------------------------------------

    trigger_preview = display.copy()

    draw_box(
        trigger_preview,
        pixel_box(
            roi,
            display_width,
            display_height,
        ),
        (255, 180, 0),
        "ROI",
        thickness=2,
    )

    print()
    print("STEP 2:")
    print("Draw the trigger zone INSIDE the ROI.")
    print()
    print(
        "Leave enough space around it so a glove "
        "can become fully visible before triggering."
    )

    trigger_selection = select_box(
        "2 - SELECT TRIGGER ZONE",
        trigger_preview,
    )

    if (
        trigger_selection[2] <= 0
        or trigger_selection[3] <= 0
    ):
        raise RuntimeError(
            "Trigger selection cancelled"
        )

    trigger = normalized_box(
        trigger_selection,
        display_width,
        display_height,
    )

    # -------------------------------------------------
    # Validate containment
    # -------------------------------------------------

    rx1, ry1, rx2, ry2 = roi
    tx1, ty1, tx2, ty2 = trigger

    if not (
        rx1 <= tx1
        and ry1 <= ty1
        and tx2 <= rx2
        and ty2 <= ry2
    ):
        raise RuntimeError(
            "Trigger zone must be fully inside ROI"
        )

    # -------------------------------------------------
    # Write calibrated config
    # -------------------------------------------------

    config.detector = replace(
        config.detector,
        roi=roi,
        trigger_zone=trigger,
    )

    config.detector.validate()

    output_config = Path(
        args.output_config
    )

    config.to_yaml(
        output_config
    )

    # -------------------------------------------------
    # Full-resolution saved preview
    # -------------------------------------------------

    result = frame.copy()

    draw_box(
        result,
        pixel_box(
            roi,
            original_width,
            original_height,
        ),
        (255, 180, 0),
        "ROI",
    )

    draw_box(
        result,
        pixel_box(
            trigger,
            original_width,
            original_height,
        ),
        (0, 255, 255),
        "TRIGGER",
    )

    preview_path = Path(
        args.preview
    )

    preview_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(preview_path),
        result,
    ):
        raise RuntimeError(
            f"Could not save preview: {preview_path}"
        )

    # -------------------------------------------------
    # Result
    # -------------------------------------------------

    print()
    print("==============================")
    print("VIDEO CALIBRATION COMPLETE")
    print("==============================")

    print()
    print("ROI:")
    print(list(roi))

    print()
    print("TRIGGER:")
    print(list(trigger))

    print()
    print("CONFIG:")
    print(output_config)

    print()
    print("PREVIEW:")
    print(preview_path)

    print()
    print(
        "NOTE: These are provisional video-camera "
        "parameters. Factory live-camera calibration "
        "is still required later."
    )


if __name__ == "__main__":
    main()

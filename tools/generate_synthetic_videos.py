"""Generate small deterministic videos for integration testing and demonstrations."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def make_video(path: Path, chirality: str, variant: int, events: int = 4):
    width, height, fps = 320, 240, 25
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {path}")
    rng = np.random.default_rng(variant)
    for event in range(events):
        for step in range(45):
            frame = np.full((height, width, 3), (65 + variant, 175, 65), dtype=np.uint8)
            noise = rng.integers(-2, 3, frame.shape, dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            cv2.rectangle(frame, (16, 12), (304, 228), (95, 205, 95), 3)
            if step < 35:
                y = -95 + step * 10
                x = 122 + (event % 2) * 3
                color = (28 + variant, 28 + variant, 28 + variant)
                cv2.rectangle(frame, (x, y), (x + 76, y + 92), color, -1)
                for finger in range(4):
                    fx = x + 2 + finger * 19
                    cv2.rectangle(frame, (fx, y - 28), (fx + 13, y + 8), color, -1)
                thumb_x = x - 25 if chirality == "left" else x + 76
                cv2.rectangle(frame, (thumb_x, y + 36), (thumb_x + 28, y + 54), color, -1)
            writer.write(frame)
    writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="synthetic_videos")
    args = parser.parse_args()
    root = Path(args.output)
    for chirality in ("left", "right"):
        for variant in (1, 2):
            make_video(root / chirality / f"{chirality}_{variant:02d}.avi", chirality, variant)
    print(f"Created four videos under {root}")


if __name__ == "__main__":
    main()

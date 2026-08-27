# Glove Chirality Pipeline

A modular computer-vision framework for turning fixed-camera conveyor videos into **one representative crop per glove passage**, then training or deploying interchangeable **left/right chirality classifiers**.

The central design rule is that dataset creation and deployment call the **same event extractor**. This prevents train/deployment crop skew.

For project continuation, read [`HANDOFF.md`](HANDOFF.md). Coding agents should also read [`AGENTS.md`](AGENTS.md), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), and [`docs/REAL_VIDEO_PLAN.md`](docs/REAL_VIDEO_PLAN.md).

## Pipeline

```text
MKV/video or camera recording
        |
interchangeable detector (classical CPU or custom YOLO)
        |
passage state machine + best-frame selector
        |
one square crop per accepted glove event
        |-----------------------------|
        |                             |
images + manifest.csv          classifier checkpoint
(training / any external use)         |
                                      left/right + confidence
```

## What is implemented

- Sequential video decoding suitable for MJPEG-in-MKV recordings.
- Configurable inspection ROI and central trigger zone.
- CPU `dark_contour` bootstrap detector for dark gloves on a bright/green belt.
- Optional Ultralytics YOLO adapter with the same detector interface.
- Temporal event state machine with confirmation, tracking distance, exit timeout, cooldown, best-frame quality scoring, padded square crops, and exactly one emission per accepted event.
- Label provenance: left-only/right-only video streams attach known source labels; detection never uses the label.
- Ordinary JPEG crops plus auditable CSV metadata.
- Grouped train/validation split by source video to prevent adjacent-event leakage.
- Interchangeable `tiny_cnn`, `resnet18`, `mobilenet_v3_small`, and `vit_b_16` classifiers.
- Image inference and full video-to-event-to-prediction deployment commands.
- Synthetic-video integration test because real recordings are not in this repository.

## Installation

Python 3.10+ is supported.

```bash
python -m venv .venv
source .venv/Scripts/activate       # Git Bash on Windows
python -m pip install -e '.[dev]'
```

Add PyTorch classifiers:

```bash
python -m pip install -e '.[ml,dev]'
```

The package does not force CPU execution. Install the PyTorch build matching the machine's CUDA driver using the official PyTorch selector, then verify:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Add the optional YOLO detector:

```bash
python -m pip install -e '.[yolo]'
```

Videos, generated images, checkpoints, and outputs are gitignored.

## 1. Calibrate on the real video

First render a frame with the configured ROI, trigger zone, and detections:

```bash
glove-pipeline preview \
  --video "D:/gloves/left/left_01.mkv" \
  --seconds 30 \
  --config configs/default.yaml \
  --output outputs/preview.jpg
```

Edit `configs/default.yaml` until:

- `roi` excludes the enclosure/light-strip borders.
- `trigger_zone` covers only the central illuminated region where a complete glove should be selected.
- red detector boxes surround gloves but not belt shadows or enclosure borders.
- `dark_threshold`, area ratios, and morphology work across early/middle/late samples from every source video.

Coordinates are normalized full-frame `[x1, y1, x2, y2]`, so the same config works at 1920x1080 and downscaled test footage.

## 2. Extract a labeled dataset

Given separate left-only and right-only folders:

```bash
glove-pipeline extract-dataset \
  --left "D:/gloves/left" \
  --right "D:/gloves/right" \
  --output data/chirality_v1 \
  --config configs/default.yaml
```

Output:

```text
data/chirality_v1/
├── images/
│   ├── left/*.jpg
│   └── right/*.jpg
└── manifest.csv
```

The crops are independent ordinary images and may be copied into any other workflow. `manifest.csv` records event ID, known-stream label and provenance, source video, representative frame/time, original bounding box, detector, quality score, and extraction-config hash.

For unlabeled extraction:

```bash
glove-pipeline extract --input video.mkv --label unknown --output outputs/events
```

## 3. Review before training

Do not assume automated extraction is correct. For a representative subset of every video, manually compare:

- physical glove passage count vs accepted crop count;
- missed and duplicate events (duplicate target: zero);
- clipped, blurred, merged, or multi-glove crops;
- crop counts per video and class.

Touching gloves may be one connected component. The bootstrap detector cannot reliably separate them; enforce spacing, review/reject ambiguous events, or train an instance detector/segmenter.

## 4. Train interchangeable classifiers

```bash
glove-pipeline train \
  --manifest data/chirality_v1/manifest.csv \
  --model resnet18 \
  --epochs 20 \
  --device cuda \
  --amp \
  --workers 4 \
  --output checkpoints/resnet18_best.pt
```

Choices: `tiny_cnn`, `resnet18`, `mobilenet_v3_small`, `vit_b_16`.

`--device auto` uses CUDA when available; `cpu`, `cuda`, and `cuda:N` are explicit alternatives. `--amp` enables CUDA mixed precision. The split is by **source video**, not random images. Horizontal flipping is intentionally absent because a reflection can alter chirality semantics. Training uses class-weighted loss, selects checkpoints by validation balanced accuracy, and writes accuracy, balanced accuracy, macro-F1, and a confusion matrix to `<checkpoint>.metrics.json`. With only three right-glove videos, report per-video results and uncertainty; collect both classes under matched recording conditions to reduce background/session leakage.

## 5. Inference

On pre-extracted images:

```bash
glove-pipeline infer-images \
  --input data/chirality_v1/images \
  --checkpoint checkpoints/resnet18_best.pt \
  --output outputs/image_predictions.csv
```

End-to-end deployment on a video uses the same extraction code as training:

```bash
glove-pipeline infer-video \
  --video new_conveyor_run.mkv \
  --checkpoint checkpoints/resnet18_best.pt \
  --device cuda \
  --config configs/default.yaml \
  --output outputs/run_001
```

This writes each event crop plus `predictions.csv` containing one left/right prediction and confidence per accepted passage.

## Swapping components

- Implement `GloveDetector.detect(frame) -> list[Detection]` and register it in `detection/factory.py` to add a detector.
- Add a classifier constructor to `models.build_model` to compare another CNN/ViT.
- The extraction manifest and image interface remain unchanged.

For YOLO, train a custom glove model and set:

```yaml
detector:
  backend: yolo
  yolo_model: checkpoints/glove_detector.pt
  yolo_confidence: 0.35
  yolo_class_id: 0
  yolo_device: 0
  yolo_half: true
```

A generic COCO nano model does not contain a glove class; `yolo11n.pt` is only a placeholder for interface testing and must be replaced by custom weights.

## Testing

```bash
pytest
```

The integration test generates a deterministic MJPEG synthetic conveyor clip and verifies that one moving object produces exactly one square event crop and a valid manifest. Real-video calibration and manually annotated passage-count validation remain required before claiming production performance.

## Current limitations and next research steps

- The classical detector assumes a dark glove against a brighter stable belt. A learned belt-color background model or instance segmentation is a better next backend if exposure, shadows, or touching objects cause errors.
- Current tracking is deliberately lightweight and optimized for one well-spaced passage through the trigger zone. For simultaneous gloves, add multi-object tracks with merge/split ambiguity rejection.
- No accuracy claim is made without the real data. Keep extraction metrics separate from chirality classification metrics.
- Because class labels come from different videos, audit models for video/session leakage and capture left and right gloves under matched conditions.

## License

MIT

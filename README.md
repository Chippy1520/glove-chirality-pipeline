# Glove Chirality Pipeline

A modular computer-vision framework for turning fixed-camera conveyor videos into **one representative crop per glove passage**, then training or deploying interchangeable **left/right chirality classifiers**.

The central design rule is that dataset creation and deployment call the **same event extractor**. This prevents train/deployment crop skew.

For project continuation, read [`HANDOFF.md`](HANDOFF.md). Coding agents should also read [`AGENTS.md`](AGENTS.md), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), and [`docs/REAL_VIDEO_PLAN.md`](docs/REAL_VIDEO_PLAN.md).

## Lightweight desktop GUI

Launch the Tkinter interface after installation:

```bash
glove-pipeline-gui
```

It provides file/directory pickers, common extraction-setting editing with YAML load/save, calibration preview, labeled/unlabeled extraction, model/GPU training controls, image/video inference, live logs, and process stopping. It calls the same CLI backend, so GUI and scripted runs remain reproducible. See [`docs/GUI.md`](docs/GUI.md).

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
- CPU `belt_foreground` detector combining Lab belt-color distance with optional temporal motion, independent of a specific glove color.
- Legacy `dark_contour` fallback for controlled dark-glove recordings.
- Optional Ultralytics YOLO adapter with the same detector interface.
- Full-containment trigger gating: a glove is eligible only when its entire detected box is inside the trigger zone; partial entry/exit frames are rejected consistently by classical and YOLO backends.
- Temporal event state machine with confirmation, tracking distance, exit timeout, cooldown, best-frame quality scoring, padded square crops, and exactly one emission per accepted event.
- Explicit no-glove behavior: empty conveyor frames and long gaps emit no crop or prediction, while adaptive background learning refreshes the belt model between passages.
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
- `color_distance_threshold`, motion assistance, area ratios, and morphology work across early/middle/late samples and every glove color.

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

Empty intervals are expected. They do not create `unknown` samples; `unknown` means an extracted glove event without a supplied source label. An entirely empty input produces a header-only manifest and zero images.

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

- The default classical detector supports arbitrary glove colors that differ visually from the dominant belt and uses motion assistance for low color contrast. A glove indistinguishable from the belt still requires temporal evidence, a contrasting belt, or a learned detector; touching objects may require instance segmentation.
- Current tracking is deliberately lightweight and optimized for one well-spaced passage through the trigger zone. For simultaneous gloves, add multi-object tracks with merge/split ambiguity rejection.
- No accuracy claim is made without the real data. Keep extraction metrics separate from chirality classification metrics.
- Because class labels come from different videos, audit models for video/session leakage and capture left and right gloves under matched conditions.

## License

MIT

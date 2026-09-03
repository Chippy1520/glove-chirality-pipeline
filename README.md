# Glove Chirality Pipeline

A modular computer-vision framework for turning fixed-camera conveyor videos into **one representative crop per glove passage**, then training or deploying interchangeable **left/right chirality classifiers**.

The central design rule is that dataset creation and deployment call the **same event extractor**. This prevents train/deployment crop skew.

For project continuation, read [`HANDOFF.md`](HANDOFF.md). Coding agents should also read [`AGENTS.md`](AGENTS.md), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/REAL_VIDEO_PLAN.md`](docs/REAL_VIDEO_PLAN.md), [`docs/CAMERA_CALIBRATION.md`](docs/CAMERA_CALIBRATION.md), [`docs/DETECTOR_MODEL_OPTIONS.md`](docs/DETECTOR_MODEL_OPTIONS.md), and [`docs/CLASSIFIER_MODEL_OPTIONS.md`](docs/CLASSIFIER_MODEL_OPTIONS.md).

## Lightweight desktop GUI

Launch the Tkinter interface after installation:

```bash
glove-pipeline-gui
```

It provides file/directory pickers, common extraction-setting editing with YAML load/save, calibration preview, labeled/unlabeled extraction, model/GPU training controls, image/video inference, live logs, and process stopping. It calls the same CLI backend, so GUI and scripted runs remain reproducible. See [`docs/GUI.md`](docs/GUI.md).

## Pipeline

```text
camera / video
        |
configured full-frame ROI -> single-class YOLO11n-seg (class 0 = glove)
        |
mask + confidence -> full-containment gate -> temporal passage processor
        |
best complete/sharp frame -> canonical bbox/masked crop -> fixed-size letterbox
        |-----------------------------|
        |                             |
images + manifest/report       classifier checkpoint
(training / offline audit)            |
                               one left/right decision per accepted passage
```

Offline extraction, offline video inference, and live inference all use the same
`PassageProcessor` and `create_event_crop()` path. The classifier is not run per frame.

## What is implemented

- Sequential video decoding suitable for MJPEG-in-MKV recordings.
- Configurable inspection ROI and central trigger zone.
- CPU `belt_foreground` detector combining Lab belt-color distance with optional temporal motion, independent of a specific glove color.
- Legacy `dark_contour` fallback for controlled dark-glove recordings.
- Optional Ultralytics YOLO adapter with full-frame or ROI-only inference and full-frame coordinate restoration.
- YOLO segmentation polygons are retained in the backend-neutral `Detection`; strict mask mode rejects box-only checkpoints instead of silently falling back.
- Full-containment trigger gating is applied by the shared passage processor, so partial gloves remain visible to diagnostics/auditing but cannot become accepted crops.
- Frame- or monotonic-time-based temporal state machine with confirmation, deterministic center/bbox-IoU association, exit timeout, and continuous-empty cooldown; detections during rearming reset cooldown to suppress duplicate passages.
- Canonical `bbox`, `masked`, and `masked_fill` crops share the same padding, square, and aspect-preserving output-size stage; `bbox` remains the compatibility default.
- Explicit no-glove behavior: empty conveyor frames and long gaps emit no crop or prediction, while adaptive background learning refreshes the belt model between passages.
- Ambiguity rejection: the single-object extractor does not arbitrarily choose among multiple simultaneous candidates.
- Label provenance: left-only/right-only video streams attach known source labels; detection never uses the label.
- Ordinary JPEG crops plus accepted-event `manifest.csv` and accepted/rejected `event_report.csv` audit metadata.
- Optional separate polygon JSON files via `event.save_masks`; ordinary CSV files never embed large polygons.
- Fixed-size, aspect-preserving crop export (`256x256` by default) shared by dataset, offline inference, and live inference.
- Grouped train/validation split by source video to prevent adjacent-event leakage.
- Interchangeable `tiny_cnn`, `resnet18`, `mobilenet_v3_small`, and `vit_b_16` classifiers.
- Image inference and full video-to-event-to-prediction deployment commands.
- `infer-live` with bounded latest-frame capture, stale-frame dropping, model warm-up, rolling performance metrics, and JSONL event output; classification runs once per accepted passage.
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
  --warmup-seconds 2 \
  --config configs/default.yaml \
  --output outputs/preview.jpg
```

Preview warms temporal detectors on preceding clean frames before evaluating the requested timestamp. This better matches sequential extraction than initializing MOG2 on a single randomly sought frame.

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
├── masks/*.json          # only when event.save_masks=true
├── manifest.csv          # accepted crops used by classifier training
└── event_report.csv      # accepted and rejected passage outcomes
```

The crops are independent ordinary images and may be copied into any other workflow. `manifest.csv` records accepted-event provenance, source frame/time, full-frame box, detector confidence, segmentation usage/area/fill ratio, candidate count, crop reference, quality, and config hash. `event_report.csv` separates accepted passages from `multiple_candidates`, `partial`, `track_lost`, `insufficient_confirmation`, `low_sharpness`, and end-of-stream rejections. Rejected outcomes never reach the classifier.

Every crop is exported at `event.output_size` square pixels (256 by default). Resizing preserves aspect ratio and letterboxes when necessary, so crop dimensions cannot become a classifier shortcut. Changing this setting requires regenerating the dataset; do not mix old variable-sized crops with the new export.

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

`--device auto` uses CUDA when available; `cpu`, `cuda`, and `cuda:N` are explicit alternatives. `--amp` enables CUDA mixed precision. The split is by **source video**, not random images. Horizontal flipping is intentionally absent because a reflection can alter chirality semantics.

Training loss and checkpoint selection are explicit. `weighted_cross_entropy` is the backward-compatible default; `cross_entropy` removes class weighting. When a missed right glove is costlier than a false right alarm, use the experimental hybrid objective and retain the checkpoint with the highest held-out right recall:

```bash
glove-pipeline train \
  --manifest data/chirality_v1/manifest.csv \
  --model resnet18 \
  --loss recall_hybrid \
  --recall-target right \
  --recall-weight 2.0 \
  --selection-metric recall_right \
  --output checkpoints/resnet18_right_recall.pt
```

The metrics sidecar records accuracy, macro recall/balanced accuracy, per-class precision and recall, macro-F1, and the confusion matrix. Increasing the recall penalty is not a substitute for source-grouped validation.

## 5. Inference

On pre-extracted images:

```bash
glove-pipeline infer-images \
  --input data/chirality_v1/images \
  --checkpoint checkpoints/resnet18_best.pt \
  --output outputs/image_predictions.csv
```

To bias deployment toward **right-glove recall**, select `right` as the thresholded class and use a threshold below `0.5`:

```bash
glove-pipeline infer-images \
  --input data/chirality_v1/images \
  --checkpoint checkpoints/resnet18_right_recall.pt \
  --decision-class right \
  --decision-threshold 0.25 \
  --output outputs/right_recall_predictions.csv
```

A right probability at or above `0.25` is then reported as right even when left has the larger probability. Lower thresholds reduce right-as-left false negatives but increase left-as-right false positives. Choose the threshold only on locked, source-grouped validation sessions and report right recall, right precision, and the confusion matrix. This option changes Layer-2 classification only; it cannot recover a glove missed by the Layer-1 detector/extractor.

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

Live camera/stream inference uses the same passage processor and crop implementation:

```bash
glove-pipeline infer-live \
  --source 0 \
  --checkpoint checkpoints/resnet18_best.pt \
  --config configs/production.yaml \
  --device cuda \
  --amp \
  --output outputs/live_events.jsonl
```

The capture thread uses a bounded queue and discards stale frames instead of accumulating latency. `event.timing_mode: time` uses monotonic capture timestamps so confirmation, exit, and cooldown durations remain physical-time quantities when frames are dropped. `runtime.detect_every_n_frames` defaults to `1`; increasing it trades detector load for temporal evidence and must be validated. Periodic capture/processed FPS and rolling YOLO/event/classifier/accepted-event latency are written to stderr, while JSONL remains machine-readable. No robot-motion commands are embedded.

## Production YOLO11n-seg configuration

The detector is deliberately single-class (`class 0 = glove`); chirality remains downstream. In the GUI, open **Extraction settings**, browse to the collaborator's trained `best.pt` under **Layer 1 — custom YOLO segmentation**, and save the configuration. Selection automatically switches the backend to YOLO, sets class 0, requires masks, and enables ROI-only inference. The checkpoint selected later on the **Inference** tab is a different Layer-2 model: it classifies the accepted crop as left/right.

A complete YAML example is committed at [`configs/production.yaml`](configs/production.yaml):

```yaml
detector:
  backend: yolo
  roi: [0.15, 0.05, 0.90, 0.99]
  trigger_zone: [0.18, 0.15, 0.90, 0.90]
  require_full_containment: true
  trigger_inner_margin_ratio: 0.0
  yolo_model: checkpoints/yolo11n_seg_glove_best.pt
  yolo_confidence: 0.25
  yolo_class_id: 0
  yolo_device: 0
  yolo_half: true
  yolo_use_masks: true
  yolo_require_masks: true
  yolo_crop_to_roi: true
  yolo_imgsz: 640
  yolo_iou: 0.50
  yolo_max_det: 5
  yolo_min_box_area_ratio: 0.0
  yolo_max_box_area_ratio: 1.0

event:
  min_detected_frames: 2
  reject_multiple_detections: true
  exit_missing_frames: 5
  cooldown_frames: 8
  crop_padding: 0.12
  make_square: true
  output_size: 256
  crop_mode: bbox
```

Keep `crop_mode: bbox` until a source-grouped experiment demonstrates that `masked` or `masked_fill` improves held-out performance. Compare modes with exactly the same source/session splits; crop mode is part of the extraction config hash.

For the fixed GRIP Aug-27 camera, [`configs/grip_aug27_seed.yaml`](configs/grip_aug27_seed.yaml) records the measured `0.03–0.40` full-frame bbox-area gate. The gate runs after ROI coordinates are restored and before trigger/event logic, so physically tiny or oversized YOLO detections cannot become events or crops. These values are camera-specific and must be remeasured after changing camera position, lens, zoom, resolution, or ROI.

The GRIP config also uses `crop_padding: 0.0` and `make_square: false`. The source crop therefore follows the selected mask-derived detection bbox rather than a padded square that may include a nearby glove. The source crop can have variable width and height; `_letterbox()` still creates a fixed 256×256 classifier input without aspect-ratio distortion. If another glove remains visible inside the selected rectangle, evaluate `masked` or `masked_fill` to retain only the selected instance polygon. Simultaneous/touching instances remain ambiguous and are rejected when `reject_multiple_detections: true`.

For a moved/current camera, preserve the Aug-27 config and calibrate [`configs/grip_current_camera.yaml`](configs/grip_current_camera.yaml) with `python scripts/realtime_detector_calibration.py --camera 0 --config configs/grip_current_camera.yaml`. The current-camera file is intentionally safe-but-uncalibrated (`roi` full frame and size gate `0.0–1.0`) until real measurements replace those values. See the [camera calibration guide](docs/CAMERA_CALIBRATION.md) for Windows backend fallback, diagnostic rejected boxes, hard-negative annotation, versioning, and model provenance.

## Swapping components

- Implement `GloveDetector.detect(frame) -> list[Detection]` and register it in `detection/factory.py` to add a detector.
- Add a classifier constructor to `models.build_model` to compare another CNN/ViT.
- The extraction manifest and image interface remain unchanged.

For YOLO, use a custom single-class segmentation checkpoint and the production example above. Generic COCO weights do not define a glove class. `yolo_require_masks: true` validates the loaded task and fails clearly for box-only weights; set it false only when box fallback is intentionally part of an experiment.

## Testing

```bash
pytest
```

The integration test generates a deterministic MJPEG synthetic conveyor clip and verifies that one moving object produces exactly one square event crop and a valid manifest. Real-video calibration and manually annotated passage-count validation remain required before claiming production performance.

## Current limitations and next research steps

- The default classical detector remains a bootstrap path; production targets a custom single-class YOLO11n-seg checkpoint. No detector weights or real-video accuracy evidence are stored here.
- Tracking remains one active deterministic passage, not a multi-object tracker. Multiple simultaneous instances are rejected and audited rather than arbitrarily selected.
- Low-confidence proposals suppressed inside YOLO NMS are not observable to the downstream audit report; evaluate detector misses separately against annotated footage.
- Real-time FPS and latency depend on camera, decoder, GPU, model, ROI, and exposure. The instrumentation is implemented, but no hardware performance claim is made without measurement.
- OpenCV camera/stream capture is supported. Use `infer-video` for deterministic file evaluation; a file supplied to `infer-live` may be decoded faster than wall time and stale frames can intentionally be dropped.
- Because class labels come from different videos, audit models for video/session leakage and capture left and right gloves under matched conditions.

## License

MIT

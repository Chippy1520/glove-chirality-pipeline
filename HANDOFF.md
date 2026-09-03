# Project handoff

This file lets another researcher or AI coding agent continue without access to the original conversation.

## Objective

A fixed 1920×1080, 25 FPS camera observes industrial gloves of potentially any color moving through a green-belt lightbox. Source recordings are separated into left-only and right-only streams. The system must:

1. detect a glove inside the illuminated inspection area;
2. identify one physical passage over time;
3. retain the best complete/sharp/central frame;
4. export one independent crop and manifest row per accepted passage;
5. use those crops to train interchangeable chirality classifiers;
6. reuse the identical extractor and preprocessing at deployment;
7. emit one left/right prediction and confidence per accepted passage.

Expected data inventory, not stored here:

- Left: 11 videos, about 157.65 minutes, 236,469 frames, 34.48 GB.
- Right: 3 videos, about 30.48 minutes, 45,716 frames, 5.64 GB.
- Video format: MJPEG in MKV, `yuvj420p`, nominal 25 FPS.

## What is complete

- Installable `src/` Python package and command-line interface.
- Color-agnostic belt-foreground, legacy dark-threshold, and custom-YOLO detector interfaces.
- Explicit no-glove state and adaptive belt-background learning during empty conveyor gaps.
- Full-bounding-box trigger containment by default; partial glove entry/exit frames are ineligible.
- Calibration detection runs before overlay rendering, and multi-candidate frames are rejected as ambiguous by default.
- YOLO segmentation retains polygons, derives tight boxes, optionally runs on the configured ROI, restores all geometry to full-frame coordinates, and can reject camera-specific implausible bbox-area ratios.
- Strict mask mode rejects accidental box-only checkpoints; classical and intentional box-only YOLO remain compatible.
- Shared `PassageProcessor` and `create_event_crop()` path for dataset extraction, offline inference, and live inference.
- `bbox`, `masked`, and `masked_fill` crops with one aspect-preserving output stage; `bbox` remains default, with a tight detector-bbox preset for packed scenes.
- Frame-count and monotonic-time event timing, deterministic center/bbox-IoU association, and opt-in segmentation quality terms.
- Accepted manifest plus explicit accepted/rejected event report; optional polygon JSON persistence.
- Segmentation preview overlays masks, tight boxes, confidence, candidate count, partial/edge diagnostics, and ambiguity.
- Real-time detector calibration reuses `ExtractionConfig`/`YoloDetector`, displays size-rejected candidates diagnostically, and saves versioned screenshots.
- Integer camera sources use verified-frame Windows backend fallback (DirectShow, MSMF, then default) and report the successful stream geometry.
- Bounded-queue `infer-live` with stale-frame dropping, one classifier call per accepted passage, JSONL output, model warm-up, and rolling runtime metrics.
- Model factory: TinyCNN, ResNet-18, MobileNetV3-Small, ViT-B/16, Swin-T, ConvNeXt V2 Pico, and DINOv3 ConvNeXt-Tiny. See `docs/CLASSIFIER_MODEL_OPTIONS.md` for roles and pretrained-weight licenses.
- Source-video-grouped validation split and selectable cross-entropy, weighted-cross-entropy, or recall-hybrid training.
- Accuracy, macro recall/balanced accuracy, per-class precision/recall, macro-F1, and confusion matrix output; checkpoints can be selected by right recall.
- Optional right-class inference threshold shared by image, offline-video, and live modes for the explicit recall/precision trade-off.
- Explicit CPU/GPU selection, CUDA mixed precision, and loader worker controls.
- Lightweight Tkinter GUI for paths, common YAML settings, extraction, training, and inference.
- Synthetic MJPEG integration test and GitHub Actions CI.

## What is not complete

No real video was available during implementation. Therefore:

- the ROI, trigger geometry, YOLO thresholds, and segmentation quality weights are uncalibrated placeholders;
- passage recall, precision, duplicates, rejection correctness, crop quality, and real-time latency/FPS are unknown on target hardware;
- multi-object tracking remains intentionally absent: simultaneous instances are rejected and audited;
- low-confidence proposals removed inside YOLO cannot be assigned a downstream reject reason;
- no production detector/classifier weights, real-data accuracy claim, or locked real-world split exists.

## Data placement

Keep data outside Git or under ignored directories:

```text
data_raw/
├── left/*.mkv
└── right/*.mkv
```

The repository ignores video, image, output, and model formats. Do not weaken those exclusions to upload private or 40 GB-scale media. If sharing data is authorized, use controlled object storage or Git LFS/DVC and commit only a source manifest with checksums and access instructions.

## First real-data session

1. Confirm both labels mean the chirality of the glove itself, not image position.
2. Record source/session/glove-lot metadata before extraction.
3. Train/validate a single-class YOLO11n-seg model (`class 0 = glove`) from source/session-grouped mask annotations, including partial, empty, glare, and multi-glove frames.
4. Run mask-aware `glove-pipeline preview` on early, middle, and late frames; tune ROI, trigger, confidence, image size, and containment before extraction.
5. Manually annotate passage counts and ambiguity on representative clips.
6. Run extraction and audit `manifest.csv` plus `event_report.csv` for precision/recall, duplicates, misses, partials, ambiguity, and crop completeness.
7. Freeze grouped splits before classifier or crop-mode comparison.
8. Compare `bbox`, `masked`, and `masked_fill` only on identical frozen splits; do not assume mask suppression is better.
9. Compare classifiers with the same crops/splits/seeds and report per-video/per-class uncertainty.
10. Evaluate end-to-end offline parity and live camera behavior, counting misses/extras/rejections as system outcomes.
11. Measure capture/processed FPS, detector/event/classifier latency, accepted-event latency, and dropped frames on target hardware before making performance claims.

Detailed gates are in `docs/REAL_VIDEO_PLAN.md` and `docs/VERIFICATION_CHECKLIST.md`.

## GPU examples

```bash
# Confirm the environment sees the GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# GPU training with mixed precision and parallel loading
glove-pipeline train \
  --manifest data/chirality_v1/manifest.csv \
  --model resnet18 \
  --epochs 30 \
  --batch-size 64 \
  --device cuda \
  --amp \
  --workers 4 \
  --output checkpoints/resnet18_best.pt

# Select another GPU
glove-pipeline train ... --device cuda:1 --amp

# GPU inference
glove-pipeline infer-video \
  --video new_run.mkv \
  --checkpoint checkpoints/resnet18_best.pt \
  --device cuda \
  --output outputs/new_run
# GPU live event inference; JSONL is suitable for a later robot/PLC adapter
# without embedding motion control in this repository.
glove-pipeline infer-live \
  --source 0 \
  --checkpoint checkpoints/resnet18_best.pt \
  --config configs/production.yaml \
  --device cuda \
  --amp \
  --output outputs/live_events.jsonl
```

For YOLO inference, configure `yolo_device`, `yolo_half`, ROI-only inference, and strict masks in `configs/production.yaml`. Use custom single-class glove segmentation weights; generic COCO weights do not define a glove class.

## Copy-paste prompt for another AI agent

> Open this repository and read `AGENTS.md`, `README.md`, `HANDOFF.md`, `docs/ARCHITECTURE.md`, `docs/REAL_VIDEO_PLAN.md`, and `docs/VERIFICATION_CHECKLIST.md`. Run the existing tests before changing code. Continue from the current baseline without rewriting the shared extractor. The first priority is real-video ROI/detector calibration and passage-level validation, not classifier tuning. Keep source-video/session grouping, avoid horizontal reflections, use the available GPU when useful, add regression tests, and do not claim accuracy without locked real-data evidence. Never commit recordings, extracted datasets, or model weights unless explicit data-governance and storage decisions authorize it.

## Repository ownership and privacy

The GitHub repository is private under `Chippy1520/glove-chirality-pipeline`. The teammate or their AI environment must be granted repository access by the owner; a private URL alone does not provide access.

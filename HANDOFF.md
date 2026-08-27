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
- Shared temporal extraction path for training and inference.
- Crop manifest with source, frame/time, geometry, quality, detector, provenance, and config hash.
- Model factory: TinyCNN, ResNet-18, MobileNetV3-Small, ViT-B/16.
- Source-video-grouped validation split and class-weighted training.
- Accuracy, balanced accuracy, macro-F1, and confusion matrix output.
- Explicit CPU/GPU selection, CUDA mixed precision, and loader worker controls.
- Synthetic MJPEG integration test and GitHub Actions CI.

## What is not complete

No real video was available during implementation. Therefore:

- the ROI and segmentation thresholds are placeholders;
- passage recall, precision, duplicates, and crop quality are unknown on real data;
- multi-object/merge/split handling remains a known limitation of the lightweight tracker;
- no production chirality model or real-data accuracy claim exists;
- no locked real-world train/validation/test split exists.

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
3. Run `glove-pipeline preview` on early, middle, and late frames of every recording condition.
4. Tune a copied YAML config; never silently alter the canonical baseline for one video.
5. Manually annotate passage counts and ambiguity on representative clips.
6. Run extraction and measure passage precision/recall, duplicates, misses, clipped crops, and rejections per source.
7. Freeze grouped splits before model comparison.
8. Compare TinyCNN, ResNet-18, MobileNetV3-Small, and ViT with the same crops/splits/seeds.
9. Report per-video and per-class metrics. Three right videos imply wide uncertainty.
10. Evaluate end-to-end inference, counting missed and extra events as system failures.

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
```

For YOLO inference, set `yolo_device: 0` and `yolo_half: true` in the extraction YAML. Use custom glove weights; generic COCO weights do not define a glove class.

## Copy-paste prompt for another AI agent

> Open this repository and read `AGENTS.md`, `README.md`, `HANDOFF.md`, `docs/ARCHITECTURE.md`, `docs/REAL_VIDEO_PLAN.md`, and `docs/VERIFICATION_CHECKLIST.md`. Run the existing tests before changing code. Continue from the current baseline without rewriting the shared extractor. The first priority is real-video ROI/detector calibration and passage-level validation, not classifier tuning. Keep source-video/session grouping, avoid horizontal reflections, use the available GPU when useful, add regression tests, and do not claim accuracy without locked real-data evidence. Never commit recordings, extracted datasets, or model weights unless explicit data-governance and storage decisions authorize it.

## Repository ownership and privacy

The GitHub repository is private under `Chippy1520/glove-chirality-pipeline`. The teammate or their AI environment must be granted repository access by the owner; a private URL alone does not provide access.

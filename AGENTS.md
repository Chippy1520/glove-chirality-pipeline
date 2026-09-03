# Instructions for coding agents

## Mission

Build a research-quality, modular system that detects each glove passage through a fixed-camera conveyor lightbox, extracts one representative crop, and classifies glove chirality as left or right. The same extraction implementation and configuration must be used for dataset creation and deployment.

## Non-negotiable design rules

1. Keep detection, event extraction, classification, training, and output sinks interchangeable.
2. Never let a source label affect detection, crop choice, or preprocessing.
3. Split by source video/session, never by adjacent crops or frames.
4. Do not use horizontal reflection augmentation unless chirality labels are transformed under an explicitly tested convention.
5. Report extraction and classification performance separately; count extraction failures in end-to-end metrics.
6. Do not claim real-world accuracy from synthetic fixtures.
7. Treat simultaneous/touching gloves as ambiguous unless an instance-aware backend resolves them.
8. Keep videos, datasets, model weights, and generated outputs out of Git. Preserve reproducibility through manifests, configs, hashes, and commands.

## Start here

Read, in order:

1. `README.md`
2. `HANDOFF.md`
3. `docs/ARCHITECTURE.md`
4. `docs/REAL_VIDEO_PLAN.md`
5. `docs/GUI.md`
6. `docs/DETECTOR_MODEL_OPTIONS.md`
7. `docs/CLASSIFIER_MODEL_OPTIONS.md`
8. `docs/VERIFICATION_CHECKLIST.md`

Install and verify:

```bash
python -m venv .venv
source .venv/Scripts/activate  # Git Bash on Windows
python -m pip install -e '.[ml,dev]'
pytest
ruff check src tests tools
```

For an NVIDIA GPU, install the PyTorch build matching the machine's CUDA driver using the official PyTorch selector, then verify `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.

## Current baseline

- Color-agnostic Lab belt-foreground detector, legacy dark-contour fallback, and optional custom Ultralytics YOLO adapter.
- Shared passage extractor for dataset and deployment modes.
- TinyCNN, ResNet-18, MobileNetV3-Small, ViT-B/16, Swin-T, ConvNeXt V2 Pico, and DINOv3 ConvNeXt-Tiny adapters.
- Explicit `auto/cpu/cuda/cuda:N` controls, CUDA AMP, and DataLoader workers.
- Lightweight Tkinter GUI that delegates all operations to the public CLI.
- Grouped splitting, class-weighted loss, balanced metrics, synthetic integration fixtures, and CI.
- No real recordings or trained production weights are in this repository.

## Immediate continuation target

When recordings arrive, do not begin by training a classifier. First calibrate the ROI/detector, annotate passage counts on representative clips, quantify misses/duplicates/ambiguous crops per video, and freeze source/session-level splits. Follow `docs/REAL_VIDEO_PLAN.md`.

## Definition of done for changes

Before committing:

```bash
pytest
ruff check src tests tools
git diff --check
```

Add regression tests for any fixed defect. Keep documentation and CLI help synchronized. Never commit private data or generated weights.

# Architecture

## System boundary

```text
video / camera recording
        ↓
FrameSource (sequential OpenCV decode)
        ↓
GloveDetector backend
        ↓ list[Detection]
Passage extractor state machine
        ↓ one best ExtractedEvent per accepted passage
        ├── Dataset sink: image + known-stream label + manifest
        └── Inference sink: classifier adapter + prediction CSV
```

`extract_video()` is the shared boundary. Training export and deployment inference both call it; there is no duplicate crop implementation.

## Package map

| Path | Responsibility |
|---|---|
| `config.py` | Validated YAML-backed detector/event settings |
| `types.py` | Backend-neutral `Detection` and `ExtractedEvent` contracts |
| `detection/base.py` | Detector abstract interface |
| `detection/classical.py` | Color-agnostic Lab/motion belt foreground plus legacy dark-object fallback |
| `detection/yolo.py` | Optional Ultralytics custom-detector adapter, CPU/GPU selectable |
| `extraction.py` | Sequential decode, temporal confirmation/tracking, quality selection, crop and manifest |
| `dataset.py` | Manifest loading, source-grouped split, chirality-safe transforms |
| `models.py` | TinyCNN/ResNet/MobileNet/ViT factory |
| `training.py` | Class-weighted PyTorch training, AMP/device controls, metrics/checkpoints |
| `inference.py` | Checkpoint loading and image predictions |
| `diagnostics.py` | ROI/trigger/detection preview image |
| `cli.py` | Public commands and end-to-end orchestration |
| `gui_commands.py` | Testable GUI-to-CLI command construction |
| `gui.py` | Lightweight Tkinter forms, settings editor, and background process log |

## Extraction lifecycle

The current lightweight extractor is intended for one well-spaced glove moving through a central trigger zone:

1. A detector candidate must have its center inside the trigger zone.
2. It is tentative until observed for `min_detected_frames`.
3. The nearest plausible detection is associated using a frame-diagonal distance gate.
4. Each candidate receives a quality score combining centrality, detector confidence, and sharpness.
5. Only the highest-scoring frame is retained in memory.
6. After `exit_missing_frames`, the event is finalized once.
7. A cooldown suppresses immediate re-triggering.
8. EOF explicitly finalizes an active event.

With no candidate, the state remains idle and no crop/classifier call is made. Long empty intervals are therefore normal. The belt-foreground backend uses plausible foreground occupancy to adapt MOG2 during empty gaps and freeze or slow learning while a glove is present, preventing a stationary passage from being absorbed into the background too quickly.

The crop expands the chosen box, clamps it to frame boundaries, and creates an exact square without stretching. Model transforms resize it later.

## Detector strategy

The default `belt_foreground` backend estimates the dominant belt color in Lab space on each frame and segments pixels by perceptual color distance. This makes crop extraction independent of whether a glove is black, white, red, blue, yellow, or another visually distinct color. When MOG2 produces a plausible temporal foreground, it is preferred so a moving target is not merged with static colored distractors; color distance is the fallback. Configuration controls ROI, trigger geometry, color distance, motion modeling, morphology, area, and solidity. The former `dark_contour` backend remains available as a controlled-scene fallback.

"Color-agnostic" does not mean visually impossible camouflage can be solved from one RGB frame. If glove and belt pixels are effectively indistinguishable, detection needs motion history, a more contrasting belt/background, another sensing modality, or a learned model that can exploit shape/context. Touching gloves remain an instance-separation problem.

Upgrade paths, in recommended order:

1. Calibrated empty-belt reference or robust spatial belt model for stronger illumination invariance.
2. Better single-object temporal association and explicit ambiguity records.
3. Custom YOLO detector for camouflage, clutter, and exposure variability.
4. Instance segmentation when touching gloves must be separated.

All upgrades should implement `GloveDetector` and preserve downstream event contracts.

## Model and preprocessing contract

- Class order is `left`, then `right`.
- Images are loaded RGB, resized to the checkpoint image size, converted to tensors, and ImageNet-normalized.
- Training may apply color jitter and small rotations.
- Horizontal reflection is intentionally excluded.
- Checkpoints store architecture name, class order, image size, preprocessing identifier, and best validation metrics.
- Inference reconstructs architectures without downloading pretrained weights, then loads the checkpoint state.

## GPU behavior

- `--device auto` selects CUDA when `torch.cuda.is_available()`, otherwise CPU.
- `--device cuda` or `cuda:N` makes accelerator intent explicit and fails if CUDA is unavailable.
- `--amp` enables CUDA mixed precision; it is safely disabled on non-CUDA devices.
- `--workers N` and pinned memory improve GPU feeding where the host allows multiprocessing.
- YOLO device and half precision are independent YAML options.

The architecture does not impose CPU as a research constraint. CPU support remains useful for extraction bootstrap, CI, and reproducibility.

## Data and leakage boundaries

Generated crops are not independent if they originate from the same recording. `grouped_split()` keeps complete source videos together. Future metadata should extend grouping to session, glove pair/lot, date, camera, and continuous parent recording.

Potential leakage channels include background condition, exposure, compression, belt dirt, source filename, crop geometry, and recording session. Because left and right originate from separate streams, nuisance-only and background-only baselines are important.

## Known structural limitations

- One active lightweight track, not a full multi-object tracker.
- No explicit accepted/rejected/ambiguous event manifest yet.
- No entry/exit line-crossing semantics or polygonal/perspective-rectified ROI yet.
- No locked test-set evaluator, calibration curves, or confidence rejection policy yet.
- Source filename is the current grouping key; richer session manifests should replace it for serious experiments.

These are deliberate continuation targets, not hidden claims.

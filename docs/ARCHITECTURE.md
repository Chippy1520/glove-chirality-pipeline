# Architecture

## Production data flow

```text
camera / video
        ↓
source adapter (sequential offline decoder or bounded latest-frame capture)
        ↓ full BGR frame + frame index + elapsed timestamp
GloveDetector
        ├── classical: box, polygon=None
        └── YOLO11n-seg: class 0, confidence, tight mask box, polygon
        ↓ list[Detection] in full-frame coordinates
PassageProcessor
        ↓ gate → associate → confirm → select best → finalize once
create_event_crop()
        ↓ bbox / masked / masked_fill → square/letterbox → fixed output size
accepted passage
        ├── dataset sink: crop + manifest.csv
        ├── audit sink: event_report.csv
        └── inference sink: one classifier call → CSV/JSONL
```

The non-negotiable boundary is `PassageProcessor` plus `create_event_crop()` in
`events.py`. `extract_video_with_report()` and `run_live_inference()` are source/sink
adapters around that same detector, gate, state machine, frame selector, and crop path.
Live mode does not contain a simplified crop implementation and never classifies every
frame.

## Package map

| Path | Responsibility |
|---|---|
| `config.py` | Validated YAML detector, event, diagnostics, and runtime settings |
| `types.py` | Immutable backend-neutral `Detection`, `ExtractedEvent`, and `EventRecord` contracts |
| `detection/base.py` | Detector interface and shared full-frame trigger policy |
| `detection/classical.py` | Lab/motion belt foreground and legacy dark-contour candidates |
| `detection/yolo.py` | Ultralytics adapter, strict segmentation checks, ROI inference, coordinate restoration |
| `events.py` | Shared passage processor, association, timing modes, quality, and canonical crop |
| `extraction.py` | Offline video adapter, image/mask persistence, manifest and event-report writers |
| `live.py` | Bounded capture queue, event-driven live classification, metrics, JSONL sink |
| `dataset.py` | Manifest loading, source-grouped split, chirality-safe transforms |
| `models.py` | TinyCNN, ResNet, MobileNet, ViT, Swin, ConvNeXt V2, and DINOv3 ConvNeXt factory |
| `training.py` | Selectable CE/weighted-CE/recall-hybrid training, AMP/device controls, per-class metrics/checkpoints |
| `inference.py` | Shared preprocessing plus optional class-specific probability thresholding |
| `diagnostics.py` | Clean-frame ROI/trigger/mask/candidate calibration preview |
| `cli.py` | Public offline, training, preview, and live commands |
| `gui_commands.py` / `gui.py` | GUI-to-CLI construction and Tkinter process controls |

## Detection contract

`Detection` always uses full-frame pixel coordinates and contains:

- tight `x1, y1, x2, y2` bounds;
- confidence and optional class ID;
- optional immutable polygon as `tuple[tuple[float, float], ...]`.

Classical and box-only backends return `polygon=None`. YOLO segmentation retains the
instance polygon and derives the tight box from it. No full-resolution binary mask is
stored per detection. Polygon area and mask/bbox fill ratio are derived lazily.

A detector reports candidates; it does not decide passage eligibility. This is necessary
for previews and audit records to show partial gloves. `PassageProcessor` applies
`inside_trigger()` consistently to every backend. With full containment enabled, a
mask-derived box crossing the trigger boundary is detected but is not eligible.

### YOLO ROI inference

With `yolo_crop_to_roi: true`:

1. normalized `roi` is converted using the original frame dimensions;
2. YOLO receives only that pixel crop;
3. local box and polygon coordinates are offset back to the original frame;
4. the restored bbox area is divided by original full-frame area and checked against the
   configured YOLO min/max physical-size ratios;
5. trigger gating, previews, crops, manifests, and live JSONL use full-frame coordinates.

`roi` and `trigger_zone` therefore retain their existing normalized full-frame meaning.
`yolo_require_masks: true` validates that the loaded model task is segmentation and
raises on any returned box without a polygon. Empty segmentation results are valid.
The area-ratio defaults `0.0–1.0` preserve historical behavior. Camera-specific nontrivial
limits reject detections inside the YOLO adapter, so rejected artifacts intentionally do not
affect ambiguity, cooldown, trigger, event, or crop logic. `detect_with_diagnostics()` exposes
raw, size-rejected, and returned counts plus rejected geometry to calibration tools only;
normal `detect()` continues returning accepted detections exclusively. Detector-level
evaluation must account for filtered candidates separately.

## Passage lifecycle

The processor intentionally represents one well-spaced physical glove:

1. All candidates are retained for ambiguity and partial diagnostics.
2. Multiple candidates are rejected by default; the processor never chooses one silently.
3. Eligible observations are associated deterministically using center displacement and
   bbox IoU. Optional mask IoU is available but disabled by default because it is more
   expensive and not yet calibrated.
4. Confirmation uses either frame counts or elapsed time.
5. Quality preserves centrality, detector confidence, and sharpness. Optional mask area,
   area stability, trigger clearance, and ROI/frame-edge penalties default to zero so
   existing selection behavior is not silently changed.
6. Only the best frame is retained in memory.
7. Exit timeout finalizes one accepted crop; rearming then requires continuous empty evidence, and any detection resets cooldown to suppress duplicates.
8. EOF accepts a confirmed passage and audits an unconfirmed/partial remainder.

Time mode uses the timestamps supplied by the source. Live mode supplies monotonic capture
time; offline mode supplies deterministic `frame_index / fps`. Dropped or intentionally
skipped live frames do not count as missing observations. At the next processed
observation, elapsed-time confirmation/exit/cooldown thresholds remain physical durations.
Frame mode remains the backward-compatible default.

## Crop contract

`create_event_crop()` is the only crop implementation:

- `bbox`: tight rectangle with configurable padding (compatibility default);
- `masked`: keep polygon pixels and zero everything outside;
- `masked_fill`: keep polygon pixels and replace outside pixels with the deterministic
  median background color from outside-mask pixels in the padded crop.

Every mode then follows the same optional square bounds and fixed-size aspect-preserving
letterbox stage. Glove geometry is never stretched. Mask modes fail clearly when a
box-only detection is supplied. Crop mode is hashed with the extraction configuration;
compare modes on the identical locked source/session split rather than mixing datasets.
For packed scenes, `crop_padding: 0.0` plus `make_square: false` uses the variable-size
selected detection bbox exactly before letterboxing, avoiding neighbor pixels introduced
only by padding or square expansion. Mask modes can suppress remaining pixels outside the
selected instance polygon; they cannot resolve a merged/incorrect instance mask.
The hash covers detector settings, event/crop settings, and detection frequency. Purely
operational diagnostics, queue size, report interval, and warm-up settings are excluded.

## Accepted and rejected event contracts

`manifest.csv` contains accepted classifier inputs and includes detector confidence,
segmentation usage, mask area, mask/bbox fill ratio, candidate count, geometry, quality,
and config hash. `event_report.csv` contains accepted and rejected outcomes such as
`multiple_candidates`, `partial`, `track_lost`, `insufficient_confirmation`,
`low_sharpness`, and `end_of_stream`.

Large polygons are not embedded in CSV. `event.save_masks: true` stores compact polygon
JSON files under `masks/` and puts their relative paths in the accepted manifest.
Detections below YOLO's configured confidence/NMS threshold are not visible downstream;
measure detector misses separately against passage annotations.

## Live operation

`LatestFrameCapture` owns a queue of one or two frames. When full, it removes the oldest
frame before inserting the newest. This bounds latency rather than buffering seconds of
stale camera data. Counters report captured, processed, and dropped frames.

At startup, detector and classifier are loaded once and optionally warmed once. YOLO runs
for tracking; the classifier runs exactly once when a passage becomes accepted. A JSONL
sink or callback receives accepted and rejected event records without embedding robot or
PLC commands. Periodic stderr reports use rolling averages for detector, event,
classifier, and accepted-event latency.

`runtime.detect_every_n_frames` defaults to one. Higher values are explicit performance
experiments and can reduce temporal evidence. For deterministic files and evaluation,
prefer `infer-video`; `infer-live` is designed for cameras and real-time streams.

## Model and preprocessing contract

- Detector class is one class: `0 = glove`; source chirality labels never affect detection.
- Classifier order is `left`, then `right`.
- Canonical crops are loaded RGB, resized to checkpoint size, tensorized, and
  ImageNet-normalized.
- Path and in-memory classifier methods call the same transform.
- Horizontal reflection is intentionally excluded.
- Checkpoints store architecture, class order, image size, preprocessing ID, training objective,
  selection metric, and validation metrics.

## Asymmetric chirality decision policy

The default classifier decision remains `argmax`, preserving historical behavior. Training can
instead use `recall_hybrid`: weighted cross-entropy plus a differentiable penalty on the
selected class's soft recall. The best checkpoint can be selected by `recall_right` when a
right-as-left false negative is the costly error.

At inference, `decision_class=right` applies a threshold directly to right probability. A
threshold below `0.5` expands the right decision region, trading lower right false negatives
for more left-as-right false positives. The same `TorchClassifier` decision path is used by
image, offline-video, and live inference. Thresholds must be selected on locked source-grouped
validation data; neither the loss nor threshold provides a formal zero-miss guarantee.

## GPU behavior

- Classifier `auto/cpu/cuda/cuda:N` controls remain explicit.
- Classifier AMP is optional and only enabled on CUDA.
- YOLO device and half precision remain independent YAML settings.
- Live startup loads and moves each model once; neither model is reloaded per frame/event.

No FPS, latency, detector accuracy, or end-to-end accuracy is claimed without measurements
on target hardware and real glove footage.

## Data and leakage boundaries

Generated crops from one recording are correlated. `grouped_split()` keeps complete source
videos together. Production experiments should group by the highest correlated unit:
session, continuous parent recording, glove pair/lot, camera, and condition. Compare bbox
and mask-suppressed crops using the exact same frozen groups.

## Known structural limitations

- One active passage, not a multi-object tracker; simultaneous instances are rejected.
- No line-crossing or perspective-rectified polygonal ROI semantics.
- Low-confidence proposals suppressed inside YOLO cannot receive downstream reject rows.
- No locked real-data test set, production weights, calibration curves, or confidence
  rejection policy is stored in this repository.
- Source filename remains the current grouping key until richer session manifests arrive.

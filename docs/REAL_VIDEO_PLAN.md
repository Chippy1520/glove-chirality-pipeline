# Real-video calibration and validation plan

Synthetic tests establish software wiring only. Follow this plan before reporting extraction or chirality performance.

## Phase 0 — definitions and provenance

- Define whether `left/right` describes glove chirality, not screen position.
- Assign stable IDs for source video, continuous recording/session, glove pair or lot, date, operator, camera, and condition where available.
- Record the known-stream label and its provenance independently of filenames.
- Identify whether left and right recordings were captured under matched conditions.
- Reserve a locked test set by session/source before model comparison.

## Phase 1 — decoder and scene audit

For every source, check decoded dimensions, FPS, frame count, duration, corruption, exposure changes, direction, speed, spacing, touching gloves, and partial events at file boundaries. Use sequential decoding; do not rely on random seeks for canonical extraction.

Measure empty intervals explicitly: empty lead-in/tail, long inter-glove gaps, and completely empty operational footage. These must yield zero glove events, while giving the adaptive background model clean belt observations.

Generate previews at early, middle, and late timestamps:

```bash
glove-pipeline preview --video VIDEO.mkv --seconds 30 --warmup-seconds 2 --config CONFIG.yaml --output preview.jpg
```

Tune ROI and trigger-zone geometry first, then the custom single-class YOLO11n-seg settings (`class 0 = glove`): confidence, image size, IoU, max detections, strict masks, ROI-only inference, and camera-specific full-frame bbox-area limits. Preview must show the mask overlay/contour, mask-derived box, confidence, candidate count, full-containment status, and ambiguity. Make the trigger zone large enough for the complete mask-derived box. Verify that partial entry/exit masks are detected but remain ineligible. Exclude light strips and enclosure edges from the ROI rather than relying on thresholds alone.

Derive nontrivial bbox-area limits from annotated real detections and false positives for one fixed camera geometry. Recompute them after changing resolution, camera pose, lens, or zoom. Add empty-belt, stain, glare, reflection, and conveyor-mark frames as hard negatives with zero glove annotations; do not create distractor classes. Physical filtering is a deterministic safeguard, not a replacement for hard-negative detector retraining.

Keep the classical detector only as a measured bootstrap baseline. Do not use generic COCO weights as a glove detector. For live deployment start from `configs/production.yaml`; preserve normalized full-frame ROI/trigger coordinates even though YOLO receives only the ROI crop.

## Phase 2 — passage ground truth

Annotate a stratified subset with:

- physical passage ID;
- entry, central-selection, and exit frames;
- target glove identity and one instance mask per visible glove, including partial gloves;
- detector class `0 = glove` only—never chirality;
- left/right chirality label for downstream classification;
- complete, clipped, merged, occluded, ambiguous, or invalid status;
- source/session grouping metadata fixed before sampling.

Include negative footage and difficult multi-glove cases. Double-annotate a subset and adjudicate disagreements without model predictions.

## Phase 3 — extraction acceptance

Run extraction without classifier tuning. Audit both `manifest.csv` and `event_report.csv`, then match accepted and rejected outcomes to annotations. Report per video:

- YOLO box/mask recall for fully visible and partial gloves plus false positives per empty hour;
- accepted crops / physical passages;
- passage precision, recall, and F1;
- duplicate rate—target zero by construction;
- false events and misses per hour;
- clipped, blurred, merged, and ambiguous crop rates;
- correctness/counts for partial, multiple-candidate, track-lost, insufficient-confirmation, low-sharpness, and end-of-stream audit outcomes;
- representative-frame quality and mask stability/clearance distributions;
- determinism across identical reruns;
- throughput on deployment hardware.

Initial engineering targets are 0.98 passage precision and recall, no duplicate emissions, and no silent decode failures. These targets must be revised according to product risk and data evidence.

If touching gloves are common, do not hide failures by relaxing thresholds. Add spacing control, ambiguity rejection, or an instance-aware backend.

## Phase 4 — split and leakage audit

Group by the highest correlated unit—prefer session or continuous parent recording over file when clips share conditions. Assert no source/session/glove-lot overlap across train, validation, and test.

Run leakage diagnostics:

- nuisance metadata-only baseline;
- background-only or glove-masked baseline;
- shuffled-label control;
- random-event split versus grouped split comparison;
- exact and perceptual duplicate checks across splits.

A large random-vs-grouped gap indicates session memorization. Report grouped results only.

## Phase 5 — classifier comparison

Use identical source/session groups, sampled passages, seeds, and evaluation code for all comparisons. First compare `crop_mode: bbox`, `masked`, and `masked_fill` with a frozen detector/extractor; do not assume mask suppression helps. Keep `bbox` as the production default until held-out evidence supports a change. Then compare TinyCNN and ResNet-18 baselines, followed by MobileNet/ViT if justified. GPU resources should be used when they improve experiment scale or turnaround.

Track:

- balanced accuracy and macro-F1;
- per-class precision, recall, and F1;
- confusion matrix;
- per-video/session metrics;
- confidence calibration and reject/coverage curves;
- clustered confidence intervals resampled by source/session;
- training time, inference latency, memory, and checkpoint size.

Do not select a model from aggregate accuracy alone. With three right videos, uncertainty and source sensitivity are central results.

## Phase 6 — end-to-end evaluation

Evaluate the frozen extractor plus frozen classifier on the locked test set. Count missed, extra, rejected-required, and wrong-target events as system failures. Report both classifier-isolated performance on ground-truth crops and end-to-end correct decisions over required physical passages.

Verify offline/deployment parity for event IDs, source frames, full-frame mask/box coordinates, crop arrays, preprocessing tensors, config hash, and model metadata. Then run the same frozen config with `infer-live` on the target camera. Use `timing_mode: time`, queue size 1–2, and detection frequency 1 initially. Report capture/processed FPS, detector/event/classifier latency, accepted-event latency, dropped frames, accepted/rejected passages, and decision timing. Do not tune away misses or ambiguity merely to improve FPS.

```bash
glove-pipeline infer-live --source 0 --checkpoint CHECKPOINT.pt \
  --config configs/production.yaml --device cuda --amp \
  --output outputs/live_events.jsonl
```

## Phase 7 — release evidence

Retain:

- source and split manifests with checksums;
- annotation guide and adjudication record;
- extractor YAML and hash;
- code commit and environment/package versions;
- model checkpoint hash and training configuration;
- passage-level, classifier-isolated, and end-to-end reports;
- subgroup metrics and confidence intervals;
- failure overlays and known limitations.

See `VERIFICATION_CHECKLIST.md` for the comprehensive checklist and initial numerical gates.

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

Tune ROI and trigger-zone geometry first, then `color_distance_threshold` and motion-background settings across every glove color. Make the trigger zone large enough for the complete glove bounding box, and verify that partial entry/exit frames are rejected while each chosen representative box remains wholly inside it. Exclude light strips, enclosure borders, and areas where incomplete distractor gloves dominate. Include gloves close to the belt color explicitly; if RGB contrast is physically insufficient, test a custom detector or change the belt/background rather than introducing color-specific thresholds.

## Phase 2 — passage ground truth

Annotate a stratified subset with:

- physical passage ID;
- entry, central-selection, and exit frames;
- target glove identity;
- left/right label;
- complete, clipped, merged, occluded, ambiguous, or invalid status.

Include negative footage and difficult multi-glove cases. Double-annotate a subset and adjudicate disagreements without model predictions.

## Phase 3 — extraction acceptance

Run extraction without classifier tuning. Match extracted events to annotations and report per video:

- accepted crops / physical passages;
- passage precision, recall, and F1;
- duplicate rate—target zero by construction;
- false events and misses per hour;
- clipped, blurred, merged, and ambiguous crop rates;
- representative-frame quality;
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

Use identical crops, split manifests, seeds, and evaluation code for all models. Start with TinyCNN and ResNet-18 baselines, then MobileNet and ViT if justified. GPU resources should be used when they improve experiment scale or turnaround.

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

Evaluate the frozen extractor plus frozen classifier on the locked test set. Count missed, extra, and wrong-target events as system failures. Report both classifier-isolated performance on ground-truth crops and end-to-end correct decisions over required physical passages.

Verify offline/deployment parity for event IDs, source frames, crop coordinates, crop arrays, preprocessing tensors, config hash, and model metadata.

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

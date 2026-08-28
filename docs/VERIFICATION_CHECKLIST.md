# Verification and Testing Checklist — Python Glove Chirality Pipeline

## 1. Scope and non-negotiable contracts

**Pipeline:** fixed-camera video (1920×1080, 25 FPS) → event detection/extraction → event crops + manifest for training; the **same event extraction implementation and configuration** → classifier inference for deployment.

Use this checklist before real videos are available to verify correctness, reproducibility, robustness, and leakage controls. Synthetic tests can establish software correctness, but **cannot establish real-world chirality accuracy**.

### 1.1 Freeze explicit contracts

- [ ] Define chirality labels and integer mapping in one versioned source (for example, `LEFT=0`, `RIGHT=1`, optionally `UNKNOWN/REJECT`).
- [ ] State whether labels describe the glove itself, the wearer’s hand, or image-side location. Never infer chirality from left/right image position.
- [ ] Define an event precisely: start rule, end rule, debounce/hysteresis, minimum/maximum duration, temporal padding, and behavior when a video starts or ends mid-event.
- [ ] Define the fixed ROI in full-frame pixel coordinates and its coordinate convention: `[x0, y0, x1, y1)` is recommended.
- [ ] Define how multiple visible gloves are handled: simultaneous/touching trigger candidates are ambiguous and rejected unless a validated instance-aware tracker is introduced; non-triggering gloves must not change event identity.
- [ ] Define crop policy: per-frame or representative frame, crop dimensions/aspect ratio, padding, boundary clipping, resizing, color order, interpolation, and normalization.
- [ ] Define manifest schema, required fields, data types, null policy, path convention, and schema version.
- [ ] Define stable identifiers: `source_video_id`, `recording_session_id`, `event_id`, `track_id` (if applicable), `frame_index`, and `extractor_version/config_hash`.
- [ ] Define timestamps from integer frame index (`frame_index / 25`) rather than accumulated floating-point increments.
- [ ] Define failure behavior for corrupt frames, unsupported resolution/FPS, missing files, empty ROI, no events, and multiple simultaneous ROI candidates.
- [ ] Store the extractor config and its cryptographic hash in both training and deployment outputs.
- [ ] Make the training exporter and deployment path call one shared extractor package/API; prohibit copied or independently reimplemented extraction logic.

### 1.2 Suggested manifest minimum

- [ ] Include: schema version, source video ID/path or content hash, session/group ID, event ID, frame start/end (inclusive/exclusive convention), representative frame(s), timestamps, original resolution/FPS, ROI, crop box(es), crop path(s), chirality label (training only), label provenance, track/object ID, extractor version/config hash, and quality/error flags.
- [ ] Validate uniqueness of `(source_video_id, event_id)` and `(source_video_id, frame_index, crop/track_id)` where applicable.
- [ ] Validate ranges: frame indices nonnegative and ordered; crop boxes inside the frame after clipping; timestamps consistent with 25 FPS.
- [ ] Keep unlabeled, rejected, and ambiguous events explicit rather than silently mapping them to left or right.

---

## 2. Deterministic synthetic-video test fixture

Build a seeded generator that writes actual videos readable by the production decoder, plus exact frame-level ground truth in JSON/CSV. Use simple textured glove proxies with an asymmetric marker (for example, a thumb-like protrusion or `L`/`R` glyph) so chirality survives motion and can be checked visually. Do not use synthetic classifier scores to test image decoding or event extraction.

### 2.1 Generator requirements

- [ ] Output 1920×1080, 25 FPS video using at least the production-intended container/codec and one lossless or near-lossless test codec when feasible.
- [ ] Record seed, codec, frame count, object trajectories, ROI contacts, chirality, object IDs, occlusion intervals, and expected events.
- [ ] Generate integer-frame event boundaries and independent expected results; do not calculate oracle events by calling production extraction code.
- [ ] Add a frame counter/timecode and object ID overlay outside model crops for debugging.
- [ ] Verify generated video by reopening it and checking decoded frame count, dimensions, FPS metadata/tolerance, and selected pixel landmarks.
- [ ] Make fixtures deterministic: same seed/config produces identical ground-truth metadata and identical decoded behavior (or a documented codec-tolerant image comparison).

### 2.2 Synthetic scenario matrix

Run every scenario for both left and right labels, entry from multiple directions, and at least three seeds unless the case is deterministic by construction.

#### Baseline and boundaries

- [ ] No glove anywhere: zero events and zero crops.
- [ ] Glove visible but never touching/entering ROI: zero events.
- [ ] One glove enters, dwells, and exits ROI: exactly one event with expected frame bounds within the configured tolerance.
- [ ] Event begins on frame 0; extractor clips pre-padding without negative indices.
- [ ] Event ends on the final frame; extractor flushes the event and clips post-padding.
- [ ] Contact for exactly minimum duration: accepted according to the documented inclusive/exclusive rule.
- [ ] Contact one frame shorter than minimum: rejected.
- [ ] Gap exactly equal to debounce/merge threshold and one frame above/below it: merge/split behavior matches specification.
- [ ] Glove touches each ROI edge/corner by 0 pixels, 1 pixel, and the configured overlap threshold.
- [ ] Crop touches each full-frame border; padding is clipped and output shape remains valid.
- [ ] Very short and very long videos; long events exercise maximum duration/splitting policy.

#### Multiple gloves and identity

- [ ] Two gloves visible, only one enters ROI: event/crop follows the entering glove, not the larger or more central distractor.
- [ ] Distractor glove passes near but outside ROI while target enters: one target event only.
- [ ] Left and right gloves enter sequentially: two ordered events with correct object IDs and no crop mixing.
- [ ] Two gloves overlap visually outside ROI, then one enters: target identity remains stable.
- [ ] Two gloves cross trajectories near ROI: no identity switch or mixed event crop.
- [ ] Two gloves enter ROI simultaneously: deterministic documented behavior (two events, one selected by a stated rule, or ambiguous/rejected); never label silently by image side.
- [ ] One glove occludes another at the ROI: extractor follows the documented occlusion/ambiguity policy.
- [ ] Static glove remains in/near ROI while another glove triggers: no repeated phantom events from the static glove.

#### Appearance, motion, and acquisition stress

- [ ] Vary glove proxy scale, rotation, perspective-like deformation, brightness, contrast, color, texture, and background clutter.
- [ ] Vary motion from subpixel/slow to fast enough for motion blur; include direction reversals and pauses at the boundary.
- [ ] Add compression artifacts, sensor-like noise, blur, shadows, illumination flicker, and partial occlusion.
- [ ] Add dropped/duplicated/corrupt-frame fixtures if deployment decoding may encounter them; verify explicit flags or controlled failure.
- [ ] Supply wrong resolution, wrong FPS, variable-frame-rate metadata, empty video, truncated video, and unreadable file; reject or normalize only as documented.
- [ ] Test non-square-pixel or rotated metadata if the chosen decoder honors these fields.

### 2.3 Synthetic oracle checks

For every fixture:

- [ ] Compare event count, order, target object ID, start/end frame, and crop box against independent ground truth.
- [ ] Check crop contents contain the expected target marker and do not contain only a distractor/background.
- [ ] Check all expected manifest rows and exact schema/types.
- [ ] Check saved crops can be reopened and have expected shape/channel order.
- [ ] Check rerunning with the same input/config is deterministic: identical manifest after normalizing run timestamps and identical crop hashes for deterministic codecs.
- [ ] Save diagnostic overlays for failures: ROI, boxes, object/track ID, event state, and frame number.

---

## 3. Unit-test checklist

Use `pytest`; parameterize frame-boundary cases and seed random/property tests.

### 3.1 Geometry and ROI

- [ ] YOLO ROI-only inference receives the intended pixel crop and restores every box/polygon vertex to full-frame coordinates before trigger gating or persistence.
- [ ] Strict-mask mode accepts empty segmentation results but rejects box-only, missing, mismatched, non-finite, and degenerate polygons without silent fallback.
- [ ] Classical and intentionally box-only backends continue to emit `polygon=None` and preserve bbox behavior.
- [ ] Point/box/polygon intersection at all edges and corners follows one documented convention.
- [ ] Overlap/IoU calculations are correct for disjoint, contained, equal, zero-area, and partially clipped boxes.
- [ ] Crop padding and clipping never produce negative indices, empty crops, wraparound, or inconsistent shapes.
- [ ] Coordinate transforms full frame → ROI → crop → resized tensor are invertible within rounding tolerance.
- [ ] Multiple-candidate selection is deterministic under ties and independent of input iteration order.

### 3.2 Temporal event state machine

- [ ] Frame-count and monotonic-time timing modes exercise equivalent passage sequences, including skipped frame indices and irregular intervals.
- [ ] Rearming requires continuous empty evidence; detections during cooldown reset it and cannot create a duplicate accepted/rejected outcome for the same occupied passage.
- [ ] Timestamp regression fails clearly rather than corrupting time-based state.
- [ ] Idle → candidate → active → closing → idle transitions are tested frame by frame.
- [ ] Minimum dwell, hysteresis, debounce, merge gap, cooldown, and max-duration thresholds test `N-1`, `N`, and `N+1` frames.
- [ ] EOF flush and start-of-file active state behave as specified.
- [ ] Frame index/timestamp conversion is exact at representative indices, including long recordings; no cumulative float drift.
- [ ] Missing/duplicate frame policy is deterministic and emits quality flags.

### 3.3 Crop and preprocessing

- [ ] `bbox`, `masked`, and `masked_fill` share identical crop bounds and letterboxing; mask modes alter only outside-polygon pixels.
- [ ] Missing polygons fail clearly in mask-required crop modes, and deterministic median fill reproduces identical arrays.
- [ ] BGR/RGB conversion, resizing interpolation, dtype, value range, channel layout, mean/std normalization, and batch dimension match the model contract.
- [ ] Representative-frame or temporal-sampling selection yields exact expected indices, including short events and padding.
- [ ] Training and inference preprocessing functions produce byte-identical or tolerance-equal tensors for the same crop/config.
- [ ] Horizontal flips are disabled unless chirality labels are swapped correctly; unit-test the relabel operation.
- [ ] Other geometric augmentations preserve label semantics and do not reveal padding/crop artifacts correlated with class.

### 3.4 Manifest/config/error handling

- [ ] Schema validator accepts a golden manifest and rejects missing fields, wrong types, invalid bounds, duplicate IDs, and unknown schema versions.
- [ ] Config serialization is canonical; semantically identical configs produce the same hash.
- [ ] Invalid ROI, FPS, resolution, codec/read failure, empty crop, and invalid label raise typed errors or emit documented reject records.
- [ ] Paths are portable/relative where intended and work with Windows path separators.
- [ ] Logging contains source/event IDs and reason codes but no nondeterministic values that contaminate golden comparisons.

### 3.5 Model adapter and decision logic

- [ ] Model input name/shape/dtype and output label order are validated at load time.
- [ ] Known fake logits/probabilities map to correct left/right labels.
- [ ] Softmax/sigmoid application is exactly once; threshold boundary behavior is tested.
- [ ] Low-confidence, tie, NaN, infinity, malformed output, and unknown-model-version cases reject safely.
- [ ] Batch and single-event inference return identical scores within numerical tolerance and preserve event ordering.

---

## 4. Integration and end-to-end tests

### 4.1 Training-export path

- [ ] Run synthetic video through the production decoder and extractor, save crops and manifest, and validate against the oracle.
- [ ] Reopen every emitted crop and validate path, dimensions, channels, and manifest linkage.
- [ ] Re-run into a clean directory: results are identical and no stale files affect output.
- [ ] Interrupted/partial runs are either atomic/resumable or fail clearly without a valid-looking partial manifest.
- [ ] Parallel processing does not change IDs, row order (after documented sort), event bounds, or crops.

### 4.2 Deployment path

- [ ] Bounded live capture never exceeds its configured queue, reports stale-frame drops, and classifies exactly once per accepted passage rather than once per frame.
- [ ] Live JSONL/event callbacks contain full-frame geometry, detector/config/model metadata, confidence, timing, and no prediction for rejected passages.
- [ ] Runtime-only display/reporting/queue/warm-up settings do not alter the extraction config hash; detection frequency and crop semantics do.
- [ ] Run the same synthetic video through deployment event extraction and a deterministic stub classifier.
- [ ] Assert deployment event IDs, start/end frames, selected frames, crop boxes, and preprocessed tensors match the training-export path event by event.
- [ ] Compare serialized extractor config hash and code/model versions in the inference record.
- [ ] Confirm deployment preserves event ordering, handles zero events, and emits one explicit result/reject per extracted event.
- [ ] Test streaming/chunked input with chunk boundaries before, inside, and after events; results match whole-file extraction after finalization.
- [ ] Test process restart/state reset so one video/stream cannot leak active-event state into another.

### 4.3 Golden parity test — required CI gate

Create one shared test that executes both paths on the same fixture and compares a canonical `ExtractedEvent` representation **before file encoding**:

- [ ] Same event count and IDs.
- [ ] Exact same source frame indices and event boundaries.
- [ ] Exact same target/track selection and crop coordinates.
- [ ] Exact same raw crop arrays.
- [ ] Exact same preprocessed model tensors (or an explicitly justified numeric tolerance).
- [ ] Same extractor/config hash.

Any mismatch fails CI. Comparing only final predicted labels is insufficient.

### 4.4 Packaging/environment

- [ ] Test supported Python and dependency versions in CI.
- [ ] Pin decoder, image library, array library, and inference-runtime versions; record them in artifacts.
- [ ] Verify CPU deployment and any intended accelerator path produce scores within a declared tolerance and identical final decisions away from threshold boundaries.
- [ ] Exercise installation in a clean environment and load a versioned model/config without relying on developer-machine paths.

---

## 5. Data leakage controls

### 5.1 Split unit and provenance

- [ ] Split by the highest correlated unit: normally `recording_session_id` or `source_video_id`; if the same person, glove pair, workstation/background, or continuous recording appears across files, group those together too.
- [ ] Never randomly split frames, crops, or events from the same source recording across train/validation/test.
- [ ] Perform group assignment **before** crop extraction/augmentation, then inherit the split in every derivative artifact.
- [ ] Keep the final test group locked and untouched by model choice, threshold selection, calibration, feature engineering, or error-driven retraining.
- [ ] If future deployment targets unseen people, glove instances, lots, dates, or sites, include those IDs in grouping and hold out entire groups accordingly.

### 5.2 Leakage assertions

- [ ] Assert disjoint group IDs across splits for source videos, sessions, people/operators, glove instances/lots, dates/batches, and sites/cameras where available.
- [ ] Store source content hash and detect duplicate or renamed videos across splits.
- [ ] Detect exact duplicate crop hashes across splits.
- [ ] Detect near-duplicates using perceptual hashes/embedding similarity; manually review high-similarity cross-split pairs.
- [ ] Assert no overlapping frame intervals from the same continuous recording appear in different splits, including clips exported from a longer parent recording.
- [ ] Fit normalization statistics, feature transforms, class weighting, oversampling, calibration, and decision thresholds on training data (or training plus a designated calibration split) only—never test.
- [ ] Apply augmentation only after splitting. Derivatives inherit the source split and source/group IDs.
- [ ] For horizontal flip augmentation, swap chirality labels and verify pixel/label pairs; otherwise prohibit flips.
- [ ] Ensure filenames, directory names, overlays, timestamps, crop padding styles, synthetic markers, and manifest-only fields cannot encode labels into model inputs.
- [ ] Strip debug glyphs/object IDs from classifier crops; synthetic text markers are for extractor verification, not classifier training/evaluation.
- [ ] Keep synthetic variants from the same base scene/seed family in one split to avoid template leakage.

### 5.3 Leakage-negative-control tests

- [ ] Train a simple classifier using only nuisance metadata (filename tokens, crop dimensions/position, event duration, source ID, timestamp, background summary). Near-chance validation performance is expected; strong performance triggers investigation.
- [ ] Train/evaluate with shuffled labels: performance should collapse to chance within statistical uncertainty.
- [ ] Evaluate a background-only or target-masked crop baseline; unexpectedly high performance indicates scene/position leakage.
- [ ] Compare random-event split versus grouped split. A large drop under grouped splitting is evidence that earlier results were optimistic; report grouped results only.
- [ ] Verify left/right class counts, nuisance distributions, and rejected-event rates by split; document unavoidable imbalances rather than moving correlated samples across groups.

---

## 6. CI test tiers and initial pass/fail gates (before real video)

### 6.1 Per-commit fast tier

- [ ] All unit tests pass.
- [ ] Lint/type/schema checks pass.
- [ ] Small synthetic baseline/boundary suite passes.
- [ ] Golden training/deployment parity test passes exactly.
- [ ] Fixed-seed rerun is deterministic.

### 6.2 Nightly/full tier

- [ ] Full synthetic scenario matrix passes across declared seeds/codecs.
- [ ] No missed or extra events in deterministic clean fixtures.
- [ ] Boundary error is within the written event policy (recommended initial gate: exact frame agreement for clean generated fixtures; otherwise at most ±1 frame only if decoder behavior justifies it).
- [ ] Target identity/crop correctness is 100% on deterministic clean multi-glove fixtures, including crossings and distractors; explicitly ambiguous simultaneous-entry cases follow the reject policy 100%.
- [ ] Crop and manifest validation has zero errors.
- [ ] Training/deployment extraction parity is 100% at event/crop/tensor level.
- [ ] Corrupt/unsupported inputs produce expected controlled errors/rejects, never crashes that leave valid-looking partial outputs.
- [ ] All leakage assertions pass with zero cross-split source/session duplicates or interval overlap.
- [ ] Performance and memory are measured on a defined reference machine; regressions over an agreed budget (for example >10%) fail only after a stable baseline is established.

### 6.3 Property/fuzz and regression tier

- [ ] Property-based geometry/state-machine tests run hundreds of valid and invalid cases without crashes or invariant violations.
- [ ] Randomized multi-object trajectories preserve invariants: event bounds ordered, boxes valid, IDs unique, output deterministic.
- [ ] Every fixed defect receives a minimal regression fixture and test.
- [ ] Retain failed random seeds for deterministic reproduction.

---

## 7. Real-video calibration plan and acceptance criteria

Do not claim production readiness from synthetic results. Before collection, write a sampling protocol and freeze the intended metrics/gates. Numerical gates below are sensible **initial engineering gates**, not substitutes for product risk requirements.

### 7.1 Real-video collection and annotation readiness

- [ ] Collect representative fixed-camera 1920×1080, 25 FPS recordings spanning both chiralities, operators, glove instances/lots, working speeds, entry directions, rotations, lighting, backgrounds, occlusion, clutter, and multiple-glove situations.
- [ ] Include negative footage (no ROI event), near misses, static gloves, partial entries, simultaneous entries, crossings, and hard/ambiguous cases.
- [ ] Preserve continuous session/source IDs and provenance needed for grouped splits.
- [ ] Predefine train, calibration/validation, and locked test groups before model tuning.
- [ ] Use a written annotation guide for event boundaries, target identity, chirality, and ambiguity/reject labels.
- [ ] Double-annotate a meaningful stratified subset; adjudicate disagreements without showing model predictions.
- [ ] Measure annotation agreement separately for chirality, event presence, boundaries, and target identity.

### 7.2 Dataset acceptance before tuning

- [ ] Decoder confirms required resolution/FPS or records explicit normalization/rejection.
- [ ] Every annotation links to an existing source/session and valid frame interval.
- [ ] No train/calibration/test group or duplicate leakage under Section 5 checks.
- [ ] Each locked-test subgroup has enough support to make its confidence interval useful; report counts and confidence intervals, not only percentages.
- [ ] Both chiralities and safety-critical conditions are represented in each evaluation split; do not fabricate balance by splitting one session across sets.
- [ ] Ambiguous cases remain labeled/rejected rather than forced to a binary ground truth.

### 7.3 Event extraction metrics and initial acceptance gates

Evaluate extraction independently of classification by matching predicted and annotated events with a predefined temporal-IoU rule (recommended starting point: IoU ≥ 0.5, plus target-ID agreement when multiple gloves are present).

- [ ] Report event precision, recall, F1, false events per hour, missed events per hour, boundary start/end error in frames, target-selection accuracy, ambiguous/reject rate, and processing failures.
- [ ] Report all metrics overall and by chirality, operator, glove instance/lot, lighting, motion-speed bucket, occlusion, entry direction, and single- vs multi-glove scene.
- [ ] **Initial gate:** event precision ≥ 0.98 and recall ≥ 0.98 on the locked test set, with two-sided 95% confidence intervals reported.
- [ ] **Initial gate:** false positives ≤ 1 per hour of negative/operational video and no silent decoder failures.
- [ ] **Initial gate:** median absolute boundary error ≤ 1 frame and 95th percentile ≤ 3 frames, unless downstream crop design demonstrates a different justified tolerance.
- [ ] **Initial gate:** target-selection accuracy ≥ 0.99 on matched multi-glove events; ambiguous simultaneous cases must be rejected according to policy, not guessed.
- [ ] Require each safety-critical subgroup to meet a separately approved floor; an aggregate pass cannot hide a failing subgroup.

### 7.4 Classifier and end-to-end metrics and initial gates

Evaluate the classifier first on ground-truth event crops (classification isolation), then on predicted extractor crops (true end-to-end performance).

- [ ] Report confusion matrix, per-class precision/recall/F1, balanced accuracy, ROC-AUC/PR-AUC where meaningful, log loss/Brier score, calibration curve/ECE, confidence coverage, and reject rate.
- [ ] Report event-level end-to-end accuracy only for correctly matched events, plus a system metric that counts missed/extra/wrong-target events as failures.
- [ ] **Initial gate:** balanced chirality accuracy ≥ 0.98 and each class recall ≥ 0.97 on locked test events after calibration.
- [ ] **Initial gate:** end-to-end correct-decision rate ≥ 0.95 across all annotated required events, counting extraction failures as incorrect unless explicitly rejected under an approved policy.
- [ ] **Initial gate:** among auto-accepted predictions, error rate ≤ 1% at a predeclared minimum coverage (recommended starting point: ≥ 90%); rejected/ambiguous events route to the approved fallback.
- [ ] **Initial gate:** no evaluated subgroup is more than 5 percentage points below overall balanced accuracy, or the subgroup has an explicit mitigation and release waiver.
- [ ] Select confidence thresholds and probability calibration using training plus designated calibration data only; evaluate once on locked test.
- [ ] Use bootstrap or appropriate clustered confidence intervals resampled by session/source, not by highly correlated event/frame alone.

### 7.5 Operational and parity acceptance

- [ ] Byte/tolerance-level extraction/preprocessing parity between offline exporter and deployed package is rechecked on sampled real events.
- [ ] Model, label map, preprocessing, ROI/extractor config, schema, and dependency versions are bundled and recorded for every run.
- [ ] Measure throughput and latency on deployment hardware with multi-glove scenes and long recordings; meet a product-defined budget with headroom.
- [ ] Long soak test completes without unbounded memory growth, state leakage between events/videos, crop/manifest loss, or ID collisions.
- [ ] Human-review overlays confirm that errors are assigned to extraction, target selection, classification, annotation, or ambiguity—not grouped into a single accuracy number.
- [ ] Define rollback, monitoring, and drift triggers (event-rate shift, reject/confidence shift, class-ratio shift, decoder errors, lighting/background changes).

### 7.6 Release decision

Release only if:

- [ ] All synthetic/CI gates pass.
- [ ] Locked real-video test data remains untouched until the final candidate is frozen.
- [ ] Extraction, classifier-isolated, and end-to-end gates all pass with uncertainty and subgroup results reported.
- [ ] No unresolved leakage, label-definition, target-identity, or train/deployment-parity issue remains.
- [ ] Every failure mode has an explicit behavior: correct prediction, rejection/fallback, or controlled error.
- [ ] A signed evaluation report records dataset version, split manifest, source/group hashes, code commit, environment, extractor/config hash, model hash, thresholds, metrics, confidence intervals, and known limitations.

---

## 8. Evidence bundle to retain for every verification run

- [ ] Test report (JUnit/pytest), coverage report, and CI environment/dependency lock.
- [ ] Synthetic generator config/seeds and oracle metadata.
- [ ] Canonical manifests, representative crops, and diagnostic overlays.
- [ ] Training/deployment parity comparison and tensor/hash evidence.
- [ ] Split manifest and leakage audit report.
- [ ] Model/config/label-map hashes and code commit.
- [ ] Real-data annotation guide, adjudication log, frozen evaluation protocol, subgroup metrics, and release report when available.

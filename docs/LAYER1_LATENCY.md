# Layer 1 latency benchmark

Measure the configured YOLO segmentation adapter plus the shared passage processor,
including mask association, duplicate-mask handling, representative-frame selection,
mask isolation and canonical crop creation. No classifier or actuator is invoked.

```bash
python tools/benchmark_layer1_latency.py \
  --video /path/to/input.mkv \
  --model /path/to/black_best.pt \
  --config /path/to/black.yaml \
  --output outputs/layer1-latency/fp32 --device cuda
```

Add `--half` and use a separate output directory for an explicitly labeled FP16
comparison. Requires the `ml` and `yolo` extras and a working CUDA PyTorch build.
The counter mode is explicitly `line`; other extractor settings are taken from the
supplied configuration, not retuned against an expected passage count.

## Measurement boundary

- Warm up the detector with 16 untimed inferences by default.
- Synchronize CUDA before the core timer, after detection, and after passage processing.
- Time the complete YOLO adapter, including preprocessing and mask postprocessing.
- Time passage processing, including **crop creation**, separately; add both stages
  for Layer 1 latency. Counter/crop time is not crop-only time.
- Exclude downloads, decoder work, PNG writes, report writes, and overlays from the
  core timer. Decoder latency is reported independently.
- Start a fresh passage processor for each measured precision mode. Detector warmup
  does not pretend to prewarm the counter's first CPU allocations.
- Use source presentation timestamps for event processing. Reject absent/non-increasing
  timing instead of silently manufacturing frame times from misleading container FPS.
- Report decoded versus advertised frame counts. EOF does not certify source completeness
  when the container advertises additional frames.
- Export every frame's timings, mask count, emitted crops and running count, alongside
  distributions for all frames, mask-present frames, no-mask frames and crop-emission
  frames. Sparse emission-frame percentiles are descriptive, not reliable tail estimates.
- Save hashes of the input, model, original configuration, counter and shared event/crop source.
- `--save-detections` includes the actual per-frame detection geometry in the timing
  JSONL, serialized **after** the core timer. This supports deterministic counter
  replay without repeating YOLO inference. It is disabled by default.
- Save rejected EOF outcomes and counter lifecycle diagnostics; do not silently
  omit incomplete source-boundary passages from the accuracy audit.

## Iterative Layer 1 acceptance

The objective is one usable, mask-isolated crop per glove for Layer 2 while keeping
Layer 1 at camera rate. Cumulative counts are audit diagnostics, not a substitute
for inspecting the emitted crops or measuring missed/duplicate passages.

- Verify the four- and nineteen-glove clips with their matching black checkpoint
  and configuration after meaningful changes. Approximately every fifth iteration,
  and before final acceptance, also verify the fifty-five-glove clip with its
  matching **green** checkpoint and configuration.
- Confirm source hashes against independently verified local clips. Different
  filenames or Drive identifiers do not establish different video contents.
- Keep weights, ROI, trigger geometry and detection thresholds fixed for a paired
  implementation comparison. Do not tune them to obtain an expected count.
- In line mode, retain one nearest valid raw candidate, isolate it at emission,
  and release it after use. No sharpness ranking is performed; outcome
  `quality_score` reports the selected detection confidence. Instance masks make
  preliminary foreign-glove suppression redundant; bbox-only backends retain it.
- Compare FP32 and FP16 with fresh detectors/processors. Audit passage identities,
  crossing times and crop contents as well as counts. Agreement with FP32 is not
  annotated mask accuracy or chirality-classifier accuracy.
- Audit startup/EOF gloves separately: an already-past-line startup glove or an
  EOF glove that never crosses is not an observed crossing. Do not seed counts or
  silently change this policy to make the total match.
- Inspect mask-present and emission-frame latency, not just the empty-belt average.
  Exceeding the average source interval is a service-time warning, not proof of a
  frame drop. Recorded-video throughput does not certify the bounded live queue or
  Layer 2 concurrency on the target workstation.

## Initial exploratory result (historical)

Private inference on a **Tesla T4**, PyTorch **2.11.0+cu128**, CUDA **12.8**, with
**two CPU threads**. External black-glove configuration: YOLO image size 640,
ROI `(0.16, 0.04, 0.96, 0.99)`, trigger zone `(0.17, 0.17, 0.96, 0.84)`, bottom-to-top
crossing, bbox crop resized to 256, gray fill 114. This is **not a hardware acceptance
result for the workstation's factory configuration**.

Input SHA-256:
`c3ff97440bee02fc50ba49e472c9714f1cb17e100800935157bd86b4060aa5b4`.
The supplied MKV advertised 1,500 frames/60 seconds, but only 1,299 frames through
46.108 seconds decoded. All recoverable frames were measured. Their average source
interval was **35.522 ms**, approximately **28.15 FPS**; the 25 FPS header was misleading.
Both modes emitted four crops at the four temporally separated passages.

| Layer 1 group | Samples | FP32 mean | FP32 median | FP32 p95 | FP32 p99 | FP32 maximum |
|---|---:|---:|---:|---:|---:|---:|
| All frames | 1299 | 20.872 ms | 20.013 ms | 26.005 ms | 40.148 ms | 78.736 ms |
| Mask-present frames | 193 | 27.263 ms | 24.418 ms | 40.554 ms | 48.378 ms | 78.736 ms |
| Crop-emission frames | 4 | 44.156 ms | 45.416 ms | 50.101 ms | 50.369 ms | 50.437 ms |

Across all FP32 frames, detector time averaged **20.194 ms**, and counter/crop work
averaged **0.678 ms**. On the four crop-emission frames, those stages averaged
**24.589 ms + 19.567 ms = 44.156 ms**. Overall averages are dominated by empty-belt
frames and must not stand in for crop-emission latency.

FP16 did not improve this run: mean core latency **22.480 ms**, p95 **27.878 ms**,
p99 **41.745 ms**, with mean crop-emission latency **44.499 ms**. This sequential
single-run comparison is not proof that FP16 is universally slower.

FP32 exceeded the source interval on **29/1299 frames (2.23%)**. The inverse mean
core latency corresponds to **47.91 FPS**, but this excludes decoding and output work
and is **not a guarantee of live-camera throughput or zero frame drops**. Average
core plus decoder time was **30.085 ms** on this recording.

### Acceptance decision

Four-passage counting correctness is encouraging, and average latency has headroom.
However, crop-emission frames exceed the 35.522 ms source interval. Do **not** certify
strict per-frame camera-rate operation from these measurements. A continuous live
capture/queue test, denser passages and tail-latency improvements remain necessary.
Keep FP32 as the measured baseline; do not enable FP16 merely on assumed speed.
No real-world classification accuracy is established by this benchmark.

## Final paired FP32 simplification comparison

Private Kaggle notebook `chatukaelapatha/grip-layer1-optimization-four-and-nineteen`,
version **5**, on the Tesla T4 environment above. Both implementations use FP32,
fresh detectors/processors and 16 untimed detector warmups. Per-frame detection
archive capture is disabled. Input, checkpoint and original configuration hashes
match within every pair; optimized counter/event source hashes match the code
prepared for this change. Private source snapshots, timings, outcomes and crops
are retained with the notebook outputs, not committed to Git.

The baseline is the **pre-simplification crop path**, with sharpness/top-five
candidate buffering and per-candidate masking. Both variants share stabilized
observed-mask association and near-identical-mask suppression. This isolates crop
path changes; it is **not a comparison against ByteTrack or every behavior in the
previous Git revision**. No ROI, model or confidence threshold was retuned.

### Mean core latency (milliseconds; lower is better)

| Clip / matching weights | Observed crossings cropped, old / new | All frames, old / new | Mask-present frames, old / new | Crop-emission frames, old / new | Emission latency reduction |
|---|---:|---:|---:|---:|---:|
| Four / black | 4 / 4 | 15.32 / 15.00 | 20.82 / 18.18 | 35.12 / 26.28 | 25.2% |
| Nineteen / black | 19 / 19 | 19.21 / 17.72 | 22.72 / 20.31 | 31.97 / 23.58 | 26.2% |
| Fifty-five / green | 53 / 53 | 28.63 / 24.32 | 28.63 / 24.32 | 38.32 / 29.13 | 24.0% |

Counter/crop work on emission frames decreased from **16.32 to 8.20 ms**, **13.31
to 5.70 ms**, and **18.50 to 10.09 ms**, respectively: approximately **50%, 57%
and 45%** reductions. YOLO inference remains the dominant cost on most frames.

### Core latency tails and source cadence

| Clip | Core p95, old / new | Core maximum, old / new | Frames above source interval, old / new | Decoded frames |
|---|---:|---:|---:|---:|
| Four | 19.57 / 18.55 ms | 58.97 / 31.72 ms | 4 / 0 | 1299 |
| Nineteen | 30.77 / 23.16 ms | 40.56 / 30.87 ms | 13 / 0 | 1229 |
| Fifty-five | 39.74 / 29.34 ms | 54.08 / 44.15 ms | 224 / 2 | 1243 |

Average source intervals were **35.52, 35.56 and 35.54 ms**. These are core-only
service times, excluding decode and file writes; over-budget frames are not
measured camera drops. A sequential paired run is useful evidence, not a repeated
randomized performance study or a target-workstation live-queue acceptance test.
The earlier detection-archive-enabled run had an isolated counter-stage spike;
it did not recur here, but its cause is not established or claimed fixed.

### Crop and boundary-policy audit

- All **76 accepted passage pairs** have identical crossing timestamps. No extra
  accepted passage or omitted observed crossing appeared on these three clips.
- **72/76 crops** select the same source frame and are **pixel-identical** to the
  baseline. The remaining four select an adjacent frame; visual inspection found
  comparable single-glove crops rather than an obvious crop regression.
- All exported crops were readable, nonblank 256-by-256 images. Contact-sheet
  inspection found no complete second glove in an accepted crop. Minor green
  segmentation fragments remain in both paths; no perfect pixel-mask or chirality
  accuracy claim is made.
- Both green runs explicitly reject one `born_past_line` startup glove and one
  `lost_before_trigger` EOF glove. The approved policy requires an **observed
  directional crossing**: no startup/EOF fallback crops or seeded counts. These
  are still two rejections among 55 visible gloves, not 55 delivered crops.
- **FP16 experiments are paused; FP32 remains the factory default.** The optional
  benchmark flag is retained, but no FP16 setting is enabled by this change.

The factory template now selects the shared `line` path. The experiments used
their external black/green configurations, not the factory template's geometry.

# Three-stage live pipeline

The live app now uses three stages by default:

1. **Layer 1 — segmentation:** detector preprocessing, inference and full-frame
   detection/mask output preparation. No passage state or crop creation here.
2. **Layer 2 — passage extraction:** one ordered owner runs the shared
   `PassageProcessor` and canonical crop implementation, using original frame IDs
   and source timestamps. This is the session worker consuming the bounded
   detection FIFO, not a second independent tracker implementation.
3. **Layer 3 — chirality and output:** classify once per accepted passage and
   deliver accepted/rejected event records, crop previews and durable outputs.

A separate latest-only observer handles frame previews, metrics callbacks and
periodic console reports. Slow presentation can skip observer frames, not already
admitted passage records or tracking updates. Frame callbacks are observational:
use `event_callback` for reliable passage delivery. Diagnostics accompany the
matching detected frame; observers never consult the detector's newer mutable state.

## Configuration and compatibility

```yaml
runtime:
  stage_pipeline: true
  capture_queue_size: 2
  classifier_queue_size: 8
  warmup: true
```

`stage_pipeline: false` retains fused detection/passage processing for comparisons.
The source capture policy is unchanged: live capture drops stale captured frames;
sequential file playback is lossless. Detected packets have a two-item FIFO and are
not independently discarded. The bounded Layer-3 passage queue faults with the
specific event ID on saturation; this is explicit overload, not a silently missing
crop or unbounded memory growth. Stop/fault cancels further physical submissions;
commands already sent to firmware cannot be recalled. Threads/native calls have
cooperative shutdown, with join timeout treated as a fault.

Normal EOF/max-frame completion also inhibits hardware before final passage records
drain. `source_exhausted` retains the completion reason, so unexpected camera EOF
still faults while video EOF is normal. Intentionally skipped detection frames do
not age tracking as synthetic empty observations; staged/fused behavior is preserved.

FP32 remains the default. ROI, thresholds, observed-crossing boundary rejections,
frame selection, mask isolation and source-label-independent extraction are unchanged.
Classifier warmup exercises the complete in-memory prediction/preprocessing path
inside the classifier worker; detector warmup uses source-sized input in its worker.
Stage durations and presentation ordering markers use high-resolution monotonic
performance counters, avoiding coarse Windows clock ticks.

Legacy metric names remain. Additional diagnostics include `layer1_ms`,
`layer2_ms`, `layer3_ms`, `detection_queue_wait_ms`, `frame_age_ms`,
`detection_queue_depth` and `observer_frames_skipped`.

## Why the split was adopted

The approved objective is **first-worker occupancy**, not a claim that the YOLO
network itself runs faster. A private FP32 Tesla-T4 experiment compared fused
segmentation+passage processing against separate segmentation and ordered passage
workers. Each had independent classification/output and the same 10-Hz preview
policy. Two repeats per mode in ABBA order were run on each clip.

| Clip | Fused first-worker mean | Segmentation-only mean | Reduction | Fused p95 | Segmentation-only p95 |
|---|---:|---:|---:|---:|---:|
| 4 / black | 21.08 ms | 20.99 ms | 0.4% | 25.48 ms | 24.63 ms |
| 19 / black | 24.24 ms | 23.01 ms | 5.1% | 31.26 ms | 27.84 ms |
| 55 / green | 32.68 ms | 29.15 ms | 10.8% | 42.79 ms | 40.36 ms |

These compare differently allocated first-worker responsibilities; **YOLO-only
call time did not improve**. The classifier workload was real DINOv3 ConvNeXt-Tiny
inference with untrained timing-only weights, not production chirality validation.
Matching inputs and crop/event identities were independently audited: 4/4, 19/19,
53 green accepted crossings plus the two explicit startup/EOF rejections.

The benchmark was paced, lossless file replay at about 28.1 FPS, not a peak-capacity
or physical-camera drop test. Green segmentation p95 still exceeded the roughly
35.54-ms source interval; end-to-end decision latency did not show a general win.
Two repeats and no clock/thermal observation limit performance confidence. No
sustained factory real-time or actuator-deadline certification is implied.

Production regression coverage exercises ordered processing, slow frame/metrics
observers, frame-matched diagnostics, owning-worker warmup/output, explicit bounded
queue faults, error propagation and late-observer decision safety. Real camera,
trained-checkpoint and attended actuator acceptance remain target-device gates.

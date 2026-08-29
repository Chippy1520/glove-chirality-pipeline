# Camera and detector calibration

This workflow calibrates Layer 1 without changing passage or chirality logic. The utility uses `ExtractionConfig` and `YoloDetector`, so ROI cropping, segmentation, confidence, class filtering, coordinate restoration, and physical size filtering match deployment.

## Install the local detector artifact

Trained checkpoints are generated artifacts and remain excluded by `.gitignore`. Put the selected file at the path recorded by the camera config, for example:

```text
checkpoints/yolo11n_seg_glove_final_v2.pt
```

Complete `models/yolo11n_seg_glove_final_v2.yaml` with the dataset version, training command/results, selected epoch, validation metrics, and SHA-256. Do not mark the record complete until those values are measured.

## Current-camera calibration

`configs/grip_aug27_seed.yaml` preserves the historical setup. Do not overwrite it. Start current-camera work from `configs/grip_current_camera.yaml`, which is intentionally uncalibrated: its ROI spans the full frame and its `0.0–1.0` size range disables physical rejection until measurements are collected.

Run:

```bash
python scripts/realtime_detector_calibration.py \
  --camera 0 \
  --config configs/grip_current_camera.yaml
```

On Windows, camera opening tries DirectShow, Media Foundation, and OpenCV's default backend. A backend succeeds only after an actual frame is read. Use `--backend "Media Foundation"` (or another listed choice) to change the first attempt without disabling fallback. The utility reports the successful backend, actual frame dimensions, and camera-reported FPS. It does not force resolution, FPS, or FOURCC.

Controls:

- **Q** or **Esc** — quit;
- **S** — save a timestamped screenshot under `outputs/calibration/`;
- **F** — enable/disable physical size filtering for calibration only;
- **R** — show/hide size-rejected boxes.

Green boxes are returned glove detections. Red `REJECT SIZE` boxes are raw YOLO detections rejected by the configured physical gate. The overlay reports raw, size-rejected, and returned counts plus bbox area ratio and confidence. Rejected diagnostics never enter `PassageProcessor` during normal extraction or live inference.

## Recalibrate physical size limits

Remeasure `yolo_min_box_area_ratio` and `yolo_max_box_area_ratio` after changing any of:

- camera height or pose;
- lens or zoom;
- capture resolution;
- conveyor geometry;
- ROI/camera alignment.

Procedure:

1. Collect representative complete-glove detections across orientation, folding, color, and passage position.
2. Record full-frame bbox-area ratios—not ROI-relative ratios.
3. Collect false-positive ratios from empty belt, stains, patches, glare, and reflections.
4. Choose conservative min/max margins between physically implausible artifacts and plausible gloves.
5. Verify on held-out recorded footage and the live camera before enabling the limits in deployment.
6. Keep true simultaneous eligible gloves ambiguous; do not weaken `reject_multiple_detections` to make calibration appear successful.

## Hard-negative detector training

For a frame containing only a patch, stain, glare, reflection, conveyor mark, or empty belt, create **zero glove annotations**. For a frame containing real gloves and distractors, annotate **every real glove** and leave distractors unannotated. Do not create patch, background, glare, or stain classes. Preserve source-video/session grouping across detector train/validation splits.

The final safeguard combines hard-negative training with the camera-specific physical gate. Neither replaces the other.

## Versioning

Never overwrite historical configurations or extraction directories. Prefer names such as:

```text
configs/grip_2026_08_29_camera_a.yaml
outputs/aug27_seed_bbox_v1/
outputs/aug27_triggeraware_sizefilter_v2/
outputs/final_detector_current_camera_v1/
```

Record the expected camera config in detector provenance. This allows later classifier comparisons to distinguish detector version, event logic, crop policy, and camera geometry.

## Ultralytics FP16 compatibility

The public config retains `yolo_half` for compatibility. When enabled, the adapter feature-detects current Ultralytics support and uses predict-time `quantize=16` (FP16). Older Ultralytics versions fall back to `half=True`. It never substitutes INT8 quantization for FP16.

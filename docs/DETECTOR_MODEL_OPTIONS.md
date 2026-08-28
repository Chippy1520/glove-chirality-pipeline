# Trainable glove detector options

## Why instance segmentation is the preferred target

A rectangular object detector always includes some belt pixels around an irregular glove. The failure in the original calibration preview was worse: diagnostic ROI/trigger graphics were drawn before detection, so the classical foreground detector boxed its own overlays. That implementation defect is fixed separately.

For production extraction, train a **single-class `glove` instance-segmentation model**. Use each predicted mask to derive a tight box and optionally suppress belt pixels in the classifier crop. Detection must remain independent of the left/right source label.

## Recommended baselines

| Priority | Model | Output | Why test it | Important constraint |
|---|---|---|---|---|
| 1 | YOLO11n-seg | boxes + retained instance polygons | Small, fast, and integrated with strict masks, ROI-only inference, full-frame coordinate restoration, mask-aware previews/crops, and shared offline/live passage processing | Ultralytics repository is AGPL-3.0; assess licensing before a closed industrial deployment |
| 2 | RF-DETR Seg Nano | boxes + instance masks | Small real-time transformer baseline, designed for fine-tuning, Apache-2.0 model/package path | Adds a new backend and should be benchmarked on the target deployment GPU |
| 3 | Torchvision Mask R-CNN R50-FPN v2 | boxes + instance masks | Mature, conservative research baseline with BSD-3-Clause torchvision | Usually slower/heavier than nano real-time models |
| Annotation aid | SAM 2.1 Small | prompted/video masks | Propagate masks through short clips to reduce manual polygon work | Use for annotation assistance, not as an unprompted passage detector |

RF-DETR's official repository currently describes detection and instance segmentation, including `RFDETRSegNano` and `RFDETRSegSmall`. Its open-source package and designated models are Apache-2.0. SAM 2 is also Apache-2.0. The Ultralytics repository exposes object detection and instance segmentation but is AGPL-3.0.

Sources:

- [Ultralytics instance segmentation](https://docs.ultralytics.com/tasks/segment/)
- [Ultralytics repository and license](https://github.com/ultralytics/ultralytics)
- [RF-DETR repository](https://github.com/roboflow/rf-detr)
- [Torchvision Mask R-CNN](https://docs.pytorch.org/vision/main/models/mask_rcnn.html)
- [SAM 2 repository](https://github.com/facebookresearch/sam2)

## Implemented YOLO11n-seg contract

The production adapter expects one detector class: `0 = glove`. Left/right is never a detector class or detector input. `Detection` retains a compact full-frame polygon rather than a 1920×1080 binary mask. The polygon drives a tight box, full-containment gate, diagnostics, optional crop background suppression, quality metadata, and optional separate JSON persistence.

Use `yolo_require_masks: true` for production so a box-only checkpoint fails clearly. `yolo_crop_to_roi: true` sends only the configured inspection ROI to YOLO, then offsets boxes and polygons back into full-frame coordinates. `roi` and `trigger_zone` remain normalized full-frame settings.

`crop_mode: bbox` remains the safe default. `masked` zeros non-glove pixels and `masked_fill` uses a deterministic median outside-mask background. These are experimental factors, not automatic improvements: compare them against bbox crops on exactly the same source/session split and keep detector/extractor settings frozen.

The shared passage processor detects partial masks but accepts only complete trigger-contained frames, rejects multiple simultaneous instances, selects one best frame, and invokes the chirality classifier once. Live capture uses monotonic time and a bounded latest-frame queue; it is not YOLO-plus-classification on every frame.

See `configs/production.yaml` for the complete strict-segmentation example.

## Public glove datasets inspected

### PPE Dataset (YOLO, five classes)

- URL: [Kaggle PPE Dataset](https://www.kaggle.com/datasets/waquarahmed1/ppe-dataset)
- Classes include `Helmet`, `Mask`, `Safety Vest`, `boots`, and `glove`.
- Approximately 612 MB according to the Kaggle API metadata inspected on 2026-08-27.
- The metadata attributes the source to [this Roboflow Universe dataset](https://universe.roboflow.com/glovesdetection/my-first-project-przjz-dj1mn/dataset/1).
- License reported by Kaggle: CC BY-SA 4.0; inspect original-source terms before reuse.

### Glove and No Gloves 2.3K

- URL: [Kaggle Glove and No Gloves 2.3K](https://www.kaggle.com/datasets/waquarahmed1/glove-and-no-gloves-2-3k)
- 2,300+ object-detection images with `Glove` and `No_Glove` classes according to its metadata.
- License reported by Kaggle: CC BY-SA 4.0.

These datasets focus on PPE compliance and hands wearing gloves in varied scenes. They are not a substitute for the fixed-camera conveyor domain, loose glove shapes, enclosure reflections, glove colors, or passage geometry. Treat them only as optional auxiliary pretraining data and never as target-domain test data.

Hugging Face searches were also inspected. `BASF-AI/SDS-Gloves-Classification` is a text-classification safety-data-sheet dataset, not images. The `pirathi2002/glove-defect-detection` and `glove-background-detection` repositories contain undocumented PatchCore checkpoints with no model card or usage evidence; they are not recommended as pipeline detectors.

## Target-domain annotation plan

1. Freeze train/validation/test splits by source video or recording session before frame sampling.
2. Sample complete passages, partial entry/exit frames, entirely empty frames, glare, belt seams, enclosure edges, touching gloves, and multiple simultaneous gloves.
3. Annotate one instance mask per visible glove, including visible partial gloves. A partial mask's derived box will cross the trigger boundary and be rejected by full-containment gating.
4. Keep empty and artifact-heavy frames as negative examples with no glove annotation.
5. Use SAM 2 mask propagation only to propose labels; manually review boundaries, merged instances, cuffs, and fingers.
6. Begin with a modest reviewed seed set, train a baseline, then use false positives and false negatives from full videos for active-learning rounds.
7. Do not encode chirality in the detector class. Detection uses one `glove` class; left/right remains the downstream classifier's task.
8. Do not use horizontal reflection augmentation unless the chirality convention and transformed labels are explicitly tested downstream.

## Evaluation required before replacing the classical bootstrap detector

Report model metrics and passage metrics separately:

- mask AP and box AP on held-out source videos;
- recall for fully visible gloves;
- false detections per empty conveyor hour;
- partial-entry rejection rate;
- duplicate crops per physical passage;
- missed passages;
- ambiguous/touching passage rate;
- extraction latency and rolling live detector/event/classifier latency on the deployment device;
- captured/processed/dropped frame counts and accepted-event latency under load;
- end-to-end chirality accuracy with extraction failures and explicit rejections included;
- bbox versus masked/masked-fill classifier results on identical source/session splits.

The recommended first comparison is classical foreground vs YOLO11n-seg vs RF-DETR Seg Nano on the same source-level split. Use Mask R-CNN as a slower reference if compute permits. Export every accepted crop through the shared fixed-size letterbox stage so crop dimensions cannot become a classifier shortcut.

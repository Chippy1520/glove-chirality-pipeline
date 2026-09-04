# Desktop GUI

The project includes a lightweight Tkinter interface that calls the same tested CLI used by scripts and deployment. It does not reimplement extraction, training, or inference logic.

## Launch

From the repository environment:

```bash
source .venv/Scripts/activate  # Git Bash on Windows
glove-pipeline-gui
```

Equivalent module command:

```bash
python -m glove_chirality.gui
```

Tkinter ships with standard Windows Python installations, so no web server or additional GUI framework is required.

## Tabs

### Extract

- Select left-only and right-only video directories and build one labeled dataset.
- Process one video or a complete directory with `left`, `right`, or `unknown` provenance.
- Render a calibration preview at a selected timestamp.
- Select the output directory and extraction YAML with native file dialogs.

### 2 · Layer 1

Edit the most frequently calibrated parameters:

- detector backend;
- normalized ROI and trigger zone;
- full-glove trigger containment and optional inner clearance margin;
- belt color-distance threshold;
- temporal motion assistance;
- adaptive empty-belt background learning and empty/foreground learning rates;
- morphology and component-area limits;
- YOLO custom model picker, confidence, glove class ID, device, half precision, image size, IoU, maximum detections, and full-frame minimum/maximum bbox-area ratios;
- one-click custom segmentation preset that selects the YOLO backend, requires masks, and enables ROI-only inference;
- event confirmation, exit, and cooldown frames;
- rejection of frames containing multiple simultaneous candidates;
- crop padding, `bbox`/`masked`/`masked_fill` mode, square policy, and fixed output size.

Load an existing YAML, save changes in place, or save a new copy. Advanced YAML values that are not shown remain intact when editing an existing configuration.

#### Use a collaborator's custom YOLO11n-seg model

The detector checkpoint and chirality-classifier checkpoint are separate:

- **Layer 1:** the custom YOLO segmentation model detects a glove, supplies its mask, and drives passage tracking/cropping.
- **Layer 2:** the classifier predicts left/right once for each accepted Layer-1 crop.

To configure Layer 1 without hand-editing YAML:

1. Open **2 · Layer 1**.
2. Under **Layer 1 — YOLO11n-seg glove model**, click **Browse…** and choose the trained segmentation checkpoint, normally `best.pt`.
3. The GUI automatically selects the `yolo` backend, sets glove class ID `0`, enables masks, requires segmentation output, and enables ROI-only inference.
4. If the annotation platform assigned glove a different class index, change **Glove class ID** to match its dataset class list.
5. Choose the YOLO device (`auto`, `cpu`, `cuda`, or `cuda:0`) and save the configuration, preferably with **Save as…**.
6. Use **Calibration preview** on representative video timestamps before extracting a dataset or starting live inference.

The default config intentionally contains no generic YOLO checkpoint. Stock COCO `yolo11n.pt` is not a glove detector. A box-only model also fails when strict masks are enabled, instead of silently changing crop behavior.

#### Remove tiny artifacts and avoid neighboring gloves

For the measured GRIP camera setup, load `configs/grip_aug27_seed.yaml`. Its YOLO bbox-area limits reject detections outside `0.03–0.40` of the complete frame before they reach trigger or passage logic. Do not copy these limits to a changed camera geometry without remeasuring real glove and false-positive boxes.

Click **Use tight detection bbox** under **Passage and crop** to set:

- crop padding to `0.0`;
- square expansion off; and
- crop mode to `bbox`.

This uses each selected mask-derived bbox as a variable-width/height source crop. It does **not** stretch the crop: aspect-preserving letterboxing still produces the classifier's fixed output dimensions. If a neighboring glove overlaps the rectangle, try `masked` or `masked_fill`, which suppresses pixels outside the selected instance polygon. The pipeline still rejects multiple eligible gloves by default rather than assigning an arbitrary passage.

### 3 · Train

Choose the manifest, checkpoint path, model, epochs, batch/image size, learning rate, grouped validation fraction, seed, device, AMP, and DataLoader workers. The Layer-2 controls also expose:

- `cross_entropy`, `weighted_cross_entropy`, or experimental `recall_hybrid` loss;
- the recall target (`right` for the current safety objective);
- recall-penalty weight; and
- the validation metric used to retain the best checkpoint (`recall_right` when missed right gloves are the costly error);
- `none`, `standard`, or `anti_spurious` chirality-safe augmentation; and
- an optional TensorBoard log directory and local dashboard launcher.

For the right-glove-recall experiment, start with `recall_hybrid`, target `right`, penalty weight `1.0`, and selection metric `recall_right`. Compare it against the weighted-cross-entropy baseline on the identical locked source/session split rather than assuming the custom loss is better.

For an NVIDIA GPU, first install the CUDA-enabled PyTorch build matching the laptop, then use `cuda` or `cuda:N` and optionally enable AMP.

`anti_spurious` strengthens color/contrast changes, occasionally removes color, applies mild blur, and erases small random regions. It is intended for controlled retraining when explanations suggest texture, print, or label shortcuts. It never reflects an image. Compare it against `standard` on the identical split; do not assume stronger augmentation is automatically better.

When a TensorBoard directory is set, training records loss, accuracy, macro recall/F1, and per-class recall. Click **Start TensorBoard** before or during training; it uses an independent process slot, so it no longer blocks training or inference. **Open dashboard** launches `127.0.0.1:<port>` in the browser, and **Stop TensorBoard** stops only the dashboard. The pipeline command has its own independent stop control in **Run log**.

### 4 · Infer

- Run the shared extractor and classifier on a full video.
- Start/stop event-driven live inference from a camera index or OpenCV stream source.
- Save machine-readable live events to JSONL and watch periodic FPS/latency/drop summaries in the run log.
- Classify one existing crop or a directory of crops.
- Select checkpoint, classifier device/AMP, output, and extraction config.
- Keep **Recall-priority class** at `argmax` for ordinary classification. To reduce right-as-left errors, select `right` and lower its probability threshold below `0.5`; this intentionally increases left-as-right false alarms.

The threshold must be selected on held-out source sessions. Do not claim zero missed right gloves from training metrics alone, and remember that this Layer-2 threshold cannot compensate for Layer-1 extraction misses.

Live mode still delegates to `glove-pipeline infer-live`; the GUI contains no camera tracker or crop implementation of its own.

### 5 · Explain

Select one accepted crop, a Layer-2 checkpoint, output image, target class, and method:

- `smoothgrad` averages input-gradient sensitivity over noisy copies and is the faster first check;
- `occlusion` hides patches and highlights regions whose removal lowers the target-class probability. It is slower but often easier to reason about.

The command writes a color overlay and adjacent JSON containing class probabilities, target class, method, and paths. Compare correct and incorrect examples from multiple held-out sessions. A bright region is diagnostic sensitivity evidence, not causal proof that the model uses a semantic glove feature. Explanations do not replace ablation tests—for example, retraining with `anti_spurious` and measuring the locked test set.

CLI equivalent:

```bash
glove-pipeline explain \
  --image outputs/dataset/images/right/example.jpg \
  --checkpoint outputs/models/classifier.pt \
  --output outputs/explanations/example.png \
  --method occlusion \
  --target-class right
```

### 6 · Compare

Choose a run archive and click **Refresh runs**. The table recursively discovers both normal `*.metrics.json` summaries and explicit-split `*.history.json` files without loading large model checkpoints. It shows:

- model and augmentation policy;
- checkpoint-selection metric;
- source split ID;
- accuracy, macro recall, macro F1, left recall, and right recall;
- validation sample count and source artifact.

Rank by any displayed metric and export the current ordering to CSV. For this project, right recall is the default because right→left errors are safety-critical, but it must be interpreted with left recall/false alarms and extraction performance.

Future runs store a stable 12-character split ID derived from the complete train/validation source-video assignment. Compare architectures directly only when their split IDs match. Older artifacts remain visible with split ID `unknown`; verify their split provenance manually before drawing conclusions.

CLI equivalent:

```bash
glove-pipeline compare-models \
  --input outputs/models runs/experiments \
  --output outputs/model_comparison.csv \
  --sort-by recall_right
```

## Execution behavior

Pipeline commands run in a background subprocess so the window remains responsive. Training, extraction, and inference remain mutually exclusive to avoid accidental GPU/camera contention, while TensorBoard has a separate concurrent process slot. The Run log labels output by process. **Stop pipeline** and **Stop TensorBoard** affect only their respective process; **Clear** clears the visible log.

The GUI does not hide errors or fabricate success. The final subprocess exit code is written to the log.

## Recommended workflow with real data

1. Open **Extraction settings**, load a copied YAML, and adjust the ROI/trigger geometry.
2. Use **Calibration preview** on early, middle, and late timestamps for every glove color.
   The detector warm-up processes preceding frames so temporal background state is not initialized from only the selected glove frame.
3. Extract a small annotated subset and verify misses, duplicates, and crop completeness.
   Include entirely empty clips and long gaps between gloves; both must produce no extra events.
4. Extract the complete labeled dataset only after passage-level calibration.
5. Train grouped model baselines.
6. Use full-video inference only with a validated extractor configuration and checkpoint.

Raw videos, crops, model checkpoints, and outputs remain excluded from Git.

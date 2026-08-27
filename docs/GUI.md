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

### Extraction settings

Edit the most frequently calibrated parameters:

- detector backend;
- normalized ROI and trigger zone;
- full-glove trigger containment and optional inner clearance margin;
- belt color-distance threshold;
- temporal motion assistance;
- adaptive empty-belt background learning and empty/foreground learning rates;
- morphology and component-area limits;
- event confirmation, exit, and cooldown frames;
- rejection of frames containing multiple simultaneous candidates;
- crop padding and square-crop policy.

Load an existing YAML, save changes in place, or save a new copy. Advanced YAML values that are not shown remain intact when editing an existing configuration.

### Train

Choose the manifest, checkpoint path, model, epochs, batch/image size, learning rate, grouped validation fraction, seed, device, AMP, and DataLoader workers.

For an NVIDIA GPU, first install the CUDA-enabled PyTorch build matching the laptop, then use `cuda` or `cuda:N` and optionally enable AMP.

### Inference

- Run the shared extractor and classifier on a full video.
- Classify one existing crop or a directory of crops.
- Select checkpoint, device, output, and extraction config.

## Execution behavior

Commands run in a background subprocess so the window remains responsive. The Run log shows the exact reproducible command and live output. Only one command runs at a time; **Stop** requests termination and **Clear** clears the visible log.

The GUI does not hide errors or fabricate success. The final subprocess exit code is written to the log.

## Recommended workflow with real data

1. Open **Extraction settings**, load a copied YAML, and adjust the ROI/trigger geometry.
2. Use **Calibration preview** on early, middle, and late timestamps for every glove color.
3. Extract a small annotated subset and verify misses, duplicates, and crop completeness.
   Include entirely empty clips and long gaps between gloves; both must produce no extra events.
4. Extract the complete labeled dataset only after passage-level calibration.
5. Train grouped model baselines.
6. Use full-video inference only with a validated extractor configuration and checkpoint.

Raw videos, crops, model checkpoints, and outputs remain excluded from Git.

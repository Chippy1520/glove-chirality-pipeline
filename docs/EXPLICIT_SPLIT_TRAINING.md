# Explicit-split classifier recovery and diagnostics

Use this path when a benchmark **already has locked `train/` and `val/`
directories**. Do not feed that benchmark through the standard single-manifest
trainer: its grouped split would create a different experiment. The browser and
Tk training forms still use the standard grouped-manifest trainer.

```text
locked_dataset/
  train/left/...
  train/right/...
  val/left/...
  val/right/...
  manifest_train.csv   # if available: image_path, label, sha256, source metadata
  manifest_val.csv
```

The maintained entry point is `scripts/train_layer2_explicit_split.py`; its
implementation is import-safe and shares `ManifestDataset`, model construction,
classification metrics and staged parameter handling with the package. No crop
regeneration, reflection, label changes, rebalancing or repartitioning is done.
Both classes must exist in both partitions. Dataset byte hashes and resolved
paths are checked for cross-partition overlap. When manifests are supplied their
membership, labels and image hashes must match; available source provenance is
also checked. **Image disjointness alone does not establish session disjointness.**
Supply source/session provenance and audit the partition before claiming a
leakage-free benchmark.

## A controlled recovery candidate

From an installed checkout (`python -m pip install -e '.[ml,dev]'`):

```bash
python scripts/train_layer2_explicit_split.py \
  --root data/locked_dataset \
  --model convnextv2_pico \
  --staged-finetune \
  --fine-tune-head-learning-rate 0.0001 \
  --device cuda --no-amp \
  --output outputs/convnext_recovery_fp32/best.pt \
  --metrics-dir outputs/convnext_recovery_fp32/metrics \
  --tensorboard-logdir outputs/convnext_recovery_fp32/tensorboard
```

Use the existing dataset root on the training machine and a **new run name**.
No data is copied into or committed to the repository. An optional
`--expected-selection-sha256` pins the known dataset selection fingerprint;
compare it to the independently recorded locked benchmark, not a newly computed
value that merely blesses unexpected changes.

`--staged-finetune` defaults to 23 total epochs, 3 head-only epochs, head LR
`3e-4` and backbone LR `2e-5`. The extra head-LR option in the example lowers the
head to `1e-4` after unfreezing; without it the head LR stays fixed. The backbone
LR must stay below the effective head LR. This is an opt-in recovery preset,
not proof of a better model. Run `--help` for the exposed learning-rate, precision,
duration, batch size and worker controls. Explicit staged options support the other registered
classifiers, including both DINOv3 backbones. The preset is architecture-specific
tuning and must not be mislabeled as the original common-recipe comparison.

Head warm-up holds the backbone in evaluation mode as well as freezing its
parameters. Unfreezing continues from the same weights and retains the head's
AdamW moments. This is not an exact numerical replay of the older workstation
DINO script, which recreated its optimizer at the stage boundary. The default
common-recipe path keeps full-model training; the standard GUI trainer remains
unchanged by this explicit-split entry point.

The initial FP32 candidate removes mixed-precision uncertainty while using a
less aggressive backbone schedule. It changes multiple factors relative to a
full-model AMP baseline, so success does **not** isolate the original cause.
To attribute a recovery, compare staged FP32 with staged AMP at otherwise equal
settings, and staged with full-model training in FP32. Hold split, preprocessing,
seed, evaluation and stopping rules constant. Do not keep retrying until one
validation score looks good; log each experiment and reserve untouched sessions
for the final evaluation.

## Safety and observability

- Existing checkpoint, history/metrics and TensorBoard destinations are protected
  from overwrite. Preserve the failed run instead of replacing its `best.pt`.
- Logits and losses must be finite before prediction/optimization. Unscaled
  gradient diagnostics distinguish finite optimizer updates from AMP overflow
  skips. An epoch with no successful updates is a failure, not progress.
- Per-epoch prediction counts and fractions expose constant LEFT/RIGHT behavior.
  A persistent one-class distribution after warm-up stops the run; this is a
  diagnostic rule, not a medical/industrial acceptance threshold.
- Checkpoint selection still uses macro recall with macro-F1 as a tiebreaker.
  This can select a constant predictor when every epoch is poor. A saved
  checkpoint is **not** by itself a successful classifier; inspect run status,
  best-epoch confusion matrix and right-glove recall.
- Histories retain stage, effective learning rates, package versions and numerical diagnostics;
  failures preserve diagnostic information rather than silently substituting
  another precision or learning rate. TensorBoard is closed on failure.
- The shared image dataset is spawn-picklable for Windows DataLoader workers;
  the script is protected by a `__main__` guard.

## How to interpret a collapse audit

1. Verify artifact sizes/hashes and distinguish the captured command/source from
   reconstructed commands and code modified after the failed run.
2. Sum confusion-matrix columns for **every epoch**, not just the best checkpoint.
   A run may switch its constant class or later produce mixed predictions.
3. Compare balanced accuracy and class recalls to constant-predictor baselines.
   Majority-class accuracy can look better than chance without discrimination.
4. Finite epoch losses do not prove every AMP update was applied. Conversely,
   constant predictions do not prove NaNs, label imbalance or architecture failure.
5. Test the data/training path on a separate tiny balanced training subset without
   augmentation as a diagnostic. Its training fit is not validation evidence.
6. Keep recovery claims separate from code tests. Random-initialized CPU probes
   validate mechanics, not pretrained CUDA behavior or factory performance.

Private audit exports match `*_Audit_Export/` and are ignored by Git. Store
run-specific audit reports under ignored `outputs/`, not in public documentation.

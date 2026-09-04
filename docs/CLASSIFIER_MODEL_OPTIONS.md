# Classifier model options

All classifier backbones consume the same accepted passage crops and ImageNet-normalized RGB preprocessing. Changing the classifier does not change detection, event selection, crop generation, source-grouped splitting, or the chirality label convention.

## Available model names

| CLI / checkpoint name | Implementation | Intended role |
|---|---|---|
| `tiny_cnn` | Local PyTorch module | From-scratch sanity baseline |
| `mobilenet_v3_small` | Torchvision | Lightweight deployment baseline |
| `resnet18` | Torchvision | Stable transfer-learning baseline |
| `vit_b_16` | Torchvision | Large legacy transformer comparator |
| `swin_t` | Torchvision | Hierarchical transformer comparator |
| `convnextv2_pico` | timm `convnextv2_pico.fcmae_ft_in1k` | Primary compact modern-CNN candidate |
| `dinov3_convnext_tiny` | timm `convnext_tiny.dinov3_lvd1689m` | Self-supervised representation candidate |

The checkpoint stores both the public model name and its implementation library/identifier. Inference reconstructs the architecture with `pretrained=False`, then loads the local checkpoint state dict; it does not download upstream weights.

## Recommended experiment

Start with a focused comparison:

1. `mobilenet_v3_small` for deployment efficiency;
2. `resnet18` as the conventional baseline;
3. `convnextv2_pico` as the primary modern CNN;
4. `swin_t` only when a transformer comparison is scientifically useful;
5. `dinov3_convnext_tiny` as the higher-capacity self-supervised challenger.

Do not select a model by overall accuracy alone. Use source/session-grouped splits and compare right-glove recall, right-to-left errors, balanced accuracy, macro-F1, calibration, per-session failures, latency, and memory. Tune any right-class decision threshold on validation data and evaluate it once on the locked test set.

Example:

```bash
glove-pipeline train \
  --manifest outputs/dataset/manifest.csv \
  --output outputs/models/convnextv2_pico.pt \
  --model convnextv2_pico \
  --device cuda \
  --amp \
  --selection-metric recall_right
```

Pretrained Swin, ConvNeXt, and DINO models should be fine-tuned rather than trained from scratch on a modest glove dataset. Horizontal reflection remains prohibited unless the chirality label is transformed under an explicitly tested convention.

## Shortcut-resistant experiments

Training exposes three policies:

- `none`: deterministic resize and normalization only;
- `standard`: mild color jitter and rotation;
- `anti_spurious`: stronger color/contrast variation, occasional grayscale and blur, plus small random erasures.

`anti_spurious` is an ablation tool for suspected print, label, color, or local-texture shortcuts. It does not certify that the model uses the thumb or silhouette, and it may hurt performance if erasing removes true chirality evidence. Compare policies with identical source-grouped splits and seeds. Horizontal reflection is absent from every policy.

Optional TensorBoard logging records training loss, accuracy, macro recall/F1, and left/right validation recall:

```bash
glove-pipeline train \
  --manifest outputs/dataset/manifest.csv \
  --output outputs/models/anti_spurious.pt \
  --model convnextv2_pico \
  --augmentation anti_spurious \
  --tensorboard-logdir outputs/tensorboard/anti_spurious

python -m tensorboard.main --logdir outputs/tensorboard --host 127.0.0.1
```

## Individual explanations

`glove-pipeline explain` supports every classifier architecture through input-level methods:

- `smoothgrad` for averaged gradient sensitivity;
- `occlusion` for target-probability drops after hiding image patches.

Each run writes an overlay and JSON metadata. Use them to formulate a shortcut hypothesis, then test that hypothesis with controlled masking or augmentation and locked held-out sessions. These maps are not transformer attention weights and are not causal explanations.

The GUI **Compare** tab and `compare-models` CLI collect normal training summaries and explicit-split histories without loading checkpoint tensors. New standard training runs store a source-partition split ID; only rank architectures as direct experiments when this ID matches. Right recall is the default ranking metric, but inspect the full class trade-off and end-to-end extraction failures.

## Weight licenses

Repository code is MIT licensed, but pretrained weights have separate terms:

- The timm metadata for `convnextv2_pico.fcmae_ft_in1k` identifies the pretrained weights as **CC BY-NC 4.0**.
- `convnext_tiny.dinov3_lvd1689m` uses Meta's **DINOv3 License**.
- Torchvision models use the terms attached to their respective pretrained weight datasets/releases.

Review the selected pretrained-weight license before commercial or industrial deployment. This project does not redistribute upstream pretrained weights or trained production checkpoints.

## Practical interpretation

A stronger backbone cannot repair extraction failures, truncated thumbs, ambiguous touching gloves, or leakage between adjacent frames. Keep extraction performance and classification performance separate, and use identical accepted crops for every architecture comparison.

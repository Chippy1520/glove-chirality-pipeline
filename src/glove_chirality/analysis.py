from __future__ import annotations

import json
from pathlib import Path

from glove_chirality.inference import TorchClassifier, decision_index

EXPLANATION_METHODS = ("smoothgrad", "occlusion")


def _select_target(classifier, probabilities, target_class: str) -> int:
    if target_class == "predicted":
        return decision_index(
            classifier.classes,
            probabilities,
            classifier.decision_class,
            classifier.decision_threshold,
        )
    if target_class not in classifier.classes:
        choices = ", ".join(("predicted", *classifier.classes))
        raise ValueError(f"target_class must be one of: {choices}")
    return classifier.classes.index(target_class)


def _normalize_map(torch, values):
    values = values.clamp_min(0)
    maximum = values.max()
    if float(maximum) <= 0.0:
        return torch.zeros_like(values)
    return values / maximum


def _smoothgrad(classifier, tensor, target_index: int, samples: int, noise_std: float):
    torch = classifier.torch
    if samples < 1:
        raise ValueError("samples must be at least 1")
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative")
    accumulated = torch.zeros(tensor.shape[-2:], device=classifier.device)
    for parameter in classifier.model.parameters():
        parameter.requires_grad_(False)
    classifier.model.zero_grad(set_to_none=True)
    for _ in range(samples):
        noisy = (tensor + torch.randn_like(tensor) * noise_std).detach().requires_grad_(True)
        classifier.model(noisy)[0, target_index].backward()
        accumulated += noisy.grad.detach().abs().mean(dim=1)[0]
        classifier.model.zero_grad(set_to_none=True)
    return _normalize_map(torch, accumulated / samples)


def _occlusion(classifier, tensor, target_index: int, patch_size: int, stride: int):
    torch = classifier.torch
    height, width = tensor.shape[-2:]
    if patch_size < 1 or stride < 1:
        raise ValueError("patch_size and stride must be positive")
    patch_size = min(patch_size, height, width)
    y_positions = list(range(0, max(height - patch_size + 1, 1), stride))
    x_positions = list(range(0, max(width - patch_size + 1, 1), stride))
    if not y_positions or y_positions[-1] != height - patch_size:
        y_positions.append(height - patch_size)
    if not x_positions or x_positions[-1] != width - patch_size:
        x_positions.append(width - patch_size)

    with torch.no_grad():
        baseline = torch.softmax(classifier.model(tensor), dim=1)[0, target_index]
    heatmap = torch.zeros((height, width), device=classifier.device)
    counts = torch.zeros_like(heatmap)
    regions = [(y, x) for y in y_positions for x in x_positions]
    batch_size = 32
    for start in range(0, len(regions), batch_size):
        batch_regions = regions[start : start + batch_size]
        occluded = tensor.repeat(len(batch_regions), 1, 1, 1)
        for index, (y, x) in enumerate(batch_regions):
            occluded[index, :, y : y + patch_size, x : x + patch_size] = 0.0
        with torch.no_grad():
            probabilities = torch.softmax(classifier.model(occluded), dim=1)[:, target_index]
        for drop, (y, x) in zip(baseline - probabilities, batch_regions, strict=True):
            heatmap[y : y + patch_size, x : x + patch_size] += drop.clamp_min(0)
            counts[y : y + patch_size, x : x + patch_size] += 1
    return _normalize_map(torch, heatmap / counts.clamp_min(1))


def explain_image(
    image_path: str | Path,
    checkpoint: str | Path,
    output_path: str | Path,
    device: str = "auto",
    method: str = "smoothgrad",
    target_class: str = "predicted",
    samples: int = 16,
    noise_std: float = 0.10,
    patch_size: int = 32,
    stride: int = 16,
    overlay_alpha: float = 0.45,
) -> dict[str, object]:
    """Save a diagnostic class-sensitivity overlay and return its metadata."""
    if method not in EXPLANATION_METHODS:
        raise ValueError(f"method must be one of: {', '.join(EXPLANATION_METHODS)}")
    if not 0.0 <= overlay_alpha <= 1.0:
        raise ValueError("overlay_alpha must be in [0.0, 1.0]")

    import cv2
    import numpy as np
    from PIL import Image

    classifier = TorchClassifier(checkpoint, device=device)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        tensor = classifier.transform(image).unsqueeze(0).to(classifier.device)
        display = np.asarray(image.resize((classifier.image_size, classifier.image_size))).copy()

    with classifier.torch.no_grad():
        probabilities_tensor = classifier.torch.softmax(classifier.model(tensor), dim=1)[0]
    probabilities = probabilities_tensor.detach().cpu().tolist()
    target_index = _select_target(classifier, probabilities, target_class)

    if method == "smoothgrad":
        heatmap = _smoothgrad(classifier, tensor, target_index, samples, noise_std)
    else:
        heatmap = _occlusion(classifier, tensor, target_index, patch_size, stride)

    heatmap_u8 = (heatmap.detach().cpu().numpy() * 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_TURBO)
    display_bgr = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(display_bgr, 1.0 - overlay_alpha, colored_bgr, overlay_alpha, 0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        raise OSError(f"Could not write explanation image: {output_path}")
    metadata = {
        "image_path": str(Path(image_path)),
        "checkpoint": str(Path(checkpoint)),
        "output_path": str(output_path),
        "method": method,
        "target_class": classifier.classes[target_index],
        "target_probability": float(probabilities[target_index]),
        "probabilities": {
            label: float(probability)
            for label, probability in zip(classifier.classes, probabilities, strict=True)
        },
        "interpretation": (
            "Diagnostic sensitivity map; highlighted regions are not causal proof of the "
            "features used by the classifier."
        ),
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata["metadata_path"] = str(metadata_path)
    return metadata

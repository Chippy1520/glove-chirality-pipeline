from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from glove_chirality.models import build_model, model_backend

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def decision_index(
    classes: list[str],
    probabilities: list[float],
    decision_class: str = "argmax",
    decision_threshold: float = 0.5,
) -> int:
    if len(classes) != len(probabilities) or not classes:
        raise ValueError("classes and probabilities must be non-empty and have equal length")
    if not 0.0 <= decision_threshold <= 1.0:
        raise ValueError("decision_threshold must be in [0.0, 1.0]")
    if decision_class == "argmax":
        return max(range(len(probabilities)), key=probabilities.__getitem__)
    if decision_class not in classes:
        raise ValueError(f"decision_class must be argmax or one of: {', '.join(classes)}")
    target_index = classes.index(decision_class)
    if probabilities[target_index] >= decision_threshold:
        return target_index
    alternatives = [index for index in range(len(classes)) if index != target_index]
    return max(alternatives, key=probabilities.__getitem__)


class TorchClassifier:
    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "auto",
        amp: bool = False,
        decision_class: str = "argmax",
        decision_threshold: float = 0.5,
    ):
        try:
            import torch
            from torchvision import transforms
        except ImportError as exc:
            raise RuntimeError("Inference requires: pip install -e .[ml]") from exc
        self.torch = torch
        self.device = torch.device(
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else ("cpu" if device == "auto" else device)
        )
        self.use_amp = amp and self.device.type == "cuda"
        saved = torch.load(checkpoint, map_location=self.device, weights_only=False)
        expected_backend = model_backend(saved["model_name"])
        if saved.get("model_backend", expected_backend) != expected_backend:
            raise RuntimeError(
                "Checkpoint model backend does not match this installation: "
                f"saved={saved['model_backend']}, expected={expected_backend}"
            )
        self.classes = saved["classes"]
        decision_index(
            self.classes,
            [1.0 / len(self.classes)] * len(self.classes),
            decision_class,
            decision_threshold,
        )
        self.decision_class = decision_class
        self.decision_threshold = decision_threshold
        self.model = build_model(saved["model_name"], len(self.classes), pretrained=False).to(
            self.device
        )
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        self.image_size = int(saved["image_size"])
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

    def _predict_pil(self, image) -> tuple[str, float]:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with self.torch.no_grad(), self.torch.autocast(
            device_type=self.device.type,
            enabled=self.use_amp,
        ):
            probabilities = self.torch.softmax(self.model(tensor), dim=1)[0]
        probability_values = probabilities.detach().cpu().tolist()
        index = decision_index(
            self.classes,
            probability_values,
            self.decision_class,
            self.decision_threshold,
        )
        return self.classes[index], float(probabilities[index])

    def predict(self, image_path: str | Path) -> tuple[str, float]:
        from PIL import Image

        with Image.open(image_path) as image:
            return self._predict_pil(image)

    def predict_array(self, image_bgr: np.ndarray) -> tuple[str, float]:
        """Classify an in-memory canonical BGR crop using checkpoint preprocessing."""
        from PIL import Image

        image_rgb = image_bgr[:, :, ::-1]
        return self._predict_pil(Image.fromarray(image_rgb))

    def warmup(self) -> None:
        tensor = self.torch.zeros(
            (1, 3, self.image_size, self.image_size),
            device=self.device,
        )
        with self.torch.no_grad(), self.torch.autocast(
            device_type=self.device.type,
            enabled=self.use_amp,
        ):
            self.model(tensor)


def infer_images(
    input_path: str | Path,
    checkpoint: str | Path,
    output_csv: str | Path,
    device: str = "auto",
    decision_class: str = "argmax",
    decision_threshold: float = 0.5,
):
    source = Path(input_path)
    images = (
        [source]
        if source.is_file()
        else sorted(item for item in source.rglob("*") if item.suffix.lower() in IMAGE_EXTENSIONS)
    )
    classifier = TorchClassifier(
        checkpoint,
        device=device,
        decision_class=decision_class,
        decision_threshold=decision_threshold,
    )
    rows = []
    for image in images:
        label, confidence = classifier.predict(image)
        rows.append(
            {
                "image_path": str(image),
                "prediction": label,
                "confidence": f"{confidence:.6f}",
            }
        )
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["image_path", "prediction", "confidence"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows

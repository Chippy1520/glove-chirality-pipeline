from __future__ import annotations

import csv
from pathlib import Path

from glove_chirality.models import build_model

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class TorchClassifier:
    def __init__(self, checkpoint: str | Path, device: str = "auto"):
        try:
            import torch
            from torchvision import transforms
        except ImportError as exc:
            raise RuntimeError("Inference requires: pip install -e .[ml]") from exc
        self.torch = torch
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
        saved = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.classes = saved["classes"]
        self.model = build_model(saved["model_name"], len(self.classes), pretrained=False).to(self.device)
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        size = int(saved["image_size"])
        self.transform = transforms.Compose([
            transforms.Resize((size, size)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(self, image_path: str | Path) -> tuple[str, float]:
        from PIL import Image
        image = self.transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            probabilities = self.torch.softmax(self.model(image), dim=1)[0]
        index = int(probabilities.argmax())
        return self.classes[index], float(probabilities[index])


def infer_images(input_path: str | Path, checkpoint: str | Path, output_csv: str | Path):
    source = Path(input_path)
    images = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    classifier = TorchClassifier(checkpoint)
    rows = []
    for image in images:
        label, confidence = classifier.predict(image)
        rows.append({"image_path": str(image), "prediction": label, "confidence": f"{confidence:.6f}"})
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_path", "prediction", "confidence"])
        writer.writeheader(); writer.writerows(rows)
    return rows

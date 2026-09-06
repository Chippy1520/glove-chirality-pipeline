"""Independent end-to-end stage-transition regression without pretrained downloads."""
import csv

import numpy as np
import pytest


def test_actual_training_loop_preserves_frozen_backbone_then_unfreezes(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    Image = pytest.importorskip("PIL.Image")
    from glove_chirality import training
    from glove_chirality.models import build_model

    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(42)
        model = build_model("tiny_cnn", pretrained=False)
        head_ids = {id(p) for p in model[-1].parameters()}
        snapshots = []

        def capture(module, inputs):
            snapshots.append({
                "frozen": not next(module.parameters()).requires_grad,
                "head_training": module[-1].training,
                "backbone_training": module[1].training,
                "parameters": {n: p.detach().clone() for n, p in module.named_parameters()},
                "buffers": {n: b.detach().clone() for n, b in module.named_buffers()},
            })

        handle = model.register_forward_pre_hook(capture)
        rows = []
        for label in ("left", "right"):
            for source in range(2):
                path = tmp_path / f"{label}_{source}.png"
                rng = np.random.default_rng(source + (10 if label == "right" else 0))
                Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(path)
                rows.append({"label": label, "image_path": path.name,
                             "source_video": f"{label}_{source}.mkv"})
        manifest = tmp_path / "manifest.csv"
        with manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["label", "image_path", "source_video"])
            writer.writeheader()
            writer.writerows(rows)
        monkeypatch.setattr(training, "build_model", lambda *args, **kwargs: model)
        training.train_classifier(
            manifest, tmp_path / "test.pt", model_name="tiny_cnn", epochs=2,
            head_only_epochs=1, backbone_learning_rate=2e-5,
            learning_rate=3e-4, image_size=32, batch_size=2,
            augmentation="none", device_name="cpu",
        )
        handle.remove()
        # One train and one validation batch per epoch; no new weights at transition.
        assert len(snapshots) == 4
        a, b, c, d = snapshots
        assert a["frozen"] and a["head_training"] and not a["backbone_training"]
        assert not c["frozen"] and c["head_training"] and c["backbone_training"]
        backbone_names = [n for n, p in model.named_parameters() if id(p) not in head_ids]
        head_names = [n for n, p in model.named_parameters() if id(p) in head_ids]
        assert all(torch.equal(a["parameters"][n], b["parameters"][n]) for n in backbone_names)
        assert all(torch.equal(a["buffers"][n], b["buffers"][n]) for n in a["buffers"])
        assert any(not torch.equal(a["parameters"][n], b["parameters"][n]) for n in head_names)
        assert all(torch.equal(b["parameters"][n], c["parameters"][n]) for n in b["parameters"])
        assert any(not torch.equal(c["parameters"][n], d["parameters"][n]) for n in backbone_names)
        assert (tmp_path / "test.pt").is_file()
        assert (tmp_path / "test.pt.metrics.json").is_file()
    finally:
        torch.set_num_threads(threads)

"""DINOv3 ViT-S integration, with no upstream weight downloads."""

import numpy as np
import pytest

from glove_chirality.cli import build_parser
from glove_chirality.models import CLASSIFIER_CHOICES, build_model, model_backend

NAME = "dinov3_vit_small"
IDENTIFIER = "vit_small_patch16_dinov3.lvd1689m"


def test_shared_cli_gui_and_browser_choices(tmp_path):
    from glove_chirality import gui, gui_commands
    from glove_chirality.web_app import create_app, create_viewer_app
    from glove_chirality.web_service import CommandService, build_web_command

    assert gui.CLASSIFIER_CHOICES is CLASSIFIER_CHOICES
    assert NAME in CLASSIFIER_CHOICES
    command = gui_commands.train(
        "manifest.csv", "model.pt", NAME, 1, 2, 224, 0.001, 0.2, 42, "cpu", 0, False
    )
    assert build_parser().parse_args(command[3:]).model == NAME
    slot, command = build_web_command(
        "train", {"manifest": "manifest.csv", "output": "model.pt", "model": NAME}
    )
    assert slot == "pipeline"
    assert build_parser().parse_args(command[3:]).model == NAME
    service = CommandService(tmp_path)
    for app in (create_app(service), create_viewer_app(service, lan_token="test-secret")):
        response = app.test_client().get("/")
        assert response.status_code == 200
        assert NAME in response.get_data(as_text=True)


def test_exact_backend_and_pretrained_forwarding(monkeypatch):
    timm = pytest.importorskip("timm")
    assert IDENTIFIER in timm.list_models("*dinov3*", pretrained=True)
    assert model_backend(NAME) == {"library": "timm", "identifier": IDENTIFIER}
    calls = []
    sentinel = object()

    def create_model(identifier, **kwargs):
        calls.append((identifier, kwargs))
        return sentinel

    monkeypatch.setattr(timm, "create_model", create_model)
    for pretrained in (True, False):
        assert build_model(NAME, num_classes=3, pretrained=pretrained) is sentinel
        assert calls[-1] == (IDENTIFIER, {"pretrained": pretrained, "num_classes": 3})


@pytest.mark.parametrize("image_size", [224, 256])
@pytest.mark.parametrize("include_backend", [False, True])
def test_checkpoint_round_trip_and_inference(tmp_path, monkeypatch, image_size, include_backend):
    torch = pytest.importorskip("torch")
    timm = pytest.importorskip("timm")
    from glove_chirality.inference import TorchClassifier

    # Small thread count avoids CPU oversubscription on CI and shared workstations.
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = build_model(NAME, pretrained=False).eval()
        saved = {
            "model_name": NAME,
            "classes": ["left", "right"],
            "image_size": image_size,
            "state_dict": model.state_dict(),
        }
        if include_backend:
            saved["model_backend"] = model_backend(NAME)
        checkpoint = tmp_path / "vit-small.pt"
        torch.save(saved, checkpoint)
        original_create = timm.create_model

        def offline_create(identifier, **kwargs):
            assert kwargs["pretrained"] is False
            return original_create(identifier, **kwargs)

        monkeypatch.setattr(timm, "create_model", offline_create)
        restored = TorchClassifier(checkpoint, device="cpu")
        tensor = torch.zeros(1, 3, image_size, image_size)
        with torch.no_grad():
            expected = model(tensor)
            actual = restored.model(tensor)
        assert actual.shape == (1, 2)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        label, probability = restored.predict_array(np.zeros((48, 64, 3), dtype=np.uint8))
        assert label in saved["classes"]
        assert 0 <= probability <= 1
        restored.warmup()
    finally:
        torch.set_num_threads(threads)

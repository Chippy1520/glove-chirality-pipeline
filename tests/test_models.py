import pytest

from glove_chirality.cli import build_parser
from glove_chirality.models import CLASSIFIER_CHOICES, build_model, model_backend


def test_new_classifier_choices_are_exposed_by_cli():
    for name in ("swin_t", "convnextv2_pico", "dinov3_convnext_tiny"):
        args = build_parser().parse_args(
            [
                "train",
                "--manifest",
                "manifest.csv",
                "--output",
                "model.pt",
                "--model",
                name,
            ]
        )
        assert args.model == name
        assert name in CLASSIFIER_CHOICES


def test_model_backend_records_exact_implementations():
    assert model_backend("swin_t") == {
        "library": "torchvision",
        "identifier": "swin_t",
    }
    assert model_backend("convnextv2_pico") == {
        "library": "timm",
        "identifier": "convnextv2_pico.fcmae_ft_in1k",
    }
    assert model_backend("dinov3_convnext_tiny") == {
        "library": "timm",
        "identifier": "convnext_tiny.dinov3_lvd1689m",
    }


def test_new_models_reconstruct_without_downloading_weights():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("timm")

    for name in ("swin_t", "convnextv2_pico", "dinov3_convnext_tiny"):
        model = build_model(name, num_classes=2, pretrained=False).eval()
        restored = build_model(name, num_classes=2, pretrained=False).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            output = restored(torch.zeros(1, 3, 64, 64))
        assert tuple(output.shape) == (1, 2)

from pathlib import Path

import numpy as np
import pytest

from glove_chirality.analysis import explain_image


def _tiny_checkpoint(path: Path, torch):
    from glove_chirality.models import build_model, model_backend

    model = build_model("tiny_cnn", num_classes=2, pretrained=False)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": "tiny_cnn",
            "model_backend": model_backend("tiny_cnn"),
            "classes": ["left", "right"],
            "image_size": 32,
        },
        path,
    )


@pytest.mark.parametrize("method", ["smoothgrad", "occlusion"])
def test_explanation_methods_write_overlay_and_metadata(tmp_path, method):
    torch = pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("torchvision")

    checkpoint = tmp_path / "tiny.pt"
    image_path = tmp_path / "glove.png"
    output = tmp_path / f"{method}.png"
    _tiny_checkpoint(checkpoint, torch)
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[8:34, 18:48] = (180, 180, 180)
    assert cv2.imwrite(str(image_path), image)

    result = explain_image(
        image_path,
        checkpoint,
        output,
        device="cpu",
        method=method,
        samples=2,
        patch_size=16,
        stride=16,
    )

    assert output.is_file()
    assert output.with_suffix(".json").is_file()
    assert result["method"] == method
    assert result["target_class"] in {"left", "right"}
    assert set(result["probabilities"]) == {"left", "right"}

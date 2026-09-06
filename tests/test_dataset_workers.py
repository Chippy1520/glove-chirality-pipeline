"""Shared datasets must work with Windows-style spawn, not only Unix fork."""
import pickle

import cv2
import numpy as np
import pytest

from glove_chirality.dataset import ManifestDataset


@pytest.mark.parametrize("training", [False, True])
def test_manifest_dataset_is_spawn_picklable(tmp_path, training):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("PIL.Image")
    path = tmp_path / "crop.png"
    assert cv2.imwrite(str(path), np.full((24, 24, 3), (25, 50, 100), dtype=np.uint8))
    dataset = ManifestDataset([{"absolute_path": str(path), "label": "right"}], 32, training)
    restored = pickle.loads(pickle.dumps(dataset))
    torch.manual_seed(42)
    expected, label = dataset[0]
    torch.manual_seed(42)
    actual, actual_label = restored[0]
    assert actual_label == label == 1
    assert torch.equal(expected, actual)


def test_manifest_dataset_actual_spawn_worker(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("PIL.Image")
    rows = []
    for label, color in [("left", (0, 0, 255)), ("right", (255, 0, 0))]:
        path = tmp_path / f"{label}.png"
        assert cv2.imwrite(str(path), np.full((24, 24, 3), color, dtype=np.uint8))
        rows.append({"absolute_path": str(path), "label": label})
    dataset = ManifestDataset(rows, 32, False)
    expected = list(torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0))
    actual = list(torch.utils.data.DataLoader(
        dataset, batch_size=2, num_workers=1, multiprocessing_context="spawn", timeout=60,
    ))
    assert len(actual) == len(expected) == 1
    assert torch.equal(actual[0][0], expected[0][0])
    assert torch.equal(actual[0][1], expected[0][1])

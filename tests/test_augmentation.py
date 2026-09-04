import pytest

from glove_chirality.dataset import AUGMENTATION_CHOICES, build_image_transform


def test_augmentation_choices_are_explicit():
    assert AUGMENTATION_CHOICES == ("none", "standard", "anti_spurious")


def test_anti_spurious_policy_never_reflects_chirality():
    pytest.importorskip("torchvision")
    transform = build_image_transform(224, training=True, augmentation="anti_spurious")
    names = {type(item).__name__ for item in transform.transforms}
    assert "RandomHorizontalFlip" not in names
    assert "RandomVerticalFlip" not in names
    assert "RandomErasing" in names
    assert "RandomGrayscale" in names


def test_unknown_augmentation_is_rejected():
    pytest.importorskip("torchvision")
    with pytest.raises(ValueError, match="augmentation must be one of"):
        build_image_transform(224, training=True, augmentation="mirror")

import pytest

from glove_chirality.inference import decision_index


def test_right_recall_threshold_can_override_argmax():
    classes = ["left", "right"]
    probabilities = [0.60, 0.40]

    assert decision_index(classes, probabilities) == 0
    assert decision_index(classes, probabilities, "right", 0.30) == 1
    assert decision_index(classes, probabilities, "right", 0.50) == 0


def test_decision_threshold_validation_is_explicit():
    with pytest.raises(ValueError, match="decision_threshold"):
        decision_index(["left", "right"], [0.5, 0.5], "right", -0.1)
    with pytest.raises(ValueError, match="decision_class"):
        decision_index(["left", "right"], [0.5, 0.5], "unknown", 0.5)

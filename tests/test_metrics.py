from glove_chirality.training import classification_metrics


def test_binary_metrics():
    metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert metrics["accuracy"] == 0.75
    assert metrics["balanced_accuracy"] == 0.75
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert 0.7 < metrics["macro_f1"] < 0.75

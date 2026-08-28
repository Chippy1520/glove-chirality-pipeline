import pytest

torch = pytest.importorskip("torch")

from glove_chirality.training import (
    build_training_loss,
    classification_metrics,
    metric_score,
)


def test_metrics_expose_right_recall_for_checkpoint_selection():
    metrics = classification_metrics(
        targets=[0, 0, 1, 1],
        predictions=[1, 1, 0, 1],
    )

    assert metrics["recall_per_class"] == [0.0, 0.5]
    assert metrics["macro_recall"] == pytest.approx(0.25)
    assert metric_score(metrics, "recall_right") == pytest.approx(0.5)


def test_recall_hybrid_penalizes_low_right_probability():
    weights = torch.tensor([1.0, 1.0])
    loss = build_training_loss(
        torch,
        "recall_hybrid",
        weights,
        recall_target_index=1,
        recall_weight=2.0,
    )
    target = torch.tensor([1])

    low_right = loss(torch.tensor([[2.0, -2.0]]), target)
    high_right = loss(torch.tensor([[-2.0, 2.0]]), target)

    assert low_right.item() > high_right.item()


def test_training_loss_validation_is_explicit():
    weights = torch.tensor([1.0, 1.0])
    with pytest.raises(ValueError, match="Unknown training loss"):
        build_training_loss(torch, "unknown", weights, 1, 1.0)
    with pytest.raises(ValueError, match="non-negative"):
        build_training_loss(torch, "recall_hybrid", weights, 1, -1.0)

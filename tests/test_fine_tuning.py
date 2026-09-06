"""Offline numerical tests; no private data or pretrained downloads."""
import json

import pytest

from glove_chirality.fine_tuning import FineTuning, classification_head, fine_tuning_config


@pytest.fixture
def torch():
    torch = pytest.importorskip("torch")
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(5)
    yield torch
    torch.set_num_threads(threads)


def fixture_model(torch):
    nn = torch.nn
    return nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(0.5), nn.Linear(4, 2))


def step(torch, model, optimizer):
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(torch.randn(8, 4)), torch.arange(8) % 2)
    loss.backward()
    optimizer.step()


def test_frozen_then_unfrozen_updates_and_optimizer_state(torch):
    model = fixture_model(torch)
    tuning = FineTuning(model, "tiny_cnn", fine_tuning_config(3, 0.01, 2, 0.001))
    optimizer = tuning.optimizer
    before = {k: v.clone() for k, v in model.state_dict().items()}
    for epoch in range(2):
        model.eval()  # mimic validation each epoch
        state = tuning.begin_epoch(epoch)
        assert state["stage"] == "head_only"
        assert state["backbone_learning_rate"] == 0
        assert model[-1].training and not model[1].training and not model[2].training
        assert all(not p.requires_grad for p in tuning.backbone_parameters)
        step(torch, model, optimizer)
    for k, v in model.state_dict().items():
        if not k.startswith("3."):
            assert torch.equal(v, before[k]), k
    assert not torch.equal(model[-1].weight, before["3.weight"])
    assert all(p.grad is None for p in tuning.backbone_parameters)
    head = model[-1].weight.detach().clone()
    moments = optimizer.state[model[-1].weight]["exp_avg"].clone()
    state = tuning.begin_epoch(2)
    assert state["stage"] == "fine_tune"
    assert tuning.optimizer is optimizer
    assert torch.equal(model[-1].weight, head)
    assert torch.equal(optimizer.state[model[-1].weight]["exp_avg"], moments)
    assert [g["lr"] for g in optimizer.param_groups] == [0.01, 0.001]
    params = [p for g in optimizer.param_groups for p in g["params"]]
    assert len(params) == len({id(p) for p in params}) == len(list(model.parameters()))
    assert all(p.requires_grad for p in params)
    assert all(m.training for m in model.modules())
    step(torch, model, optimizer)
    assert not torch.equal(model[0].weight, before["0.weight"])
    assert not torch.equal(model[-1].weight, head)
    assert not torch.equal(model[1].running_mean, before["1.running_mean"])


def test_default_matches_original_adamw_numerically(torch):
    import copy

    model = fixture_model(torch)
    original = copy.deepcopy(model)
    tuning = FineTuning(model, "tiny_cnn", fine_tuning_config(1, 0.001))
    optimizer = torch.optim.AdamW(original.parameters(), lr=0.001)
    assert tuning.begin_epoch(0)["stage"] == "full"
    assert len(tuning.optimizer.param_groups) == 1
    torch.manual_seed(7)
    step(torch, model, tuning.optimizer)
    torch.manual_seed(7)
    step(torch, original, optimizer)
    for a, b in zip(model.state_dict().values(), original.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_differential_lr_without_warmup(torch):
    tuning = FineTuning(fixture_model(torch), "tiny_cnn", fine_tuning_config(1, 0.01, 0, 0.001))
    assert tuning.begin_epoch(0)["stage"] == "fine_tune"
    assert len(tuning.optimizer.param_groups) == 2
    assert fine_tuning_config(2, 0.01, 1)["backbone_learning_rate"] == 0.001


def test_numerical_differential_update_scale(torch):
    model = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False),
                                torch.nn.Linear(1, 1, bias=False))
    for p in model.parameters():
        torch.nn.init.ones_(p)
    tuning = FineTuning(model, "tiny_cnn", fine_tuning_config(1, 0.01, 0, 0.001))
    tuning.begin_epoch(0)
    model(torch.ones(1, 1)).square().sum().backward()
    tuning.optimizer.step()
    # Equal starting weights/gradients isolate the actual AdamW LR-group effect.
    head_delta = 1 - model[1].weight.item()
    backbone_delta = 1 - model[0].weight.item()
    assert head_delta / backbone_delta == pytest.approx(10, rel=1e-4)


@pytest.mark.parametrize("kwargs", [
    {"epochs": 0}, {"epochs": 1.5}, {"head_only_epochs": -1},
    {"head_only_epochs": 2}, {"head_only_epochs": 3}, {"head_only_epochs": 0.5},
    {"head_only_epochs": True}, {"learning_rate": 0}, {"learning_rate": float("nan")},
    {"learning_rate": float("inf")}, {"backbone_learning_rate": 0},
    {"backbone_learning_rate": -1}, {"backbone_learning_rate": 0.01},
    {"backbone_learning_rate": float("nan")}, {"backbone_learning_rate": float("inf")},
])
def test_validation_before_io(kwargs):
    from glove_chirality.training import train_classifier

    options = {"epochs": 2, "learning_rate": 0.01, **kwargs}
    with pytest.raises(ValueError):
        train_classifier("nonexistent.csv", "must-not-write.pt", **options)


@pytest.mark.parametrize("name,path", [
    ("tiny_cnn", "14"), ("resnet18", "fc"), ("mobilenet_v3_small", "classifier"),
    ("vit_b_16", "heads"), ("swin_t", "head"), ("convnextv2_pico", "head.fc"),
    ("dinov3_convnext_tiny", "head.fc"), ("dinov3_vit_small", "head"),
])
def test_real_architecture_head_discovery_offline(torch, name, path):
    from glove_chirality.models import build_model

    if name in {"convnextv2_pico", "dinov3_convnext_tiny", "dinov3_vit_small"}:
        pytest.importorskip("timm")
    model = build_model(name, pretrained=False)
    assert classification_head(model, name) is model.get_submodule(path)
    tuning = FineTuning(model, name, fine_tuning_config(2, 0.01, 1))
    tuning.begin_epoch(0)
    assert sum(p.requires_grad for p in model.parameters()) == len(tuning.head_parameters)


@pytest.mark.parametrize("warmup", [0, 1])
def test_training_metadata_and_tensorboard(torch, tmp_path, monkeypatch, warmup):
    from glove_chirality import training

    tb = pytest.importorskip("torch.utils.tensorboard")
    texts, scalars = {}, {}

    class Writer:
        def __init__(self, **kwargs):
            pass

        def add_text(self, key, value, epoch=0):
            texts[key, epoch] = value

        def add_scalar(self, key, value, epoch):
            scalars[key, epoch] = value

        def close(self):
            pass

    monkeypatch.setattr(tb, "SummaryWriter", Writer)
    rows = [{"label": label, "source_video": label} for label in ("left", "right")]
    monkeypatch.setattr(training, "read_manifest", lambda _: rows)
    monkeypatch.setattr(training, "grouped_split", lambda *args: (rows, rows))
    monkeypatch.setattr(training, "ManifestDataset", lambda *args: torch.utils.data.TensorDataset(
        torch.randn(8, 4), torch.arange(8) % 2
    ))
    monkeypatch.setattr(training, "build_model", lambda *args, **kw: fixture_model(torch))
    output = tmp_path / "fixture.pt"
    summary = training.train_classifier(
        "fixture.csv", output, epochs=2, head_only_epochs=warmup,
        learning_rate=0.01, device_name="cpu", tensorboard_logdir=tmp_path / "tb",
    )
    checkpoint = torch.load(output, weights_only=True)
    assert checkpoint["fine_tuning"] == summary["fine_tuning"]
    assert checkpoint["epoch"] == summary["best_epoch"]["epoch"]
    assert checkpoint["stage"] == summary["best_epoch"]["stage"]
    assert summary == json.loads(output.with_suffix(".pt.metrics.json").read_text())
    expected = ["head_only", "fine_tune"] if warmup else ["full", "full"]
    assert [r["stage"] for r in summary["history"]] == expected
    assert json.loads(texts["run/config", 0])["fine_tuning"] == summary["fine_tuning"]
    for epoch, stage in enumerate(expected, 1):
        assert texts["training/stage", epoch] == stage
        assert scalars["training/head_learning_rate", epoch] == 0.01
        assert scalars["training/backbone_learning_rate", epoch] == (
            0 if stage == "head_only" else 0.001 if warmup else 0.01
        )

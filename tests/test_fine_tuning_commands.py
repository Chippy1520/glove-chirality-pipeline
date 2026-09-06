import pytest

from glove_chirality import cli, gui_commands
from glove_chirality.web_service import build_web_command


@pytest.mark.parametrize("options", [{}, {"head_only_epochs": 2}, {
    "head_only_epochs": 2, "backbone_learning_rate": 0.0001,
}, {"backbone_learning_rate": 0.0001}])
def test_gui_web_cli_forwarding(monkeypatch, options):
    command = gui_commands.train(
        "manifest.csv", "model.pt", "dinov3_vit_small", 4, 8, 224,
        0.001, 0.2, 42, "cpu", 0, False, **options,
    )
    _, web = build_web_command("train", {
        "manifest": "manifest.csv", "output": "model.pt", "epochs": 4,
        "model": "dinov3_vit_small", **options,
    })
    captured = []
    monkeypatch.setattr("glove_chirality.training.train_classifier",
                        lambda **kwargs: captured.append(kwargs) or {})
    for args in (command, web):
        parsed = cli.build_parser().parse_args(args[3:])
        assert parsed.head_only_epochs == options.get("head_only_epochs", 0)
        assert parsed.backbone_learning_rate == options.get("backbone_learning_rate")
        cli.main(args[3:])
        assert captured[-1]["head_only_epochs"] == parsed.head_only_epochs
        assert captured[-1]["backbone_learning_rate"] == parsed.backbone_learning_rate
        if not options:
            assert "--head-only-epochs" not in args
            assert "--backbone-learning-rate" not in args


@pytest.mark.parametrize("value", [None, "", "  "])
def test_blank_web_backbone_lr(value):
    _, command = build_web_command("train", {
        "manifest": "m.csv", "output": "m.pt", "backbone_learning_rate": value,
    })
    assert "--backbone-learning-rate" not in command


@pytest.mark.parametrize("options", [
    {"head_only_epochs": -1}, {"head_only_epochs": 20},
    {"head_only_epochs": 1.5}, {"head_only_epochs": True},
    {"backbone_learning_rate": "nan"}, {"backbone_learning_rate": "inf"},
    {"backbone_learning_rate": 0}, {"backbone_learning_rate": 0.001},
])
def test_web_rejects_invalid_before_launch(options):
    with pytest.raises(ValueError):
        build_web_command("train", {"manifest": "m.csv", "output": "m.pt", **options})


def test_web_ui_fields(tmp_path):
    from glove_chirality.web_app import create_app
    from glove_chirality.web_service import CommandService

    html = create_app(CommandService(tmp_path)).test_client().get("/").get_data(as_text=True)
    assert 'name="head_only_epochs"' in html
    assert 'name="backbone_learning_rate"' in html
    assert "Total epochs (both stages)" in html


def test_tk_training_form_callback_without_display():
    """Exercise the actual nested Tk method with minimal display-free widgets."""
    import ast
    import inspect
    from types import SimpleNamespace

    from glove_chirality import gui
    from glove_chirality.models import CLASSIFIER_CHOICES

    tree = ast.parse(inspect.getsource(gui))
    method = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "_training_tab")
    namespace = {"gui_commands": gui_commands, "CLASSIFIER_CHOICES": CLASSIFIER_CHOICES}
    # Only compile the repository's own Tk method, never user-supplied source.
    exec(compile(ast.Module(body=[method], type_ignores=[]), gui.__file__, "exec"), namespace)  # noqa: S102
    buttons = {}

    class Variable:
        def __init__(self, value=""):
            self.value = value

        def get(self):
            return self.value

    class Widget:
        def __init__(self, *args, **kwargs):
            if "command" in kwargs:
                buttons[kwargs["text"]] = kwargs["command"]

        def grid(self, **kwargs):
            pass

        def pack(self, **kwargs):
            pass

    commands = []
    app = SimpleNamespace(
        _scrollable_tab=lambda *args: object(), _path_row=lambda *args: None,
        _guard=lambda _messagebox, callback, **kw: commands.append(callback()),
    )
    tk = SimpleNamespace(StringVar=Variable, IntVar=Variable, DoubleVar=Variable,
                         BooleanVar=Variable)
    ttk = SimpleNamespace(**{k: Widget for k in (
        "Label", "Combobox", "Entry", "Checkbutton", "Frame", "Button",
    )})
    namespace["_training_tab"](app, tk, ttk, None, None)
    app.train_manifest.value = "m.csv"
    app.train_output.value = "m.pt"
    buttons["Start training"]()
    assert "--head-only-epochs" not in commands[-1]
    app.train_head_epochs.value = 3
    app.train_backbone_lr.value = "0.0001"
    buttons["Start training"]()
    parsed = cli.build_parser().parse_args(commands[-1][3:])
    assert parsed.head_only_epochs == 3
    assert parsed.backbone_learning_rate == 0.0001

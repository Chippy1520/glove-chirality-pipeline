"""UI-neutral native form metadata derived from the existing workstation contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

from flask import render_template

from glove_chirality.comparison import COMPARISON_METRICS
from glove_chirality.models import CLASSIFIER_CHOICES


@dataclass
class _Node:
    tag: str
    attrs: dict[str, str | None] = field(default_factory=dict)
    parent: _Node | None = None
    children: list[_Node] = field(default_factory=list)
    parts: list[str] = field(default_factory=list)

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()

    def text(self) -> str:
        return " ".join(" ".join(self.parts + [child.text() for child in self.children]).split())

    def closest(self, tag: str):
        node = self.parent
        while node is not None:
            if node.tag == tag:
                return node
            node = node.parent
        return None


class _Document(HTMLParser):
    def __init__(self, html: str):
        super().__init__(convert_charrefs=True)
        self.root = _Node("document")
        self.stack = [self.root]
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        node = _Node(tag, dict(attrs), self.stack[-1])
        self.stack[-1].children.append(node)
        if tag not in {"input", "img", "br", "meta", "link", "hr", "source"}:
            self.stack.append(node)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        if data.strip():
            self.stack[-1].parts.append(data.strip())


_TITLES = {
    "extract_dataset": "Build a labeled dataset", "extract_single": "Extract a source",
    "preview": "Calibration preview", "train": "Train a classifier",
    "infer_video": "Batch video inference", "infer_images": "Classify images",
    "infer_live": "Live CLI inference", "explain": "Explain a prediction",
    "tensorboard": "TensorBoard", "audit_dataset": "Audit dataset",
}
_ADVANCED_TRAIN = {
    "head_only_epochs", "backbone_learning_rate", "image_size", "learning_rate",
    "validation_fraction", "seed", "workers", "loss", "recall_target", "recall_weight",
    "selection_metric", "augmentation", "tensorboard_logdir", "amp",
}
_PATH_KEYS = {"input", "output", "video", "image", "manifest", "checkpoint", "config", "left", "right", "logdir"}
_FACTORY_KEYS = {
    "camera": "source", "source-type": "source_type", "checkpoint": "checkpoint",
    "config": "config", "device": "device", "amp": "amp", "mode": "mode",
    "reject-class": "reject_class", "decision-class": "decision_class",
    "decision-threshold": "decision_threshold", "belt-direction": "belt_direction",
    "trigger-enabled": "trigger_line_enabled", "trigger-fraction": "trigger_line_fraction",
    "delay": "delay_ms", "continue-counters": "continue_counters",
    "camera-preset": "camera_preset", "camera-fps": "camera_fps",
    "camera-backend": "camera_backend", "camera-width": "camera_width",
    "camera-height": "camera_height", "camera-fourcc": "camera_fourcc",
    "serial-port": "port", "baud": "baud", "models-root": "models_root",
    "show-rejected": "show_size_rejected", "audio": "audio",
}


def _field(node: _Node, name: str, *, advanced: bool = False) -> dict[str, Any]:
    label = node.closest("label")
    title = name.replace("_", " ").capitalize()
    help_text = ""
    if label:
        title_node = next((child for child in label.children if child.tag == "span"), None)
        if title_node and title_node.text():
            title = title_node.text()
        else:
            title = " ".join(label.parts) or title
        help_text = " ".join(child.text() for child in label.walk() if child.tag == "small")
    kind = "select" if node.tag == "select" else node.attrs.get("type", "text")
    choices = []
    value: Any = node.attrs.get("value", "") or ""
    if kind == "select":
        options = [child for child in node.walk() if child.tag == "option"]
        choices = [{"value": option.attrs.get("value") or option.text(), "label": option.text()} for option in options]
        chosen = next((option for option in options if "selected" in option.attrs), options[0] if options else None)
        value = (chosen.attrs.get("value") or chosen.text()) if chosen else ""
    elif kind == "checkbox":
        value = "checked" in node.attrs
    elif node.tag == "textarea":
        kind, value = "text", node.text()
    browse = ""
    if label:
        button = next((child for child in label.walk() if child.tag == "button" and "data-kind" in child.attrs), None)
        if button:
            browse = button.attrs.get("data-kind", "file") or "file"
    if not browse and name in _PATH_KEYS:
        browse = "directory" if name in {"left", "right", "logdir", "input"} else "file"
    return {
        "name": name, "title": title, "kind": kind, "value": value,
        "choices": choices, "help": help_text, "required": "required" in node.attrs,
        "advanced": advanced, "browse": browse,
        "minimum": node.attrs.get("min"), "maximum": node.attrs.get("max"),
        "step": node.attrs.get("step"),
    }


def workstation_schema() -> dict[str, Any]:
    """Call in the host Flask request context; render metadata, never an embedded UI."""
    html = render_template("index.html", classifier_choices=CLASSIFIER_CHOICES,
                           comparison_metrics=COMPARISON_METRICS, initial_can_edit=True)
    nodes = list(_Document(html).root.walk())
    forms = []
    for form in nodes:
        action = form.attrs.get("data-action")
        if form.tag != "form" or not action:
            continue
        fields = []
        for node in form.walk():
            name = node.attrs.get("name")
            if node.tag in {"input", "select", "textarea"} and name:
                advanced = (action == "train" and name in _ADVANCED_TRAIN) or name in {"amp", "decision_class", "decision_threshold", "warmup_seconds"}
                fields.append(_field(node, name, advanced=advanced))
        forms.append({"action": action, "title": _TITLES.get(action, action), "fields": fields})
    forms.append({"action": "audit_dataset", "title": _TITLES["audit_dataset"], "fields": [
        {"name": "manifest", "title": "Dataset manifest", "kind": "text", "value": "", "required": True, "browse": "file"},
        {"name": "output", "title": "Audit output", "kind": "text", "value": "outputs/audit.json", "required": True, "browse": "file"},
    ]})
    factory = []
    for node in nodes:
        identity = node.attrs.get("id", "") or ""
        short = identity.removeprefix("factory-")
        if node.tag not in {"input", "select"} or not identity.startswith("factory-") or short not in _FACTORY_KEYS:
            continue
        key = _FACTORY_KEYS[short]
        item = _field(node, key, advanced=key not in {"source_type", "source", "checkpoint", "config", "mode"})
        if key in {"source", "checkpoint", "config", "port"}:
            item.update(kind="text", choices=[])
            item["value"] = "0" if key == "source" else "configs/factory.yaml" if key == "config" else ""
            item["browse"] = "file" if key in {"checkpoint", "config"} else ""
        factory.append(item)
    factory.append({"name": "geometry", "title": "Geometry profile", "kind": "select", "value": "yaml", "advanced": True,
                    "choices": [{"value": "yaml", "label": "Use YAML"}, {"value": "grip", "label": "GRIP current-camera override"}]})
    return {"forms": forms, "factory": factory, "models": list(CLASSIFIER_CHOICES), "comparison_metrics": COMPARISON_METRICS}

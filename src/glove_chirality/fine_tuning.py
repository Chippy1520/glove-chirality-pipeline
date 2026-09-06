"""Opt-in head warm-up and differential learning rates (no eager ML imports)."""
from __future__ import annotations

import math


def fine_tuning_config(epochs, learning_rate, head_only_epochs=0, backbone_learning_rate=None):
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer (total across both stages)")
    if (isinstance(head_only_epochs, bool) or not isinstance(head_only_epochs, int)
            or not 0 <= head_only_epochs < epochs):
        raise ValueError("head_only_epochs must be an integer >= 0 and less than epochs")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if backbone_learning_rate is not None and (
        not math.isfinite(backbone_learning_rate) or not 0 < backbone_learning_rate < learning_rate
    ):
        raise ValueError("backbone_learning_rate must be finite, positive and below learning_rate")
    enabled = head_only_epochs > 0 or backbone_learning_rate is not None
    return {
        "enabled": enabled,
        "epochs": epochs,
        "head_only_epochs": head_only_epochs,
        "learning_rate": learning_rate,
        "requested_backbone_learning_rate": backbone_learning_rate,
        "backbone_learning_rate": (
            backbone_learning_rate if backbone_learning_rate is not None
            else learning_rate / 10 if enabled else learning_rate
        ),
    }


def classification_head(model, model_name):
    """Locate the classification module, not a transformer/ConvNeXt backbone block."""
    from glove_chirality.models import model_backend

    backend = model_backend(model_name)
    if backend["library"] == "timm":
        head = model.get_classifier()
        if isinstance(head, str):
            head = model.get_submodule(head)
    elif model_name == "tiny_cnn":
        head = model[-1]
    else:
        head = getattr(model, {
            "resnet18": "fc", "mobilenet_v3_small": "classifier",
            "vit_b_16": "heads", "swin_t": "head",
        }[model_name])
    if not hasattr(head, "parameters") or not list(head.parameters()):
        raise ValueError(f"No parameterized classification head found for {model_name}")
    return head


class FineTuning:
    """Keep one AdamW: adding backbone later preserves learned head weights AND moments."""

    def __init__(self, model, model_name, config):
        import torch

        self.model, self.config = model, config
        self.stage = None
        self.head = classification_head(model, model_name) if config["enabled"] else None
        if self.head is None:
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
            return
        self.head_parameters = list(self.head.parameters())
        head_ids = {id(p) for p in self.head_parameters}
        all_parameters = list(model.parameters())
        if not head_ids <= {id(p) for p in all_parameters}:
            raise ValueError("Classification head parameters must belong to the model")
        self.backbone_parameters = [p for p in all_parameters if id(p) not in head_ids]
        if not self.backbone_parameters:
            raise ValueError("Staged fine-tuning requires a separate backbone")
        for p in all_parameters:
            p.requires_grad_(id(p) in head_ids)
            p.grad = None
        self.optimizer = torch.optim.AdamW([
            {"params": self.head_parameters, "lr": config["learning_rate"], "name": "head"},
        ], lr=config["learning_rate"])

    def begin_epoch(self, epoch):
        """Apply training modes every epoch, including after validation's model.eval()."""
        frozen = self.head is not None and epoch < self.config["head_only_epochs"]
        self.stage = "head_only" if frozen else "fine_tune" if self.head is not None else "full"
        if self.head is not None and not frozen and len(self.optimizer.param_groups) == 1:
            for p in self.backbone_parameters:
                p.requires_grad_(True)
            self.optimizer.add_param_group({
                "params": self.backbone_parameters,
                "lr": self.config["backbone_learning_rate"], "name": "backbone",
            })
        if frozen:
            self.model.eval()  # Freeze BN buffers, dropout and stochastic depth as well as gradients.
            self.head.train()
        else:
            self.model.train()
        return {
            "epoch": epoch + 1, "stage": self.stage,
            "head_learning_rate": self.optimizer.param_groups[0]["lr"],
            "backbone_learning_rate": 0.0 if frozen else self.optimizer.param_groups[-1]["lr"],
        }

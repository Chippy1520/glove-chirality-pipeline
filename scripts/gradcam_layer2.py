from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from glove_chirality.models import build_model


def find_last_conv(model):

    last_name = None
    last_module = None

    for name, module in model.named_modules():

        if isinstance(module, torch.nn.Conv2d):
            last_name = name
            last_module = module

    if last_module is None:
        raise RuntimeError(
            "Could not find Conv2d layer."
        )

    return last_name, last_module


def normalize_cam(cam):

    cam = np.maximum(cam, 0)

    minimum = cam.min()
    maximum = cam.max()

    if maximum - minimum < 1e-12:
        return np.zeros_like(cam)

    return (
        (cam - minimum)
        / (maximum - minimum)
    )


def overlay_cam(
    original_bgr,
    cam,
    alpha=0.45,
):

    h, w = original_bgr.shape[:2]

    cam = cv2.resize(
        cam,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )

    heatmap = np.uint8(
        np.clip(cam, 0, 1) * 255
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET,
    )

    overlay = cv2.addWeighted(
        original_bgr,
        1.0 - alpha,
        heatmap,
        alpha,
        0,
    )

    return overlay


parser = argparse.ArgumentParser()

parser.add_argument(
    "--image",
    required=True,
)

parser.add_argument(
    "--checkpoint",
    default=r"checkpoints\chirality_mobilenet_v3_small_aug27_v1.pt",
)

parser.add_argument(
    "--true-label",
    default="left",
    choices=["left", "right"],
)

parser.add_argument(
    "--output",
    default=r"runs\gradcam",
)

parser.add_argument(
    "--tensorboard-logdir",
    default=r"runs\tensorboard\chirality_mobilenet_v3_small_aug27_v1",
)

args = parser.parse_args()


# ---------------------------------------------------------
# DEVICE
# ---------------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ---------------------------------------------------------
# LOAD CHECKPOINT
# ---------------------------------------------------------

saved = torch.load(
    args.checkpoint,
    map_location=device,
    weights_only=False,
)

classes = saved["classes"]
image_size = int(saved["image_size"])


model = build_model(
    saved["model_name"],
    num_classes=len(classes),
    pretrained=False,
).to(device)

model.load_state_dict(
    saved["state_dict"]
)

model.eval()


# ---------------------------------------------------------
# FIND GRAD-CAM LAYER
# ---------------------------------------------------------

layer_name, target_layer = (
    find_last_conv(model)
)

print("Grad-CAM target layer:")
print(layer_name)


# ---------------------------------------------------------
# IMAGE
# ---------------------------------------------------------

image_path = Path(args.image)

original_bgr = cv2.imread(
    str(image_path)
)

if original_bgr is None:
    raise RuntimeError(
        f"Could not read image: {image_path}"
    )


pil = Image.open(
    image_path
).convert("RGB")


transform = transforms.Compose(
    [
        transforms.Resize(
            (image_size, image_size)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)


tensor = (
    transform(pil)
    .unsqueeze(0)
    .to(device)
)


# ---------------------------------------------------------
# GRAD-CAM STORAGE
# ---------------------------------------------------------

activation_store = {}
gradient_store = {}


def forward_hook(
    module,
    inputs,
    output,
):

    activation_store["value"] = output

    def save_gradient(gradient):
        gradient_store["value"] = gradient

    output.register_hook(
        save_gradient
    )


handle = target_layer.register_forward_hook(
    forward_hook
)


# ---------------------------------------------------------
# FIRST FORWARD PASS: PREDICTION
# ---------------------------------------------------------

logits = model(tensor)

probabilities = torch.softmax(
    logits,
    dim=1,
)[0]


pred_index = int(
    probabilities.argmax()
)

pred_label = classes[pred_index]

print()
print("==============================")
print("PREDICTION")
print("==============================")

for i, label in enumerate(classes):

    print(
        label,
        "=",
        round(
            float(probabilities[i]),
            6,
        ),
    )

print()

print(
    "PREDICTED:",
    pred_label,
)

print(
    "TRUE:",
    args.true_label,
)


# ---------------------------------------------------------
# FUNCTION FOR CLASS-SPECIFIC CAM
# ---------------------------------------------------------

def compute_cam(target_index):

    activation_store.clear()
    gradient_store.clear()

    model.zero_grad(
        set_to_none=True
    )

    logits = model(tensor)

    score = logits[
        0,
        target_index,
    ]

    score.backward()

    activations = (
        activation_store["value"]
        .detach()
    )

    gradients = (
        gradient_store["value"]
        .detach()
    )

    weights = gradients.mean(
        dim=(2, 3),
        keepdim=True,
    )

    cam = (
        weights
        * activations
    ).sum(
        dim=1
    )

    cam = torch.relu(cam)[0]

    cam = (
        cam.cpu()
        .numpy()
    )

    return normalize_cam(cam)


# ---------------------------------------------------------
# PREDICTED CLASS CAM
# ---------------------------------------------------------

pred_cam = compute_cam(
    pred_index
)

pred_overlay = overlay_cam(
    original_bgr,
    pred_cam,
)


# ---------------------------------------------------------
# TRUE CLASS CAM
# ---------------------------------------------------------

true_index = classes.index(
    args.true_label
)

true_cam = compute_cam(
    true_index
)

true_overlay = overlay_cam(
    original_bgr,
    true_cam,
)


handle.remove()


# ---------------------------------------------------------
# LABEL IMAGES
# ---------------------------------------------------------

def label_image(
    image,
    text,
):

    result = image.copy()

    cv2.rectangle(
        result,
        (0, 0),
        (result.shape[1], 30),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        result,
        text,
        (7, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255,255,255),
        1,
        cv2.LINE_AA,
    )

    return result


original_labelled = label_image(
    original_bgr,
    "Original",
)

pred_labelled = label_image(
    pred_overlay,
    f"Grad-CAM predicted: {pred_label}",
)

true_labelled = label_image(
    true_overlay,
    f"Grad-CAM true: {args.true_label}",
)


panel = np.hstack(
    [
        original_labelled,
        pred_labelled,
        true_labelled,
    ]
)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

output = Path(args.output)

output.mkdir(
    parents=True,
    exist_ok=True,
)

stem = image_path.stem


pred_path = (
    output
    / f"{stem}_gradcam_pred_{pred_label}.jpg"
)

true_path = (
    output
    / f"{stem}_gradcam_true_{args.true_label}.jpg"
)

panel_path = (
    output
    / f"{stem}_gradcam_panel.jpg"
)


cv2.imwrite(
    str(pred_path),
    pred_overlay,
)

cv2.imwrite(
    str(true_path),
    true_overlay,
)

cv2.imwrite(
    str(panel_path),
    panel,
)


print()
print("==============================")
print("OUTPUT")
print("==============================")

print("Pred CAM:", pred_path)
print("True CAM:", true_path)
print("Panel:", panel_path)


# ---------------------------------------------------------
# ADD IMAGES TO TENSORBOARD
# ---------------------------------------------------------

try:

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(
        args.tensorboard_logdir
    )

    original_rgb = cv2.cvtColor(
        original_bgr,
        cv2.COLOR_BGR2RGB,
    )

    pred_rgb = cv2.cvtColor(
        pred_overlay,
        cv2.COLOR_BGR2RGB,
    )

    true_rgb = cv2.cvtColor(
        true_overlay,
        cv2.COLOR_BGR2RGB,
    )

    panel_rgb = cv2.cvtColor(
        panel,
        cv2.COLOR_BGR2RGB,
    )

    writer.add_image(
        "GradCAM/original",
        original_rgb,
        dataformats="HWC",
    )

    writer.add_image(
        f"GradCAM/predicted_{pred_label}",
        pred_rgb,
        dataformats="HWC",
    )

    writer.add_image(
        f"GradCAM/true_{args.true_label}",
        true_rgb,
        dataformats="HWC",
    )

    writer.add_image(
        "GradCAM/comparison",
        panel_rgb,
        dataformats="HWC",
    )

    writer.flush()
    writer.close()

    print(
        "Grad-CAM added to TensorBoard."
    )

except Exception as exc:

    print(
        "TensorBoard image logging failed:",
        exc,
    )

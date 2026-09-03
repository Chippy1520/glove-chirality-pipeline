import argparse
from pathlib import Path
from collections import Counter, defaultdict

import torch
from PIL import Image
from torchvision import transforms

from glove_chirality.models import build_model


parser = argparse.ArgumentParser(
    description="Audit a trained Layer-2 chirality classifier on the fixed validation split."
)

parser.add_argument(
    "--root",
    type=Path,
    required=True,
    help="Layer-2 dataset root containing train/ and val/ directories.",
)

parser.add_argument(
    "--checkpoint",
    type=Path,
    required=True,
    help="Trained classifier checkpoint.",
)

args = parser.parse_args()

ROOT = args.root
CHECKPOINT = args.checkpoint

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


saved = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)

classes = saved["classes"]
image_size = int(saved["image_size"])

model = build_model(
    saved["model_name"],
    len(classes),
    pretrained=False,
).to(DEVICE)

model.load_state_dict(
    saved["state_dict"]
)

model.eval()


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


results = []

for true_label in classes:

    base = ROOT / "val" / true_label

    images = sorted(
        p
        for p in base.rglob("*.jpg")
        if "images" in p.parts
    )

    for path in images:

        with Image.open(path) as image:

            tensor = (
                transform(
                    image.convert("RGB")
                )
                .unsqueeze(0)
                .to(DEVICE)
            )

        with torch.no_grad():

            logits = model(tensor)

            probs = torch.softmax(
                logits,
                dim=1,
            )[0]

        pred_index = int(
            probs.argmax()
        )

        pred = classes[pred_index]

        confidence = float(
            probs[pred_index]
        )

        right_prob = float(
            probs[classes.index("right")]
        )

        results.append(
            {
                "path": path,
                "true": true_label,
                "pred": pred,
                "confidence": confidence,
                "right_prob": right_prob,
            }
        )


# --------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------

matrix = {
    true: Counter()
    for true in classes
}

for r in results:
    matrix[r["true"]][r["pred"]] += 1


print()
print("==============================")
print("CONFUSION MATRIX")
print("==============================")

print("              PRED LEFT  PRED RIGHT")

for true in classes:

    print(
        f"TRUE {true.upper():5s}",
        f"{matrix[true]['left']:10d}",
        f"{matrix[true]['right']:11d}",
    )


# --------------------------------------------
# ERRORS
# --------------------------------------------

errors = [
    r
    for r in results
    if r["true"] != r["pred"]
]

print()
print("==============================")
print("ERRORS")
print("==============================")

print("TOTAL:", len(errors))

for r in errors:

    print()
    print("TRUE:", r["true"])
    print("PRED:", r["pred"])
    print(
        "CONF:",
        round(r["confidence"], 6)
    )
    print(
        "RIGHT_PROB:",
        round(r["right_prob"], 6)
    )
    print("PATH:", r["path"])


# --------------------------------------------
# LOW-CONFIDENCE CORRECT RESULTS
# --------------------------------------------

correct = [
    r
    for r in results
    if r["true"] == r["pred"]
]

correct.sort(
    key=lambda r: r["confidence"]
)

print()
print("==============================")
print("20 LOWEST-CONFIDENCE CORRECT")
print("==============================")

for r in correct[:20]:

    print(
        r["true"],
        "pred=" + r["pred"],
        "conf=" + str(
            round(r["confidence"], 5)
        ),
        "right_prob=" + str(
            round(r["right_prob"], 5)
        ),
        r["path"].name,
    )


# --------------------------------------------
# CONFIDENCE DISTRIBUTION
# --------------------------------------------

import numpy as np

print()
print("==============================")
print("CONFIDENCE DISTRIBUTION")
print("==============================")

for label in classes:

    values = np.array(
        [
            r["confidence"]
            for r in results
            if r["true"] == label
        ]
    )

    print()
    print(label.upper())

    for p in [
        0,
        1,
        5,
        10,
        50,
        90,
        95,
        99,
        100,
    ]:

        print(
            f"P{p}:",
            round(
                np.percentile(
                    values,
                    p,
                ),
                6,
            ),
        )

from pathlib import Path
import cv2
import hashlib
import random
import csv
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description="Audit Layer-2 chirality crop dataset quality and duplicates."
)

parser.add_argument(
    "--root",
    type=Path,
    required=True,
    help="Layer-2 dataset root containing train/ and val/ directories.",
)

args = parser.parse_args()

ROOT = args.root

AUDIT = ROOT / "_audit"
AUDIT.mkdir(parents=True, exist_ok=True)

random.seed(42)

groups = {
    "train_left": ROOT / "train" / "left",
    "train_right": ROOT / "train" / "right",
    "val_left": ROOT / "val" / "left",
    "val_right": ROOT / "val" / "right",
}

records = []
hash_map = {}


def find_images(base):
    if not base.exists():
        return []

    return sorted(
        p
        for p in base.rglob("*.jpg")
        if "images" in p.parts
    )


for group, base in groups.items():

    images = find_images(base)

    print()
    print("============================")
    print(group.upper())
    print("============================")
    print("images:", len(images))

    bad = 0
    wrong_size = 0

    for path in images:

        img = cv2.imread(str(path))

        if img is None:
            bad += 1
            records.append(
                [group, str(path), "CORRUPT", "", "", ""]
            )
            continue

        h, w = img.shape[:2]

        if (w, h) != (256, 256):
            wrong_size += 1

        digest = hashlib.sha1(
            path.read_bytes()
        ).hexdigest()

        hash_map.setdefault(
            digest, []
        ).append((group, path))

        records.append(
            [
                group,
                str(path),
                "OK",
                w,
                h,
                digest,
            ]
        )

    print("corrupt:", bad)
    print("wrong size:", wrong_size)


# --------------------------------------------------
# DUPLICATE CHECK
# --------------------------------------------------

duplicate_groups = [
    items
    for items in hash_map.values()
    if len(items) > 1
]

cross_class_duplicates = []

for items in duplicate_groups:

    labels = {
        group.split("_")[1]
        for group, _ in items
    }

    if len(labels) > 1:
        cross_class_duplicates.append(items)


print()
print("============================")
print("DUPLICATES")
print("============================")

print(
    "exact duplicate hash groups:",
    len(duplicate_groups)
)

print(
    "cross-class duplicate groups:",
    len(cross_class_duplicates)
)

for items in cross_class_duplicates[:20]:

    print()
    print("CROSS-CLASS DUPLICATE:")

    for group, path in items:
        print(" ", group, path)


# --------------------------------------------------
# WRITE CSV
# --------------------------------------------------

with (AUDIT / "crop_integrity.csv").open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "group",
            "path",
            "status",
            "width",
            "height",
            "sha1",
        ]
    )

    writer.writerows(records)


# --------------------------------------------------
# CONTACT SHEETS
# --------------------------------------------------

def make_sheet(group, paths, n=40):

    if not paths:
        return

    sample = random.sample(
        paths,
        min(n, len(paths)),
    )

    thumb_size = 180
    cols = 5

    rows = int(np.ceil(len(sample) / cols))

    cell_h = thumb_size + 28
    cell_w = thumb_size

    sheet = np.zeros(
        (
            rows * cell_h,
            cols * cell_w,
            3,
        ),
        dtype=np.uint8,
    )

    for i, path in enumerate(sample):

        img = cv2.imread(str(path))

        if img is None:
            continue

        img = cv2.resize(
            img,
            (thumb_size, thumb_size),
            interpolation=cv2.INTER_AREA,
        )

        row = i // cols
        col = i % cols

        y = row * cell_h
        x = col * cell_w

        sheet[
            y:y + thumb_size,
            x:x + thumb_size,
        ] = img

        source = path.parents[2].name

        cv2.putText(
            sheet,
            source,
            (x + 4, y + thumb_size + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255,255,255),
            1,
            cv2.LINE_AA,
        )

    out = AUDIT / f"{group}_random40.jpg"

    cv2.imwrite(
        str(out),
        sheet,
    )

    print(
        group,
        "contact sheet:",
        out,
    )


print()
print("============================")
print("CONTACT SHEETS")
print("============================")

for group, base in groups.items():
    make_sheet(
        group,
        find_images(base),
    )


print()
print("AUDIT FOLDER:")
print(AUDIT)

from pathlib import Path
import csv
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description="Audit temporally close accepted glove passage events."
)

parser.add_argument(
    "--root",
    type=Path,
    required=True,
    help="Extraction output directory containing manifest.csv.",
)

args = parser.parse_args()

ROOT = args.root

manifest = ROOT / "manifest.csv"
outdir = ROOT / "close_pair_audit"
outdir.mkdir(parents=True, exist_ok=True)

rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
rows.sort(key=lambda r: float(r["timestamp_s"]))

pairs = []

for a, b in zip(rows[:-1], rows[1:]):

    ta = float(a["timestamp_s"])
    tb = float(b["timestamp_s"])
    gap = tb - ta

    if gap < 1.0:
        pairs.append((a, b, gap))

print("CLOSE PAIRS:", len(pairs))

for index, (a, b, gap) in enumerate(pairs, 1):

    pa = ROOT / a["image_path"]
    pb = ROOT / b["image_path"]

    ia = cv2.imread(str(pa))
    ib = cv2.imread(str(pb))

    if ia is None or ib is None:
        print("Could not read:", pa, pb)
        continue

    h = 350

    def resize(img):
        scale = h / img.shape[0]
        return cv2.resize(
            img,
            (
                int(round(img.shape[1] * scale)),
                h,
            ),
            interpolation=cv2.INTER_AREA,
        )

    ia = resize(ia)
    ib = resize(ib)

    cv2.putText(
        ia,
        f"A  t={float(a['timestamp_s']):.2f}s",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2,
    )

    cv2.putText(
        ib,
        f"B  t={float(b['timestamp_s']):.2f}s",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2,
    )

    canvas = np.hstack([ia, ib])

    cv2.putText(
        canvas,
        f"gap = {gap:.2f}s",
        (10, canvas.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2,
    )

    output = outdir / f"pair_{index:02d}_gap_{gap:.2f}s.jpg"

    cv2.imwrite(str(output), canvas)

print("OUTPUT:", outdir)

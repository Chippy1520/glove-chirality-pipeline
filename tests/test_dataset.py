import pytest

from glove_chirality.dataset import grouped_split


def _row(label, source):
    return {"label": label, "source_video": source}


def test_grouped_split_has_no_video_leakage():
    rows = []
    for label in ("left", "right"):
        for source in (f"{label}_a.mkv", f"{label}_b.mkv", f"{label}_c.mkv"):
            rows.extend([_row(label, source)] * 4)
    train, validation = grouped_split(rows, 0.34, 7)
    train_sources = {r["source_video"] for r in train}
    validation_sources = {r["source_video"] for r in validation}
    assert train_sources.isdisjoint(validation_sources)
    assert {r["label"] for r in train} == {"left", "right"}
    assert {r["label"] for r in validation} == {"left", "right"}


def test_grouped_split_rejects_one_video_class():
    with pytest.raises(ValueError, match="at least two source videos"):
        grouped_split([_row("left", "only.mkv"), _row("right", "r1.mkv"), _row("right", "r2.mkv")], 0.2, 1)

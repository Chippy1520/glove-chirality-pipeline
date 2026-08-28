import numpy as np

from glove_chirality.detection.yolo import _polygon_box


def test_polygon_box_uses_tight_mask_bounds_and_clamps_to_frame():
    polygon = np.array([
        [-3.2, 12.4],
        [82.7, 8.1],
        [105.3, 54.8],
        [20.2, 70.6],
    ])
    assert _polygon_box(polygon, width=100, height=60) == (0, 8, 100, 60)


def test_polygon_box_rejects_degenerate_mask():
    assert _polygon_box(np.array([[1, 1], [2, 2]]), 100, 100) is None

from types import SimpleNamespace

import numpy as np
import pytest

from glove_chirality.config import DetectorConfig, EventConfig, ExtractionConfig
from glove_chirality.detection.yolo import YoloDetector, _polygon_box
from glove_chirality.events import PassageProcessor
from glove_chirality.types import Detection


class _Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _Coordinates:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, _index):
        return self

    def tolist(self):
        return self.values


class _Box:
    def __init__(self, xyxy, confidence=0.9, class_id=0):
        self.xyxy = _Coordinates(xyxy)
        self.conf = _Scalar(confidence)
        self.cls = _Scalar(class_id)


class _Model:
    def __init__(self, result):
        self.result = result
        self.frame = None
        self.options = None

    def predict(self, frame, **options):
        self.frame = frame.copy()
        self.options = options
        return [self.result]


def _detector(config, result):
    detector = YoloDetector.__new__(YoloDetector)
    detector.config = config
    detector.model = _Model(result)
    return detector


def test_polygon_box_uses_tight_mask_bounds_and_clamps_to_frame():
    polygon = np.array(
        [
            [-3.2, 12.4],
            [82.7, 8.1],
            [105.3, 54.8],
            [20.2, 70.6],
        ]
    )
    assert _polygon_box(polygon, width=100, height=60) == (0, 8, 100, 60)


def test_polygon_box_rejects_degenerate_mask():
    assert _polygon_box(np.array([[1, 1], [2, 2]]), 100, 100) is None


def test_yolo_retains_polygon_and_maps_roi_coordinates_to_full_frame():
    polygon = np.array([[5, 6], [30, 6], [30, 40], [5, 40]], dtype=np.float32)
    result = SimpleNamespace(
        boxes=[_Box([4, 5, 32, 42])],
        masks=SimpleNamespace(xy=[polygon]),
    )
    config = DetectorConfig(
        backend="yolo", yolo_model="custom.pt",
        roi=(0.25, 0.20, 0.75, 0.80),
        yolo_crop_to_roi=True,
        yolo_class_id=0,
        yolo_imgsz=512,
        yolo_iou=0.4,
        yolo_max_det=3,
        yolo_max_box_area_ratio=0.10,
    )
    detector = _detector(config, result)

    detections = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))

    assert detector.model.frame.shape[:2] == (60, 100)
    assert detector.model.options["imgsz"] == 512
    assert detector.model.options["iou"] == 0.4
    assert detector.model.options["max_det"] == 3
    assert detector.model.options["classes"] == [0]
    assert "half" not in detector.model.options
    assert len(detections) == 1
    detection = detections[0]
    assert (detection.x1, detection.y1, detection.x2, detection.y2) == (55, 26, 80, 60)
    assert detection.polygon == (
        (55.0, 26.0),
        (80.0, 26.0),
        (80.0, 60.0),
        (55.0, 60.0),
    )
    assert detection.class_id == 0
    assert detection.confidence == pytest.approx(0.9)


def test_yolo_rejects_boxes_outside_full_frame_area_gate():
    polygons = [
        np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32),
        np.array([[20, 20], [50, 20], [50, 40], [20, 40]], dtype=np.float32),
        np.array([[10, 10], [70, 10], [70, 70], [10, 70]], dtype=np.float32),
    ]
    result = SimpleNamespace(
        boxes=[_Box([0, 0, 10, 10]), _Box([20, 20, 50, 40]), _Box([10, 10, 70, 70])],
        masks=SimpleNamespace(xy=polygons),
    )
    config = DetectorConfig(
        backend="yolo",
        yolo_model="custom.pt",
        yolo_min_box_area_ratio=0.03,
        yolo_max_box_area_ratio=0.20,
    )

    detector = _detector(config, result)
    detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert (detections[0].x1, detections[0].y1, detections[0].x2, detections[0].y2) == (
        20,
        20,
        50,
        40,
    )
    diagnostics = detector.last_diagnostics
    assert diagnostics.raw_yolo_count == 3
    assert diagnostics.size_rejected_count == 2
    assert diagnostics.returned_detection_count == 1
    assert [item.box_area_ratio for item in diagnostics.size_rejected] == pytest.approx(
        [0.01, 0.36]
    )

    processor = PassageProcessor(
        detector,
        ExtractionConfig(
            detector=config,
            event=EventConfig(
                min_detected_frames=1,
                exit_missing_frames=1,
                cooldown_frames=0,
                output_size=32,
            ),
        ),
        "diagnostic-isolation",
        "test",
    )
    frame_result = processor.process(np.zeros((100, 100, 3), dtype=np.uint8), 0, 0.0)
    assert len(frame_result.detections) == 1
    assert all(outcome.reject_reason != "multiple_candidates" for outcome in frame_result.outcomes)

    unfiltered, unfiltered_diagnostics = detector.detect_with_diagnostics(
        np.zeros((100, 100, 3), dtype=np.uint8),
        apply_size_filter=False,
    )
    assert len(unfiltered) == 3
    assert unfiltered_diagnostics.size_rejected_count == 2
    assert unfiltered_diagnostics.returned_detection_count == 3


def test_default_size_limits_preserve_all_valid_boxes_and_half_true_is_compatible():
    result = SimpleNamespace(
        boxes=[_Box([0, 0, 100, 100])],
        masks=SimpleNamespace(
            xy=[np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)]
        ),
    )
    detector = _detector(
        DetectorConfig(backend="yolo", yolo_model="custom.pt", yolo_half=True),
        result,
    )

    assert len(detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))) == 1
    assert detector.last_diagnostics.size_rejected_count == 0
    assert detector.model.options["half"] is True


def test_half_uses_current_ultralytics_fp16_quantize_option_when_supported():
    result = SimpleNamespace(boxes=[], masks=None)
    detector = _detector(
        DetectorConfig(backend="yolo", yolo_model="custom.pt", yolo_half=True),
        result,
    )
    detector._supports_quantize = True

    detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert detector.model.options["quantize"] == 16
    assert "half" not in detector.model.options


def test_yolo_require_masks_rejects_box_only_detection():
    result = SimpleNamespace(boxes=[_Box([10, 10, 30, 30])], masks=None)
    detector = _detector(
        DetectorConfig(backend="yolo", yolo_model="custom.pt", yolo_require_masks=True),
        result,
    )
    with pytest.raises(RuntimeError, match="box-only"):
        detector.detect(np.zeros((60, 80, 3), dtype=np.uint8))


def test_yolo_require_masks_allows_empty_segmentation_result():
    result = SimpleNamespace(boxes=[], masks=None)
    detector = _detector(
        DetectorConfig(backend="yolo", yolo_model="custom.pt", yolo_require_masks=True),
        result,
    )
    assert detector.detect(np.zeros((60, 80, 3), dtype=np.uint8)) == []


def test_box_only_yolo_fallback_has_no_polygon():
    result = SimpleNamespace(boxes=[_Box([10, 10, 30, 30])], masks=None)
    detector = _detector(DetectorConfig(backend="yolo", yolo_model="custom.pt"), result)
    detection = detector.detect(np.zeros((60, 80, 3), dtype=np.uint8))[0]
    assert detection.polygon is None
    assert (detection.x1, detection.y1, detection.x2, detection.y2) == (10, 10, 30, 30)


def test_box_fallback_uses_floor_ceil_instead_of_shrinking_bounds():
    result = SimpleNamespace(boxes=[_Box([10.8, 11.2, 30.1, 31.9])], masks=None)
    detector = _detector(DetectorConfig(backend="yolo", yolo_model="custom.pt"), result)
    detection = detector.detect(np.zeros((60, 80, 3), dtype=np.uint8))[0]
    assert (detection.x1, detection.y1, detection.x2, detection.y2) == (10, 11, 31, 32)


def test_strict_masks_skip_nonfinite_polygon():
    polygon = np.array([[5, 5], [np.nan, 20], [20, 20]], dtype=np.float32)

    result = SimpleNamespace(
        boxes=[_Box([5, 5, 20, 20])],
        masks=SimpleNamespace(xy=[polygon]),
    )

    detector = _detector(
        DetectorConfig(
            backend="yolo",
            yolo_model="custom.pt",
            yolo_require_masks=True,
        ),
        result,
    )

    detections, diagnostics = detector.detect_with_diagnostics(
        np.zeros((60, 80, 3), dtype=np.uint8)
    )

    assert detections == []
    assert diagnostics.raw_yolo_count == 1
    assert diagnostics.returned_detection_count == 0

def test_strict_masks_skip_only_invalid_detection_and_keep_valid_one():
    invalid_polygon = np.array(
        [[5, 5], [np.nan, 20], [20, 20]],
        dtype=np.float32,
    )

    valid_polygon = np.array(
        [[30, 30], [50, 30], [50, 50], [30, 50]],
        dtype=np.float32,
    )

    result = SimpleNamespace(
        boxes=[
            _Box([5, 5, 20, 20]),
            _Box([30, 30, 50, 50]),
        ],
        masks=SimpleNamespace(
            xy=[
                invalid_polygon,
                valid_polygon,
            ]
        ),
    )

    detector = _detector(
        DetectorConfig(
            backend="yolo",
            yolo_model="custom.pt",
            yolo_require_masks=True,
        ),
        result,
    )

    detections = detector.detect(
        np.zeros((80, 80, 3), dtype=np.uint8)
    )

    assert len(detections) == 1

    detection = detections[0]

    assert (
        detection.x1,
        detection.y1,
        detection.x2,
        detection.y2,
    ) == (30, 30, 50, 50)

    assert detection.polygon is not None


def test_detection_canonicalizes_polygon_to_deeply_immutable_tuple():
    source = [[1, 2], [3, 4], [5, 6]]
    detection = Detection(1, 2, 5, 6, 0.9, polygon=source)
    source[0][0] = 99
    assert detection.polygon == ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))

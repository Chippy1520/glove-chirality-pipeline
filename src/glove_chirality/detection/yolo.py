from __future__ import annotations

import numpy as np

from glove_chirality.config import DetectorConfig
from glove_chirality.detection.base import GloveDetector
from glove_chirality.types import Detection, Polygon


def _polygon_points(polygon: np.ndarray) -> np.ndarray | None:
    points = np.asarray(polygon, dtype=np.float32)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) < 3
        or not np.isfinite(points).all()
    ):
        return None
    return points


def _polygon_box(
    polygon: np.ndarray,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    points = _polygon_points(polygon)
    if points is None:
        return None
    x1 = max(0, int(np.floor(points[:, 0].min())))
    y1 = max(0, int(np.floor(points[:, 1].min())))
    x2 = min(width, int(np.ceil(points[:, 0].max())))
    y2 = min(height, int(np.ceil(points[:, 1].max())))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _model_box(
    values: list[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    coordinates = np.asarray(values, dtype=np.float64)
    if coordinates.shape != (4,) or not np.isfinite(coordinates).all():
        return None
    x1 = max(0, min(width, int(np.floor(coordinates[0]))))
    y1 = max(0, min(height, int(np.floor(coordinates[1]))))
    x2 = max(0, min(width, int(np.ceil(coordinates[2]))))
    y2 = max(0, min(height, int(np.ceil(coordinates[3]))))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _roi_box(
    roi: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    return (
        max(0, min(width, round(x1 * width))),
        max(0, min(height, round(y1 * height))),
        max(0, min(width, round(x2 * width))),
        max(0, min(height, round(y2 * height))),
    )


def _full_polygon(
    polygon: np.ndarray,
    offset_x: int,
    offset_y: int,
    width: int,
    height: int,
) -> Polygon | None:
    points = _polygon_points(polygon)
    if points is None:
        return None
    points = points.copy()
    points[:, 0] = np.clip(points[:, 0] + offset_x, 0, width)
    points[:, 1] = np.clip(points[:, 1] + offset_y, 0, height)
    return tuple((float(x), float(y)) for x, y in points)


class YoloDetector(GloveDetector):
    """Ultralytics adapter returning backend-neutral full-frame detections."""

    name = "yolo"

    def __init__(self, config: DetectorConfig):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install the YOLO option with: pip install -e .[yolo]") from exc
        self.config = config
        self.model = YOLO(config.yolo_model)
        model_task = getattr(self.model, "task", None)
        if config.yolo_require_masks and model_task != "segment":
            raise RuntimeError(
                f"yolo_require_masks=true but {config.yolo_model!r} has task "
                f"{model_task!r}, not 'segment'; use segmentation weights or "
                "disable yolo_require_masks"
            )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        offset_x = offset_y = 0
        inference_frame = frame
        if self.config.yolo_crop_to_roi:
            x1, y1, x2, y2 = _roi_box(self.config.roi, width, height)
            inference_frame = frame[y1:y2, x1:x2]
            if inference_frame.size == 0:
                return []
            offset_x, offset_y = x1, y1

        options: dict[str, object] = {
            "conf": self.config.yolo_confidence,
            "iou": self.config.yolo_iou,
            "imgsz": self.config.yolo_imgsz,
            "max_det": self.config.yolo_max_det,
            "verbose": False,
            "half": self.config.yolo_half,
        }
        if self.config.yolo_class_id is not None:
            options["classes"] = [self.config.yolo_class_id]
        if self.config.yolo_device != "auto":
            options["device"] = self.config.yolo_device
        result = self.model.predict(inference_frame, **options)[0]
        masks = getattr(result, "masks", None)
        if self.config.yolo_require_masks and masks is None and len(result.boxes):
            raise RuntimeError(
                "yolo_require_masks=true, but the loaded YOLO model returned box-only output; "
                "load a segmentation checkpoint or disable yolo_require_masks"
            )
        polygons = masks.xy if self.config.yolo_use_masks and masks is not None else []

        detections: list[Detection] = []
        frame_area = max(1, width * height)
        local_height, local_width = inference_frame.shape[:2]
        for index, box in enumerate(result.boxes):
            class_id = int(box.cls.item())
            if self.config.yolo_class_id is not None and class_id != self.config.yolo_class_id:
                continue
            local_box = _model_box(box.xyxy[0].tolist(), local_width, local_height)
            polygon = None
            if index < len(polygons):
                mask_box = _polygon_box(polygons[index], local_width, local_height)
                mask_polygon = _full_polygon(
                    polygons[index], offset_x, offset_y, width, height
                )
                if mask_box is not None and mask_polygon is not None:
                    local_box = mask_box
                    polygon = mask_polygon
                elif self.config.yolo_require_masks:
                    raise RuntimeError(
                        "yolo_require_masks=true, but a YOLO detection had an "
                        "invalid or degenerate segmentation polygon"
                    )
            elif self.config.yolo_require_masks:
                raise RuntimeError(
                    "yolo_require_masks=true, but a YOLO detection had no segmentation polygon"
                )
            if local_box is None:
                continue
            bx1, by1, bx2, by2 = local_box
            bx1 = max(0, min(width, bx1 + offset_x))
            by1 = max(0, min(height, by1 + offset_y))
            bx2 = max(0, min(width, bx2 + offset_x))
            by2 = max(0, min(height, by2 + offset_y))
            if bx2 <= bx1 or by2 <= by1:
                continue
            box_area_ratio = ((bx2 - bx1) * (by2 - by1)) / frame_area
            if not (
                self.config.yolo_min_box_area_ratio
                <= box_area_ratio
                <= self.config.yolo_max_box_area_ratio
            ):
                continue
            detections.append(
                Detection(
                    bx1,
                    by1,
                    bx2,
                    by2,
                    float(box.conf.item()),
                    class_id,
                    polygon,
                )
            )
        return detections

    def warmup(self, frame: np.ndarray) -> None:
        self.detect(frame)

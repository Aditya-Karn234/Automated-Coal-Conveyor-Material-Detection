"""
Pure geometry helpers: YOLO <-> pixel coordinate conversion, IoU, and
alpha-mask-based tight bounding boxes.

CRITICAL FIX: annotations must be derived from the object's actual
non-transparent (alpha) footprint, never from the transformed canvas
size. Rotating a rectangular cutout expands its bounding canvas with
transparent corners, and edge feathering can bleed alpha a few pixels
past the original crop -- both would otherwise get baked into the
annotation as extra, empty area.
"""

from typing import List, Optional, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> BBox:
    """Normalized YOLO (cx,cy,w,h) -> absolute pixel (x1,y1,x2,y2)."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> BBox:
    """Absolute pixel (x1,y1,x2,y2) -> normalized YOLO (cx,cy,w,h), clipped to image bounds."""
    x1c, y1c = max(0.0, x1), max(0.0, y1)
    x2c, y2c = min(float(img_w), x2), min(float(img_h), y2)
    w = max(0.0, x2c - x1c)
    h = max(0.0, y2c - y1c)
    cx = x1c + w / 2
    cy = y1c + h / 2
    return cx / img_w, cy / img_h, w / img_w, h / img_h


def compute_iou(box_a: BBox, box_b: BBox) -> float:
    """Standard IoU between two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def alpha_tight_bbox(rgba: np.ndarray, alpha_threshold: int = 1) -> Optional[Tuple[int, int, int, int]]:
    """
    Returns the tight (x1, y1, x2, y2) bounding box, in the array's own
    local pixel coordinates, of every pixel whose alpha exceeds
    `alpha_threshold`. Returns None if no pixel qualifies (fully
    transparent array).
    """
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
        return None
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > alpha_threshold)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return x1, y1, x2, y2


def tight_crop_to_alpha(rgba: np.ndarray, alpha_threshold: int = 1) -> Optional[np.ndarray]:
    """Crops rgba to the tight bounding box of its non-transparent pixels."""
    bbox = alpha_tight_bbox(rgba, alpha_threshold)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return rgba[y1:y2, x1:x2]


def clipped_box_is_valid(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int, min_size_px: float = 1.0) -> bool:
    """
    CRITICAL FIX: explicit post-clip validity check. A box that lands
    fully (or almost fully) outside the frame will clip to zero or
    near-zero area -- that must be treated as a failed placement, not
    silently written as a degenerate annotation.
    """
    x1c, y1c = max(0.0, x1), max(0.0, y1)
    x2c, y2c = min(float(img_w), x2), min(float(img_h), y2)
    return (x2c - x1c) >= min_size_px and (y2c - y1c) >= min_size_px

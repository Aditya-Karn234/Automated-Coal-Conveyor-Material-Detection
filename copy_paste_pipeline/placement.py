"""
Placement search: where in the image can we paste an object without
violating the IoU constraint against existing boxes, restricted to a
plausible conveyor / active-object region.

HIGH PRIORITY FIX: placement is no longer sampled uniformly across the
whole frame. Real foreign-object detections only make sense on the
conveyor / material surface, not e.g. in the sky or on machinery
frames at the image edges. We infer that active region from the union
of existing annotations (expanded by a margin), or fall back to a
configurable central-frame default when an image has no existing boxes
to infer from.
"""

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .config import Config
from .geometry import BBox, compute_iou

ROI = Tuple[float, float, float, float]  # x1, y1, x2, y2 in absolute pixels


@dataclass
class PlacementResult:
    x: Optional[int]
    y: Optional[int]
    attempts: int
    max_iou_seen: float
    success: bool
    failure_reason: str = ""


def compute_conveyor_roi(existing_boxes: List[BBox], img_w: int, img_h: int, cfg: Config) -> ROI:
    """
    Infers the active/conveyor region for this image. If existing
    annotations are present, the ROI is their union bounding box
    expanded by ROI_MARGIN_FRACTION of the image size (so pastes can
    land near, not just exactly on top of, known object locations).
    Otherwise falls back to Config.DEFAULT_ROI_FRACTION, a conservative
    central crop.
    """
    if existing_boxes:
        x1 = min(b[0] for b in existing_boxes)
        y1 = min(b[1] for b in existing_boxes)
        x2 = max(b[2] for b in existing_boxes)
        y2 = max(b[3] for b in existing_boxes)

        margin_x = img_w * cfg.ROI_MARGIN_FRACTION
        margin_y = img_h * cfg.ROI_MARGIN_FRACTION
        x1 = max(0.0, x1 - margin_x)
        y1 = max(0.0, y1 - margin_y)
        x2 = min(float(img_w), x2 + margin_x)
        y2 = min(float(img_h), y2 + margin_y)
        return x1, y1, x2, y2

    fx1, fy1, fx2, fy2 = cfg.DEFAULT_ROI_FRACTION
    return fx1 * img_w, fy1 * img_h, fx2 * img_w, fy2 * img_h


def find_candidate_position(
    img_w: int,
    img_h: int,
    obj_w: int,
    obj_h: int,
    existing_boxes: List[BBox],
    cfg: Config,
    roi: Optional[ROI] = None,
) -> PlacementResult:
    """
    Randomly searches, within `roi` (or the full frame minus border
    margin if ROI restriction is disabled), for a top-left (x, y) such
    that the pasted object stays fully inside the image and its IoU
    with every existing box stays below MAX_IOU_WITH_EXISTING.

    Tracks attempts used and the maximum IoU observed across rejected
    attempts, so callers can log placement-difficulty statistics.
    """
    margin = cfg.BORDER_MARGIN_PX

    if cfg.RESTRICT_TO_CONVEYOR_ROI and roi is not None:
        rx1, ry1, rx2, ry2 = roi
    else:
        rx1, ry1, rx2, ry2 = 0, 0, img_w, img_h

    x_min = max(margin, int(rx1))
    y_min = max(margin, int(ry1))
    x_max = min(img_w - margin, int(rx2)) - obj_w
    y_max = min(img_h - margin, int(ry2)) - obj_h

    if x_max <= x_min or y_max <= y_min:
        return PlacementResult(None, None, 0, 0.0, False, "object_too_large_for_roi")

    max_iou_seen = 0.0
    for attempt in range(1, cfg.MAX_PLACEMENT_TRIES + 1):
        x = random.randint(x_min, x_max)
        y = random.randint(y_min, y_max)
        candidate_box = (x, y, x + obj_w, y + obj_h)

        if not existing_boxes:
            return PlacementResult(x, y, attempt, 0.0, True)

        ious = [compute_iou(candidate_box, existing) for existing in existing_boxes]
        local_max = max(ious)
        max_iou_seen = max(max_iou_seen, local_max)

        if local_max < cfg.MAX_IOU_WITH_EXISTING:
            return PlacementResult(x, y, attempt, local_max, True)

    return PlacementResult(None, None, cfg.MAX_PLACEMENT_TRIES, max_iou_seen, False, "iou_retry_budget_exhausted")

"""
Visualization & validation helpers.

VISUALIZATION FIX: the pipeline now automatically writes before/after
preview pairs for a random subset of augmented images, with boxes
drawn so a human can eyeball placement quality (feathering, background
match, ROI sanity) before committing to a training run.
"""

import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

EXISTING_COLOR = (60, 180, 75)     # green - original annotations
PASTED_COLOR = (0, 0, 230)         # red - newly pasted foreign objects


def draw_boxes(image: np.ndarray, boxes: List[Tuple[float, float, float, float]], color: Tuple[int, int, int], label: str = "") -> np.ndarray:
    out = image.copy()
    for (x1, y1, x2, y2) in boxes:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, color, 2)
        if label:
            cv2.putText(out, label, (p1[0], max(0, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def save_before_after(
    before_img: np.ndarray,
    after_img: np.ndarray,
    before_boxes: List[Tuple[float, float, float, float]],
    existing_boxes_in_after: List[Tuple[float, float, float, float]],
    pasted_boxes: List[Tuple[float, float, float, float]],
    out_path: Path,
) -> None:
    """Draws boxes on both frames and writes them side by side as one image."""
    before_vis = draw_boxes(before_img, before_boxes, EXISTING_COLOR, "orig")
    after_vis = draw_boxes(after_img, existing_boxes_in_after, EXISTING_COLOR, "orig")
    after_vis = draw_boxes(after_vis, pasted_boxes, PASTED_COLOR, "pasted")

    # Pad to equal height before concatenating, in case of any mismatch.
    h = max(before_vis.shape[0], after_vis.shape[0])
    before_vis = cv2.copyMakeBorder(before_vis, 0, h - before_vis.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    after_vis = cv2.copyMakeBorder(after_vis, 0, h - after_vis.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    divider = np.full((h, 4, 3), 255, dtype=np.uint8)
    combined = cv2.hconcat([before_vis, divider, after_vis])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combined)


def sample_preview_image_ids(candidate_ids: List[str], n: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    if len(candidate_ids) <= n:
        return list(candidate_ids)
    return rng.sample(candidate_ids, n)

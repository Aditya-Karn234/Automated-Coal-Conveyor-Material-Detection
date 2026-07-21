"""YOLO label file reading/writing helpers."""

from pathlib import Path
from typing import List, Tuple

from .geometry import BBox, yolo_to_xyxy


def read_yolo_label(label_path: Path, img_w: int, img_h: int) -> Tuple[List[str], List[BBox], List[int]]:
    """
    Returns (raw_lines, xyxy_boxes, class_ids).

    raw_lines preserves the original text lines verbatim (so unrelated
    annotations are never rewritten); xyxy_boxes / class_ids are the
    parsed, pixel-space equivalent used for IoU checks and for counting
    existing foreign_object instances.
    """
    if not label_path.exists():
        return [], [], []
    raw_lines = [ln.strip() for ln in label_path.read_text().splitlines() if ln.strip()]
    boxes: List[BBox] = []
    class_ids: List[int] = []
    for ln in raw_lines:
        parts = ln.split()
        if len(parts) < 5:
            continue
        cls, cx, cy, w, h = parts[:5]
        boxes.append(yolo_to_xyxy(float(cx), float(cy), float(w), float(h), img_w, img_h))
        class_ids.append(int(float(cls)))
    return raw_lines, boxes, class_ids


def update_yolo_labels(raw_lines: List[str], class_id: int, box_yolo: Tuple[float, float, float, float]) -> List[str]:
    """Appends one new YOLO-format annotation line to the list of label lines."""
    cx, cy, w, h = box_yolo
    new_line = f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
    return raw_lines + [new_line]


def count_class_instances(class_ids: List[int], class_id: int) -> int:
    return sum(1 for c in class_ids if c == class_id)

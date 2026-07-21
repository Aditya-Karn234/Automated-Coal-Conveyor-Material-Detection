"""Compositing: alpha blending (with feathered edges) or optional Poisson blending."""

import cv2
import numpy as np

from .config import Config


def alpha_blend(base_img: np.ndarray, obj_rgba: np.ndarray, x: int, y: int) -> np.ndarray:
    """
    Alpha-composites obj_rgba onto base_img (BGR) with its top-left
    corner at (x, y). The alpha channel may already be feathered
    (soft edges) by the transform stage, which is what removes the
    hard seam around pasted objects. Returns a new image.
    """
    out = base_img.copy()
    oh, ow = obj_rgba.shape[:2]
    bh, bw = out.shape[:2]

    x2, y2 = min(x + ow, bw), min(y + oh, bh)
    ow_clip, oh_clip = x2 - x, y2 - y
    if ow_clip <= 0 or oh_clip <= 0:
        return out

    roi = out[y:y2, x:x2].astype(np.float32)
    obj_crop = obj_rgba[:oh_clip, :ow_clip].astype(np.float32)

    alpha = obj_crop[:, :, 3:4] / 255.0
    blended = obj_crop[:, :, :3] * alpha + roi * (1.0 - alpha)
    out[y:y2, x:x2] = blended.astype(np.uint8)
    return out


def seamless_clone_blend(base_img: np.ndarray, obj_rgba: np.ndarray, x: int, y: int) -> np.ndarray:
    """
    AUGMENTATION QUALITY (optional): Poisson blending via
    cv2.seamlessClone for more realistic compositing than flat alpha
    blending -- smooths local gradients at the boundary instead of
    just cross-fading pixel values. Falls back to alpha_blend if the
    object mask is degenerate (seamlessClone requires a non-trivial
    mask fully inside the destination).
    """
    oh, ow = obj_rgba.shape[:2]
    bh, bw = base_img.shape[:2]

    x2, y2 = min(x + ow, bw), min(y + oh, bh)
    ow_clip, oh_clip = x2 - x, y2 - y
    if ow_clip <= 1 or oh_clip <= 1:
        return base_img.copy()

    obj_bgr = obj_rgba[:oh_clip, :ow_clip, :3]
    mask = obj_rgba[:oh_clip, :ow_clip, 3]

    if mask.max() == 0:
        return alpha_blend(base_img, obj_rgba, x, y)

    center = (x + ow_clip // 2, y + oh_clip // 2)
    # seamlessClone requires the mask/src to sit strictly inside dst;
    # nudge the center in slightly if it's touching an edge.
    center = (min(max(center[0], 1), bw - 2), min(max(center[1], 1), bh - 2))

    try:
        cloned = cv2.seamlessClone(obj_bgr, base_img, mask, center, cv2.NORMAL_CLONE)
        return cloned
    except cv2.error:
        # Poisson solve can fail on tiny/degenerate masks; fall back gracefully.
        return alpha_blend(base_img, obj_rgba, x, y)


def composite(base_img: np.ndarray, obj_rgba: np.ndarray, x: int, y: int, cfg: Config) -> np.ndarray:
    """Dispatches to the configured compositing method."""
    if cfg.USE_SEAMLESS_CLONE:
        return seamless_clone_blend(base_img, obj_rgba, x, y)
    return alpha_blend(base_img, obj_rgba, x, y)

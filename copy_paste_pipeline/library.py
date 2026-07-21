"""
Foreign-object cutout library loading, and the empirical object-size
distribution used to sample realistic paste scales.

HIGH PRIORITY FIX: object scale is sampled from the real size
distribution of existing foreign_object annotations in the training
set (relative width/height, so it's resolution-independent), instead
of a single fixed 0.8-1.2x multiplier applied blindly to whatever size
the cutout happens to be.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from .config import Config

logger = logging.getLogger("copy_paste_pipeline")


def load_object_library(cfg: Config) -> pd.DataFrame:
    files = sorted(cfg.TIGHT_OBJECTS_DIR.glob("*.png"))

    if not files:
        raise RuntimeError(
            f"No PNG files found in {cfg.TIGHT_OBJECTS_DIR}"
        )

    return pd.DataFrame({
        "resolved_path": files
    })


def build_empirical_relative_sizes(cfg: Config) -> List[Tuple[float, float]]:
    """
    Scans every training label file and collects the (relative_w,
    relative_h) of every existing box whose class matches
    FOREIGN_OBJECT_CLASS_ID. These are resolution-independent (0-1
    fractions of image width/height) so they can be reapplied to any
    image size later.
    """
    sizes: List[Tuple[float, float]] = []
    if not cfg.TRAIN_LABELS.exists():
        return sizes

    for label_path in cfg.TRAIN_LABELS.glob("*.txt"):
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            if cls != cfg.FOREIGN_OBJECT_CLASS_ID:
                continue
            w, h = float(parts[3]), float(parts[4])
            if w > 0 and h > 0:
                sizes.append((w, h))

    logger.info("Found %d existing foreign_object boxes in train/ to build empirical size distribution", len(sizes))
    return sizes


def sample_target_size_px(
    empirical_relative_sizes: List[Tuple[float, float]],
    img_w: int,
    img_h: int,
    cfg: Config,
) -> Optional[Tuple[float, float]]:
    """
    Returns a (target_w_px, target_h_px) sampled from the empirical
    distribution scaled to this image's resolution, or None if the
    distribution is too small to trust (caller should fall back to
    SCALE_RANGE_FALLBACK in that case).
    """
    if not cfg.USE_EMPIRICAL_SCALE or len(empirical_relative_sizes) < cfg.MIN_EMPIRICAL_SAMPLES:
        return None
    import random

    rel_w, rel_h = random.choice(empirical_relative_sizes)
    return rel_w * img_w, rel_h * img_h

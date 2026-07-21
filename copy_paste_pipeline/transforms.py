"""
Object transforms, split into two stages so background-matching can
see the actual paste location:

1. geometric_transform_object -- scale, rotate, feather edges, and
   tight-crop to the alpha mask (AUGMENTATION QUALITY FIX: cropping is
   redone *after* feathering, since blurring the alpha channel can
   bleed a few pixels of non-zero alpha past the original crop).
2. photometric_transform_object -- applied once the paste location
   (and therefore the background patch under it) is known: matches the
   object's brightness/contrast to the local background (HIGH PRIORITY
   FIX), then layers the small random brightness/contrast/blur jitter
   on top for variety.
"""

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import Config
from .geometry import tight_crop_to_alpha


@dataclass
class GeometricTransformMeta:
    scale: float
    rotation_deg: float


@dataclass
class PhotometricTransformMeta:
    brightness_delta: float
    contrast_delta: float
    background_matched: bool
    blurred: bool


def _rotate_rgba(obj_rgba: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotates an RGBA image around its center, expanding the canvas so
    nothing clips. Cropping to content happens later, after feathering."""
    h, w = obj_rgba.shape[:2]
    diag = int(np.ceil(np.sqrt(w ** 2 + h ** 2)))
    pad_x = (diag - w) // 2 + 1
    pad_y = (diag - h) // 2 + 1
    padded = cv2.copyMakeBorder(obj_rgba, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))

    ph, pw = padded.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(
        padded, rot_mat, (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )
    return rotated


def _feather_alpha(rgba: np.ndarray, feather_px: int) -> np.ndarray:
    """
    Softens the alpha channel's edges with a Gaussian blur so pasted
    objects don't show a hard cut line (AUGMENTATION QUALITY FIX).
    Pads first so the blur has room to bleed outward without being
    clipped by the array boundary.
    """
    if feather_px <= 0:
        return rgba
    pad = feather_px * 2
    padded = cv2.copyMakeBorder(rgba, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    k = feather_px * 2 + 1  # odd kernel size
    alpha = padded[:, :, 3].astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    padded[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    return padded


def geometric_transform_object(
    obj_rgba: np.ndarray,
    cfg: Config,
    target_size_px: Optional[Tuple[float, float]] = None,
) -> Tuple[Optional[np.ndarray], Optional[GeometricTransformMeta]]:
    """
    Applies scale + rotation + edge feathering + final tight alpha crop.

    If `target_size_px` (w, h) is given (from the empirical size
    distribution), the object is scaled uniformly so its area matches
    that target while preserving its own aspect ratio. Otherwise falls
    back to Config.SCALE_RANGE_FALLBACK, a uniform random multiplier
    applied to the cutout's native size.
    """
    if obj_rgba is None or obj_rgba.size == 0 or obj_rgba.ndim != 3 or obj_rgba.shape[2] != 4:
        return None, None

    h, w = obj_rgba.shape[:2]

    if target_size_px is not None:
        target_w, target_h = target_size_px
        # Preserve the cutout's own aspect ratio; match area via the
        # geometric mean of the two implied scale factors.
        scale = float(np.sqrt(max(target_w, 1.0) / max(w, 1) * max(target_h, 1.0) / max(h, 1)))
    else:
        scale = random.uniform(*cfg.SCALE_RANGE_FALLBACK)

    scale = max(0.05, scale)  # guard against degenerate near-zero scale
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    scaled = cv2.resize(obj_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

    angle = random.uniform(*cfg.ROTATION_RANGE_DEG)
    rotated = _rotate_rgba(scaled, angle)

    feathered = _feather_alpha(rotated, cfg.FEATHER_EDGE_PX)

    # CRITICAL FIX: recompute the tight crop from the alpha mask *after*
    # feathering, not from the pre-feather canvas -- feathering can
    # expand the non-transparent footprint by a few pixels.
    final = tight_crop_to_alpha(feathered, alpha_threshold=1)
    if final is None or final.shape[0] < 2 or final.shape[1] < 2:
        return None, None

    return final, GeometricTransformMeta(scale=scale, rotation_deg=angle)


def _adjust_brightness_contrast(rgb: np.ndarray, brightness_delta: float, contrast_delta: float) -> np.ndarray:
    contrast_factor = 1.0 + contrast_delta
    brightness_offset = brightness_delta * 255.0
    out = rgb.astype(np.float32) * contrast_factor + brightness_offset
    return np.clip(out, 0, 255).astype(np.uint8)


def _match_background_stats(obj_bgr: np.ndarray, obj_alpha: np.ndarray, background_patch: np.ndarray, strength: float) -> np.ndarray:
    """
    HIGH PRIORITY FIX: shifts the object's per-channel mean/std toward
    the local background patch's mean/std (mean-std / "Reinhard-style"
    color transfer), blended with the object's own original statistics
    by `strength` (0 = untouched, 1 = fully matched). Only the
    non-transparent object pixels are used to compute the object's own
    statistics, so background bleed-through in the cutout doesn't
    skew the match.
    """
    if background_patch is None or background_patch.size == 0 or strength <= 0:
        return obj_bgr

    mask = obj_alpha > 10
    if mask.sum() < 10:
        return obj_bgr

    obj_f = obj_bgr.astype(np.float32)
    bg_f = background_patch.astype(np.float32)

    out = obj_f.copy()
    for c in range(3):
        obj_channel = obj_f[:, :, c][mask]
        obj_mean, obj_std = obj_channel.mean(), obj_channel.std() + 1e-6
        bg_mean, bg_std = bg_f[:, :, c].mean(), bg_f[:, :, c].std() + 1e-6

        target_mean = obj_mean * (1 - strength) + bg_mean * strength
        target_std = obj_std * (1 - strength) + bg_std * strength

        channel = obj_f[:, :, c]
        normalized = (channel - obj_mean) / obj_std
        out[:, :, c] = normalized * target_std + target_mean

    return np.clip(out, 0, 255).astype(np.uint8)


def photometric_transform_object(
    obj_rgba: np.ndarray,
    cfg: Config,
    background_patch: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, PhotometricTransformMeta]:
    """
    Applies (in order): background-stat matching, random brightness /
    contrast jitter, and probabilistic Gaussian blur. Alpha channel is
    never touched here.
    """
    bgr = obj_rgba[:, :, :3].copy()
    alpha = obj_rgba[:, :, 3]

    background_matched = False
    if cfg.MATCH_BACKGROUND_STATS and background_patch is not None:
        bgr = _match_background_stats(bgr, alpha, background_patch, cfg.BACKGROUND_MATCH_STRENGTH)
        background_matched = True

    brightness_delta = random.uniform(*cfg.BRIGHTNESS_RANGE)
    contrast_delta = random.uniform(*cfg.CONTRAST_RANGE)
    bgr = _adjust_brightness_contrast(bgr, brightness_delta, contrast_delta)

    blurred = False
    if random.random() < cfg.BLUR_PROB:
        k = random.choice(cfg.BLUR_KERNEL_CHOICES)
        bgr = cv2.GaussianBlur(bgr, (k, k), 0)
        blurred = True

    result = np.dstack([bgr, alpha])
    meta = PhotometricTransformMeta(
        brightness_delta=brightness_delta,
        contrast_delta=contrast_delta,
        background_matched=background_matched,
        blurred=blurred,
    )
    return result, meta

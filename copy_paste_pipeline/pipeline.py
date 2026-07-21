"""Top-level orchestration: wires every module together into a full run."""

import logging
import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .blending import composite
from .config import Config
from .geometry import clipped_box_is_valid, xyxy_to_yolo
from .labels import count_class_instances, read_yolo_label, update_yolo_labels
from .library import build_empirical_relative_sizes, load_object_library, sample_target_size_px
from .placement import compute_conveyor_roi, find_candidate_position
from .reporting import AttemptRecord, ImageSummaryRecord, print_summary, write_image_summary_csv, write_report_csv
from .transforms import geometric_transform_object, photometric_transform_object
from .visualization import sample_preview_image_ids, save_before_after

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("copy_paste_pipeline")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def resolve_foreign_object_class_id(cfg: Config) -> int:
    """
    If CLASS_NAMES_FILE is configured, resolve the 'foreign_object'
    line index from it and prefer that over the hardcoded default --
    this is the safest way to avoid the class-id mixup that caused
    pasted objects to be mislabeled as 'coal' in earlier runs.
    """
    if cfg.CLASS_NAMES_FILE and cfg.CLASS_NAMES_FILE.exists():
        names = [ln.strip() for ln in cfg.CLASS_NAMES_FILE.read_text().splitlines() if ln.strip()]
        for idx, name in enumerate(names):
            if name.lower() == "foreign_object":
                logger.info("Resolved foreign_object class id = %d from %s", idx, cfg.CLASS_NAMES_FILE)
                return idx
        logger.warning(
            "classes file %s did not contain 'foreign_object'; using configured id %d",
            cfg.CLASS_NAMES_FILE, cfg.FOREIGN_OBJECT_CLASS_ID,
        )
    return cfg.FOREIGN_OBJECT_CLASS_ID


def make_output_dirs(cfg: Config) -> None:
    for split in ("train", "val", "test"):
        (cfg.OUTPUT_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (cfg.OUTPUT_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)


def copy_split_unchanged(images_dir: Path, labels_dir: Path, out_images_dir: Path, out_labels_dir: Path, split_name: str, cfg: Config) -> None:
    if not images_dir.exists():
        logger.warning("%s images directory not found at %s; skipping copy.", split_name, images_dir)
        return
    image_files = [p for p in images_dir.iterdir() if p.suffix.lower() in cfg.IMAGE_EXTENSIONS]
    for img_path in tqdm(image_files, desc=f"Copying {split_name} (unchanged)"):
        shutil.copy2(img_path, out_images_dir / img_path.name)
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, out_labels_dir / label_path.name)


def _safe_background_patch(image: np.ndarray, x: int, y: int, w: int, h: int) -> Optional[np.ndarray]:
    bh, bw = image.shape[:2]
    x2, y2 = min(x + w, bw), min(y + h, bh)
    if x2 <= x or y2 <= y:
        return None
    return image[y:y2, x:x2]


def _should_augment_image(existing_fo_count: int, cfg: Config) -> bool:
    """
    Per-image coin flip deciding whether this training image is
    augmented at all. Images that already contain a foreign_object
    annotation get their probability scaled down by
    EXISTING_FO_BIAS_MULTIPLIER, biasing augmentation effort toward
    images that don't have one yet.
    """
    prob = cfg.AUGMENTATION_PROBABILITY
    if existing_fo_count > 0:
        prob *= cfg.EXISTING_FO_BIAS_MULTIPLIER
    return random.random() < prob


def process_single_image(
    image_path: Path,
    label_path: Path,
    library_df: pd.DataFrame,
    empirical_sizes: List[Tuple[float, float]],
    cfg: Config,
    class_id: int,
) -> Tuple[List[str], np.ndarray, List[AttemptRecord], ImageSummaryRecord]:
    image_id = image_path.stem

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image {image_path}")

    img_h, img_w = img.shape[:2]
    raw_lines, existing_boxes, class_ids = read_yolo_label(label_path, img_w, img_h)
    existing_fo_count = count_class_instances(class_ids, class_id)

    # Per-image probability gate: only a fraction of train images are
    # augmented at all, biased toward images that don't already have a
    # foreign_object annotation. Images that lose the coin flip are
    # written through completely unchanged.
    augmentation_selected = _should_augment_image(existing_fo_count, cfg)

    working_img = img
    pasted_names: List[str] = []
    attempts: List[AttemptRecord] = []
    n_to_try = 0

    if augmentation_selected:
        # HIGH PRIORITY FIX: cap additional pastes for images that already
        # contain foreign_object annotations.
        if existing_fo_count > 0:
            n_to_try = min(random.randint(cfg.MIN_OBJECTS_PER_IMAGE, cfg.MAX_OBJECTS_PER_IMAGE), cfg.MAX_ADDITIONAL_IF_EXISTING_FOREIGN_OBJECT)
        else:
            n_to_try = random.randint(cfg.MIN_OBJECTS_PER_IMAGE, cfg.MAX_OBJECTS_PER_IMAGE)

    roi = compute_conveyor_roi(existing_boxes, img_w, img_h, cfg) if n_to_try > 0 else None

    for obj_idx in range(n_to_try):
        row = library_df.sample(n=1).iloc[0]
        obj_path = row["resolved_path"]
        object_name = Path(obj_path).stem

        obj_rgba = cv2.imread(str(obj_path), cv2.IMREAD_UNCHANGED)
        if obj_rgba is None or obj_rgba.ndim != 3 or obj_rgba.shape[2] != 4:
            attempts.append(AttemptRecord(image_id, obj_idx, object_name, False, failure_reason="unreadable_or_non_rgba_cutout"))
            continue

        target_size = sample_target_size_px(empirical_sizes, img_w, img_h, cfg)
        transformed, geo_meta = geometric_transform_object(obj_rgba, cfg, target_size)
        if transformed is None:
            attempts.append(AttemptRecord(image_id, obj_idx, object_name, False, failure_reason="degenerate_geometric_transform"))
            continue

        obj_h, obj_w = transformed.shape[:2]
        placement = find_candidate_position(img_w, img_h, obj_w, obj_h, existing_boxes, cfg, roi)
        if not placement.success:
            attempts.append(AttemptRecord(
                image_id, obj_idx, object_name, False,
                rotation_deg=geo_meta.rotation_deg, scale=geo_meta.scale,
                retry_count=placement.attempts, failure_reason=placement.failure_reason,
            ))
            continue

        x, y = placement.x, placement.y
        background_patch = _safe_background_patch(working_img, x, y, obj_w, obj_h)
        final_obj, photo_meta = photometric_transform_object(transformed, cfg, background_patch)

        new_box_xyxy = (x, y, x + obj_w, y + obj_h)
        if not clipped_box_is_valid(*new_box_xyxy, img_w, img_h):
            # CRITICAL FIX: never write an annotation that clips to zero/near-zero area.
            attempts.append(AttemptRecord(
                image_id, obj_idx, object_name, False,
                rotation_deg=geo_meta.rotation_deg, scale=geo_meta.scale,
                retry_count=placement.attempts, failure_reason="clipped_outside_bounds",
            ))
            continue

        working_img = composite(working_img, final_obj, x, y, cfg)
        box_yolo = xyxy_to_yolo(*new_box_xyxy, img_w, img_h)
        raw_lines = update_yolo_labels(raw_lines, class_id, box_yolo)
        existing_boxes.append(new_box_xyxy)
        pasted_names.append(object_name)

        attempts.append(AttemptRecord(
            image_id, obj_idx, object_name, True,
            rotation_deg=geo_meta.rotation_deg, scale=geo_meta.scale,
            brightness_delta=photo_meta.brightness_delta, contrast_delta=photo_meta.contrast_delta,
            paste_x=x, paste_y=y, paste_w=obj_w, paste_h=obj_h,
            retry_count=placement.attempts,
        ))

    summary = ImageSummaryRecord(
        image_id=image_id,
        added_objects=len(pasted_names),
        attempted_objects=n_to_try,
        pasted_object_names=";".join(pasted_names),
        final_object_count=len(existing_boxes),
        existing_foreign_object_count=existing_fo_count,
        augmentation_selected=augmentation_selected,
    )

    save_augmented_sample(image_path, working_img, raw_lines, cfg)
    return raw_lines, working_img, attempts, summary


def save_augmented_sample(original_image_path: Path, image: np.ndarray, label_lines: List[str], cfg: Config) -> None:
    out_img_path = cfg.OUTPUT_ROOT / "train" / "images" / original_image_path.name
    out_label_path = cfg.OUTPUT_ROOT / "train" / "labels" / f"{original_image_path.stem}.txt"
    cv2.imwrite(str(out_img_path), image)
    out_label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))


def generate_previews(cfg: Config, image_ids_with_pastes: List[str]) -> None:
    """
    VISUALIZATION FIX: writes side-by-side before/after images (boxes
    drawn, existing vs pasted color-coded) for a random sample of
    augmented images, so a human can spot-check placement/blend quality
    before training. Run `_preview/` review manually -- recommended
    50-100 samples per the required-fixes checklist.
    """
    if not cfg.GENERATE_PREVIEWS or not image_ids_with_pastes:
        return

    sampled_ids = sample_preview_image_ids(image_ids_with_pastes, cfg.N_PREVIEW_SAMPLES, cfg.SEED)
    logger.info("Generating %d before/after preview pairs in %s", len(sampled_ids), cfg.PREVIEW_DIR)

    for image_id in tqdm(sampled_ids, desc="Generating previews"):
        # Find the original image under any supported extension.
        before_path = None
        for ext in cfg.IMAGE_EXTENSIONS:
            candidate = cfg.TRAIN_IMAGES / f"{image_id}{ext}"
            if candidate.exists():
                before_path = candidate
                break
        after_path = None
        for ext in cfg.IMAGE_EXTENSIONS:
            candidate = cfg.OUTPUT_ROOT / "train" / "images" / f"{image_id}{ext}"
            if candidate.exists():
                after_path = candidate
                break
        if before_path is None or after_path is None:
            continue

        before_img = cv2.imread(str(before_path), cv2.IMREAD_COLOR)
        after_img = cv2.imread(str(after_path), cv2.IMREAD_COLOR)
        if before_img is None or after_img is None:
            continue

        img_h, img_w = before_img.shape[:2]
        before_lines, before_boxes, _ = read_yolo_label(cfg.TRAIN_LABELS / f"{image_id}.txt", img_w, img_h)

        after_h, after_w = after_img.shape[:2]
        after_lines, after_boxes, after_class_ids = read_yolo_label(
            cfg.OUTPUT_ROOT / "train" / "labels" / f"{image_id}.txt", after_w, after_h
        )
        n_original = len(before_boxes)
        existing_boxes_in_after = after_boxes[:n_original]
        pasted_boxes = after_boxes[n_original:]

        out_path = cfg.PREVIEW_DIR / f"{image_id}_before_after.jpg"
        save_before_after(before_img, after_img, before_boxes, existing_boxes_in_after, pasted_boxes, out_path)


def run_pipeline(cfg: Optional[Config] = None) -> pd.DataFrame:
    cfg = cfg or Config()
    set_seed(cfg.SEED)

    logger.info("Preparing output directory tree at %s", cfg.OUTPUT_ROOT)
    make_output_dirs(cfg)

    class_id = resolve_foreign_object_class_id(cfg)

    logger.info("Copying val/ and test/ splits unchanged...")
    copy_split_unchanged(cfg.VAL_IMAGES, cfg.VAL_LABELS, cfg.OUTPUT_ROOT / "val" / "images", cfg.OUTPUT_ROOT / "val" / "labels", "val", cfg)
    copy_split_unchanged(cfg.TEST_IMAGES, cfg.TEST_LABELS, cfg.OUTPUT_ROOT / "test" / "images", cfg.OUTPUT_ROOT / "test" / "labels", "test", cfg)

    logger.info("Loading foreign-object library...")
    library_df = load_object_library(cfg)

    logger.info("Building empirical object-size distribution from existing train/ annotations...")
    empirical_sizes = build_empirical_relative_sizes(cfg)

    if not cfg.TRAIN_IMAGES.exists():
        raise FileNotFoundError(f"Train images directory not found at {cfg.TRAIN_IMAGES}")

    train_image_files = sorted(p for p in cfg.TRAIN_IMAGES.iterdir() if p.suffix.lower() in cfg.IMAGE_EXTENSIONS)
    logger.info("Found %d training images.", len(train_image_files))

    all_attempts: List[AttemptRecord] = []
    all_summaries: List[ImageSummaryRecord] = []
    images_with_pastes: List[str] = []
    n_errors = 0

    for image_path in tqdm(train_image_files, desc="Augmenting train split"):
        label_path = cfg.TRAIN_LABELS / f"{image_path.stem}.txt"
        try:
            _, _, attempts, summary = process_single_image(image_path, label_path, library_df, empirical_sizes, cfg, class_id)
            all_attempts.extend(attempts)
            all_summaries.append(summary)
            if summary.added_objects > 0:
                images_with_pastes.append(summary.image_id)
        except Exception as exc:  # noqa: BLE001 - keep going over the whole dataset
            n_errors += 1
            logger.exception("Failed to process %s: %s", image_path.name, exc)
            try:
                shutil.copy2(image_path, cfg.OUTPUT_ROOT / "train" / "images" / image_path.name)
                if label_path.exists():
                    shutil.copy2(label_path, cfg.OUTPUT_ROOT / "train" / "labels" / label_path.name)
            except Exception:  # noqa: BLE001
                logger.error("Also failed to fall back to copying %s untouched.", image_path.name)

    report_df = write_report_csv(all_attempts, cfg.OUTPUT_ROOT / cfg.REPORT_CSV_NAME)
    write_image_summary_csv(all_summaries, cfg.OUTPUT_ROOT / cfg.IMAGE_SUMMARY_CSV_NAME)
    logger.info("Wrote %s and %s", cfg.REPORT_CSV_NAME, cfg.IMAGE_SUMMARY_CSV_NAME)

    generate_previews(cfg, images_with_pastes)

    print_summary(all_attempts, all_summaries, len(train_image_files), n_errors)
    return report_df

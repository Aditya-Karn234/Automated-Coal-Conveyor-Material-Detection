"""
Central configuration for the Context-Aware Copy-Paste pipeline.

Every tunable parameter lives here. Nothing downstream should hard-code
a magic number that belongs in this file.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class Config:
    SEED: int = 42

    # ------------------------------------------------------------------ #
    # Input dataset
    # ------------------------------------------------------------------ #
    DATASET_ROOT: Path = Path("/content/extracted_data/productive_state")
    TRAIN_IMAGES: Path = field(init=False)
    TRAIN_LABELS: Path = field(init=False)
    VAL_IMAGES: Path = field(init=False)
    VAL_LABELS: Path = field(init=False)
    TEST_IMAGES: Path = field(init=False)
    TEST_LABELS: Path = field(init=False)

    # ------------------------------------------------------------------ #
    # Foreign object library
    # ------------------------------------------------------------------ #
    FOREIGN_OBJECT_LIB: Path = Path("/content/foreign_object_library")
    TIGHT_OBJECTS_DIR: Path = field(init=False)
    METADATA_CSV: Path = field(init=False)

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    OUTPUT_ROOT: Path = Path("/content/augmented_dataset")
    PREVIEW_DIR: Path = field(init=False)

    # ------------------------------------------------------------------ #
    # CRITICAL FIX: class id.
    # In this dataset's label schema class 0 is "coal". foreign_object
    # is class 2. Getting this wrong silently mislabels every pasted
    # object -- verify against your own classes.txt / data.yaml before
    # trusting the default below.
    # ------------------------------------------------------------------ #
    FOREIGN_OBJECT_CLASS_ID: int = 2
    CLASS_NAMES_FILE: Optional[Path] = None  # optional override, one name per line, YOLO index order

    # ------------------------------------------------------------------ #
    # How many objects to paste per image
    # ------------------------------------------------------------------ #
    MIN_OBJECTS_PER_IMAGE: int = 1
    MAX_OBJECTS_PER_IMAGE: int = 3
    # HIGH PRIORITY FIX: images that already contain >=1 foreign_object
    # ground-truth annotation are capped to this many *additional*
    # pastes, instead of the full MAX_OBJECTS_PER_IMAGE range, so we
    # don't over-saturate already-positive images.
    MAX_ADDITIONAL_IF_EXISTING_FOREIGN_OBJECT: int = 1

    # ------------------------------------------------------------------ #
    # Whether to augment a given training image at all.
    #
    # AUGMENTATION_PROBABILITY is the base chance any train image gets
    # augmented; images that don't get selected are copied through to
    # the output train split completely unchanged (same as val/test).
    #
    # EXISTING_FO_BIAS_MULTIPLIER scales that probability down for
    # images that already contain >=1 foreign_object annotation, so
    # augmentation effort concentrates on images that don't have one
    # yet -- e.g. base 0.5 * multiplier 0.25 = 0.125 effective chance
    # for an already-positive image, vs 0.5 for a negative one.
    # ------------------------------------------------------------------ #
    AUGMENTATION_PROBABILITY: float = 0.5
    EXISTING_FO_BIAS_MULTIPLIER: float = 0.25

    # ------------------------------------------------------------------ #
    # Geometric transforms
    # ------------------------------------------------------------------ #
    # HIGH PRIORITY FIX: fixed range is now only a fallback, used when
    # too few real foreign_object boxes exist in train/ to build an
    # empirical size distribution. See library.build_empirical_sizes().
    SCALE_RANGE_FALLBACK: Tuple[float, float] = (0.8, 1.2)
    USE_EMPIRICAL_SCALE: bool = True
    MIN_EMPIRICAL_SAMPLES: int = 5  # below this count, fall back to the fixed range
    ROTATION_RANGE_DEG: Tuple[float, float] = (-15.0, 15.0)

    # ------------------------------------------------------------------ #
    # Photometric transforms
    # ------------------------------------------------------------------ #
    BRIGHTNESS_RANGE: Tuple[float, float] = (-0.15, 0.15)
    CONTRAST_RANGE: Tuple[float, float] = (-0.10, 0.10)
    BLUR_PROB: float = 0.20
    BLUR_KERNEL_CHOICES: Tuple[int, ...] = (3, 5)
    # HIGH PRIORITY FIX: match the pasted object's mean/std to the local
    # background patch it will sit on top of, before applying the small
    # random jitter above. 0 = no matching (old behavior), 1 = full match.
    MATCH_BACKGROUND_STATS: bool = True
    BACKGROUND_MATCH_STRENGTH: float = 0.7

    # ------------------------------------------------------------------ #
    # Edge / compositing quality
    # ------------------------------------------------------------------ #
    FEATHER_EDGE_PX: int = 3           # 0 disables feathering
    USE_SEAMLESS_CLONE: bool = False   # cv2.seamlessClone Poisson blending instead of alpha blend

    # ------------------------------------------------------------------ #
    # Placement
    # ------------------------------------------------------------------ #
    BORDER_MARGIN_PX: int = 10
    MAX_IOU_WITH_EXISTING: float = 0.10
    MAX_PLACEMENT_TRIES: int = 30
    # HIGH PRIORITY FIX: restrict the placement search to a conveyor /
    # active-object ROI instead of the full frame.
    RESTRICT_TO_CONVEYOR_ROI: bool = True
    ROI_MARGIN_FRACTION: float = 0.15   # expand the ROI inferred from existing boxes by this fraction of image size
    # Used only when an image has no existing boxes to infer an ROI from.
    DEFAULT_ROI_FRACTION: Tuple[float, float, float, float] = (0.05, 0.05, 0.95, 0.95)

    IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")

    # ------------------------------------------------------------------ #
    # Reporting / visualization
    # ------------------------------------------------------------------ #
    REPORT_CSV_NAME: str = "report.csv"                  # per-paste-attempt detail
    IMAGE_SUMMARY_CSV_NAME: str = "image_summary.csv"     # per-image aggregate
    GENERATE_PREVIEWS: bool = True
    N_PREVIEW_SAMPLES: int = 50
    PREVIEW_DIRNAME: str = "_preview"

    def __post_init__(self):
        self.TRAIN_IMAGES = self.DATASET_ROOT / "train" / "images"
        self.TRAIN_LABELS = self.DATASET_ROOT / "train" / "labels"
        self.VAL_IMAGES = self.DATASET_ROOT / "val" / "images"
        self.VAL_LABELS = self.DATASET_ROOT / "val" / "labels"
        self.TEST_IMAGES = self.DATASET_ROOT / "test" / "images"
        self.TEST_LABELS = self.DATASET_ROOT / "test" / "labels"
        self.TIGHT_OBJECTS_DIR = self.FOREIGN_OBJECT_LIB / "tight_objects"
        self.METADATA_CSV = self.FOREIGN_OBJECT_LIB / "metadata.csv"
        self.PREVIEW_DIR = self.OUTPUT_ROOT / self.PREVIEW_DIRNAME

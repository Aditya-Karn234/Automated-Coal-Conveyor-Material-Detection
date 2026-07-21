"""
Reporting: two CSVs plus a printed run summary.

REPORTING FIX: report.csv is extended to per-paste-*attempt*
granularity (object name, rotation, scale, brightness, contrast, paste
coordinates, retry count, success/failure), and a second CSV,
image_summary.csv, keeps the original per-image rollup so existing
downstream tooling that expects one row per image still works.
"""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd


@dataclass
class AttemptRecord:
    image_id: str
    object_index: int
    object_name: str
    success: bool
    rotation_deg: Optional[float] = None
    scale: Optional[float] = None
    brightness_delta: Optional[float] = None
    contrast_delta: Optional[float] = None
    paste_x: Optional[int] = None
    paste_y: Optional[int] = None
    paste_w: Optional[int] = None
    paste_h: Optional[int] = None
    retry_count: int = 0
    failure_reason: str = ""


@dataclass
class ImageSummaryRecord:
    image_id: str
    added_objects: int
    attempted_objects: int
    pasted_object_names: str
    final_object_count: int
    existing_foreign_object_count: int
    augmentation_selected: bool = True


def write_report_csv(attempts: List[AttemptRecord], out_path: Path) -> pd.DataFrame:
    df = pd.DataFrame([a.__dict__ for a in attempts])
    df.to_csv(out_path, index=False)
    return df


def write_image_summary_csv(summaries: List[ImageSummaryRecord], out_path: Path) -> pd.DataFrame:
    df = pd.DataFrame([s.__dict__ for s in summaries])
    df.to_csv(out_path, index=False)
    return df


def print_summary(attempts: List[AttemptRecord], summaries: List[ImageSummaryRecord], n_images_total: int, n_errors: int) -> None:
    total_attempts = len(attempts)
    successes = [a for a in attempts if a.success]
    failures = [a for a in attempts if not a.success]
    failure_reasons = Counter(a.failure_reason for a in failures)

    added_per_image = [s.added_objects for s in summaries]
    avg_added = sum(added_per_image) / len(added_per_image) if added_per_image else 0.0
    images_with_additions = sum(1 for n in added_per_image if n > 0)

    distribution = Counter(added_per_image)

    n_selected = sum(1 for s in summaries if s.augmentation_selected)
    n_selected_with_existing_fo = sum(1 for s in summaries if s.augmentation_selected and s.existing_foreign_object_count > 0)
    n_selected_without_existing_fo = n_selected - n_selected_with_existing_fo
    n_with_existing_fo = sum(1 for s in summaries if s.existing_foreign_object_count > 0)
    n_without_existing_fo = len(summaries) - n_with_existing_fo

    print("\n" + "=" * 64)
    print("CONTEXT-AWARE COPY-PASTE AUGMENTATION -- SUMMARY")
    print("=" * 64)
    print(f"Training images processed        : {n_images_total}")
    print(f"Images that failed processing     : {n_errors}")
    print(f"Images selected for augmentation  : {n_selected} ({n_selected / max(len(summaries), 1):.1%})")
    print(f"  - without existing foreign_object: {n_selected_without_existing_fo}/{n_without_existing_fo}")
    print(f"  - with existing foreign_object   : {n_selected_with_existing_fo}/{n_with_existing_fo}")
    print(f"Total placement attempts          : {total_attempts}")
    print(f"Successful pastes                 : {len(successes)}")
    print(f"Failed placements                 : {len(failures)}")
    if failure_reasons:
        print("  Failure reason breakdown:")
        for reason, count in failure_reasons.most_common():
            print(f"    - {reason or '(unspecified)'}: {count}")
    print(f"Images with >=1 object pasted     : {images_with_additions} ({images_with_additions / max(n_images_total, 1):.1%})")
    print(f"Average objects pasted per image  : {avg_added:.2f}")
    print("  Additions-per-image distribution:")
    for n_added in sorted(distribution):
        print(f"    - {n_added} object(s): {distribution[n_added]} image(s)")
    print("=" * 64 + "\n")

"""
NeuroGuide AI - Dataset Ingestion, Verification & Stratified Splitting

Scans the raw Brain MRI dataset, validates data integrity, and creates a
stratified, reproducible manifest (70% Train, 15% Validation, 15% Test)
guaranteeing ZERO data leakage between splits.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Tuple
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from backend.app.config import (
    ARCHIVE_DIR,
    CLASS_NAMES,
    CLASS_TO_IDX,
    DATA_DIR,
    MANIFEST_PATH,
    RANDOM_SEED,
    SPLIT_RATIOS,
)


def scan_raw_images(archive_dir: Path) -> List[Dict]:
    """
    Recursively scans the archive directory to catalog all Brain MRI scans.
    """
    records = []
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    if not archive_dir.exists():
        raise FileNotFoundError(f"Archive directory does not exist: {archive_dir}")

    print(f"Scanning raw images in: {archive_dir} ...")

    # Traverse through Training and Testing subfolders
    for split_folder in ["Training", "Testing"]:
        split_path = archive_dir / split_folder
        if not split_path.exists():
            continue

        for class_name in CLASS_NAMES:
            class_dir = split_path / class_name
            if not class_dir.exists():
                print(f"Warning: Subdirectory not found: {class_dir}")
                continue

            for file_path in class_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    records.append({
                        "file_path": str(file_path.resolve()),
                        "relative_path": str(file_path.relative_to(archive_dir.parent)),
                        "class_name": class_name,
                        "class_idx": CLASS_TO_IDX[class_name],
                        "source_split": split_folder,
                        "file_size_kb": round(file_path.stat().st_size / 1024, 2),
                    })

    print(f"Discovered total of {len(records)} image files across all subdirectories.")
    return records


def verify_image_integrity(records: List[Dict]) -> Tuple[List[Dict], List[str]]:
    """
    Validates that each discovered file can be cleanly opened and decoded as an image.
    Appends image dimensions (width, height, channels) to valid records.
    """
    valid_records = []
    corrupted_files = []

    print("Verifying image readability and extracting dimensions...")
    for record in tqdm(records, desc="Integrity Verification"):
        img_path = record["file_path"]
        try:
            # Quick decode using OpenCV
            img = cv2.imread(img_path)
            if img is None:
                corrupted_files.append(img_path)
                continue

            h, w = img.shape[:2]
            c = img.shape[2] if len(img.shape) == 3 else 1

            record["height"] = h
            record["width"] = w
            record["channels"] = c
            record["aspect_ratio"] = round(w / h, 3) if h > 0 else 0.0
            record["image_id"] = Path(img_path).stem
            valid_records.append(record)
        except Exception as e:
            corrupted_files.append(f"{img_path} (Error: {str(e)})")

    if corrupted_files:
        print(f"Warning: Found {len(corrupted_files)} unreadable files!")
    else:
        print("All image files passed decode verification without corruption.")

    return valid_records, corrupted_files


def create_stratified_splits(
    df: pd.DataFrame,
    ratios: Dict[str, float] = SPLIT_RATIOS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Performs stratified train/val/test splitting across all target classes.

    Default split: 70% Train, 15% Validation, 15% Test.
    """
    print(f"\nPerforming stratified splitting (seed={seed}): {ratios} ...")

    train_ratio = ratios.get("train", 0.70)
    val_ratio = ratios.get("val", 0.15)
    test_ratio = ratios.get("test", 0.15)

    # First split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df["class_idx"],
        random_state=seed,
    )

    # Second split: Val vs Test
    relative_test_ratio = test_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df["class_idx"],
        random_state=seed,
    )

    # Assign split labels
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Verify zero overlap between splits
    train_paths = set(train_df["file_path"])
    val_paths = set(val_df["file_path"])
    test_paths = set(test_df["file_path"])

    assert len(train_paths.intersection(val_paths)) == 0, "Data leakage detected: Train and Val overlap!"
    assert len(train_paths.intersection(test_paths)) == 0, "Data leakage detected: Train and Test overlap!"
    assert len(val_paths.intersection(test_paths)) == 0, "Data leakage detected: Val and Test overlap!"

    print("Leakage verification passed: zero overlap between Train, Val, and Test splits.")
    return combined_df


def summarize_splits(manifest_df: pd.DataFrame) -> None:
    """
    Prints detailed tabular summary of split and class distributions.
    """
    print("\n" + "=" * 65)
    print("DATASET SPLIT & CLASS BALANCE SUMMARY")
    print("=" * 65)

    summary_table = pd.crosstab(
        manifest_df["class_name"],
        manifest_df["split"],
        margins=True,
        margins_name="Total",
    )
    print(summary_table.to_string())

    print("\nSplit Proportions:")
    for split, count in manifest_df["split"].value_counts().items():
        pct = (count / len(manifest_df)) * 100
        print(f"  - {split.upper():5s}: {count:5d} images ({pct:.2f}%)")

    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Prepare, verify, and split Brain MRI dataset.")
    parser.add_argument("--archive-dir", type=str, default=str(ARCHIVE_DIR), help="Path to archive directory.")
    parser.add_argument("--output-manifest", type=str, default=str(MANIFEST_PATH), help="Path to output manifest CSV.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility.")
    args = parser.parse_args()

    archive_path = Path(args.archive_dir)
    manifest_out = Path(args.output_manifest)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    # 1. Catalog raw files
    records = scan_raw_images(archive_path)
    if not records:
        print("Error: No images found. Check archive path.")
        return

    # 2. Verify file integrity and obtain image properties
    valid_records, corrupted = verify_image_integrity(records)
    if not valid_records:
        print("Error: No valid images found.")
        return

    df = pd.DataFrame(valid_records)

    # 3. Create stratified splits
    manifest_df = create_stratified_splits(df, seed=args.seed)

    # 4. Save manifest CSV
    manifest_df.to_csv(manifest_out, index=False)
    print(f"\nSaved dataset manifest successfully to:\n  -> {manifest_out.resolve()}")

    # 5. Display tabular summary
    summarize_splits(manifest_df)


if __name__ == "__main__":
    main()

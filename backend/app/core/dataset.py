"""
NeuroGuide AI - Brain MRI PyTorch Dataset & DataLoaders

Implements strict ML best practices:
- Separate training and validation/test pipelines to prevent data leakage.
- Training split: Preprocessing + Data Augmentations (Rotation, Flips, Affine, Jitter).
- Validation & Test splits: Pure deterministic preprocessing (Skull crop, Resize, CLAHE, Normalize).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False
    Dataset = object
    DataLoader = Any

from backend.app.config import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RANDOM_SEED,
)
from backend.app.core.preprocessor import BrainMRIPreprocessor


class BrainMRIAugmenter:
    """
    Augmentation pipeline applied STRICTLY to training data only.

    Includes:
    - Random Horizontal Flip (p=0.5)
    - Random Rotation (+-15 degrees)
    - Subtle Random Affine translation (+-5%)
    - Random Brightness and Contrast jitter (+-10%)
    """

    def __init__(
        self,
        rotation_degrees: float = 15.0,
        flip_prob: float = 0.5,
        affine_translate: float = 0.05,
        jitter_factor: float = 0.10,
    ):
        self.rotation_degrees = rotation_degrees
        self.flip_prob = flip_prob
        self.affine_translate = affine_translate
        self.jitter_factor = jitter_factor

    def __call__(self, image_rgb_uint8: np.ndarray) -> np.ndarray:
        """
        Applies stochastic spatial and photometric augmentations on uint8 RGB array.
        """
        img = image_rgb_uint8.copy()
        h, w = img.shape[:2]

        # 1. Random Horizontal Flip
        if np.random.rand() < self.flip_prob:
            img = cv2.flip(img, 1)

        # 2. Random Rotation & Affine Translation
        angle = np.random.uniform(-self.rotation_degrees, self.rotation_degrees)
        tx = np.random.uniform(-self.affine_translate * w, self.affine_translate * w)
        ty = np.random.uniform(-self.affine_translate * h, self.affine_translate * h)

        # Center for rotation
        center = (w / 2.0, h / 2.0)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot_mat[0, 2] += tx
        rot_mat[1, 2] += ty

        img = cv2.warpAffine(
            img,
            rot_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # 3. Brightness & Contrast Jitter
        # Alpha: contrast [1 - factor, 1 + factor]
        # Beta: brightness [-factor * 255, factor * 255]
        alpha = 1.0 + np.random.uniform(-self.jitter_factor, self.jitter_factor)
        beta = np.random.uniform(-self.jitter_factor * 128.0, self.jitter_factor * 128.0)
        img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        return img


class BrainMRIDataset(Dataset):
    """
    PyTorch Dataset for Brain MRI classification.
    """

    def __init__(
        self,
        dataframe_or_records: Union[pd.DataFrame, List[Dict]],
        split: str = "train",
        preprocessor: Optional[BrainMRIPreprocessor] = None,
        augment: Optional[bool] = None,
    ):
        """
        Args:
            dataframe_or_records: DataFrame or list of dicts with 'file_path' and 'class_name'/'class_idx'.
            split: 'train', 'val', or 'test'.
            preprocessor: Instance of BrainMRIPreprocessor.
            augment: If None, automatically True for 'train' and False for 'val'/'test'.
        """
        if isinstance(dataframe_or_records, pd.DataFrame):
            self.data = dataframe_or_records.reset_index(drop=True)
        else:
            self.data = pd.DataFrame(dataframe_or_records)

        self.split = split.lower()
        self.preprocessor = preprocessor or BrainMRIPreprocessor()

        # Augmentation strictly for train by default
        if augment is None:
            self.augment = (self.split == "train")
        else:
            self.augment = augment

        self.augmenter = BrainMRIAugmenter() if self.augment else None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", "torch.Tensor", Dict]:
        row = self.data.iloc[idx]
        file_path = str(row["file_path"])
        class_name = str(row["class_name"])
        class_idx = int(row.get("class_idx", CLASS_TO_IDX[class_name]))

        # Read image
        image_bgr = cv2.imread(file_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Image not found or unreadable: {file_path}")

        # Step 1: Skull stripping & Extreme Contour Cropping
        cropped_bgr, bbox = self.preprocessor.crop_skull_contour(image_bgr)

        # Step 2: Dimension standardization (224x224)
        resized_bgr = self.preprocessor.resize(cropped_bgr)

        # Step 3: Contrast Normalization (CLAHE)
        clahe_bgr = self.preprocessor.apply_clahe(resized_bgr)

        # Convert to RGB
        rgb_uint8 = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB)

        # Step 4: Augmentation (Training Split Only)
        if self.augment and self.augmenter:
            rgb_uint8 = self.augmenter(rgb_uint8)

        # Step 5: Normalization
        normalized_rgb = self.preprocessor.normalize(rgb_uint8)

        # Convert to PyTorch Tensor: HWC -> CHW
        chw = np.transpose(normalized_rgb, (2, 0, 1)).astype(np.float32)
        image_tensor = torch.from_numpy(chw)
        label_tensor = torch.tensor(class_idx, dtype=torch.long)

        meta = {
            "file_path": file_path,
            "class_name": class_name,
            "class_idx": class_idx,
            "split": self.split,
            "image_id": row.get("image_id", Path(file_path).stem),
            "crop_bbox": bbox,
        }

        return image_tensor, label_tensor, meta


def create_dataloaders(
    manifest_df: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    pin_memory: bool = False,
) -> Dict[str, DataLoader]:
    """
    Creates DataLoaders for train, val, and test splits from a manifest DataFrame.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to create DataLoaders.")

    loaders = {}
    splits = ["train", "val", "test"]

    for split in splits:
        split_df = manifest_df[manifest_df["split"] == split].reset_index(drop=True)
        if len(split_df) == 0:
            continue

        dataset = BrainMRIDataset(
            dataframe_or_records=split_df,
            split=split,
            augment=(split == "train"),
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == "train" and len(dataset) > batch_size),
        )
        loaders[split] = loader

    return loaders

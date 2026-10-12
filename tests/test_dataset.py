"""
Unit tests for Brain MRI Dataset & DataLoader Pipeline (Phase 1)
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import (
    CLASS_NAMES,
    IMAGE_SIZE,
    MANIFEST_PATH,
)
from backend.app.core.dataset import (
    BrainMRIAugmenter,
    BrainMRIDataset,
    create_dataloaders,
)
from backend.app.core.preprocessor import BrainMRIPreprocessor


@pytest.fixture
def dummy_manifest(tmp_path):
    """
    Creates a temporary manifest with dummy image files across classes and splits.
    """
    records = []
    classes = CLASS_NAMES
    splits = ["train", "val", "test"]

    for split in splits:
        for c_idx, c_name in enumerate(classes):
            for i in range(2):
                img_path = tmp_path / f"{split}_{c_name}_{i}.png"
                dummy_img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
                import cv2
                cv2.imwrite(str(img_path), dummy_img)

                records.append({
                    "file_path": str(img_path),
                    "class_name": c_name,
                    "class_idx": c_idx,
                    "split": split,
                    "image_id": f"{split}_{c_name}_{i}",
                })

    return pd.DataFrame(records)


def test_augmenter_output_shape_and_dtype():
    """
    Verifies that BrainMRIAugmenter preserves uint8 dtype and image dimensions.
    """
    augmenter = BrainMRIAugmenter(rotation_degrees=15.0, flip_prob=1.0)
    sample_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    augmented = augmenter(sample_img)
    assert augmented.shape == (224, 224, 3)
    assert augmented.dtype == np.uint8
    assert augmented.min() >= 0
    assert augmented.max() <= 255


def test_dataset_item_generation(dummy_manifest):
    """
    Verifies that BrainMRIDataset yields valid PyTorch tensors and metadata.
    """
    preprocessor = BrainMRIPreprocessor()
    dataset = BrainMRIDataset(dummy_manifest, split="train", preprocessor=preprocessor)

    assert len(dataset) == len(dummy_manifest)

    img_tensor, label_tensor, meta = dataset[0]

    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert img_tensor.dtype == torch.float32

    assert isinstance(label_tensor, torch.Tensor)
    assert label_tensor.dtype == torch.long
    assert 0 <= label_tensor.item() < len(CLASS_NAMES)

    assert "file_path" in meta
    assert "class_name" in meta
    assert "crop_bbox" in meta
    assert meta["split"] == "train"


def test_dataset_strict_split_augmentation_rule(dummy_manifest):
    """
    Verifies that training split enables augmentations, while val/test disables them.
    """
    train_dataset = BrainMRIDataset(dummy_manifest, split="train")
    val_dataset = BrainMRIDataset(dummy_manifest, split="val")
    test_dataset = BrainMRIDataset(dummy_manifest, split="test")

    assert train_dataset.augment is True
    assert val_dataset.augment is False
    assert test_dataset.augment is False


def test_dataloader_creation(dummy_manifest):
    """
    Verifies DataLoader batch creation and tensor collation.
    """
    loaders = create_dataloaders(dummy_manifest, batch_size=4, num_workers=0)

    assert "train" in loaders
    assert "val" in loaders
    assert "test" in loaders

    train_loader = loaders["train"]
    for batch_images, batch_labels, batch_metas in train_loader:
        assert batch_images.shape[0] <= 4
        assert batch_images.shape[1:] == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert batch_labels.shape[0] <= 4
        assert isinstance(batch_metas, dict)
        break


def test_preprocessor_to_tensor():
    """
    Verifies BrainMRIPreprocessor.to_tensor with and without batch dimension.
    """
    preprocessor = BrainMRIPreprocessor()
    sample_bgr = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)

    t_batched = preprocessor.to_tensor(sample_bgr, add_batch_dim=True)
    assert t_batched.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert isinstance(t_batched, torch.Tensor)

    t_single = preprocessor.to_tensor(sample_bgr, add_batch_dim=False)
    assert t_single.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert isinstance(t_single, torch.Tensor)

"""
Unit tests for Brain MRI Preprocessing Pipeline (Phase 1)
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import (
    ARCHIVE_DIR,
    CLASS_NAMES,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from backend.app.core.preprocessor import BrainMRIPreprocessor


@pytest.fixture
def preprocessor():
    return BrainMRIPreprocessor()


@pytest.fixture
def synthetic_mri_image():
    """
    Creates a synthetic 512x512 brain-like image:
    - Black canvas border (background)
    - Elliptical white/gray cranial skull perimeter
    - Internal brain-like texture
    """
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    center = (256, 256)
    axes = (180, 220)  # Cranial ellipse

    # Draw skull border
    cv2.ellipse(img, center, axes, 0, 0, 360, (200, 200, 200), -1)

    # Draw internal brain parenchyma structure with varying intensities
    cv2.ellipse(img, center, (160, 200), 0, 0, 360, (120, 120, 120), -1)
    cv2.circle(img, (220, 240), 30, (180, 180, 180), -1)  # Synthetic tumor lesion
    return img


def test_skull_contour_cropping(preprocessor, synthetic_mri_image):
    """
    Verifies that the black margin is cropped and cranial perimeter is isolated.
    """
    cropped, bbox = preprocessor.crop_skull_contour(synthetic_mri_image)
    x_min, y_min, x_max, y_max = bbox

    # Bounding box must be within original dimensions
    assert 0 <= x_min < x_max <= 512
    assert 0 <= y_min < y_max <= 512

    # Cropped dimensions must be tighter than the original 512x512 black canvas
    assert cropped.shape[0] < 512
    assert cropped.shape[1] < 512
    assert cropped.shape[0] > 100
    assert cropped.shape[1] > 100
    assert (x_max - x_min) == cropped.shape[1]
    assert (y_max - y_min) == cropped.shape[0]


def test_skull_cropping_fallback(preprocessor):
    """
    Verifies graceful fallback to full image if image is completely black or uniform.
    """
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    cropped, bbox = preprocessor.crop_skull_contour(blank)
    assert cropped.shape == (300, 300, 3)
    assert bbox == (0, 0, 300, 300)


def test_dimension_standardization(preprocessor, synthetic_mri_image):
    """
    Verifies that image is resized to standard 224x224 target.
    """
    resized = preprocessor.resize(synthetic_mri_image)
    assert resized.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)


def test_clahe_contrast_enhancement(preprocessor, synthetic_mri_image):
    """
    Verifies CLAHE normalization enhances local contrast.
    """
    clahe_out = preprocessor.apply_clahe(synthetic_mri_image)
    assert clahe_out.shape == synthetic_mri_image.shape
    assert clahe_out.dtype == np.uint8
    # Standard deviation should be maintained or enhanced
    assert np.std(clahe_out) > 0


def test_imagenet_normalization(preprocessor):
    """
    Verifies that normalization applies ImageNet mean and std.
    """
    ones_rgb = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255, dtype=np.uint8)
    normalized = preprocessor.normalize(ones_rgb)

    assert normalized.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert normalized.dtype == np.float32

    # For 1.0 (255/255), (1.0 - mean) / std
    expected_channel_0 = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    assert np.allclose(normalized[:, :, 0], expected_channel_0, atol=1e-3)


def test_full_preprocessing_pipeline_dict(preprocessor, synthetic_mri_image):
    """
    Verifies full preprocessing pipeline with return_intermediates=True.
    """
    result = preprocessor.preprocess_image(synthetic_mri_image, return_intermediates=True)

    required_keys = [
        "raw_bgr",
        "raw_rgb",
        "cropped_bgr",
        "cropped_rgb",
        "resized_bgr",
        "clahe_bgr",
        "preprocessed_uint8_rgb",
        "normalized_rgb",
        "crop_bbox",
    ]
    for key in required_keys:
        assert key in result, f"Missing key in preprocessing result: {key}"

    assert result["normalized_rgb"].shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert result["preprocessed_uint8_rgb"].shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert result["preprocessed_uint8_rgb"].dtype == np.uint8


def test_real_dataset_sample(preprocessor):
    """
    Verifies preprocessing on an actual scan from archive directory.
    """
    sample_dir = ARCHIVE_DIR / "Testing" / "glioma"
    if not sample_dir.exists():
        pytest.skip("Dataset archive not found.")

    sample_files = list(sample_dir.glob("*.jpg"))
    if not sample_files:
        pytest.skip("No sample images in Testing/glioma.")

    sample_path = str(sample_files[0])
    result = preprocessor.preprocess_image(sample_path, return_intermediates=True)

    assert result["normalized_rgb"].shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert result["preprocessed_uint8_rgb"].shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert len(result["crop_bbox"]) == 4


def test_invalid_input_error(preprocessor):
    """
    Verifies proper error handling on invalid or nonexistent input.
    """
    with pytest.raises(ValueError):
        preprocessor.preprocess_image("nonexistent_image_path_12345.jpg")

    with pytest.raises(TypeError):
        preprocessor.preprocess_image(12345)

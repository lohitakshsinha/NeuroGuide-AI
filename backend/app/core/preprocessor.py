"""
NeuroGuide AI - Brain MRI Preprocessing Pipeline

Implements clinical-grade MRI preprocessing:
1. Grayscale conversion & Gaussian smoothing (5x5).
2. Otsu automatic thresholding.
3. Skull perimeter contour detection & extreme coordinate cropping.
4. Bilinear/bicubic spatial standardization (224x224).
5. CLAHE (Contrast Limited Adaptive Histogram Equalization) contrast calibration.
6. ImageNet normalization & PyTorch FloatTensor generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False

from backend.app.config import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    BLUR_KERNEL_SIZE,
    CONTOUR_MIN_AREA_RATIO,
    CONTOUR_PADDING_PERCENT,
)


class BrainMRIPreprocessor:
    """
    Standardized, reproducible preprocessing pipeline for Brain MRI scans.
    """

    def __init__(
        self,
        target_size: int = IMAGE_SIZE,
        clahe_clip_limit: float = CLAHE_CLIP_LIMIT,
        clahe_tile_grid_size: Tuple[int, int] = CLAHE_TILE_GRID_SIZE,
        blur_kernel: Tuple[int, int] = BLUR_KERNEL_SIZE,
        min_area_ratio: float = CONTOUR_MIN_AREA_RATIO,
        padding_ratio: float = CONTOUR_PADDING_PERCENT,
    ):
        self.target_size = target_size
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.blur_kernel = blur_kernel
        self.min_area_ratio = min_area_ratio
        self.padding_ratio = padding_ratio

        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size,
        )

        self.imagenet_mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.imagenet_std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)

    def crop_skull_contour(
        self,
        image_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Detects cranial perimeter and crops redundant dark canvas background.

        Args:
            image_bgr: BGR or Grayscale uint8 numpy image.

        Returns:
            cropped_image: Cropped BGR image.
            bbox: (x_min, y_min, x_max, y_max) bounding box coordinates.
        """
        h, w = image_bgr.shape[:2]
        total_area = h * w

        # Convert to grayscale
        if len(image_bgr.shape) == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr.copy()
            image_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Gaussian smoothing to suppress sensor speckle noise
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        # Otsu automatic thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological opening/closing to bridge small skull discontinuities
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find external contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # Fallback to full image if no contour detected
            return image_bgr, (0, 0, w, h)

        # Largest external contour corresponds to the cranium
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)

        # If detected contour is too minuscule (e.g. noise artifact), avoid over-cropping
        if contour_area < (total_area * self.min_area_ratio):
            return image_bgr, (0, 0, w, h)

        # Extract extreme coordinates
        x_min = int(largest_contour[:, :, 0].min())
        x_max = int(largest_contour[:, :, 0].max())
        y_min = int(largest_contour[:, :, 1].min())
        y_max = int(largest_contour[:, :, 1].max())

        # Add fractional margin padding
        pad_x = int(w * self.padding_ratio)
        pad_y = int(h * self.padding_ratio)

        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(w, x_max + pad_x)
        y_max = min(h, y_max + pad_y)

        # Safety check: ensure positive width and height
        if (x_max - x_min) < 20 or (y_max - y_min) < 20:
            return image_bgr, (0, 0, w, h)

        cropped = image_bgr[y_min:y_max, x_min:x_max]
        return cropped, (x_min, y_min, x_max, y_max)

    def apply_clahe(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization.

        Transforms BGR image to LAB color space, equalizes the L (Luminance) channel,
        and converts back to BGR. This preserves anatomical tissue gradient without
        distorting color balance or introducing artificial saturation.

        Args:
            image_bgr: BGR uint8 numpy array.

        Returns:
            clahe_bgr: Contrast-normalized BGR image.
        """
        if len(image_bgr.shape) == 2:
            return self.clahe.apply(image_bgr)

        # Convert to LAB space
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE exclusively to luminance
        equalized_l = self.clahe.apply(l_channel)

        # Re-merge and convert back to BGR
        merged_lab = cv2.merge((equalized_l, a_channel, b_channel))
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    def resize(
        self,
        image_bgr: np.ndarray,
        interpolation: int = cv2.INTER_CUBIC,
    ) -> np.ndarray:
        """
        Standardizes image resolution to (target_size, target_size).
        """
        return cv2.resize(image_bgr, (self.target_size, self.target_size), interpolation=interpolation)

    def normalize(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Normalizes RGB image to [0, 1] then applies ImageNet mean/std scaling.

        Args:
            image_rgb: RGB uint8 numpy array in range [0, 255].

        Returns:
            normalized_float32: Standardized float32 numpy array.
        """
        image_float = image_rgb.astype(np.float32) / 255.0
        normalized = (image_float - self.imagenet_mean) / self.imagenet_std
        return normalized

    def preprocess_image(
        self,
        image_input: Union[str, np.ndarray, Image.Image, bytes],
        crop_skull: bool = True,
        use_clahe: bool = True,
        return_intermediates: bool = False,
    ) -> Union[np.ndarray, Dict[str, Union[np.ndarray, Tuple[int, int, int, int]]]]:
        """
        Full end-to-end numpy preprocessing pipeline.

        Args:
            image_input: Filepath string, numpy array, PIL Image, or raw image bytes.
            crop_skull: Whether to detect and crop skull contour.
            use_clahe: Whether to apply CLAHE contrast equalization.
            return_intermediates: If True, returns dict with each stage's output.

        Returns:
            If return_intermediates is False:
                preprocessed_rgb: Normalized float32 array (target_size, target_size, 3).
            If return_intermediates is True:
                Dictionary with intermediate visual representations for inspection/UI.
        """
        # Load / convert input into BGR uint8 numpy array
        raw_bgr = self._load_to_bgr(image_input)

        # 1. Skull Stripping & Extreme Contour Cropping
        if crop_skull:
            cropped_bgr, bbox = self.crop_skull_contour(raw_bgr)
        else:
            cropped_bgr = raw_bgr.copy()
            bbox = (0, 0, raw_bgr.shape[1], raw_bgr.shape[0])

        # 2. Dimension Standardisation
        resized_bgr = self.resize(cropped_bgr)

        # 3. Contrast Normalization (CLAHE)
        if use_clahe:
            clahe_bgr = self.apply_clahe(resized_bgr)
        else:
            clahe_bgr = resized_bgr.copy()

        # Convert to RGB
        rgb_uint8 = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB)

        # 4. ImageNet Normalization
        normalized_rgb = self.normalize(rgb_uint8)

        if not return_intermediates:
            return normalized_rgb

        return {
            "raw_bgr": raw_bgr,
            "raw_rgb": cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB),
            "cropped_bgr": cropped_bgr,
            "cropped_rgb": cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB),
            "resized_bgr": resized_bgr,
            "clahe_bgr": clahe_bgr,
            "preprocessed_uint8_rgb": rgb_uint8,
            "normalized_rgb": normalized_rgb,
            "crop_bbox": bbox,
        }

    def to_tensor(
        self,
        image_input: Union[str, np.ndarray, Image.Image, bytes],
        add_batch_dim: bool = True,
        crop_skull: bool = True,
        use_clahe: bool = True,
    ) -> "torch.Tensor":
        """
        Executes preprocessing and returns a PyTorch FloatTensor.

        Returns:
            Tensor of shape (1, 3, target_size, target_size) if add_batch_dim is True,
            or (3, target_size, target_size) if False.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for to_tensor(). Please install torch.")

        normalized_rgb = self.preprocess_image(
            image_input,
            crop_skull=crop_skull,
            use_clahe=use_clahe,
            return_intermediates=False,
        )

        # HWC (224, 224, 3) -> CHW (3, 224, 224)
        chw = np.transpose(normalized_rgb, (2, 0, 1)).astype(np.float32)
        tensor = torch.from_numpy(chw)

        if add_batch_dim:
            tensor = tensor.unsqueeze(0)

        return tensor

    def _load_to_bgr(self, image_input: Union[str, Path, np.ndarray, Image.Image, bytes]) -> np.ndarray:
        """
        Helper to convert various input formats into standardized BGR uint8 array.
        """
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise ValueError(f"Unable to read image at path: {image_input}")
            return img

        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Unable to decode image bytes into valid image.")
            return img

        elif isinstance(image_input, Image.Image):
            rgb_arr = np.array(image_input.convert("RGB"))
            return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

        elif isinstance(image_input, np.ndarray):
            if image_input.dtype != np.uint8:
                if image_input.max() <= 1.0:
                    image_input = (image_input * 255.0).astype(np.uint8)
                else:
                    image_input = image_input.astype(np.uint8)

            if len(image_input.shape) == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif len(image_input.shape) == 3:
                if image_input.shape[2] == 1:
                    return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
                elif image_input.shape[2] == 4:
                    return cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
                return image_input
            raise ValueError(f"Unsupported numpy image shape: {image_input.shape}")

        raise TypeError(f"Unsupported image input type: {type(image_input)}")

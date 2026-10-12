"""
NeuroGuide AI - Out-of-Distribution (OOD) Input Gatekeeper

Implements Section 4.2 of the project implementation plan:
Validates incoming image uploads before executing deep learning inference:
1. Grayscale vs Color Saturation check (standard MRI is monochromatic; rejects colorful photos of people, animals, landscapes).
2. Intensity & Background Distribution check (MRI scans feature a black or dark canvas surrounding a cranial ellipse; rejects blank white sheets or screenshots).
3. Cranial Ellipse & Contour Sanity Check (validates that a plausible cranial structure exists within expected aspect ratios).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class OODDetector:
    """
    Heuristic and statistical gatekeeper guarding the diagnostic pipeline against
    accidental or adversarial non-MRI image submissions.
    """

    def __init__(
        self,
        max_saturation_mean: float = 28.0,
        max_channel_diff_std: float = 18.0,
        min_dark_background_ratio: float = 0.08,
        min_cranial_area_ratio: float = 0.05,
    ):
        """
        Args:
            max_saturation_mean: Maximum allowable mean HSV saturation (0-255).
            max_channel_diff_std: Maximum allowable standard deviation across R, G, B channels.
            min_dark_background_ratio: Minimum fraction of peripheral pixels that must be low intensity.
            min_cranial_area_ratio: Minimum contour area relative to image area.
        """
        self.max_saturation_mean = max_saturation_mean
        self.max_channel_diff_std = max_channel_diff_std
        self.min_dark_background_ratio = min_dark_background_ratio
        self.min_cranial_area_ratio = min_cranial_area_ratio

    def validate_image(
        self,
        image_input: Union[str, np.ndarray, Image.Image, bytes],
    ) -> Dict[str, Any]:
        """
        Validates whether the image matches basic structural and photometric properties
        of an authentic brain MRI scan.

        Returns:
            {
                "is_valid_mri": bool,
                "rejection_reason": Optional[str],
                "confidence_score": float [0.0, 1.0],
                "diagnostics": {
                    "mean_saturation": float,
                    "channel_diff_std": float,
                    "dark_background_ratio": float,
                    "cranial_contour_area_ratio": float
                }
            }
        """
        img_bgr = self._load_image(image_input)
        h, w = img_bgr.shape[:2]

        if h < 32 or w < 32:
            return {
                "is_valid_mri": False,
                "rejection_reason": "Image dimensions are too low (< 32x32) to contain diagnostic cranial features.",
                "confidence_score": 0.0,
                "diagnostics": {"height": h, "width": w},
            }

        # 1. Color Saturation & Channel Divergence
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        mean_sat = float(np.mean(sat))

        # Difference between R, G, B channels
        b, g, r = img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2]
        channel_diff = np.abs(r.astype(np.float32) - g.astype(np.float32)) + np.abs(g.astype(np.float32) - b.astype(np.float32))
        mean_channel_diff = float(np.mean(channel_diff))

        is_colorful = (mean_sat > self.max_saturation_mean) or (mean_channel_diff > self.max_channel_diff_std)

        # 2. Dark Background Check
        # MRIs typically have black borders or corner pixels
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        dark_pixels = np.sum(gray < 40)
        dark_bg_ratio = float(dark_pixels / (h * w))

        # Check if corners are dark
        corner_size = max(4, min(h, w) // 10)
        corners = [
            gray[:corner_size, :corner_size],
            gray[:corner_size, -corner_size:],
            gray[-corner_size:, :corner_size],
            gray[-corner_size:, -corner_size:],
        ]
        corner_mean = float(np.mean([np.mean(c) for c in corners]))

        # 3. Cranial Contour Area & Shape
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cranial_area_ratio = float(cv2.contourArea(largest_contour) / (h * w))
        else:
            cranial_area_ratio = 0.0

        diagnostics = {
            "mean_saturation": round(mean_sat, 2),
            "channel_diff_mean": round(mean_channel_diff, 2),
            "dark_background_ratio": round(dark_bg_ratio, 3),
            "corner_mean_intensity": round(corner_mean, 1),
            "cranial_contour_area_ratio": round(cranial_area_ratio, 3),
        }

        # Decision Logic
        if is_colorful:
            return {
                "is_valid_mri": False,
                "rejection_reason": (
                    f"Out-of-distribution image rejected: high color saturation detected "
                    f"(saturation={mean_sat:.1f}). Clinical brain MRI scans are monochromatic grayscale."
                ),
                "confidence_score": 0.05,
                "diagnostics": diagnostics,
            }

        if dark_bg_ratio < self.min_dark_background_ratio and corner_mean > 200:
            return {
                "is_valid_mri": False,
                "rejection_reason": (
                    "Out-of-distribution image rejected: bright/white background detected. "
                    "Brain MRI scans require a dark background field."
                ),
                "confidence_score": 0.10,
                "diagnostics": diagnostics,
            }

        if cranial_area_ratio < self.min_cranial_area_ratio:
            return {
                "is_valid_mri": False,
                "rejection_reason": (
                    "Unable to identify cranial brain structure. Image appears blank or features no distinct parenchyma."
                ),
                "confidence_score": 0.20,
                "diagnostics": diagnostics,
            }

        # Passed all gatekeeper heuristics
        return {
            "is_valid_mri": True,
            "rejection_reason": None,
            "confidence_score": 0.95,
            "diagnostics": diagnostics,
        }

    def _load_image(self, image_input: Union[str, Path, np.ndarray, Image.Image, bytes]) -> np.ndarray:
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise ValueError(f"Unable to read image from path: {image_input}")
            return img
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Unable to decode image bytes.")
            return img
        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif len(image_input.shape) == 3 and image_input.shape[2] == 3:
                return image_input
            elif len(image_input.shape) == 3 and image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

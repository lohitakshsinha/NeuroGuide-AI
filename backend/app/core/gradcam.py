"""
NeuroGuide AI - Explainable AI (Grad-CAM) & ROI Localization Engine

Implements Phase 3 of the NeuroGuide AI architecture:
1. Target feature layer hooking for activations and gradients.
2. Gradient-weighted Class Activation Mapping (Grad-CAM) with ReLU rectification.
3. Spatial normalization, bicubic/bilinear upsampling, and OpenCV colormap blending.
4. Automated Region of Interest (ROI) extraction:
   - Top 20% activation thresholding.
   - Bounding box coordinates (pixel and normalized).
   - Centroid, peak coordinates, and area coverage fraction.
   - Neuroanatomical quadrant & radiological localization heuristic.
5. Base64 serialization for zero-latency web API transmission.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False
    nn = object

from backend.app.config import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    IMAGE_SIZE,
    NUM_CLASSES,
)
from backend.app.core.preprocessor import BrainMRIPreprocessor

logger = logging.getLogger(__name__)


# Supported OpenCV Colormaps for Neuroimaging
COLORMAP_REGISTRY: Dict[str, int] = {
    "turbo": cv2.COLORMAP_TURBO,
    "jet": cv2.COLORMAP_JET,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "hot": cv2.COLORMAP_HOT,
    "plasma": cv2.COLORMAP_PLASMA,
}


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation
    tailored for convolutional neural networks in clinical neuroimaging.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
    ):
        """
        Initializes Grad-CAM with a PyTorch model and registers hooks on the target layer.

        Args:
            model: PyTorch model instance (e.g. NeuroBrainClassifier).
            target_layer: The target convolutional layer. If None, queries model.get_target_layer().
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to instantiate GradCAM.")

        self.model = model
        self.model.eval()

        if target_layer is None:
            if hasattr(model, "get_target_layer"):
                self.target_layer = model.get_target_layer()
            else:
                raise ValueError(
                    "target_layer was not provided and model has no get_target_layer() method."
                )
        else:
            self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks: List[Any] = []

        self._register_hooks()

    def _register_hooks(self) -> None:
        """Registers forward and backward hooks on the target convolutional layer."""
        self.remove_hooks()

        def forward_hook(module: nn.Module, input: Any, output: torch.Tensor):
            self.activations = output.detach()

        def backward_hook(module: nn.Module, grad_input: Any, grad_output: Any):
            # grad_output is a tuple of gradients with respect to output
            if isinstance(grad_output, tuple) and len(grad_output) > 0:
                self.gradients = grad_output[0].detach()
            else:
                self.gradients = grad_output.detach()

        f_handle = self.target_layer.register_forward_hook(forward_hook)
        b_handle = self.target_layer.register_full_backward_hook(backward_hook)
        self._hooks = [f_handle, b_handle]

    def remove_hooks(self) -> None:
        """Removes registered PyTorch hooks to prevent memory leaks."""
        for handle in self._hooks:
            try:
                handle.remove()
            except Exception:
                pass
        self._hooks = []
        self.activations = None
        self.gradients = None

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove_hooks()

    def __del__(self) -> None:
        self.remove_hooks()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        target_size: Optional[Tuple[int, int]] = None,
        counterfactual: bool = False,
    ) -> Tuple[np.ndarray, int, float, Dict[str, float]]:
        """
        Computes the Grad-CAM activation heatmap for a given input tensor.

        Args:
            input_tensor: PyTorch tensor (1, C, H, W) or (C, H, W).
            target_class: Integer index of target class. If None, uses top predicted class.
            target_size: Desired (width, height) for the output heatmap. Defaults to input tensor size.
            counterfactual: If True, highlights regions that negatively contribute to class.

        Returns:
            heatmap: 2D numpy array with float32 values normalized to [0.0, 1.0].
            predicted_class_idx: Integer index of the model's top prediction.
            confidence: Confidence probability of the top predicted class.
            all_probabilities: Dict mapping class names or indices to calibrated probabilities.
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad_(True)

        self.model.zero_grad()
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        # Select class score to backpropagate
        if target_class is None:
            class_to_explain = pred_idx
        else:
            class_to_explain = int(target_class)

        target_score = logits[0, class_to_explain]
        target_score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Grad-CAM failed to capture activations or gradients. "
                "Ensure target_layer is part of the computation graph."
            )

        # Gradients: (1, C, H_feat, W_feat)
        # Activations: (1, C, H_feat, W_feat)
        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()

        # Step 2: Global Average Pooling of gradients over spatial dimensions (alpha weights)
        weights = np.mean(gradients, axis=(1, 2))  # shape: (C,)

        if counterfactual:
            weights = -weights

        # Step 3: Weighted linear combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Step 4: Rectified Linear Activation (ReLU)
        cam = np.maximum(cam, 0)

        # Step 5: Normalize to [0.0, 1.0]
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam = np.zeros_like(cam)

        # Step 6: Upsample heatmap to target resolution
        if target_size is None:
            target_size = (input_tensor.shape[-1], input_tensor.shape[-2])  # (width, height)

        cam_resized = cv2.resize(
            cam,
            target_size,
            interpolation=cv2.INTER_LINEAR,
        )

        # Re-ensure bounds [0.0, 1.0] after interpolation
        cam_resized = np.clip(cam_resized, 0.0, 1.0).astype(np.float32)

        # Class probabilities mapping
        all_probs: Dict[str, float] = {}
        for idx in range(len(probs)):
            cname = IDX_TO_CLASS.get(idx, f"class_{idx}")
            all_probs[cname] = float(probs[idx])

        return cam_resized, pred_idx, confidence, all_probs

    def generate_all_classes_heatmaps(
        self,
        input_tensor: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generates individual Grad-CAM heatmaps for each class to compare activations.
        """
        heatmaps = {}
        for idx, class_name in enumerate(CLASS_NAMES):
            cam, _, _, _ = self.generate_heatmap(
                input_tensor,
                target_class=idx,
                target_size=target_size,
            )
            heatmaps[class_name] = cam
        return heatmaps

    @staticmethod
    def apply_colormap(
        heatmap: np.ndarray,
        colormap_name: str = "turbo",
    ) -> np.ndarray:
        """
        Applies an OpenCV neuroimaging colormap to a [0.0, 1.0] float heatmap.

        Args:
            heatmap: 2D numpy array [0.0, 1.0].
            colormap_name: One of 'turbo', 'jet', 'viridis', 'inferno', 'magma', etc.

        Returns:
            RGB uint8 numpy array (H, W, 3).
        """
        cmap_code = COLORMAP_REGISTRY.get(colormap_name.lower(), cv2.COLORMAP_TURBO)
        heatmap_uint8 = (np.clip(heatmap, 0.0, 1.0) * 255.0).astype(np.uint8)

        # cv2.applyColorMap returns BGR
        bgr_map = cv2.applyColorMap(heatmap_uint8, cmap_code)
        rgb_map = cv2.cvtColor(bgr_map, cv2.COLOR_BGR2RGB)
        return rgb_map

    @staticmethod
    def blend_overlay(
        base_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap_name: str = "turbo",
    ) -> np.ndarray:
        """
        Blends an anatomical brain image with a colored Grad-CAM heatmap.

        Formula:
            I_blended = alpha * I_colormap + (1 - alpha) * I_base

        Args:
            base_image: Grayscale or RGB uint8 image (H, W) or (H, W, 3).
            heatmap: 2D float32 array [0.0, 1.0] matching or rescalable to base_image.
            alpha: Heatmap blend opacity weight [0.0, 1.0].
            colormap_name: Colormap name to use.

        Returns:
            RGB uint8 numpy array (H, W, 3) with blended overlay.
        """
        alpha = float(np.clip(alpha, 0.0, 1.0))
        h, w = base_image.shape[:2]

        # Prepare base image as RGB
        if len(base_image.shape) == 2:
            base_rgb = cv2.cvtColor(base_image, cv2.COLOR_GRAY2RGB)
        elif base_image.shape[2] == 1:
            base_rgb = cv2.cvtColor(base_image, cv2.COLOR_GRAY2RGB)
        elif base_image.shape[2] == 3:
            base_rgb = base_image.copy()
        elif base_image.shape[2] == 4:
            base_rgb = cv2.cvtColor(base_image, cv2.COLOR_RGBA2RGB)
        else:
            raise ValueError(f"Unexpected base_image shape: {base_image.shape}")

        if base_rgb.dtype != np.uint8:
            if base_rgb.max() <= 1.0:
                base_rgb = (base_rgb * 255.0).astype(np.uint8)
            else:
                base_rgb = base_rgb.astype(np.uint8)

        # Resize heatmap if shape mismatch
        if heatmap.shape[:2] != (h, w):
            heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        colored_cam = GradCAM.apply_colormap(heatmap, colormap_name=colormap_name)

        # Alpha blending
        blended = (alpha * colored_cam.astype(np.float32) + (1.0 - alpha) * base_rgb.astype(np.float32))
        blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)
        return blended

    @staticmethod
    def to_base64_png(image_array: np.ndarray, include_prefix: bool = True) -> str:
        """
        Encodes a uint8 RGB/Grayscale image array to a Base64 PNG data URI string.

        Args:
            image_array: RGB (H, W, 3), Grayscale (H, W), or float heatmap [0, 1].
            include_prefix: If True, prepends 'data:image/png;base64,'.

        Returns:
            Base64 encoded string.
        """
        if image_array.dtype != np.uint8:
            if image_array.max() <= 1.0:
                image_array = (image_array * 255.0).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)

        pil_img = Image.fromarray(image_array)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG", optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        if include_prefix:
            return f"data:image/png;base64,{encoded}"
        return encoded


# =========================================================================
# Region of Interest (ROI) & Neuroanatomical Localization
# =========================================================================

def extract_roi(
    heatmap: np.ndarray,
    threshold_ratio: float = 0.80,
    min_area_pixels: int = 16,
    original_shape: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """
    Extracts the Region of Interest (ROI) from the Grad-CAM activation heatmap.

    Follows Section 3.2:
        Mask(x, y) = 1 if L_Grad-CAM(x, y) >= threshold_ratio * max(L) else 0

    Args:
        heatmap: 2D float numpy array [0.0, 1.0].
        threshold_ratio: Intensity cutoff fraction relative to maximum activation (default 0.80).
        min_area_pixels: Minimum contour area in pixels to reject isolated noise artifacts.
        original_shape: Optional (height, width) of the raw uploaded image to scale bbox.

    Returns:
        Structured dictionary containing:
        - has_focal_activation: bool
        - threshold_used: float
        - peak_value: float
        - peak_coords: [x, y] in heatmap pixel grid
        - peak_coords_norm: [x_norm, y_norm] in [0, 1]
        - centroid: [x, y] center of mass
        - centroid_norm: [cx_norm, cy_norm] in [0, 1]
        - bbox: [x_min, y_min, x_max, y_max] in heatmap pixel grid
        - bbox_norm: [x_min, y_min, x_max, y_max] normalized [0, 1]
        - bbox_original: [x_min, y_min, x_max, y_max] in original scan resolution (if provided)
        - area_pixels: int
        - area_fraction: float (% of image occupied by high activation)
        - anatomical_region: Clinical localization string
        - anatomical_quadrant: High-level quadrant string
        - contour_polygon: List of [x, y] polygon vertices for SVG / Canvas rendering
    """
    h, w = heatmap.shape[:2]
    max_val = float(np.max(heatmap))

    if max_val <= 1e-6:
        return _empty_roi_response(h, w, original_shape)

    cutoff = float(threshold_ratio * max_val)

    # Binary mask of top activation
    binary_mask = (heatmap >= cutoff).astype(np.uint8) * 255

    # Peak coordinates (maximum intensity location)
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    peak_x, peak_y = int(peak_x), int(peak_y)

    # Find contours of activation islands
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area_pixels]

    if not valid_contours:
        # If no single contour meets minimum area, check if any activation exists
        if np.sum(binary_mask > 0) >= 4:
            points = np.column_stack(np.where(binary_mask > 0))  # y, x
            y_min, x_min = int(np.min(points[:, 0])), int(np.min(points[:, 1]))
            y_max, x_max = int(np.max(points[:, 0])), int(np.max(points[:, 1]))
            cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
            area_px = int(np.sum(binary_mask > 0))
            poly_points = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        else:
            return _empty_roi_response(h, w, original_shape)
    else:
        # Combine all valid contours into primary bounding region
        all_points = np.vstack(valid_contours)
        x, y, bw, bh = cv2.boundingRect(all_points)
        x_min, y_min, x_max, y_max = x, y, x + bw, y + bh

        # Compute centroid using spatial moments
        total_m00 = sum(cv2.moments(cnt)["m00"] for cnt in valid_contours)
        if total_m00 > 0:
            cx = sum(cv2.moments(cnt)["m10"] for cnt in valid_contours) / total_m00
            cy = sum(cv2.moments(cnt)["m01"] for cnt in valid_contours) / total_m00
        else:
            cx, cy = float(peak_x), float(peak_y)

        area_px = int(sum(cv2.contourArea(cnt) for cnt in valid_contours))

        # Simplify largest contour for fast frontend rendering
        largest_cnt = max(valid_contours, key=cv2.contourArea)
        epsilon = 0.015 * cv2.arcLength(largest_cnt, True)
        approx = cv2.approxPolyDP(largest_cnt, epsilon, True)
        poly_points = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]

    # Normalized coordinates
    bbox_norm = [
        float(np.clip(x_min / w, 0.0, 1.0)),
        float(np.clip(y_min / h, 0.0, 1.0)),
        float(np.clip(x_max / w, 0.0, 1.0)),
        float(np.clip(y_max / h, 0.0, 1.0)),
    ]
    centroid_norm = [
        float(np.clip(cx / w, 0.0, 1.0)),
        float(np.clip(cy / h, 0.0, 1.0)),
    ]
    peak_coords_norm = [
        float(np.clip(peak_x / w, 0.0, 1.0)),
        float(np.clip(peak_y / h, 0.0, 1.0)),
    ]

    area_fraction = float(area_px / (h * w))

    # Scale to original shape if provided
    if original_shape is not None:
        orig_h, orig_w = original_shape[:2]
        bbox_original = [
            int(bbox_norm[0] * orig_w),
            int(bbox_norm[1] * orig_h),
            int(bbox_norm[2] * orig_w),
            int(bbox_norm[3] * orig_h),
        ]
    else:
        bbox_original = [x_min, y_min, x_max, y_max]

    # Clinical anatomical localization heuristic
    anatomical_desc, quadrant_desc = determine_anatomical_location(
        cx_norm=centroid_norm[0],
        cy_norm=centroid_norm[1],
        bbox_norm=bbox_norm,
        area_fraction=area_fraction,
    )

    return {
        "has_focal_activation": True,
        "threshold_used": cutoff,
        "threshold_ratio": threshold_ratio,
        "peak_value": max_val,
        "peak_coords": [peak_x, peak_y],
        "peak_coords_norm": peak_coords_norm,
        "centroid": [float(round(cx, 1)), float(round(cy, 1))],
        "centroid_norm": [round(c, 4) for c in centroid_norm],
        "bbox": [x_min, y_min, x_max, y_max],
        "bbox_norm": [round(b, 4) for b in bbox_norm],
        "bbox_original": bbox_original,
        "area_pixels": area_px,
        "area_fraction": round(area_fraction, 4),
        "anatomical_region": anatomical_desc,
        "anatomical_quadrant": quadrant_desc,
        "contour_polygon": poly_points,
    }


def _empty_roi_response(h: int, w: int, original_shape: Optional[Tuple[int, int]]) -> Dict[str, Any]:
    """Helper for scans with no focal hyperactivation (e.g. normal parenchyma)."""
    orig = [0, 0, original_shape[1], original_shape[0]] if original_shape else [0, 0, w, h]
    return {
        "has_focal_activation": False,
        "threshold_used": 0.0,
        "threshold_ratio": 0.80,
        "peak_value": 0.0,
        "peak_coords": [w // 2, h // 2],
        "peak_coords_norm": [0.5, 0.5],
        "centroid": [float(w // 2), float(h // 2)],
        "centroid_norm": [0.5, 0.5],
        "bbox": [0, 0, w, h],
        "bbox_norm": [0.0, 0.0, 1.0, 1.0],
        "bbox_original": orig,
        "area_pixels": 0,
        "area_fraction": 0.0,
        "anatomical_region": "Diffuse / Normal parenchymal baseline (No focal mass)",
        "anatomical_quadrant": "Non-localized / Symmetrical",
        "contour_polygon": [],
    }


def determine_anatomical_location(
    cx_norm: float,
    cy_norm: float,
    bbox_norm: List[float],
    area_fraction: float,
) -> Tuple[str, str]:
    """
    Determines the neuroanatomical location and quadrant based on standard
    axial/coronal brain MRI coordinates and radiological orientation conventions.

    Radiological Convention Note:
    - Left side of image = Patient's Anatomical Right.
    - Right side of image = Patient's Anatomical Left.
    - Top of image = Anterior (Frontal) or Superior.
    - Bottom of image = Posterior (Occipital) or Inferior (Skull Base / Cerebellum).

    Args:
        cx_norm: Normalized horizontal centroid [0.0, 1.0].
        cy_norm: Normalized vertical centroid [0.0, 1.0].
        bbox_norm: [x_min, y_min, x_max, y_max] normalized coordinates.
        area_fraction: Fractional area of the high-activation region.

    Returns:
        (anatomical_region, quadrant): Descriptive medical localization strings.
    """
    # Lateral Hemisphere determination (Radiological View)
    if cx_norm < 0.42:
        hemi_anatomical = "Right Hemisphere (Radiological Left)"
        hemi_short = "Right"
    elif cx_norm > 0.58:
        hemi_anatomical = "Left Hemisphere (Radiological Right)"
        hemi_short = "Left"
    else:
        hemi_anatomical = "Midline / Central"
        hemi_short = "Midline"

    # Anterior-Posterior / Superior-Inferior axis
    if cy_norm < 0.38:
        axis_region = "Anterior / Frontal lobe"
        axis_short = "Anterior-Superior"
    elif cy_norm > 0.65:
        axis_region = "Posterior / Occipital-Cerebellar region"
        axis_short = "Posterior-Inferior"
    else:
        axis_region = "Mid-axial / Temporal-Parietal region"
        axis_short = "Mid-axial"

    # Distance from image center (radius)
    dist_from_center = np.sqrt((cx_norm - 0.5) ** 2 + (cy_norm - 0.5) ** 2)

    # Heuristic 1: Sellar / Pituitary Fossa localization
    # Pituitary adenomas typically occupy the sellar/suprasellar region at the lower midline skull base
    is_sellar = (0.40 <= cx_norm <= 0.60) and (0.45 <= cy_norm <= 0.72) and (dist_from_center < 0.22)

    # Heuristic 2: Falcine / Parasagittal (Meningioma predilection site)
    is_falcine = (0.45 <= cx_norm <= 0.55) and (cy_norm < 0.35 or cy_norm > 0.70)

    # Heuristic 3: Convexity / Extra-axial Dural Margin (Peripheral edge near skull)
    is_convexity = dist_from_center > 0.30

    if is_sellar:
        anatomical_desc = "Sellar / Suprasellar region (Pituitary fossa)"
        quadrant_desc = "Central Skull Base (Sellar)"
    elif is_falcine:
        anatomical_desc = f"Midline Parasagittal / Falcine dural region ({axis_short})"
        quadrant_desc = f"Midline Falx ({axis_short})"
    elif is_convexity:
        anatomical_desc = f"{hemi_short} Convexity / Peripheral dural margin ({axis_region})"
        quadrant_desc = f"{hemi_short} {axis_short} Convexity"
    else:
        anatomical_desc = f"Deep parenchymal {hemi_anatomical} ({axis_region})"
        quadrant_desc = f"{hemi_short} {axis_short} Parenchyma"

    return anatomical_desc, quadrant_desc


# =========================================================================
# Test-Time Augmentation (TTA) Utilities
# =========================================================================

def _tta_predict(
    model: nn.Module,
    base_tensor: "torch.Tensor",
    n_augments: int = 8,
) -> Tuple[int, float, Dict[str, float]]:
    """
    Runs inference on N augmented views of the input tensor and returns
    averaged (ensemble) softmax probabilities.

    Augmentation views applied (all within the normalized tensor space):
      - Original
      - Horizontal flip (left-right mirror)
      - Rotations: 90°, 180°, 270°
      - Horizontal flip of 90°, 180°, 270° rotations
      - Optional: slight gamma adjustment via clamping

    Using more views reduces prediction variance on borderline / low-contrast
    images without retraining the model.

    Args:
        model: Trained NeuroBrainClassifier in eval mode.
        base_tensor: Preprocessed input tensor of shape (1, 3, H, W).
        n_augments: Number of augmented views to generate (max 8).

    Returns:
        pred_idx: Integer index of the averaged top prediction.
        confidence: Confidence of the top-averaged class.
        all_probs: Dict mapping class names to averaged probabilities.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for TTA.")

    device = next(model.parameters()).device
    tensor = base_tensor.to(device)  # (1, 3, H, W)

    # Build augmented variants using torch operations
    # k=0 → 0°, k=1 → 90°, k=2 → 180°, k=3 → 270°
    views = []
    for k in range(4):
        rotated = torch.rot90(tensor, k=k, dims=[2, 3])
        views.append(rotated)
        if len(views) < n_augments:
            views.append(torch.flip(rotated, dims=[3]))  # horizontal flip
        if len(views) >= n_augments:
            break

    views = views[:n_augments]

    model.eval()
    accumulated_probs = np.zeros(NUM_CLASSES, dtype=np.float64)

    with torch.no_grad():
        for view in views:
            logits = model(view)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy().astype(np.float64)
            accumulated_probs += probs

    averaged_probs = accumulated_probs / len(views)
    averaged_probs /= averaged_probs.sum()  # re-normalize for floating point safety

    pred_idx = int(np.argmax(averaged_probs))
    confidence = float(averaged_probs[pred_idx])

    all_probs: Dict[str, float] = {
        IDX_TO_CLASS.get(i, f"class_{i}"): float(averaged_probs[i])
        for i in range(NUM_CLASSES)
    }

    return pred_idx, confidence, all_probs


# =========================================================================
# High-Level Explainability Orchestrator
# =========================================================================

def explain_scan(
    image_input: Union[str, np.ndarray, Image.Image, bytes],
    model: nn.Module,
    preprocessor: Optional[BrainMRIPreprocessor] = None,
    target_class: Optional[int] = None,
    target_layer: Optional[nn.Module] = None,
    overlay_alpha: float = 0.55,
    threshold_ratio: float = 0.80,
    tta_views: int = 8,
) -> Dict[str, Any]:
    """
    High-level, end-to-end explainability pipeline for a single brain MRI scan.

    Executes:
    1. Preprocessing (skull stripping, standardisation, CLAHE).
    2. Test-Time Augmentation (TTA) ensemble inference over N augmented views.
    3. Grad-CAM backpropagation on the TTA-consensus predicted class.
    4. Region of Interest (ROI) extraction and neuroanatomical quadrant detection.
    5. Multi-colormap overlay rendering (turbo, jet, viridis).
    6. Base64 serialization for immediate frontend consumption.

    Args:
        image_input: File path, numpy array, PIL Image, or raw bytes.
        model: Trained NeuroBrainClassifier.
        preprocessor: BrainMRIPreprocessor instance (created if None).
        target_class: Specific class index to explain (None = TTA-consensus class).
        target_layer: Convolutional layer for Grad-CAM (None = model.get_target_layer()).
        overlay_alpha: Heatmap blend opacity weight [0.0, 1.0].
        threshold_ratio: Cutoff ratio for high-activation mask (default 0.80).
        tta_views: Number of TTA augmented inference views (default 8).

    Returns:
        Structured dictionary matching the Phase 3 & Phase 6 API specification.
    """
    if preprocessor is None:
        preprocessor = BrainMRIPreprocessor()

    # 1. Preprocess input
    intermediates = preprocessor.preprocess_image(
        image_input,
        crop_skull=True,
        use_clahe=True,
        return_intermediates=True,
    )
    tensor = preprocessor.to_tensor(image_input, add_batch_dim=True)

    # 2. TTA ensemble: average probabilities over N augmented views for stability
    tta_pred_idx, tta_confidence, tta_all_probs = _tta_predict(
        model=model,
        base_tensor=tensor,
        n_augments=tta_views,
    )

    # If caller specifies a target class, honour it; otherwise use TTA consensus
    explain_class = target_class if target_class is not None else tta_pred_idx

    # 3. Grad-CAM on TTA-consensus class for interpretability
    with GradCAM(model=model, target_layer=target_layer) as cam_engine:
        cam_heatmap, _gcam_pred_idx, _gcam_conf, _gcam_probs = cam_engine.generate_heatmap(
            input_tensor=tensor,
            target_class=explain_class,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
        )

    # Use TTA-averaged values as the authoritative output
    pred_idx = tta_pred_idx
    confidence = tta_confidence
    all_probs = tta_all_probs

    pred_class_name = IDX_TO_CLASS.get(pred_idx, CLASS_NAMES[pred_idx])

    # 3. ROI Extraction on the preprocessed 224x224 grid
    roi = extract_roi(
        heatmap=cam_heatmap,
        threshold_ratio=threshold_ratio,
        original_shape=intermediates["raw_rgb"].shape[:2],
    )

    # 4. Generate Multi-Colormap Overlays
    clahe_rgb = intermediates["preprocessed_uint8_rgb"]
    overlays_base64: Dict[str, str] = {}
    for colormap in ["turbo", "jet", "viridis"]:
        blended = GradCAM.blend_overlay(
            base_image=clahe_rgb,
            heatmap=cam_heatmap,
            alpha=overlay_alpha,
            colormap_name=colormap,
        )
        overlays_base64[colormap] = GradCAM.to_base64_png(blended)

    # Standalone heatmap in Turbo
    colored_heatmap = GradCAM.apply_colormap(cam_heatmap, colormap_name="turbo")
    heatmap_base64 = GradCAM.to_base64_png(colored_heatmap)

    # Base64 encodings of raw and preprocessed images for dual viewer
    raw_base64 = GradCAM.to_base64_png(intermediates["raw_rgb"])
    preprocessed_base64 = GradCAM.to_base64_png(clahe_rgb)

    return {
        "predicted_class": pred_class_name,
        "predicted_idx": pred_idx,
        "confidence": float(round(confidence, 4)),
        "probabilities": {k: round(v, 4) for k, v in all_probs.items()},
        "heatmap": cam_heatmap,  # 2D float32 array
        "roi": roi,
        "overlays_base64": overlays_base64,
        "heatmap_base64": heatmap_base64,
        "raw_image_base64": raw_base64,
        "preprocessed_image_base64": preprocessed_base64,
        "crop_bbox": list(intermediates["crop_bbox"]),
    }

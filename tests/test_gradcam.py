"""
Unit & Integration Tests for NeuroGuide AI Grad-CAM & ROI Localization Pipeline (Phase 3)
"""

import base64
import io
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pytest
import torch

from backend.app.config import CLASS_NAMES, NUM_CLASSES
from backend.app.core.gradcam import (
    COLORMAP_REGISTRY,
    GradCAM,
    determine_anatomical_location,
    explain_scan,
    extract_roi,
)
from backend.app.core.model import NeuroBrainClassifier, build_model
from backend.app.core.preprocessor import BrainMRIPreprocessor


@pytest.fixture(scope="module")
def cpu_model():
    """Provides an initialized model on CPU for testing Grad-CAM hooks."""
    model = build_model(backbone_name="efficientnet_b0", num_classes=NUM_CLASSES, pretrained=False)
    model.eval()
    return model


@pytest.fixture
def sample_tensor():
    """Creates a synthetic standard preprocessed tensor (1, 3, 224, 224)."""
    torch.manual_seed(42)
    # Simulate a brain-like circular structure
    img = np.zeros((224, 224, 3), dtype=np.float32)
    cv2.circle(img, (112, 112), 80, (0.7, 0.7, 0.7), -1)
    # Add a bright simulated lesion in upper-right quadrant
    cv2.circle(img, (140, 80), 20, (1.2, 1.2, 1.2), -1)
    tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0)
    return tensor


@pytest.fixture
def sample_synthetic_mri():
    """Creates a realistic synthetic 224x224 grayscale uint8 brain MRI scan."""
    img = np.zeros((224, 224), dtype=np.uint8)
    # Skull ellipse
    cv2.ellipse(img, (112, 112), (75, 95), 0, 0, 360, 180, -1)
    # Parenchyma texture
    cv2.ellipse(img, (112, 112), (65, 85), 0, 0, 360, 120, -1)
    # Lesion
    cv2.circle(img, (135, 90), 18, 240, -1)
    return img


class TestGradCAMCore:
    """Verifies Grad-CAM forward/backward hooks, gradient capture, and heatmap math."""

    def test_hook_registration_and_cleanup(self, cpu_model):
        """Verifies hooks are attached on init and cleanly removed on exit."""
        with GradCAM(cpu_model) as cam:
            assert len(cam._hooks) == 2
            assert cam.target_layer is not None
        # Verify removal
        assert len(cam._hooks) == 0
        assert cam.activations is None
        assert cam.gradients is None

    def test_gradients_non_zero_and_activation_captured(self, cpu_model, sample_tensor):
        """Verifies that backpropagation computes non-zero gradients on target layer."""
        with GradCAM(cpu_model) as cam:
            cam_map, pred_idx, conf, probs = cam.generate_heatmap(
                input_tensor=sample_tensor,
                target_class=None,
            )

            assert cam.activations is not None
            assert cam.gradients is not None
            assert cam.gradients.shape == cam.activations.shape
            assert torch.any(cam.gradients != 0).item(), "Gradients should be non-zero for target layer."

    def test_heatmap_values_normalized_to_0_1(self, cpu_model, sample_tensor):
        """Verifies Grad-CAM output is strictly bounded in [0.0, 1.0]."""
        with GradCAM(cpu_model) as cam:
            cam_map, pred_idx, conf, probs = cam.generate_heatmap(sample_tensor)

            assert isinstance(cam_map, np.ndarray)
            assert cam_map.shape == (224, 224)
            assert cam_map.dtype == np.float32
            assert cam_map.min() >= 0.0
            assert cam_map.max() <= 1.0
            assert 0 <= pred_idx < NUM_CLASSES
            assert 0.0 <= conf <= 1.0
            assert len(probs) == NUM_CLASSES

    def test_specific_target_class_gradcam(self, cpu_model, sample_tensor):
        """Verifies generating Grad-CAM for a specific requested class index."""
        with GradCAM(cpu_model) as cam:
            for target_c in range(NUM_CLASSES):
                cam_map, pred_idx, _, _ = cam.generate_heatmap(sample_tensor, target_class=target_c)
                assert cam_map.shape == (224, 224)
                assert cam_map.min() >= 0.0
                assert cam_map.max() <= 1.0

    def test_all_classes_heatmaps_generation(self, cpu_model, sample_tensor):
        """Verifies generating comparative heatmaps across all 4 classes."""
        with GradCAM(cpu_model) as cam:
            all_cams = cam.generate_all_classes_heatmaps(sample_tensor)
            assert set(all_cams.keys()) == set(CLASS_NAMES)
            for cname, heatmap in all_cams.items():
                assert heatmap.shape == (224, 224)
                assert heatmap.min() >= 0.0
                assert heatmap.max() <= 1.0

    def test_counterfactual_heatmap(self, cpu_model, sample_tensor):
        """Verifies negative/counterfactual activation calculation."""
        with GradCAM(cpu_model) as cam:
            cf_cam, _, _, _ = cam.generate_heatmap(sample_tensor, counterfactual=True)
            assert cf_cam.shape == (224, 224)
            assert cf_cam.min() >= 0.0
            assert cf_cam.max() <= 1.0


class TestColormapAndBlending:
    """Verifies OpenCV colormapping, alpha blending, and Base64 serialization."""

    def test_colormaps_supported(self):
        """Ensures all documented colormaps generate valid RGB uint8 images."""
        dummy_cam = np.linspace(0, 1, 224 * 224, dtype=np.float32).reshape(224, 224)

        for cmap_name in ["turbo", "jet", "viridis", "inferno", "magma"]:
            colored = GradCAM.apply_colormap(dummy_cam, colormap_name=cmap_name)
            assert isinstance(colored, np.ndarray)
            assert colored.shape == (224, 224, 3)
            assert colored.dtype == np.uint8

    def test_blend_overlay_math(self):
        """Verifies alpha blending produces exact weighted combination."""
        base = np.full((100, 100, 3), 100, dtype=np.uint8)
        heatmap = np.ones((100, 100), dtype=np.float32)

        # At alpha = 0.0, result equals base image
        blended_0 = GradCAM.blend_overlay(base, heatmap, alpha=0.0, colormap_name="turbo")
        np.testing.assert_array_equal(blended_0, base)

        # At alpha = 1.0, result equals colored heatmap
        colored = GradCAM.apply_colormap(heatmap, colormap_name="turbo")
        blended_1 = GradCAM.blend_overlay(base, heatmap, alpha=1.0, colormap_name="turbo")
        np.testing.assert_array_equal(blended_1, colored)

        # At alpha = 0.5, result is midpoint
        blended_half = GradCAM.blend_overlay(base, heatmap, alpha=0.5, colormap_name="turbo")
        assert blended_half.shape == (100, 100, 3)
        assert blended_half.dtype == np.uint8

    def test_to_base64_png(self):
        """Verifies base64 serialization decodes into a valid PNG image."""
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        b64_uri = GradCAM.to_base64_png(img, include_prefix=True)

        assert b64_uri.startswith("data:image/png;base64,")
        raw_b64 = b64_uri.replace("data:image/png;base64,", "")
        decoded_bytes = base64.b64decode(raw_b64)

        pil_img = Image.open(io.BytesIO(decoded_bytes))
        assert pil_img.size == (64, 64)
        assert pil_img.format == "PNG"


class TestROILocalization:
    """Verifies Region of Interest (ROI) extraction and anatomical classification heuristics."""

    def test_roi_extraction_focal_lesion(self):
        """Tests bounding box, centroid, peak coordinates, and area fraction on a focal peak."""
        heatmap = np.zeros((224, 224), dtype=np.float32)
        # Focal Gaussian peak centered at (150, 70)
        y, x = np.ogrid[:224, :224]
        heatmap = np.exp(-((x - 150) ** 2 + (y - 70) ** 2) / (2 * 15 ** 2)).astype(np.float32)

        roi = extract_roi(heatmap, threshold_ratio=0.80)

        assert roi["has_focal_activation"] is True
        assert roi["peak_coords"] == [150, 70]
        # Centroid should be within 2 pixels of peak
        assert abs(roi["centroid"][0] - 150) <= 2
        assert abs(roi["centroid"][1] - 70) <= 2

        # Bounding box should enclose the peak
        x_min, y_min, x_max, y_max = roi["bbox"]
        assert x_min < 150 < x_max
        assert y_min < 70 < y_max
        assert 0.0 <= roi["bbox_norm"][0] <= 1.0
        assert 0.0 < roi["area_fraction"] < 0.20
        assert len(roi["contour_polygon"]) >= 3

    def test_roi_flat_or_zero_heatmap(self):
        """Tests that a flat zero heatmap safely returns the non-focal baseline response."""
        flat_map = np.zeros((224, 224), dtype=np.float32)
        roi = extract_roi(flat_map)

        assert roi["has_focal_activation"] is False
        assert "Normal" in roi["anatomical_region"] or "Diffuse" in roi["anatomical_region"]
        assert roi["area_fraction"] == 0.0

    def test_anatomical_sellar_pituitary_location(self):
        """Verifies that central skull base activations classify as Sellar / Pituitary fossa."""
        # cx = 0.50 (midline), cy = 0.60 (lower skull base)
        region, quad = determine_anatomical_location(
            cx_norm=0.50, cy_norm=0.60, bbox_norm=[0.42, 0.52, 0.58, 0.68], area_fraction=0.03
        )
        assert "Sellar" in region or "Pituitary" in region
        assert "Sellar" in quad

    def test_anatomical_radiological_hemisphere_convention(self):
        """Verifies radiological convention (left of image = Right patient hemisphere)."""
        # cx = 0.25 (left of scan) -> Right anatomical hemisphere
        region_right, quad_right = determine_anatomical_location(
            cx_norm=0.25, cy_norm=0.30, bbox_norm=[0.15, 0.20, 0.35, 0.40], area_fraction=0.04
        )
        assert "Right" in region_right
        assert "Right" in quad_right

        # cx = 0.75 (right of scan) -> Left anatomical hemisphere
        region_left, quad_left = determine_anatomical_location(
            cx_norm=0.75, cy_norm=0.30, bbox_norm=[0.65, 0.20, 0.85, 0.40], area_fraction=0.04
        )
        assert "Left" in region_left
        assert "Left" in quad_left


class TestEndToEndExplainScan:
    """Verifies the complete explain_scan orchestrator pipeline."""

    def test_explain_scan_synthetic(self, cpu_model, sample_synthetic_mri):
        """Verifies full explain_scan flow on a synthetic scan array."""
        result = explain_scan(
            image_input=sample_synthetic_mri,
            model=cpu_model,
            overlay_alpha=0.6,
            threshold_ratio=0.80,
        )

        assert "predicted_class" in result
        assert result["predicted_class"] in CLASS_NAMES
        assert 0.0 <= result["confidence"] <= 1.0
        assert isinstance(result["heatmap"], np.ndarray)
        assert result["heatmap"].shape == (224, 224)

        # ROI checks
        assert "roi" in result
        assert "bbox" in result["roi"]
        assert "anatomical_region" in result["roi"]

        # Base64 assets checks
        assert "turbo" in result["overlays_base64"]
        assert "jet" in result["overlays_base64"]
        assert "viridis" in result["overlays_base64"]
        assert result["overlays_base64"]["turbo"].startswith("data:image/png;base64,")
        assert result["heatmap_base64"].startswith("data:image/png;base64,")
        assert result["raw_image_base64"].startswith("data:image/png;base64,")
        assert result["preprocessed_image_base64"].startswith("data:image/png;base64,")

    def test_explain_scan_with_real_archive_image(self, cpu_model):
        """Verifies explain_scan with an authentic MRI file from the dataset."""
        all_jpgs = list(Path("archive/Training/glioma").glob("*.jpg"))
        if not all_jpgs:
            all_jpgs = list(Path("archive/Training").rglob("*.jpg"))
        if not all_jpgs:
            pytest.skip("No dataset files found in archive/Training.")
        sample_path = all_jpgs[0]

        from backend.app.config import MODELS_DIR
        from backend.app.core.model import load_checkpoint
        ckpt_file = MODELS_DIR / "model_best.pt"
        if ckpt_file.exists():
            eval_model, _ = load_checkpoint(ckpt_file)
        else:
            eval_model = cpu_model

        result = explain_scan(
            image_input=str(sample_path),
            model=eval_model,
        )

        assert result["predicted_class"] in CLASS_NAMES
        assert result["roi"]["has_focal_activation"] is True
        assert result["roi"]["anatomical_region"] is not None
        assert len(result["overlays_base64"]) >= 3


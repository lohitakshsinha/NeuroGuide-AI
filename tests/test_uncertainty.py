"""
Unit & Integration Tests for NeuroGuide AI Uncertainty & OOD Gatekeeper (Phase 4 / M3)
"""

import math
import numpy as np
import pytest

from backend.app.core.ood_detector import OODDetector
from backend.app.core.uncertainty import (
    CLINICAL_LIMITATIONS_CATALOG,
    UncertaintyEvaluator,
    calculate_probability_margin,
    calculate_shannon_entropy,
)


class TestUncertaintyMetrics:
    """Verifies Shannon entropy and probability margin mathematical properties."""

    def test_shannon_entropy_extreme_certainty(self):
        """Pure one-hot distribution [1.0, 0.0, 0.0, 0.0] must have entropy 0.0."""
        probs = [1.0, 0.0, 0.0, 0.0]
        h = calculate_shannon_entropy(probs)
        assert h == 0.0

    def test_shannon_entropy_extreme_uncertainty(self):
        """Uniform distribution [0.25, 0.25, 0.25, 0.25] must have maximum entropy log2(4) = 2.0."""
        probs = [0.25, 0.25, 0.25, 0.25]
        h = calculate_shannon_entropy(probs)
        assert abs(h - 2.0) < 1e-3

    def test_shannon_entropy_dict_input(self):
        """Accepts dictionary input correctly."""
        probs = {"glioma": 0.8, "meningioma": 0.1, "notumor": 0.05, "pituitary": 0.05}
        h = calculate_shannon_entropy(probs)
        assert 0.0 < h < 2.0

    def test_probability_margin_calculation(self):
        """Top-1 minus Top-2 difference."""
        probs = {"glioma": 0.85, "meningioma": 0.10, "pituitary": 0.03, "notumor": 0.02}
        delta, top1, top2 = calculate_probability_margin(probs)
        assert delta == 0.75
        assert top1 == ("glioma", 0.85)
        assert top2 == ("meningioma", 0.10)


class TestUncertaintyEvaluator:
    """Verifies confidence categorization (HIGH, MODERATE, AMBIGUOUS) and clinical caveats."""

    @pytest.fixture
    def evaluator(self):
        return UncertaintyEvaluator()

    def test_high_confidence_case(self, evaluator):
        """High confidence: Low entropy (< 0.60) and wide margin (> 0.40)."""
        probs = {"meningioma": 0.94, "glioma": 0.03, "pituitary": 0.02, "notumor": 0.01}
        report = evaluator.evaluate(probs)

        assert report["confidence_level"] == "HIGH"
        assert report["is_ambiguous"] is False
        assert report["entropy"] < 0.60
        assert report["margin_delta"] > 0.40
        assert report["top_prediction"]["class"] == "meningioma"
        assert len(report["clinical_limitations"]) == 5

    def test_ambiguous_case_narrow_margin(self, evaluator):
        """Ambiguous case: narrow margin between top two classes (e.g. glioma 48% vs meningioma 44%)."""
        probs = {"glioma": 0.48, "meningioma": 0.44, "notumor": 0.05, "pituitary": 0.03}
        report = evaluator.evaluate(probs)

        assert report["confidence_level"] == "AMBIGUOUS"
        assert report["is_ambiguous"] is True
        assert report["clinical_caveat"] is not None
        assert "overlapping" in report["clinical_caveat"].lower()
        assert "Glioma" in report["clinical_caveat"]
        assert "Meningioma" in report["clinical_caveat"]

    def test_moderate_confidence_case(self, evaluator):
        """Moderate confidence case (0.6 <= H <= 1.2 and Delta >= 0.15)."""
        probs = {"pituitary": 0.80, "glioma": 0.12, "notumor": 0.05, "meningioma": 0.03}
        report = evaluator.evaluate(probs)

        assert report["confidence_level"] == "MODERATE"
        assert report["is_ambiguous"] is False
        assert 0.60 <= report["entropy"] <= 1.20



class TestOODGatekeeper:
    """Verifies OOD input detection (color saturation, background contrast, contour)."""

    @pytest.fixture
    def ood(self):
        return OODDetector()

    def test_rejects_colorful_natural_image(self, ood):
        """Color landscape / photo should be rejected due to high color saturation."""
        # Highly saturated color image (e.g. green grass, blue sky)
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[:112, :] = [255, 100, 20]  # Blue/Orange
        img[112:, :] = [20, 220, 20]  # Green grass

        res = ood.validate_image(img)
        assert res["is_valid_mri"] is False
        assert "color saturation" in res["rejection_reason"].lower()

    def test_rejects_blank_white_image(self, ood):
        """Pure white image should be rejected due to lack of dark background and cranial contour."""
        white_img = np.full((224, 224, 3), 255, dtype=np.uint8)
        res = ood.validate_image(white_img)
        assert res["is_valid_mri"] is False

    def test_accepts_synthetic_mri(self, ood):
        """Monochromatic MRI with dark corners and central cranial ellipse should pass."""
        import cv2
        mri = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.ellipse(mri, (112, 112), (70, 90), 0, 0, 360, (160, 160, 160), -1)
        res = ood.validate_image(mri)
        assert res["is_valid_mri"] is True
        assert res["rejection_reason"] is None

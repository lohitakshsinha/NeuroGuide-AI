"""
NeuroGuide AI - Unit Tests for Evaluation & ONNX Export Pipeline
"""

import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import CLASS_NAMES, NUM_CLASSES
from backend.app.core.model import build_model, save_checkpoint
from training.evaluate import (
    generate_markdown_report,
    plot_confusion_matrices,
    plot_multiclass_roc_curves,
)
from training.export_onnx import export_to_onnx


def test_plot_confusion_matrices():
    """Verify that confusion matrices plot is generated properly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "test_cm.png"
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 2, 0, 1, 3, 3])

        cm_raw = plot_confusion_matrices(y_true, y_pred, CLASS_NAMES, out_path)

        assert out_path.exists()
        assert out_path.stat().st_size > 0
        assert cm_raw.shape == (NUM_CLASSES, NUM_CLASSES)
        assert cm_raw.sum() == len(y_true)


def test_plot_multiclass_roc_curves():
    """Verify that multi-class ROC curves and AUC scores are computed accurately."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "test_roc.png"
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        # Synthetic probabilities (8 samples, 4 classes)
        y_probs = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.7, 0.1, 0.1],
            [0.05, 0.05, 0.8, 0.1],
            [0.05, 0.05, 0.1, 0.8],
            [0.75, 0.1, 0.1, 0.05],
            [0.15, 0.65, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.05, 0.15, 0.7],
        ])

        auc_scores = plot_multiclass_roc_curves(y_true, y_probs, CLASS_NAMES, out_path)

        assert out_path.exists()
        assert "macro_avg" in auc_scores
        assert auc_scores["macro_avg"] > 0.8  # Strong synthetic signal
        for c in CLASS_NAMES:
            assert c in auc_scores


def test_generate_markdown_report():
    """Verify markdown evaluation report structure and generation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        report_path = Path(tmp_dir) / "test_report.md"
        metrics = {
            "total_samples": 100,
            "overall_accuracy": 0.94,
            "macro_f1": 0.938,
            "weighted_f1": 0.941,
            "macro_precision": 0.942,
            "macro_recall": 0.935,
            "backbone_name": "efficientnet_b0",
            "per_class": {
                c: {"precision": 0.94, "recall": 0.93, "f1-score": 0.935, "support": 25}
                for c in CLASS_NAMES
            },
        }
        cm_raw = np.diag([24, 23, 25, 22])
        auc_scores = {c: 0.98 for c in CLASS_NAMES}
        auc_scores["macro_avg"] = 0.98
        error_cases = [
            {
                "image_id": "img_001",
                "true_class": "glioma",
                "pred_class": "meningioma",
                "confidence": 0.82,
            }
        ]

        generate_markdown_report(metrics, cm_raw, auc_scores, error_cases, report_path)

        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "NeuroGuide AI - Model Evaluation & Performance Benchmark Report" in content
        assert "94.00%" in content
        assert "Glioma" in content
        assert "img_001" in content


def test_export_to_onnx():
    """Verify that export_to_onnx produces a valid ONNX model file passing validation."""
    model = build_model(backbone_name="efficientnet_b0", num_classes=4, pretrained=False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "model_test.pt"
        onnx_path = Path(tmp_dir) / "model_test.onnx"

        save_checkpoint(model=model, file_path=ckpt_path)

        out = export_to_onnx(
            model_path=ckpt_path,
            output_path=onnx_path,
            opset_version=17,
            include_softmax=True,
            verify_parity=True,
        )

        assert out.exists()
        assert out.stat().st_size > 1_000_000  # ONNX file should be > 1MB

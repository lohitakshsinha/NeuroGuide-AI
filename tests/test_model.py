"""
NeuroGuide AI - Unit Tests for Deep Learning Model Architecture
"""

import os
import sys
import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import NUM_CLASSES, CLASS_NAMES
from backend.app.core.model import (
    NeuroBrainClassifier,
    build_model,
    load_checkpoint,
    save_checkpoint,
)


def test_model_build_efficientnet():
    """Verify that build_model builds an EfficientNet-B0 classifier with the expected head."""
    model = build_model(backbone_name="efficientnet_b0", num_classes=4, pretrained=False)
    assert isinstance(model, NeuroBrainClassifier)
    assert model.backbone_name == "efficientnet_b0"
    assert model.num_classes == 4
    assert isinstance(model.classifier, nn.Sequential)

    # Check classifier structure
    assert len(model.classifier) == 6
    assert isinstance(model.classifier[0], nn.Dropout)
    assert isinstance(model.classifier[1], nn.Linear)
    assert isinstance(model.classifier[2], nn.BatchNorm1d)
    assert isinstance(model.classifier[3], nn.SiLU)
    assert isinstance(model.classifier[4], nn.Dropout)
    assert isinstance(model.classifier[5], nn.Linear)
    assert model.classifier[5].out_features == 4


def test_model_forward_pass_shape():
    """Verify that forward pass produces logits of shape (B, num_classes)."""
    model = build_model(backbone_name="efficientnet_b0", num_classes=4, pretrained=False)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)

    assert logits.shape == (2, 4)
    assert not torch.isnan(logits).any()


def test_predict_scan_output_format():
    """Verify that predict_scan outputs correct dictionary structure and probabilities."""
    model = build_model(backbone_name="efficientnet_b0", num_classes=4, pretrained=False)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    res = model.predict_scan(dummy_input)

    assert "prediction" in res
    assert "class_idx" in res
    assert "confidence" in res
    assert "probabilities" in res
    assert res["prediction"] in CLASS_NAMES
    assert 0 <= res["class_idx"] < 4
    assert 0.0 <= res["confidence"] <= 1.0

    # Probabilities must sum to ~1.0
    prob_sum = sum(res["probabilities"].values())
    assert abs(prob_sum - 1.0) < 1e-4


def test_freeze_backbone():
    """Verify that freeze_backbone freezes all backbone layers but keeps head trainable."""
    model = build_model(backbone_name="efficientnet_b0", num_classes=4, pretrained=False)
    model.freeze_backbone()

    # Backbone parameters should have requires_grad = False
    for param in model.backbone.parameters():
        assert not param.requires_grad

    # Classifier parameters should have requires_grad = True
    for param in model.classifier.parameters():
        assert param.requires_grad


def test_unfreeze_stages():
    """Verify that unfreeze_stages unfreezes the requested top stages."""
    model = build_model(backbone_name="efficientnet_b0", num_classes=4, pretrained=False)
    model.freeze_backbone()

    model.unfreeze_stages(stages_to_unfreeze=2)

    # Last 2 blocks of features should have requires_grad = True
    last_block = model.backbone.features[-1]
    for param in last_block.parameters():
        assert param.requires_grad

    # First block of features should still be frozen
    first_block = model.backbone.features[0]
    for param in first_block.parameters():
        assert not param.requires_grad


def test_gradcam_target_layer():
    """Verify that get_target_layer returns a valid convolutional module."""
    model = build_model(backbone_name="efficientnet_b0", num_classes=4, pretrained=False)
    target_layer = model.get_target_layer()

    assert target_layer is not None
    # For EfficientNet, the last features block is Conv2dNormActivation
    assert isinstance(target_layer, nn.Module)


def test_save_and_load_checkpoint():
    """Verify roundtrip saving and loading of model weights and metadata."""
    model = build_model(backbone_name="efficientnet_b0", num_classes=4, pretrained=False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_model.pt"
        test_metrics = {"val_f1": 0.94, "val_acc": 0.95}

        saved_path = save_checkpoint(
            model=model,
            epoch=5,
            metrics=test_metrics,
            file_path=ckpt_path,
        )
        assert saved_path.exists()

        loaded_model, ckpt = load_checkpoint(file_path=ckpt_path)
        assert loaded_model.backbone_name == "efficientnet_b0"
        assert ckpt["epoch"] == 5
        assert ckpt["metrics"]["val_f1"] == 0.94


def test_single_training_step():
    """Verify that a single backprop step updates model weights properly."""
    model = build_model(backbone_name="efficientnet_b0", num_classes=4, pretrained=False)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    inputs = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([0, 2], dtype=torch.long)

    # Initial loss
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    assert loss.item() > 0.0

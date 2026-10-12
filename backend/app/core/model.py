"""
NeuroGuide AI - Deep Learning Model Architecture & Inference Core

Implements:
1. Transfer-learning convolutional backbones (EfficientNet-B0 primary, ResNet-50 alternative).
2. Custom medical classification head with Dropout, BatchNorm, and SiLU activations.
3. Grad-CAM target layer resolution for interpretability.
4. Two-stage fine-tuning hooks (backbone freezing & partial unfreezing).
5. Safe checkpoint saving, loading, and inference serialization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as models
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False
    nn = object

from backend.app.config import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    MODELS_DIR,
    NUM_CLASSES,
)

logger = logging.getLogger(__name__)


class NeuroBrainClassifier(nn.Module if TORCH_AVAILABLE else object):
    """
    Production-grade Brain MRI Classifier utilizing pre-trained CNN backbones
    with a specialized classification head designed for medical imaging.
    """

    SUPPORTED_BACKBONES = {
        "efficientnet_b0": "EfficientNet-B0 (Fast, lightweight, ~5.3M parameters)",
        "resnet50": "ResNet-50 (Deep residual network, ~25.6M parameters)",
    }

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout_p1: float = 0.3,
        dropout_p2: float = 0.2,
        hidden_dim: int = 256,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to instantiate NeuroBrainClassifier.")

        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes
        self.dropout_p1 = dropout_p1
        self.dropout_p2 = dropout_p2
        self.hidden_dim = hidden_dim

        if self.backbone_name not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{backbone_name}'. "
                f"Supported options: {list(self.SUPPORTED_BACKBONES.keys())}"
            )

        # 1. Initialize backbone
        if self.backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            # Remove original classifier
            self.backbone.classifier = nn.Identity()

        elif self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            # Remove original FC layer
            self.backbone.fc = nn.Identity()

        self.in_features = in_features

        # 2. Build custom medical classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_p1),
            nn.Linear(in_features, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=self.dropout_p2),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

        logger.info(
            f"Initialized NeuroBrainClassifier (backbone={self.backbone_name}, "
            f"in_features={self.in_features}, num_classes={self.num_classes})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing raw logits (batch_size, num_classes).
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract penultimate feature representations before the classification head.
        """
        return self.backbone(x)

    def get_target_layer(self) -> nn.Module:
        """
        Returns the primary target convolutional layer for Grad-CAM activation mapping.
        """
        if self.backbone_name == "efficientnet_b0":
            # Last convolutional block in features sequential container
            return self.backbone.features[-1]
        elif self.backbone_name == "resnet50":
            # Last bottleneck block of layer4
            return self.backbone.layer4[-1]
        else:
            raise NotImplementedError(f"Target layer not defined for {self.backbone_name}")

    def freeze_backbone(self) -> None:
        """
        Freezes all backbone feature extractor weights for Stage 1 warmup training.
        Only the classification head parameters retain requires_grad = True.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.classifier.parameters():
            param.requires_grad = True

        logger.info("Backbone frozen. Only classification head is trainable.")

    def unfreeze_stages(self, stages_to_unfreeze: int = 2) -> None:
        """
        Unfreezes the top convolutional stages of the backbone for Stage 2 fine-tuning.
        """
        if self.backbone_name == "efficientnet_b0":
            # EfficientNet features has 9 blocks (0 through 8)
            total_blocks = len(self.backbone.features)
            cutoff = max(0, total_blocks - stages_to_unfreeze)

            for i, child in enumerate(self.backbone.features):
                requires_grad = (i >= cutoff)
                for param in child.parameters():
                    param.requires_grad = requires_grad

        elif self.backbone_name == "resnet50":
            # ResNet residual layers: layer1, layer2, layer3, layer4
            layers = [
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4,
            ]
            cutoff = max(0, len(layers) - stages_to_unfreeze)
            for i, layer in enumerate(layers):
                requires_grad = (i >= cutoff)
                for param in layer.parameters():
                    param.requires_grad = requires_grad

        # Ensure classifier is always trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"Unfroze top {stages_to_unfreeze} stages. "
            f"Trainable parameters: {trainable_params:,} / {total_params:,} "
            f"({(trainable_params / total_params) * 100:.1f}%)"
        )

    def unfreeze_all(self) -> None:
        """
        Unfreezes all parameters across the entire model.
        """
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Entire model unfrozen.")

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes calibrated softmax probabilities: (batch_size, num_classes).
        """
        self.eval()
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    @torch.no_grad()
    def predict_scan(
        self,
        x: torch.Tensor,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        High-level inference helper for a single scan tensor (1, 3, H, W).
        Returns a dictionary with prediction metadata.
        """
        self.eval()
        names = class_names or CLASS_NAMES

        if x.dim() == 3:
            x = x.unsqueeze(0)

        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_class = names[pred_idx]
        confidence = float(probs[pred_idx])

        probabilities_dict = {
            names[i]: float(probs[i]) for i in range(len(names))
        }

        # Sort probabilities descending
        sorted_probs = sorted(
            probabilities_dict.items(), key=lambda item: item[1], reverse=True
        )

        return {
            "prediction": pred_class,
            "class_idx": pred_idx,
            "confidence": confidence,
            "probabilities": probabilities_dict,
            "sorted_probabilities": sorted_probs,
            "logits": logits[0].cpu().numpy().tolist(),
        }


def build_model(
    backbone_name: str = "efficientnet_b0",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> NeuroBrainClassifier:
    """
    Factory function to construct a NeuroBrainClassifier.
    """
    return NeuroBrainClassifier(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_p1=dropout,
        dropout_p2=max(0.1, dropout * 0.67),
    )


def save_checkpoint(
    model: NeuroBrainClassifier,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    file_path: Union[str, Path] = MODELS_DIR / "model_best.pt",
) -> Path:
    """
    Serializes model weights and training metadata.
    """
    save_path = Path(file_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "backbone_name": model.backbone_name,
        "num_classes": model.num_classes,
        "class_names": CLASS_NAMES,
        "model_state_dict": model.state_dict(),
        "metrics": metrics or {},
    }

    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(state, save_path)
    logger.info(f"Checkpoint successfully saved to {save_path}")
    return save_path


def load_checkpoint(
    file_path: Union[str, Path] = MODELS_DIR / "model_best.pt",
    device: Union[str, torch.device] = "cpu",
    model: Optional[NeuroBrainClassifier] = None,
) -> Tuple[NeuroBrainClassifier, Dict[str, Any]]:
    """
    Deserializes model weights and metadata from disk.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    backbone_name = checkpoint.get("backbone_name", "efficientnet_b0")
    num_classes = checkpoint.get("num_classes", NUM_CLASSES)

    if model is None:
        model = build_model(
            backbone_name=backbone_name,
            num_classes=num_classes,
            pretrained=False,
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Loaded checkpoint from {path} onto device={device}")
    return model, checkpoint

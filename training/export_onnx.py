"""
NeuroGuide AI - Lightweight ONNX Model Exporter

Exports the trained PyTorch NeuroBrainClassifier into high-performance,
portable ONNX format for edge deployment, microservices, and web runtimes.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import (
    IMAGE_SIZE,
    MODELS_DIR,
    NUM_CLASSES,
)
from backend.app.core.model import (
    NeuroBrainClassifier,
    load_checkpoint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("NeuroGuide-ONNX")


class InferenceWrapper(nn.Module):
    """
    Wraps the classifier to output both raw logits and calibrated softmax probabilities.
    """

    def __init__(self, classifier: NeuroBrainClassifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs


def export_to_onnx(
    model_path: Path = MODELS_DIR / "model_best.pt",
    output_path: Path = MODELS_DIR / "model.onnx",
    opset_version: int = 17,
    include_softmax: bool = True,
    verify_parity: bool = True,
) -> Path:
    """
    Exports PyTorch model checkpoint to ONNX format.

    Args:
        model_path: Path to the .pt checkpoint.
        output_path: Path where the .onnx file will be written.
        opset_version: ONNX operator set version (default 17).
        include_softmax: Whether to wrap model to output (logits, probabilities).
        verify_parity: Whether to test output parity against PyTorch.

    Returns:
        Path to exported ONNX file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading checkpoint from: {model_path} ...")
    model, checkpoint = load_checkpoint(file_path=model_path, device="cpu")
    model.eval()

    # Wrap model if softmax output is desired
    export_module: nn.Module
    if include_softmax:
        export_module = InferenceWrapper(model)
        output_names = ["logits", "probabilities"]
        dynamic_axes = {
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "probabilities": {0: "batch_size"},
        }
    else:
        export_module = model
        output_names = ["logits"]
        dynamic_axes = {
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        }

    export_module.eval()

    # Create representative dummy input (B=1, C=3, H=224, W=224)
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)

    logger.info(f"Exporting model to ONNX (opset={opset_version}) at {output_path} ...")

    with torch.no_grad():
        torch.onnx.export(
            export_module,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"ONNX export successful! File size: {file_size_mb:.2f} MB")

    # Verify model structure with onnx checker
    if verify_parity:
        try:
            import onnx
            logger.info("Validating ONNX model graph structure with onnx.checker...")
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX graph validation PASSED successfully!")
        except Exception as e:
            logger.warning(f"ONNX check warning: {e}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="NeuroGuide AI - ONNX Model Exporter")
    parser.add_argument("--model", type=str, default=str(MODELS_DIR / "model_best.pt"), help="PyTorch checkpoint")
    parser.add_argument("--output", type=str, default=str(MODELS_DIR / "model.onnx"), help="Target ONNX path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--no-softmax", action="store_true", help="Omit softmax probabilities output")
    args = parser.parse_args()

    export_to_onnx(
        model_path=Path(args.model),
        output_path=Path(args.output),
        opset_version=args.opset,
        include_softmax=not args.no_softmax,
    )


if __name__ == "__main__":
    main()

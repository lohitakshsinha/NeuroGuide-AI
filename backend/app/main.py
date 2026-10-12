"""
NeuroGuide AI - FastAPI Application Server Entrypoint
Integrates deep learning inference, Grad-CAM visual explainability,
Shannon uncertainty quantification, and medical limitations catalog.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.predict import router as predict_router
from backend.app.api.samples import router as samples_router
from backend.app.config import (
    CLASS_NAMES,
    MODELS_DIR,
    PROJECT_ROOT,
    SAMPLES_DIR,
)
from backend.app.core.model import build_model, load_checkpoint
from backend.app.core.ood_detector import OODDetector
from backend.app.core.preprocessor import BrainMRIPreprocessor
from backend.app.core.uncertainty import UncertaintyEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("neuroguide.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle event handler to load PyTorch model weights and
    core analysis utilities on application startup.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initializing NeuroGuide AI inference engine on device: {device}")

    # 1. Initialize preprocessing, OOD detector, and uncertainty evaluator
    app.state.device = str(device)
    app.state.preprocessor = BrainMRIPreprocessor()
    app.state.ood_detector = OODDetector()
    app.state.uncertainty_evaluator = UncertaintyEvaluator()

    # 2. Load PyTorch model weights
    best_weights_path = MODELS_DIR / "model_best.pt"
    last_weights_path = MODELS_DIR / "model_last.pt"

    model = None
    if best_weights_path.exists():
        try:
            model, _ = load_checkpoint(file_path=best_weights_path, device=device)
            logger.info(f"Successfully loaded best checkpoint from {best_weights_path}")
        except Exception as exc:
            logger.warning(f"Failed to load best checkpoint: {exc}. Trying fallback...")

    if model is None and last_weights_path.exists():
        try:
            model, _ = load_checkpoint(file_path=last_weights_path, device=device)
            logger.info(f"Successfully loaded fallback checkpoint from {last_weights_path}")
        except Exception as exc:
            logger.error(f"Failed to load fallback checkpoint: {exc}")

    if model is None:
        logger.warning("No serialized checkpoint found; initializing pretrained EfficientNet-B0 backbone.")
        model = build_model(backbone_name="efficientnet_b0", num_classes=len(CLASS_NAMES), pretrained=True)
        model.to(device)
        model.eval()

    app.state.model = model
    app.state.model_loaded = True
    app.state.backbone = getattr(model, "backbone_name", "efficientnet_b0")

    yield

    logger.info("NeuroGuide AI application shutting down.")


# Create FastAPI application instance
app = FastAPI(
    title="NeuroGuide AI - Brain MRI Tumor Classification & Explainability Suite",
    description=(
        "Production-ready medical-educational AI system that bridges deep learning predictions, "
        "Grad-CAM visual explainability, Shannon uncertainty quantification, and clinical limitations."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for interactive development and API consumers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(predict_router, prefix="/api")
app.include_router(samples_router, prefix="/api")


@app.get("/health", tags=["Health & Status"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for service readiness and monitoring.
    """
    sample_files = list(SAMPLES_DIR.glob("*.jpg"))
    return {
        "status": "ok",
        "device": getattr(app.state, "device", "unknown"),
        "model_loaded": getattr(app.state, "model_loaded", False),
        "backbone": getattr(app.state, "backbone", "unknown"),
        "classes": CLASS_NAMES,
        "samples_available": len(sample_files),
        "version": "1.0.0",
    }


# Frontend static files mounting
FRONTEND_DIR = PROJECT_ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    logger.info(f"Frontend directory not yet found at {FRONTEND_DIR}; will serve API only.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)

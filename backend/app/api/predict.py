"""
NeuroGuide AI - Prediction & Diagnostic Explainability API
Provides endpoints for brain MRI tumor classification, Grad-CAM heatmap generation,
Shannon uncertainty quantification, and clinical limitation framing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status

from backend.app.config import (
    CLASS_METADATA,
    SAMPLES_DIR,
)
from backend.app.core.gradcam import COLORMAP_REGISTRY, explain_scan
from backend.app.core.uncertainty import CLINICAL_LIMITATIONS_CATALOG

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["Diagnostic Inference"])

# Supported upload MIME types
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "application/octet-stream",  # Fallback for raw binary uploads
}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB limit


def _build_prediction_response(
    image_bytes: bytes,
    request: Request,
    colormap: str = "turbo",
    overlay_alpha: float = 0.55,
    threshold_ratio: float = 0.80,
) -> Dict[str, Any]:
    """
    Executes the complete diagnostic pipeline:
    1. OOD Input Gatekeeper
    2. Deep Learning Inference & Probabilities
    3. Grad-CAM Activation & ROI Localization
    4. Quantitative Shannon Uncertainty Calculation
    5. Clinical Limitations Catalog Injection
    """
    ood_detector = getattr(request.app.state, "ood_detector", None)
    model = getattr(request.app.state, "model", None)
    preprocessor = getattr(request.app.state, "preprocessor", None)
    uncertainty_evaluator = getattr(request.app.state, "uncertainty_evaluator", None)

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Diagnostic model is not loaded or currently unavailable.",
        )

    # 1. OOD Gatekeeper Validation
    if ood_detector is not None:
        try:
            ood_result = ood_detector.validate_image(image_bytes)
        except Exception as ood_err:
            logger.warning(f"OOD detector check failed to decode: {ood_err}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unable to process or decode image file: {str(ood_err)}",
            )

        if not ood_result.get("is_valid_mri", True):
            reason = ood_result.get(
                "rejection_reason",
                "Submitted image does not appear to be an authentic brain MRI scan.",
            )
            logger.info(f"OOD check rejected scan: {reason}")
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "OOD_REJECTION",
                    "message": reason,
                    "confidence_score": ood_result.get("confidence_score", 0.0),
                    "diagnostics": ood_result.get("diagnostics", {}),
                },
            )

    # 2. Grad-CAM & Model Inference
    try:
        explain_result = explain_scan(
            image_input=image_bytes,
            model=model,
            preprocessor=preprocessor,
            overlay_alpha=overlay_alpha,
            threshold_ratio=threshold_ratio,
        )
    except Exception as exc:
        logger.error(f"Inference error during explain_scan: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing neural network inference: {str(exc)}",
        )

    # 3. Uncertainty & Confidence Evaluation
    probs = explain_result["probabilities"]
    if uncertainty_evaluator is not None:
        uncertainty = uncertainty_evaluator.evaluate(probs)
    else:
        from backend.app.core.uncertainty import UncertaintyEvaluator
        evaluator = UncertaintyEvaluator()
        uncertainty = evaluator.evaluate(probs)

    pred_class = explain_result["predicted_class"]
    meta = CLASS_METADATA.get(pred_class, {})

    # Sort probabilities descending
    sorted_probs = [
        {
            "class": c_name,
            "display_name": CLASS_METADATA.get(c_name, {}).get("display_name", c_name.title()),
            "probability": p_val,
        }
        for c_name, p_val in sorted(probs.items(), key=lambda x: x[1], reverse=True)
    ]

    roi_data = explain_result.get("roi", {})

    return {
        "success": True,
        "predicted_class": pred_class,
        "predicted_idx": explain_result["predicted_idx"],
        "display_name": meta.get("display_name", pred_class.title()),
        "description": meta.get("description", ""),
        "who_grade_typical": meta.get("who_grade_typical", "N/A"),
        "clinical_urgency": meta.get("clinical_urgency", "Routine"),
        "confidence": explain_result["confidence"],
        "probabilities": probs,
        "sorted_probabilities": sorted_probs,
        "uncertainty": uncertainty,
        "explainability": {
            "roi": roi_data,
            "anatomical_region": roi_data.get("anatomical_region", "Undetermined"),
            "anatomical_quadrant": roi_data.get("anatomical_quadrant", "Undetermined"),
            "peak_value": roi_data.get("peak_value", 0.0),
            "area_fraction": roi_data.get("area_fraction", 0.0),
            "bbox": roi_data.get("bbox", [0, 0, 224, 224]),
            "bbox_norm": roi_data.get("bbox_norm", [0.0, 0.0, 1.0, 1.0]),
            "has_focal_activation": roi_data.get("has_focal_activation", True),
        },
        "images": {
            "raw_image_base64": explain_result.get("raw_image_base64"),
            "preprocessed_image_base64": explain_result.get("preprocessed_image_base64"),
            "heatmap_base64": explain_result.get("heatmap_base64"),
            "overlays_base64": explain_result.get("overlays_base64", {}),
        },
        "clinical_limitations": CLINICAL_LIMITATIONS_CATALOG,
    }


@router.post(
    "/predict",
    summary="Predict tumor classification with Grad-CAM and uncertainty analysis",
    description="Upload a brain MRI scan image to obtain multi-class probabilities, Grad-CAM visual heatmaps, Shannon entropy uncertainty, and clinical caveats.",
)
async def predict_mri(
    request: Request,
    file: UploadFile = File(..., description="Brain MRI scan image file (JPEG, PNG, etc.)"),
    colormap: str = Query("turbo", description="Grad-CAM colormap: turbo, jet, viridis"),
    overlay_alpha: float = Query(0.55, ge=0.0, le=1.0, description="Heatmap blend transparency"),
    threshold_ratio: float = Query(0.80, ge=0.1, le=0.99, description="Grad-CAM ROI threshold ratio"),
) -> Dict[str, Any]:
    """
    Main inference endpoint for client uploads.
    """
    # 1. Validate file metadata
    if file.filename:
        ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
        valid_extensions = {"jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"}
        if ext not in valid_extensions and file.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format '{file.filename}'. Please upload a standard image file (JPEG, PNG).",
            )

    try:
        image_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file data: {str(exc)}",
        )

    if not image_bytes or len(image_bytes) < 64:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty or corrupted.",
        )

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds the 15 MB limit.",
        )

    return _build_prediction_response(
        image_bytes=image_bytes,
        request=request,
        colormap=colormap,
        overlay_alpha=overlay_alpha,
        threshold_ratio=threshold_ratio,
    )


@router.post(
    "/predict/sample/{sample_id}",
    summary="Instant prediction on curated sample MRI scan",
    description="Executes instant inference and XAI visualization on a curated verified sample scan.",
)
async def predict_sample(
    sample_id: str,
    request: Request,
    colormap: str = Query("turbo", description="Grad-CAM colormap"),
    overlay_alpha: float = Query(0.55, ge=0.0, le=1.0),
    threshold_ratio: float = Query(0.80, ge=0.1, le=0.99),
) -> Dict[str, Any]:
    """
    Directly predicts on curated demo sample scans by ID.
    """
    # Find sample file
    sample_filename = f"{sample_id}.jpg"
    sample_path = SAMPLES_DIR / sample_filename

    if not sample_path.exists():
        # Try finding without sample_ prefix or checking files
        matches = list(SAMPLES_DIR.glob(f"*{sample_id}*"))
        if matches and matches[0].is_file():
            sample_path = matches[0]
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Curated sample '{sample_id}' not found.",
            )

    image_bytes = sample_path.read_bytes()

    return _build_prediction_response(
        image_bytes=image_bytes,
        request=request,
        colormap=colormap,
        overlay_alpha=overlay_alpha,
        threshold_ratio=threshold_ratio,
    )

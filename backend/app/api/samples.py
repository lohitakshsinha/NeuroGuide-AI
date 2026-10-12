"""
NeuroGuide AI - Curated Sample Scans API
Provides endpoints to list and inspect pre-loaded verified brain MRI sample scans.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from backend.app.config import CLASS_METADATA, SAMPLES_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/samples", tags=["Curated Samples"])


def _load_manifest() -> List[Dict[str, Any]]:
    """Loads and returns the verified samples manifest."""
    manifest_path = SAMPLES_DIR / "samples_manifest.json"
    if not manifest_path.exists():
        logger.warning(f"Samples manifest not found at {manifest_path}")
        return []

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            return manifest
    except Exception as exc:
        logger.error(f"Error reading samples manifest: {exc}")
        return []


@router.get(
    "",
    response_model=List[Dict[str, Any]],
    summary="List curated sample MRI scans",
    description="Returns preloaded brain MRI scans with ground truth diagnoses for instant 1-click clinical testing.",
)
async def list_samples() -> List[Dict[str, Any]]:
    """
    Returns curated demo samples with diagnostic metadata and download endpoints.
    """
    raw_samples = _load_manifest()
    enriched_samples = []

    for item in raw_samples:
        sample_id = item.get("id")
        class_name = item.get("class_name", "")
        meta = CLASS_METADATA.get(class_name, {})

        enriched_samples.append({
            "id": sample_id,
            "class_name": class_name,
            "display_name": meta.get("display_name", item.get("display_name", class_name.title())),
            "description": meta.get("description", ""),
            "who_grade_typical": meta.get("who_grade_typical", "N/A"),
            "clinical_urgency": meta.get("clinical_urgency", "Routine"),
            "filename": item.get("filename"),
            "file_size_kb": item.get("file_size_kb", 0.0),
            "image_url": f"/api/samples/{sample_id}/image",
        })

    return enriched_samples


@router.get(
    "/{sample_id}/image",
    summary="Download or view curated sample image",
    description="Returns the JPEG/PNG image binary for the requested sample scan ID.",
)
async def get_sample_image(sample_id: str):
    """
    Streams the raw image file for the requested sample.
    """
    samples = _load_manifest()
    target_sample = next((s for s in samples if s.get("id") == sample_id), None)

    if not target_sample:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample '{sample_id}' not found in curated library.",
        )

    filename = target_sample.get("filename")
    image_path = SAMPLES_DIR / filename

    if not image_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image file for sample '{sample_id}' does not exist on disk.",
        )

    return FileResponse(
        path=str(image_path),
        media_type="image/jpeg",
        filename=filename,
        headers={"Cache-Control": "public, max-age=86400"},
    )

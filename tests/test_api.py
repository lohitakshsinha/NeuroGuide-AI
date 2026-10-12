"""
FastAPI Endpoints Verification Tests (Phase 5 / Phase 6.1)
Verifies:
1. /health readiness check
2. /api/samples retrieval & curated sample streaming
3. /api/predict multipart image uploads
4. /api/predict/sample/{id} instant 1-click execution
5. OOD gatekeeper HTTP 422 rejection of colorful photos
6. Non-image file HTTP 400 rejection
7. Frontend static assets serving at root
"""

import io
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import SAMPLES_DIR
from backend.app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


class TestAPIEndpoints:
    """Verifies FastAPI core REST services."""

    def test_health_check(self, client):
        """Verifies /health endpoint returns service metadata and model status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert "classes" in data
        assert len(data["classes"]) == 4
        assert data["samples_available"] >= 4

    def test_list_samples(self, client):
        """Verifies /api/samples returns the 4 curated demo scans."""
        response = client.get("/api/samples")
        assert response.status_code == 200
        samples = response.json()
        assert len(samples) == 4

        sample_classes = {s["class_name"] for s in samples}
        assert sample_classes == {"glioma", "meningioma", "pituitary", "notumor"}

        for s in samples:
            assert "id" in s
            assert "display_name" in s
            assert "image_url" in s
            assert s["image_url"].startswith("/api/samples/")

    def test_get_sample_image(self, client):
        """Verifies /api/samples/{id}/image streams a valid JPEG binary."""
        response = client.get("/api/samples/sample_glioma/image")
        assert response.status_code == 200
        assert "image/jpeg" in response.headers["content-type"]
        assert len(response.content) > 1000

    def test_get_sample_image_not_found(self, client):
        """Verifies 404 on non-existent sample."""
        response = client.get("/api/samples/unknown_tumor_sample/image")
        assert response.status_code == 404

    def test_predict_sample_glioma(self, client):
        """Verifies 1-click prediction on curated demo sample."""
        response = client.post("/api/predict/sample/sample_glioma")
        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["predicted_class"] == "glioma"
        assert data["confidence"] > 0.5
        assert "probabilities" in data
        assert "uncertainty" in data
        assert "explainability" in data
        assert "images" in data

        # Check explainability payload
        exp = data["explainability"]
        assert "roi" in exp
        assert "anatomical_region" in exp
        assert "anatomical_quadrant" in exp
        assert "bbox" in exp

        # Check uncertainty payload
        unc = data["uncertainty"]
        assert 0.0 <= unc["entropy"] <= 2.0
        assert unc["confidence_level"] in {"HIGH", "MODERATE", "AMBIGUOUS"}

        # Check base64 images
        imgs = data["images"]
        assert "raw_image_base64" in imgs
        assert "preprocessed_image_base64" in imgs
        assert "heatmap_base64" in imgs
        assert "turbo" in imgs["overlays_base64"]

    def test_predict_sample_meningioma(self, client):
        """Verifies prediction on meningioma sample."""
        response = client.post("/api/predict/sample/sample_meningioma")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["predicted_class"] == "meningioma"
        assert data["confidence"] > 0.5

    def test_predict_sample_pituitary(self, client):
        """Verifies prediction on pituitary sample."""
        response = client.post("/api/predict/sample/sample_pituitary")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["predicted_class"] == "pituitary"

    def test_predict_sample_notumor(self, client):
        """Verifies prediction on normal (no tumor) sample."""
        response = client.post("/api/predict/sample/sample_notumor")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["predicted_class"] == "notumor"

    def test_predict_upload_valid_mri(self, client):
        """Verifies multipart file upload for prediction."""
        glioma_path = SAMPLES_DIR / "sample_glioma.jpg"
        with open(glioma_path, "rb") as f:
            file_bytes = f.read()

        response = client.post(
            "/api/predict",
            files={"file": ("brain_scan.jpg", file_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["predicted_class"] == "glioma"

    def test_predict_upload_non_image_rejected(self, client):
        """Verifies HTTP 400 when uploading a text document."""
        response = client.post(
            "/api/predict",
            files={"file": ("notes.txt", b"Patient clinical chart notes", "text/plain")},
        )
        assert response.status_code == 400

    def test_predict_upload_ood_colorful_photo_rejected(self, client):
        """Verifies OOD gatekeeper rejects colorful photos with HTTP 422."""
        # Create a bright saturated RGB image (e.g. scenery or animal photo simulation)
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:, :, 0] = 255  # Full Blue
        img[:, :, 1] = 50   # Green
        img[:, :, 2] = 220  # Full Red

        is_success, buffer = cv2.imencode(".jpg", img)
        assert is_success

        response = client.post(
            "/api/predict",
            files={"file": ("colorful_sunset.jpg", buffer.tobytes(), "image/jpeg")},
        )
        assert response.status_code == 422
        data = response.json()
        detail = data.get("detail", {})
        if isinstance(detail, dict):
            assert detail.get("error") == "OOD_REJECTION"
            assert "Out-of-distribution" in detail.get("message", "")

    def test_static_frontend_serving(self, client):
        """Verifies FastAPI mounts and serves frontend single-page app."""
        index_res = client.get("/")
        assert index_res.status_code == 200
        assert "NEUROGUIDE AI" in index_res.text
        assert "mriDropzone" in index_res.text
        assert "overlayScanCanvas" in index_res.text

        css_res = client.get("/styles/main.css")
        assert css_res.status_code == 200

        viewer_css = client.get("/styles/viewer.css")
        assert viewer_css.status_code == 200

        js_res = client.get("/scripts/app.js")
        assert js_res.status_code == 200

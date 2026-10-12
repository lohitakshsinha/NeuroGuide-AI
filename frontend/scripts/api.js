/**
 * NeuroGuide AI - Backend API Client
 * Manages communication with FastAPI server endpoints.
 */

const API_BASE_URL = window.location.origin;

export class NeuroGuideAPI {
  /**
   * Health & readiness check.
   */
  static async checkHealth() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (!response.ok) {
        throw new Error(`Health check failed with HTTP ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.warn('Backend service offline or unreachable:', error);
      return { status: 'offline', error: error.message };
    }
  }

  /**
   * Retrieves verified curated demo sample scans.
   */
  static async getCuratedSamples() {
    const response = await fetch(`${API_BASE_URL}/api/samples`);
    if (!response.ok) {
      throw new Error(`Failed to load samples with HTTP ${response.status}`);
    }
    return await response.json();
  }

  /**
   * Predicts on a curated demo sample by ID.
   */
  static async predictSample(sampleId, options = {}) {
    const { colormap = 'turbo', overlayAlpha = 0.55, thresholdRatio = 0.80 } = options;
    const url = new URL(`${API_BASE_URL}/api/predict/sample/${sampleId}`);
    url.searchParams.set('colormap', colormap);
    url.searchParams.set('overlay_alpha', overlayAlpha);
    url.searchParams.set('threshold_ratio', thresholdRatio);

    const response = await fetch(url.toString(), {
      method: 'POST',
    });

    const data = await response.json();
    if (!response.ok) {
      const err = new Error(data.detail?.message || data.detail || 'Prediction failed');
      err.status = response.status;
      err.data = data;
      throw err;
    }

    return data;
  }

  /**
   * Uploads and runs inference on a user-submitted MRI scan file.
   */
  static async predictUploadedFile(file, options = {}) {
    const { colormap = 'turbo', overlayAlpha = 0.55, thresholdRatio = 0.80 } = options;
    const url = new URL(`${API_BASE_URL}/api/predict`);
    url.searchParams.set('colormap', colormap);
    url.searchParams.set('overlay_alpha', overlayAlpha);
    url.searchParams.set('threshold_ratio', thresholdRatio);

    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(url.toString(), {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      const err = new Error(data.detail?.message || data.detail || 'Prediction failed');
      err.status = response.status;
      err.data = data;
      throw err;
    }

    return data;
  }
}

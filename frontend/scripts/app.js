/**
 * NeuroGuide AI - Frontend Application Orchestrator
 * Connects API client, DualDynamicViewer, drag-and-drop, sample chips,
 * and diagnostic analytics rendering.
 */

import { NeuroGuideAPI } from './api.js';
import { DualDynamicViewer } from './viewer.js';

class NeuroGuideApp {
  constructor() {
    this.viewer = null;
    this.currentData = null;
    this.curatedSamples = [];
    this.activeSampleId = null;

    // DOM Elements
    this.dropzone = document.getElementById('mriDropzone');
    this.fileInput = document.getElementById('fileInput');
    this.sampleCardsContainer = document.getElementById('sampleCardsContainer');
    this.loadingOverlay = document.getElementById('loadingOverlay');
    this.alertBanner = document.getElementById('alertBanner');
    this.alertMessage = document.getElementById('alertMessage');

    // Controls
    this.opacitySlider = document.getElementById('opacitySlider');
    this.opacityVal = document.getElementById('opacityVal');
    this.colormapPills = document.querySelectorAll('.colormap-pill');
    this.bboxToggle = document.getElementById('bboxToggle');
    this.btnSideBySide = document.getElementById('btnSideBySide');
    this.btnBlended = document.getElementById('btnBlended');

    // Diagnostics Elements
    this.diagnosisName = document.getElementById('diagnosisName');
    this.diagnosisConfidence = document.getElementById('diagnosisConfidence');
    this.diagnosisWhoBadge = document.getElementById('diagnosisWhoBadge');
    this.diagnosisUrgencyBadge = document.getElementById('diagnosisUrgencyBadge');
    this.entropyVal = document.getElementById('entropyVal');
    this.marginVal = document.getElementById('marginVal');
    this.confidenceBadge = document.getElementById('confidenceBadge');
    this.clinicalCaveatBox = document.getElementById('clinicalCaveatBox');
    this.clinicalCaveatText = document.getElementById('clinicalCaveatText');
    this.probabilitiesContainer = document.getElementById('probabilitiesContainer');

    // ROI Elements
    this.roiQuadrant = document.getElementById('roiQuadrant');
    this.roiRegion = document.getElementById('roiRegion');
    this.roiBbox = document.getElementById('roiBbox');
    this.roiPeak = document.getElementById('roiPeak');
    this.roiArea = document.getElementById('roiArea');

    // Limitations Container
    this.limitationsContainer = document.getElementById('limitationsContainer');
    this.healthBadge = document.getElementById('healthBadge');
    this.healthText = document.getElementById('healthText');

    this.init();
  }

  async init() {
    // 1. Initialize Canvas Viewer
    this.viewer = new DualDynamicViewer({
      rawCanvasId: 'rawScanCanvas',
      overlayCanvasId: 'overlayScanCanvas',
      viewportContainerId: 'viewportContainer',
      placeholderLeftId: 'placeholderLeft',
      placeholderRightId: 'placeholderRight',
      coordDisplayId: 'coordDisplay',
      statusDisplayId: 'statusDisplay',
    });

    // 2. Attach Event Handlers
    this._attachEventListeners();

    // 3. Health Check
    await this._checkServerHealth();

    // 4. Load Samples Gallery
    await this._loadSamples();
  }

  _attachEventListeners() {
    // Dropzone Drag & Drop
    if (this.dropzone && this.fileInput) {
      this.dropzone.addEventListener('click', () => this.fileInput.click());

      this.dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        this.dropzone.classList.add('drag-over');
      });

      this.dropzone.addEventListener('dragleave', () => {
        this.dropzone.classList.remove('drag-over');
      });

      this.dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        this.dropzone.classList.remove('drag-over');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
          this.handleFile(e.dataTransfer.files[0]);
        }
      });

      this.fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
          this.handleFile(e.target.files[0]);
        }
      });
    }

    // Opacity Slider
    if (this.opacitySlider) {
      this.opacitySlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        if (this.opacityVal) this.opacityVal.textContent = `${Math.round(val * 100)}%`;
        this.viewer.setOpacity(val);
      });
    }

    // Colormap Pills
    this.colormapPills.forEach((pill) => {
      pill.addEventListener('click', () => {
        this.colormapPills.forEach((p) => p.classList.remove('active'));
        pill.classList.add('active');
        const cmap = pill.dataset.colormap;
        this.viewer.setColormap(cmap);
      });
    });

    // Bounding Box Checkbox
    if (this.bboxToggle) {
      this.bboxToggle.addEventListener('change', (e) => {
        this.viewer.setBoundingBox(e.target.checked);
      });
    }

    // View Mode Segmented Buttons
    if (this.btnSideBySide && this.btnBlended) {
      this.btnSideBySide.addEventListener('click', () => {
        this.btnSideBySide.classList.add('active');
        this.btnBlended.classList.remove('active');
        this.viewer.setViewMode('side-by-side');
      });

      this.btnBlended.addEventListener('click', () => {
        this.btnBlended.classList.add('active');
        this.btnSideBySide.classList.remove('active');
        this.viewer.setViewMode('blended');
      });
    }
  }

  async _checkServerHealth() {
    const health = await NeuroGuideAPI.checkHealth();
    if (health.status === 'ok') {
      if (this.healthBadge) {
        this.healthBadge.className = 'status-badge';
      }
      if (this.healthText) {
        const dev = health.device ? health.device.toUpperCase() : 'CPU';
        this.healthText.textContent = `Online • ${health.backbone || 'EfficientNet'} (${dev})`;
      }
    } else {
      if (this.healthText) this.healthText.textContent = 'Server Offline';
    }
  }

  async _loadSamples() {
    try {
      this.curatedSamples = await NeuroGuideAPI.getCuratedSamples();
      this._renderSampleCards();

      // Automatically select and demonstrate the first sample (Glioma or Meningioma)
      if (this.curatedSamples && this.curatedSamples.length > 0) {
        // Pre-load Glioma for instant demonstration
        const gliomaSample = this.curatedSamples.find((s) => s.class_name === 'glioma') || this.curatedSamples[0];
        this.handleSampleSelect(gliomaSample.id);
      }
    } catch (err) {
      console.warn('Could not load samples gallery:', err);
    }
  }

  _renderSampleCards() {
    if (!this.sampleCardsContainer) return;
    this.sampleCardsContainer.innerHTML = '';

    this.curatedSamples.forEach((sample) => {
      const card = document.createElement('div');
      card.className = `sample-chip ${sample.id === this.activeSampleId ? 'active' : ''}`;
      card.id = `sample-${sample.id}`;
      card.innerHTML = `
        <div class="sample-thumb-wrap">
          <img src="${sample.image_url}" alt="${sample.display_name}" loading="lazy" />
        </div>
        <div class="sample-chip-info">
          <span class="sample-chip-name">${sample.display_name}</span>
          <span class="sample-chip-sub">Verified Demo Case</span>
        </div>
      `;

      card.addEventListener('click', () => this.handleSampleSelect(sample.id));
      this.sampleCardsContainer.appendChild(card);
    });
  }

  async handleSampleSelect(sampleId) {
    this.activeSampleId = sampleId;
    // Update active class on chips
    document.querySelectorAll('.sample-chip').forEach((c) => c.classList.remove('active'));
    const targetChip = document.getElementById(`sample-${sampleId}`);
    if (targetChip) targetChip.classList.add('active');

    this._setLoading(true);
    this._hideAlert();

    try {
      const currentAlpha = this.opacitySlider ? parseFloat(this.opacitySlider.value) : 0.55;
      const activeCmapPill = document.querySelector('.colormap-pill.active');
      const currentCmap = activeCmapPill ? activeCmapPill.dataset.colormap : 'turbo';

      const data = await NeuroGuideAPI.predictSample(sampleId, {
        colormap: currentCmap,
        overlayAlpha: currentAlpha,
      });

      this.currentData = data;
      await this.viewer.loadResult(data);
      this._renderPredictionAnalytics(data);
    } catch (err) {
      this._showAlert(err.message, 'error');
    } finally {
      this._setLoading(false);
    }
  }

  async handleFile(file) {
    if (!file) return;

    // Deselect sample chip
    this.activeSampleId = null;
    document.querySelectorAll('.sample-chip').forEach((c) => c.classList.remove('active'));

    this._setLoading(true);
    this._hideAlert();

    try {
      const currentAlpha = this.opacitySlider ? parseFloat(this.opacitySlider.value) : 0.55;
      const activeCmapPill = document.querySelector('.colormap-pill.active');
      const currentCmap = activeCmapPill ? activeCmapPill.dataset.colormap : 'turbo';

      const data = await NeuroGuideAPI.predictUploadedFile(file, {
        colormap: currentCmap,
        overlayAlpha: currentAlpha,
      });

      this.currentData = data;
      await this.viewer.loadResult(data);
      this._renderPredictionAnalytics(data);
    } catch (err) {
      let errMsg = err.message || 'Error processing scan';
      if (err.data?.detail?.message) {
        errMsg = err.data.detail.message;
      } else if (typeof err.data?.detail === 'string') {
        errMsg = err.data.detail;
      }
      this._showAlert(errMsg, 'error');
    } finally {
      this._setLoading(false);
    }
  }

  _renderPredictionAnalytics(data) {
    const {
      display_name,
      confidence,
      who_grade_typical,
      clinical_urgency,
      uncertainty,
      sorted_probabilities,
      explainability,
      clinical_limitations,
    } = data;

    // 1. Hero Diagnosis & Score
    if (this.diagnosisName) this.diagnosisName.textContent = display_name;
    if (this.diagnosisConfidence) {
      this.diagnosisConfidence.textContent = `${(confidence * 100).toFixed(1)}%`;
    }

    if (this.diagnosisWhoBadge) {
      this.diagnosisWhoBadge.textContent = who_grade_typical;
    }

    if (this.diagnosisUrgencyBadge) {
      this.diagnosisUrgencyBadge.textContent = `${clinical_urgency} Urgency`;
      this.diagnosisUrgencyBadge.className = 'badge-tag';
      if (clinical_urgency === 'High') {
        this.diagnosisUrgencyBadge.classList.add('badge-urgency-high');
      } else if (clinical_urgency === 'Moderate') {
        this.diagnosisUrgencyBadge.classList.add('badge-urgency-moderate');
      } else {
        this.diagnosisUrgencyBadge.classList.add('badge-urgency-routine');
      }
    }

    // 2. Shannon Uncertainty & Entropy
    if (this.entropyVal) {
      this.entropyVal.textContent = `${uncertainty.entropy.toFixed(2)} bits`;
    }
    if (this.marginVal) {
      this.marginVal.textContent = `Δ ${(uncertainty.margin_delta * 100).toFixed(1)}%`;
    }
    if (this.confidenceBadge) {
      this.confidenceBadge.textContent = uncertainty.confidence_level;
      this.confidenceBadge.style.backgroundColor = `${uncertainty.confidence_color}22`;
      this.confidenceBadge.style.color = uncertainty.confidence_color;
      this.confidenceBadge.style.border = `1px solid ${uncertainty.confidence_color}55`;
    }

    // Clinical Advisory Note
    if (this.clinicalCaveatBox && this.clinicalCaveatText) {
      if (uncertainty.clinical_caveat) {
        this.clinicalCaveatText.textContent = uncertainty.clinical_caveat;
        this.clinicalCaveatBox.style.display = 'flex';
      } else {
        this.clinicalCaveatBox.style.display = 'none';
      }
    }

    // 3. Probabilities Breakdown Bars
    if (this.probabilitiesContainer && sorted_probabilities) {
      this.probabilitiesContainer.innerHTML = '';
      sorted_probabilities.forEach((item, index) => {
        const pct = (item.probability * 100).toFixed(1);
        const row = document.createElement('div');
        row.className = 'prob-row-item';
        row.innerHTML = `
          <div class="prob-row-labels">
            <span class="prob-name">${item.display_name}</span>
            <span class="prob-percentage">${pct}%</span>
          </div>
          <div class="prob-bar-track">
            <div class="prob-bar-fill ${index === 0 ? 'lead' : ''}" style="width: ${pct}%"></div>
          </div>
        `;
        this.probabilitiesContainer.appendChild(row);
      });
    }

    // 4. ROI & Anatomical Localization
    const roi = explainability?.roi;
    if (this.roiQuadrant) {
      this.roiQuadrant.textContent = explainability?.anatomical_quadrant || 'Unlocalized';
    }
    if (this.roiRegion) {
      this.roiRegion.textContent = explainability?.anatomical_region || 'Diffuse parenchymal baseline';
    }
    if (this.roiBbox && roi?.bbox) {
      const [x1, y1, x2, y2] = roi.bbox;
      this.roiBbox.textContent = `[${x1}, ${y1}, ${x2}, ${y2}]`;
    }
    if (this.roiPeak && roi?.peak_value !== undefined) {
      this.roiPeak.textContent = `${(roi.peak_value * 100).toFixed(0)}%`;
    }
    if (this.roiArea && roi?.area_fraction !== undefined) {
      this.roiArea.textContent = `${(roi.area_fraction * 100).toFixed(1)}%`;
    }

    // 5. Clinical Limitations Catalog
    if (this.limitationsContainer && clinical_limitations) {
      this.limitationsContainer.innerHTML = '';
      clinical_limitations.forEach((lim) => {
        const limEl = document.createElement('div');
        limEl.className = 'limitation-item';

        let severityClass = 'severity-warning';
        if (lim.severity === 'Critical') severityClass = 'severity-critical';
        if (lim.severity === 'Notice') severityClass = 'severity-notice';
        if (lim.severity === 'Mandatory Disclaimer') severityClass = 'severity-mandatory';

        limEl.innerHTML = `
          <div class="limitation-icon-col">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 9v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div class="limitation-content">
            <div class="limitation-head">
              <span class="limitation-title">${lim.title}</span>
              <span class="limitation-severity-badge ${severityClass}">${lim.severity}</span>
            </div>
            <div class="limitation-body">${lim.description}</div>
          </div>
        `;
        this.limitationsContainer.appendChild(limEl);
      });
    }
  }

  _setLoading(isLoading) {
    if (this.loadingOverlay) {
      if (isLoading) {
        this.loadingOverlay.classList.add('active');
      } else {
        this.loadingOverlay.classList.remove('active');
      }
    }
  }

  _showAlert(message, type = 'error') {
    if (this.alertBanner && this.alertMessage) {
      this.alertMessage.textContent = message;
      this.alertBanner.className = `alert-banner ${type}`;
      this.alertBanner.style.display = 'flex';
    }
  }

  _hideAlert() {
    if (this.alertBanner) {
      this.alertBanner.style.display = 'none';
    }
  }
}

// Instantiate on DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
  new NeuroGuideApp();
});

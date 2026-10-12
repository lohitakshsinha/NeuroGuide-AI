/**
 * NeuroGuide AI - Canvas Rendering & Dual Dynamic Viewer Engine
 * Manages raw MRI scan, Grad-CAM overlays, dynamic opacity blending,
 * colormap switching, and bounding box drawing.
 */

export class DualDynamicViewer {
  constructor({
    rawCanvasId,
    overlayCanvasId,
    viewportContainerId,
    placeholderLeftId,
    placeholderRightId,
    coordDisplayId,
    statusDisplayId,
  }) {
    this.rawCanvas = document.getElementById(rawCanvasId);
    this.overlayCanvas = document.getElementById(overlayCanvasId);
    this.viewportContainer = document.getElementById(viewportContainerId);
    this.placeholderLeft = document.getElementById(placeholderLeftId);
    this.placeholderRight = document.getElementById(placeholderRightId);
    this.coordDisplay = document.getElementById(coordDisplayId);
    this.statusDisplay = document.getElementById(statusDisplayId);

    this.rawCtx = this.rawCanvas?.getContext('2d');
    this.overlayCtx = this.overlayCanvas?.getContext('2d');

    // State
    this.opacity = 0.55;
    this.colormap = 'turbo';
    this.showBoundingBox = true;
    this.viewMode = 'side-by-side'; // 'side-by-side' or 'blended'

    this.rawImage = null;
    this.preprocessedImage = null;
    this.heatmapImage = null;
    this.overlayImages = {}; // colormap -> Image
    this.roiData = null;

    this._bindEvents();
  }

  _bindEvents() {
    if (!this.overlayCanvas) return;

    this.overlayCanvas.addEventListener('mousemove', (e) => this._handleMouseMove(e));
    this.overlayCanvas.addEventListener('mouseleave', () => this._handleMouseLeave());
    if (this.rawCanvas) {
      this.rawCanvas.addEventListener('mousemove', (e) => this._handleMouseMove(e));
      this.rawCanvas.addEventListener('mouseleave', () => this._handleMouseLeave());
    }
  }

  _handleMouseMove(e) {
    const canvas = e.target;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const x = Math.floor((e.clientX - rect.left) * scaleX);
    const y = Math.floor((e.clientY - rect.top) * scaleY);

    if (this.coordDisplay) {
      this.coordDisplay.textContent = `X: ${x}px | Y: ${y}px`;
    }
  }

  _handleMouseLeave() {
    if (this.coordDisplay) {
      this.coordDisplay.textContent = 'X: -- | Y: --';
    }
  }

  /**
   * Loads new prediction explainability payload into viewer.
   */
  async loadResult(data) {
    const { images, explainability } = data;
    this.roiData = explainability?.roi || null;

    // Load Base64 images into Image objects
    const loadImg = (base64Str) =>
      new Promise((resolve, reject) => {
        if (!base64Str) return resolve(null);
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = base64Str.startsWith('data:') ? base64Str : `data:image/png;base64,${base64Str}`;
      });

    try {
      this.rawImage = await loadImg(images.raw_image_base64);
      this.preprocessedImage = await loadImg(images.preprocessed_image_base64);
      this.heatmapImage = await loadImg(images.heatmap_base64);

      this.overlayImages = {};
      if (images.overlays_base64) {
        for (const [cmap, b64] of Object.entries(images.overlays_base64)) {
          this.overlayImages[cmap] = await loadImg(b64);
        }
      }

      this._togglePlaceholders(false);
      this.render();

      if (this.statusDisplay) {
        this.statusDisplay.textContent = `224x224 • Normalized • ${this.colormap.toUpperCase()}`;
      }
    } catch (err) {
      console.error('Failed to load viewer images:', err);
    }
  }

  setOpacity(newVal) {
    this.opacity = parseFloat(newVal);
    this.render();
  }

  setColormap(cmap) {
    this.colormap = cmap;
    if (this.statusDisplay) {
      this.statusDisplay.textContent = `224x224 • Normalized • ${this.colormap.toUpperCase()}`;
    }
    this.render();
  }

  setBoundingBox(show) {
    this.showBoundingBox = Boolean(show);
    this.render();
  }

  setViewMode(mode) {
    this.viewMode = mode;
    if (this.viewportContainer) {
      if (mode === 'blended') {
        this.viewportContainer.classList.add('single-mode');
        if (this.rawCanvas?.parentElement) {
          this.rawCanvas.parentElement.style.display = 'none';
        }
      } else {
        this.viewportContainer.classList.remove('single-mode');
        if (this.rawCanvas?.parentElement) {
          this.rawCanvas.parentElement.style.display = 'flex';
        }
      }
    }
    this.render();
  }

  render() {
    if (!this.preprocessedImage && !this.rawImage) return;

    const baseImg = this.preprocessedImage || this.rawImage;
    const w = baseImg.naturalWidth || 224;
    const h = baseImg.naturalHeight || 224;

    // 1. Render Left Canvas (Raw / Preprocessed Anatomical Scan)
    if (this.rawCanvas && this.rawCtx && this.viewMode === 'side-by-side') {
      this.rawCanvas.width = w;
      this.rawCanvas.height = h;
      this.rawCtx.clearRect(0, 0, w, h);
      this.rawCtx.drawImage(baseImg, 0, 0, w, h);

      // Subtle border label
      this.rawCtx.font = '10px JetBrains Mono, monospace';
      this.rawCtx.fillStyle = 'rgba(255, 255, 255, 0.6)';
      this.rawCtx.fillText('ANATOMICAL (CLAHE)', 8, 16);
    }

    // 2. Render Right / Blended Canvas (Grad-CAM with live alpha & BBox)
    if (this.overlayCanvas && this.overlayCtx) {
      this.overlayCanvas.width = w;
      this.overlayCanvas.height = h;
      this.overlayCtx.clearRect(0, 0, w, h);

      // Base anatomical layer
      this.overlayCtx.globalAlpha = 1.0;
      this.overlayCtx.drawImage(baseImg, 0, 0, w, h);

      // Grad-CAM Heatmap layer with custom opacity
      const overlayImg = this.overlayImages[this.colormap] || this.heatmapImage;
      if (overlayImg) {
        this.overlayCtx.save();
        this.overlayCtx.globalAlpha = this.opacity;
        this.overlayCtx.drawImage(overlayImg, 0, 0, w, h);
        this.overlayCtx.restore();
      }

      // Draw ROI Bounding Box & Annotation
      if (this.showBoundingBox && this.roiData && this.roiData.has_focal_activation) {
        this._drawBoundingBox(this.overlayCtx, this.roiData, w, h);
      }
    }
  }

  _drawBoundingBox(ctx, roi, w, h) {
    const bbox = roi.bbox; // [x_min, y_min, x_max, y_max]
    if (!bbox || bbox.length < 4) return;

    const [x1, y1, x2, y2] = bbox;
    const boxW = x2 - x1;
    const boxH = y2 - y1;

    ctx.save();
    // Glowing cyan stroke
    ctx.strokeStyle = '#06B6D4';
    ctx.lineWidth = 2;
    ctx.shadowColor = '#06B6D4';
    ctx.shadowBlur = 8;
    ctx.strokeRect(x1, y1, boxW, boxH);

    // Corner brackets
    const bracketLen = Math.min(10, boxW / 4, boxH / 4);
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#38BDF8';
    // Top-left
    ctx.beginPath();
    ctx.moveTo(x1, y1 + bracketLen);
    ctx.lineTo(x1, y1);
    ctx.lineTo(x1 + bracketLen, y1);
    // Top-right
    ctx.moveTo(x2 - bracketLen, y1);
    ctx.lineTo(x2, y1);
    ctx.lineTo(x2, y1 + bracketLen);
    // Bottom-left
    ctx.moveTo(x1, y2 - bracketLen);
    ctx.lineTo(x1, y2);
    ctx.lineTo(x1 + bracketLen, y2);
    // Bottom-right
    ctx.moveTo(x2 - bracketLen, y2);
    ctx.lineTo(x2, y2);
    ctx.lineTo(x2, y2 - bracketLen);
    ctx.stroke();

    // Centroid Crosshair
    if (roi.centroid) {
      const [cx, cy] = roi.centroid;
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
      ctx.fillStyle = '#F43F5E';
      ctx.fill();
    }

    // Badge label atop bounding box
    const labelText = `ROI Peak: ${Math.round((roi.peak_value || 1.0) * 100)}%`;
    ctx.font = 'bold 9px JetBrains Mono, monospace';
    ctx.shadowBlur = 0;
    const textWidth = ctx.measureText(labelText).width;
    const labelY = Math.max(14, y1 - 4);

    ctx.fillStyle = 'rgba(6, 182, 212, 0.9)';
    ctx.fillRect(x1, labelY - 11, textWidth + 8, 14);

    ctx.fillStyle = '#000000';
    ctx.fillText(labelText, x1 + 4, labelY);
    ctx.restore();
  }

  _togglePlaceholders(show) {
    if (this.placeholderLeft) this.placeholderLeft.style.display = show ? 'flex' : 'none';
    if (this.placeholderRight) this.placeholderRight.style.display = show ? 'flex' : 'none';
    if (this.rawCanvas) this.rawCanvas.style.display = show ? 'none' : 'block';
    if (this.overlayCanvas) this.overlayCanvas.style.display = show ? 'none' : 'block';
  }

  reset() {
    this._togglePlaceholders(true);
    if (this.rawCtx && this.rawCanvas) {
      this.rawCtx.clearRect(0, 0, this.rawCanvas.width, this.rawCanvas.height);
    }
    if (this.overlayCtx && this.overlayCanvas) {
      this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
    }
    if (this.coordDisplay) this.coordDisplay.textContent = 'X: -- | Y: --';
    if (this.statusDisplay) this.statusDisplay.textContent = 'Awaiting MRI Scan...';
  }
}

"""
NeuroGuide AI - Grad-CAM & ROI Localization Showcase Generator

Generates a publication-grade multi-panel diagnostic figure comparing
raw MRI scans, preprocessed inputs, Grad-CAM activation maps, colormaps
(Turbo, Jet, Viridis), and automated anatomical ROI bounding boxes.
"""

from __future__ import annotations

import base64
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from backend.app.config import ARTIFACTS_DIR, SAMPLES_DIR
from backend.app.core.gradcam import explain_scan
from backend.app.core.model import load_checkpoint
from backend.app.core.uncertainty import UncertaintyEvaluator


def generate_gradcam_showcase(output_path: Path = ARTIFACTS_DIR / "gradcam_explainability_showcase.png") -> Path:
    model, _ = load_checkpoint()
    evaluator = UncertaintyEvaluator()

    sample_items = [
        ("Glioma", SAMPLES_DIR / "sample_glioma.jpg"),
        ("Meningioma", SAMPLES_DIR / "sample_meningioma.jpg"),
        ("Pituitary", SAMPLES_DIR / "sample_pituitary.jpg"),
        ("No Tumor", SAMPLES_DIR / "sample_notumor.jpg"),
    ]

    fig, axes = plt.subplots(4, 7, figsize=(22, 13), facecolor="#0B0F19")
    plt.subplots_adjust(wspace=0.08, hspace=0.25, left=0.08, right=0.96, top=0.91, bottom=0.05)

    fig.suptitle(
        "NeuroGuide AI — Explainable AI (Grad-CAM) & Anatomical ROI Localization Showcase",
        fontsize=18,
        fontweight="bold",
        color="#F8FAFC",
        y=0.96,
    )

    column_headers = [
        "Original MRI Scan",
        "Preprocessed (CLAHE)",
        "Grad-CAM Heatmap",
        "Turbo Overlay (55%)",
        "Jet Overlay (55%)",
        "Viridis Overlay (55%)",
        "Automated ROI Localization",
    ]

    for col_idx, header in enumerate(column_headers):
        axes[0, col_idx].set_title(header, fontsize=12, fontweight="bold", color="#38BDF8", pad=10)

    for row_idx, (ground_truth_label, filepath) in enumerate(sample_items):
        exp = explain_scan(filepath, model=model)
        raw_bgr = cv2.imread(str(filepath))
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

        pred_class = exp["predicted_class"].capitalize()
        conf_pct = exp["confidence"] * 100
        uncertainty = evaluator.evaluate(exp["probabilities"])
        entropy = uncertainty["entropy"]

        # 1. Original Scan
        axes[row_idx, 0].imshow(raw_rgb)
        label_text = (
            f"True: {ground_truth_label}\n"
            f"Pred: {pred_class} ({conf_pct:.1f}%)\n"
            f"Entropy: {entropy:.2f} bits\n"
            f"[{uncertainty['confidence_level']}]"
        )
        axes[row_idx, 0].set_ylabel(
            label_text,
            color="#F1F5F9",
            fontsize=10.5,
            fontweight="bold",
            rotation=0,
            labelpad=85,
            va="center",
        )

        # 2. Preprocessed CLAHE
        prep_bytes = base64.b64decode(exp["preprocessed_image_base64"].split(",")[1])
        prep_arr = cv2.imdecode(np.frombuffer(prep_bytes, np.uint8), cv2.IMREAD_COLOR)
        prep_rgb = cv2.cvtColor(prep_arr, cv2.COLOR_BGR2RGB)
        axes[row_idx, 1].imshow(prep_rgb)

        # 3. Raw Grad-CAM Heatmap
        axes[row_idx, 2].imshow(exp["heatmap"], cmap="inferno")

        # 4. Turbo Overlay
        turbo_bytes = base64.b64decode(exp["overlays_base64"]["turbo"].split(",")[1])
        turbo_arr = cv2.cvtColor(
            cv2.imdecode(np.frombuffer(turbo_bytes, np.uint8), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        )
        axes[row_idx, 3].imshow(turbo_arr)

        # 5. Jet Overlay
        jet_bytes = base64.b64decode(exp["overlays_base64"]["jet"].split(",")[1])
        jet_arr = cv2.cvtColor(
            cv2.imdecode(np.frombuffer(jet_bytes, np.uint8), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        )
        axes[row_idx, 4].imshow(jet_arr)

        # 6. Viridis Overlay
        vir_bytes = base64.b64decode(exp["overlays_base64"]["viridis"].split(",")[1])
        vir_arr = cv2.cvtColor(
            cv2.imdecode(np.frombuffer(vir_bytes, np.uint8), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        )
        axes[row_idx, 5].imshow(vir_arr)

        # 7. ROI Localization on Turbo
        axes[row_idx, 6].imshow(turbo_arr)
        roi = exp["roi"]
        if roi["has_focal_activation"]:
            bx_min, by_min, bx_max, by_max = roi["bbox"]
            rect = patches.Rectangle(
                (bx_min, by_min),
                bx_max - bx_min,
                by_max - by_min,
                linewidth=2,
                edgecolor="#22C55E",
                facecolor="none",
                linestyle="--",
            )
            axes[row_idx, 6].add_patch(rect)

            # Peak coordinates marker
            px, py = roi["peak_coords"]
            axes[row_idx, 6].plot(
                px, py, "o", color="#EF4444", markersize=6, markeredgecolor="white", markeredgewidth=1.5
            )

            # Anatomical region annotation callout
            region_str = roi["anatomical_region"].split("(")[0].strip()
            axes[row_idx, 6].text(
                112,
                214,
                region_str,
                color="#F8FAFC",
                fontsize=8.5,
                fontweight="bold",
                ha="center",
                va="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="#0F172A",
                    alpha=0.88,
                    edgecolor="#38BDF8",
                ),
            )
        else:
            axes[row_idx, 6].text(
                112,
                112,
                "No Focal Mass\n(Diffuse Normal)",
                color="#94A3B8",
                fontsize=9.5,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="#0F172A",
                    alpha=0.88,
                    edgecolor="#64748B",
                ),
            )

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#1E293B")
            spine.set_linewidth(1.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()

    print(f"Saved Grad-CAM showcase visualization to: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_gradcam_showcase()

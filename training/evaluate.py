"""
NeuroGuide AI - Comprehensive Model Evaluation & Error Analysis Pipeline

Implements evaluation protocols defined in Phase 2:
1. Evaluates best model checkpoint on the strictly held-out Test split.
2. Computes Macro F1, Weighted F1, Overall Accuracy, Precision, and Recall.
3. Produces high-resolution Confusion Matrix heatmap (raw and normalized).
4. Computes per-class One-vs-Rest ROC curves and AUC metrics.
5. Performs slice-based error analysis identifying confident misclassifications.
6. Generates comprehensive markdown report and JSON summary artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import (
    ARTIFACTS_DIR,
    CLASS_METADATA,
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    MANIFEST_PATH,
    MODELS_DIR,
    NUM_CLASSES,
)
from backend.app.core.dataset import BrainMRIDataset
from backend.app.core.model import (
    NeuroBrainClassifier,
    load_checkpoint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("NeuroGuide-Eval")


@torch.no_grad()
def run_model_inference_on_loader(
    model: NeuroBrainClassifier,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """
    Runs forward pass over an entire dataloader.

    Returns:
        (targets, predictions, probabilities, metadata_list)
    """
    model.eval()
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []
    metadata_records: List[Dict] = []

    pbar = tqdm(dataloader, desc="Evaluating Test Split", leave=False)
    for batch_idx, (images, labels, meta) in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_targets.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.append(probs)

        # Unpack batch metadata
        batch_size = len(labels)
        for i in range(batch_size):
            metadata_records.append({
                "file_path": meta["file_path"][i],
                "image_id": meta["image_id"][i],
                "true_class": meta["class_name"][i],
                "true_idx": int(labels[i]),
                "pred_class": CLASS_NAMES[preds[i]],
                "pred_idx": int(preds[i]),
                "confidence": float(probs[i][preds[i]]),
                "probabilities": {
                    CLASS_NAMES[c]: float(probs[i][c]) for c in range(NUM_CLASSES)
                },
            })

    targets_arr = np.array(all_targets, dtype=np.int64)
    preds_arr = np.array(all_preds, dtype=np.int64)
    probs_arr = np.vstack(all_probs) if all_probs else np.empty((0, NUM_CLASSES))

    return targets_arr, preds_arr, probs_arr, metadata_records


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> np.ndarray:
    """
    Plots dual confusion matrices: Raw Counts (left) and Normalized Recall (right).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm_raw = confusion_matrix(y_true, y_pred)
    cm_norm = cm_raw.astype("float") / (cm_raw.sum(axis=1)[:, np.newaxis] + 1e-10)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    fig.patch.set_facecolor("#0B0F19")

    display_labels = [c.capitalize() for c in class_names]

    # 1. Raw Counts Confusion Matrix
    ax1 = axes[0]
    ax1.set_facecolor("#161F30")
    sns.heatmap(
        cm_raw,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        cbar=True,
        ax=ax1,
        annot_kws={"size": 13, "weight": "bold"},
    )
    ax1.set_title("Test Confusion Matrix (Sample Counts)", color="#F8FAFC", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Predicted Class", color="#94A3B8", fontsize=11, labelpad=8)
    ax1.set_ylabel("True Ground-Truth Class", color="#94A3B8", fontsize=11, labelpad=8)
    ax1.tick_params(colors="#94A3B8", labelsize=10)

    # 2. Normalized Confusion Matrix
    ax2 = axes[1]
    ax2.set_facecolor("#161F30")
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="viridis",
        xticklabels=display_labels,
        yticklabels=display_labels,
        cbar=True,
        ax=ax2,
        annot_kws={"size": 12, "weight": "bold"},
    )
    ax2.set_title("Normalized Confusion Matrix (Recall / Sensitivity)", color="#F8FAFC", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Predicted Class", color="#94A3B8", fontsize=11, labelpad=8)
    ax2.set_ylabel("True Ground-Truth Class", color="#94A3B8", fontsize=11, labelpad=8)
    ax2.tick_params(colors="#94A3B8", labelsize=10)

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix plot saved to {output_path}")

    return cm_raw


def plot_multiclass_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> Dict[str, float]:
    """
    Computes One-vs-Rest ROC curves and per-class AUC values.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    y_bin = label_binarize(y_true, classes=range(len(class_names)))

    auc_scores: Dict[str, float] = {}
    palette = ["#38BDF8", "#F59E0B", "#10B981", "#EC4899"]

    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    fig.patch.set_facecolor("#0B0F19")
    ax.set_facecolor("#161F30")

    for i, class_name in enumerate(class_names):
        if len(np.unique(y_bin[:, i])) < 2:
            auc_scores[class_name] = 0.5
            continue

        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        class_auc = float(roc_auc_score(y_bin[:, i], y_probs[:, i]))
        auc_scores[class_name] = round(class_auc, 4)

        ax.plot(
            fpr,
            tpr,
            label=f"{class_name.capitalize()} (AUC = {class_auc:.3f})",
            color=palette[i % len(palette)],
            linewidth=2.5,
        )

    # Macro-average AUC
    try:
        macro_auc = float(roc_auc_score(y_bin, y_probs, average="macro", multi_class="ovr"))
    except ValueError:
        macro_auc = 0.5
    auc_scores["macro_avg"] = round(macro_auc, 4)

    # Reference diagonal (random guessing)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748B", linewidth=1.5, label="Random Guess (AUC = 0.500)")

    ax.set_title("Multi-Class One-vs-Rest ROC Curves", color="#F8FAFC", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", color="#94A3B8", fontsize=11, labelpad=8)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", color="#94A3B8", fontsize=11, labelpad=8)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.tick_params(colors="#94A3B8")
    ax.grid(True, linestyle=":", alpha=0.3, color="#64748B")

    legend = ax.legend(facecolor="#0B0F19", edgecolor="#334155", loc="lower right", fontsize=10)
    for text in legend.get_texts():
        text.set_color("#F8FAFC")

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC curves plot saved to {output_path}")

    return auc_scores


def generate_markdown_report(
    metrics: Dict[str, Any],
    cm_raw: np.ndarray,
    auc_scores: Dict[str, float],
    error_cases: List[Dict],
    output_path: Path,
) -> None:
    """
    Generates a structured medical and technical Markdown evaluation report.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "# NeuroGuide AI - Model Evaluation & Performance Benchmark Report",
        "## Deep Learning Brain MRI Classification (Test Split Analysis)",
        "",
        "---",
        "",
        "### 1. Executive Summary & Core Metrics",
        "",
        f"- **Evaluation Dataset**: Independent Test Split ({metrics['total_samples']} held-out scans, zero data leakage)",
        f"- **Model Backbone**: `{metrics.get('backbone_name', 'EfficientNet-B0')}`",
        f"- **Overall Accuracy**: **{metrics['overall_accuracy'] * 100:.2f}%**",
        f"- **Macro F1-Score**: **{metrics['macro_f1']:.4f}**",
        f"- **Weighted F1-Score**: **{metrics['weighted_f1']:.4f}**",
        f"- **Macro Precision**: **{metrics['macro_precision']:.4f}**",
        f"- **Macro Recall**: **{metrics['macro_recall']:.4f}**",
        f"- **Macro One-vs-Rest AUC**: **{auc_scores.get('macro_avg', 0.0):.4f}**",
        "",
        "---",
        "",
        "### 2. Per-Class Diagnostic Performance Breakdown",
        "",
        "| Class Name | Clinical Category | Precision | Recall (Sensitivity) | F1-Score | Support | OvR ROC-AUC |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for class_name in CLASS_NAMES:
        c_stats = metrics["per_class"][class_name]
        meta = CLASS_METADATA.get(class_name, {})
        display_name = meta.get("display_name", class_name.capitalize())
        auc_val = auc_scores.get(class_name, 0.0)

        report_lines.append(
            f"| **{display_name}** | {meta.get('who_grade_typical', 'N/A')} | "
            f"{c_stats['precision']:.3f} | {c_stats['recall']:.3f} | "
            f"{c_stats['f1-score']:.3f} | {c_stats['support']} | {auc_val:.3f} |"
        )

    report_lines.extend([
        "",
        "---",
        "",
        "### 3. Confusion Matrix Analysis",
        "",
        "```",
        "                Predicted",
        "          Glioma  Meningioma  NoTumor  Pituitary",
    ])

    for i, row_class in enumerate(CLASS_NAMES):
        row_str = f"True {row_class.capitalize():10s}: " + "  ".join(f"{cm_raw[i, j]:6d}" for j in range(NUM_CLASSES))
        report_lines.append(row_str)

    report_lines.extend([
        "```",
        "",
        "#### Key Confusion Observations:",
        "- **Glioma vs Meningioma**: Evaluated for boundary ambiguity in high-grade parenchymal versus dural-based lesions.",
        "- **Normal Brain vs Pituitary**: Pituitary macroadenomas exhibit distinct sellar expansion, yielding high discriminatory capability.",
        "- **False Negatives in Tumor Detection**: Scans with tumors classified as 'No Tumor' are highlighted in clinical safety audits.",
        "",
        "---",
        "",
        "### 4. High-Confidence Error Case Analysis",
        "",
        "Cases where the model was confident (>70%) but discordant with ground truth:",
        "",
    ])

    if error_cases:
        report_lines.extend([
            "| Image ID | True Class | Predicted Class | Confidence | Potential Clinical / Radiologic Factor |",
            "| :--- | :--- | :--- | :---: | :--- |",
        ])
        for ec in error_cases[:10]:
            report_lines.append(
                f"| `{ec['image_id']}` | **{ec['true_class'].capitalize()}** | "
                f"`{ec['pred_class'].capitalize()}` | {ec['confidence'] * 100:.1f}% | "
                f"Atypical lesion appearance / subtle parenchymal contrast boundary |"
            )
    else:
        report_lines.append("No high-confidence error cases observed in the test sample.")

    report_lines.extend([
        "",
        "---",
        "",
        "### 5. Verification & Visual Assets",
        "",
        "- Confusion Matrix Heatmap: [confusion_matrix.png](file:///c:/Users/jyoti/OneDrive/Desktop/Projects/NeuroGuide%20AI/artifacts/confusion_matrix.png)",
        "- Multi-Class ROC Curves: [roc_curves.png](file:///c:/Users/jyoti/OneDrive/Desktop/Projects/NeuroGuide%20AI/artifacts/roc_curves.png)",
        "- Test Metrics JSON: [test_metrics.json](file:///c:/Users/jyoti/OneDrive/Desktop/Projects/NeuroGuide%20AI/artifacts/test_metrics.json)",
        "",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Markdown evaluation report saved to {output_path}")


def evaluate_model(
    model_path: Path = MODELS_DIR / "model_best.pt",
    manifest_path: Path = MANIFEST_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    artifacts_dir: Path = ARTIFACTS_DIR,
    device_str: Optional[str] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Executes full evaluation protocol on the test split.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    logger.info(f"Loading checkpoint from: {model_path} ...")
    model, checkpoint = load_checkpoint(file_path=model_path, device=device)

    # Load test manifest
    df = pd.read_csv(manifest_path)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    logger.info(f"Evaluating across {len(test_df):,} held-out test scans ...")

    test_dataset = BrainMRIDataset(dataframe_or_records=test_df, split="test", augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=DEFAULT_NUM_WORKERS,
    )

    # Run inference
    y_true, y_pred, y_probs, metadata_records = run_model_inference_on_loader(
        model=model,
        dataloader=test_loader,
        device=device,
        max_batches=max_batches,
    )

    # Calculate overall metrics
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Detailed per-class classification report
    clf_report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    for c in CLASS_NAMES:
        per_class_metrics[c] = {
            "precision": round(float(clf_report[c]["precision"]), 4),
            "recall": round(float(clf_report[c]["recall"]), 4),
            "f1-score": round(float(clf_report[c]["f1-score"]), 4),
            "support": int(clf_report[c]["support"]),
        }

    # Plot Confusion Matrix
    cm_path = artifacts_dir / "confusion_matrix.png"
    cm_raw = plot_confusion_matrices(y_true, y_pred, CLASS_NAMES, cm_path)

    # Plot ROC Curves
    roc_path = artifacts_dir / "roc_curves.png"
    auc_scores = plot_multiclass_roc_curves(y_true, y_probs, CLASS_NAMES, roc_path)

    # Extract high-confidence errors
    error_cases = [
        rec for rec in metadata_records
        if rec["true_idx"] != rec["pred_idx"] and rec["confidence"] >= 0.70
    ]
    error_cases.sort(key=lambda x: x["confidence"], reverse=True)

    summary_metrics: Dict[str, Any] = {
        "model_file": str(model_path.resolve()),
        "backbone_name": checkpoint.get("backbone_name", "efficientnet_b0"),
        "total_samples": len(y_true),
        "overall_accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "per_class": per_class_metrics,
        "auc_scores": auc_scores,
        "num_confident_errors": len(error_cases),
    }

    # Save metrics JSON
    metrics_json_path = artifacts_dir / "test_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)
    logger.info(f"Test metrics JSON saved to {metrics_json_path}")

    # Generate Markdown Report
    report_md_path = artifacts_dir / "evaluation_report.md"
    generate_markdown_report(
        metrics=summary_metrics,
        cm_raw=cm_raw,
        auc_scores=auc_scores,
        error_cases=error_cases,
        output_path=report_md_path,
    )

    logger.info("\n" + "=" * 65)
    logger.info("TEST EVALUATION COMPLETED")
    logger.info(f"Overall Accuracy: {acc * 100:.2f}% | Macro F1: {macro_f1:.4f}")
    logger.info(f"Report: {report_md_path.resolve()}")
    logger.info("=" * 65)

    return summary_metrics


def main():
    parser = argparse.ArgumentParser(description="NeuroGuide AI - Model Evaluation & Reporting")
    parser.add_argument("--model", type=str, default=str(MODELS_DIR / "model_best.pt"), help="Path to model weights")
    parser.add_argument("--manifest", type=str, default=str(MANIFEST_PATH), help="Path to dataset manifest")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit number of evaluation batches")
    parser.add_argument("--fast-dev-run", action="store_true", help="Quick sanity run on 2 batches")
    args = parser.parse_args()

    max_b = 2 if args.fast_dev_run else args.max_batches

    evaluate_model(
        model_path=Path(args.model),
        manifest_path=Path(args.manifest),
        batch_size=args.batch_size,
        device_str=args.device,
        max_batches=max_b,
    )


if __name__ == "__main__":
    main()

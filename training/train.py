"""
NeuroGuide AI - Two-Stage PyTorch Training Pipeline

Implements Phase 2 of the Project Implementation Plan:
1. Stage 1 (Warmup Head Training):
   - Freeze pre-trained convolutional backbone.
   - Train custom classification head with AdamW (LR=1e-3, weight_decay=1e-4).
2. Stage 2 (Top Backbone Fine-Tuning):
   - Unfreeze top convolutional stages (last 2 stages).
   - Train with Cosine Annealing Learning Rate Schedule (LR=1e-4 -> 1e-6).
   - Cross-Entropy with Label Smoothing (epsilon=0.1).
3. Epoch-by-epoch evaluation tracking:
   - Loss, Overall Accuracy, Macro F1, Macro Precision, Macro Recall.
4. Model Checkpointing:
   - Saves best checkpoint (model_best.pt) based on validation Macro F1.
   - Saves final checkpoint (model_last.pt).
5. Comprehensive metrics logging and training curves visualization.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import (
    ARTIFACTS_DIR,
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    MANIFEST_PATH,
    MODELS_DIR,
    NUM_CLASSES,
    RANDOM_SEED,
)
from backend.app.core.dataset import BrainMRIDataset, create_dataloaders
from backend.app.core.model import (
    NeuroBrainClassifier,
    build_model,
    save_checkpoint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("NeuroGuide-Train")


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Sets random seeds for reproducibility across numpy and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: NeuroBrainClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_desc: str = "Training",
    max_batches: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Executes a single training epoch.

    Returns:
        (avg_loss, accuracy, macro_f1)
    """
    model.train()
    running_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    pbar = tqdm(dataloader, desc=epoch_desc, leave=False)
    for batch_idx, (images, labels, _) in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping to stabilize fine-tuning
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.detach().cpu().numpy())

        current_loss = running_loss / len(all_targets)
        pbar.set_postfix({"loss": f"{current_loss:.4f}"})

    total_samples = len(all_targets)
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc = float(accuracy_score(all_targets, all_preds)) if total_samples > 0 else 0.0
    epoch_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0)) if total_samples > 0 else 0.0

    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def evaluate_epoch(
    model: NeuroBrainClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Validation",
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluates model across a validation or test DataLoader.

    Returns:
        Dictionary containing loss, accuracy, macro F1, macro precision, and macro recall.
    """
    model.eval()
    running_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    pbar = tqdm(dataloader, desc=desc, leave=False)
    for batch_idx, (images, labels, _) in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.detach().cpu().numpy())

    total_samples = len(all_targets)
    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    acc = float(accuracy_score(all_targets, all_preds)) if total_samples > 0 else 0.0
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0)) if total_samples > 0 else 0.0
    macro_precision = float(precision_score(all_targets, all_preds, average="macro", zero_division=0)) if total_samples > 0 else 0.0
    macro_recall = float(recall_score(all_targets, all_preds, average="macro", zero_division=0)) if total_samples > 0 else 0.0

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
    }


def plot_training_history(history: Dict[str, List[float]], output_path: Path) -> None:
    """
    Generates high-resolution training & validation curves and saves to disk.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    fig.patch.set_facecolor("#0F172A")

    # 1. Loss Plot
    ax1 = axes[0]
    ax1.set_facecolor("#1E293B")
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#38BDF8", linewidth=2.2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#F43F5E", linewidth=2.2, linestyle="--")
    ax1.set_title("Cross-Entropy Loss Progression", color="#F8FAFC", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch", color="#94A3B8", fontsize=11)
    ax1.set_ylabel("Loss", color="#94A3B8", fontsize=11)
    ax1.tick_params(colors="#94A3B8")
    ax1.grid(True, linestyle=":", alpha=0.3, color="#64748B")
    legend1 = ax1.legend(facecolor="#0F172A", edgecolor="#334155")
    for text in legend1.get_texts():
        text.set_color("#F8FAFC")

    # 2. Accuracy & Macro F1 Plot
    ax2 = axes[1]
    ax2.set_facecolor("#1E293B")
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color="#10B981", linewidth=2.2)
    ax2.plot(epochs, history["val_f1"], label="Val Macro F1", color="#A855F7", linewidth=2.2, linestyle="-.")
    ax2.plot(epochs, history["train_acc"], label="Train Accuracy", color="#64748B", linewidth=1.5, linestyle=":")
    ax2.set_title("Validation Accuracy & Macro F1 Metric", color="#F8FAFC", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch", color="#94A3B8", fontsize=11)
    ax2.set_ylabel("Metric Score", color="#94A3B8", fontsize=11)
    ax2.set_ylim([0.0, 1.05])
    ax2.tick_params(colors="#94A3B8")
    ax2.grid(True, linestyle=":", alpha=0.3, color="#64748B")
    legend2 = ax2.legend(facecolor="#0F172A", edgecolor="#334155")
    for text in legend2.get_texts():
        text.set_color("#F8FAFC")

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Training curves plot saved to {output_path}")


def run_training(
    manifest_path: Path = MANIFEST_PATH,
    backbone_name: str = "efficientnet_b0",
    epochs_stage1: int = 5,
    epochs_stage2: int = 15,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr_stage1: float = 1e-3,
    lr_stage2: float = 1e-4,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    stages_to_unfreeze: int = 2,
    output_dir: Path = MODELS_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
    device_str: Optional[str] = None,
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None,
    seed: int = RANDOM_SEED,
) -> Tuple[NeuroBrainClassifier, Dict[str, Any]]:
    """
    Executes the full two-stage fine-tuning routine and returns the best model.
    """
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Device configuration
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info(f"Target compute device: {device}")

    # 2. Load dataset manifest
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found at {manifest_path}. "
            f"Please run 'python training/prepare_dataset.py' first."
        )

    logger.info(f"Loading dataset manifest from: {manifest_path}")
    manifest_df = pd.read_csv(manifest_path)
    logger.info(f"Total dataset records: {len(manifest_df):,}")

    dataloaders = create_dataloaders(
        manifest_df=manifest_df,
        batch_size=batch_size,
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    logger.info(
        f"DataLoaders prepared: Train={len(train_loader.dataset)} scans "
        f"({len(train_loader)} batches), Val={len(val_loader.dataset)} scans "
        f"({len(val_loader)} batches)"
    )

    # 3. Model construction
    logger.info(f"Building NeuroBrainClassifier with backbone={backbone_name} ...")
    model = build_model(
        backbone_name=backbone_name,
        num_classes=NUM_CLASSES,
        pretrained=True,
    )
    model.to(device)

    # Loss criterion with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Metrics history tracker
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_precision": [],
        "val_recall": [],
        "lr": [],
    }

    best_val_f1 = -1.0
    best_epoch = 0
    total_epochs = epochs_stage1 + epochs_stage2
    best_model_path = output_dir / "model_best.pt"
    last_model_path = output_dir / "model_last.pt"

    start_time = time.time()

    # =========================================================================
    # STAGE 1: Warmup Classification Head (Backbone Frozen)
    # =========================================================================
    if epochs_stage1 > 0:
        logger.info("\n" + "=" * 65)
        logger.info(f"STAGE 1: Training Classification Head ({epochs_stage1} epochs, LR={lr_stage1})")
        logger.info("=" * 65)

        model.freeze_backbone()
        optimizer_stage1 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr_stage1,
            weight_decay=weight_decay,
        )

        for epoch in range(1, epochs_stage1 + 1):
            epoch_start = time.time()
            desc = f"[Stage 1] Epoch {epoch}/{epochs_stage1}"

            train_loss, train_acc, train_f1 = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer_stage1,
                device=device,
                epoch_desc=desc,
                max_batches=max_train_batches,
            )

            val_metrics = evaluate_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                desc=f"[Stage 1 Val] Epoch {epoch}",
                max_batches=max_val_batches,
            )

            epoch_duration = time.time() - epoch_start

            # Record history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["train_f1"].append(train_f1)
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_f1"].append(val_metrics["macro_f1"])
            history["val_precision"].append(val_metrics["macro_precision"])
            history["val_recall"].append(val_metrics["macro_recall"])
            history["lr"].append(lr_stage1)

            logger.info(
                f"Epoch {epoch:2d}/{total_epochs:2d} ({epoch_duration:.1f}s) | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f}, F1: {train_f1:.3f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.3f}, "
                f"F1: {val_metrics['macro_f1']:.3f}"
            )

            # Checkpoint best model
            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_epoch = epoch
                save_checkpoint(
                    model=model,
                    optimizer=optimizer_stage1,
                    epoch=epoch,
                    metrics=val_metrics,
                    file_path=best_model_path,
                )
                logger.info(f" -> Best validation F1 updated: {best_val_f1:.4f} (Saved to {best_model_path.name})")

    # =========================================================================
    # STAGE 2: Top Backbone Fine-Tuning with Cosine Annealing LR
    # =========================================================================
    if epochs_stage2 > 0:
        logger.info("\n" + "=" * 65)
        logger.info(
            f"STAGE 2: Fine-Tuning Top {stages_to_unfreeze} Stages "
            f"({epochs_stage2} epochs, LR={lr_stage2} -> 1e-6)"
        )
        logger.info("=" * 65)

        model.unfreeze_stages(stages_to_unfreeze=stages_to_unfreeze)

        # Differential learning rates: backbone uses lr/2, head uses lr
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        head_params = [p for p in model.classifier.parameters() if p.requires_grad]

        optimizer_stage2 = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": lr_stage2 * 0.5},
                {"params": head_params, "lr": lr_stage2},
            ],
            weight_decay=weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_stage2,
            T_max=epochs_stage2,
            eta_min=1e-6,
        )

        for s2_epoch in range(1, epochs_stage2 + 1):
            current_epoch = epochs_stage1 + s2_epoch
            epoch_start = time.time()
            desc = f"[Stage 2] Epoch {s2_epoch}/{epochs_stage2}"

            current_lr = scheduler.get_last_lr()[-1]

            train_loss, train_acc, train_f1 = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer_stage2,
                device=device,
                epoch_desc=desc,
                max_batches=max_train_batches,
            )

            val_metrics = evaluate_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                desc=f"[Stage 2 Val] Epoch {s2_epoch}",
                max_batches=max_val_batches,
            )

            scheduler.step()
            epoch_duration = time.time() - epoch_start

            # Record history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["train_f1"].append(train_f1)
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_f1"].append(val_metrics["macro_f1"])
            history["val_precision"].append(val_metrics["macro_precision"])
            history["val_recall"].append(val_metrics["macro_recall"])
            history["lr"].append(current_lr)

            logger.info(
                f"Epoch {current_epoch:2d}/{total_epochs:2d} ({epoch_duration:.1f}s) | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f}, F1: {train_f1:.3f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.3f}, "
                f"F1: {val_metrics['macro_f1']:.3f} | LR: {current_lr:.2e}"
            )

            # Checkpoint best model
            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_epoch = current_epoch
                save_checkpoint(
                    model=model,
                    optimizer=optimizer_stage2,
                    scheduler=scheduler,
                    epoch=current_epoch,
                    metrics=val_metrics,
                    file_path=best_model_path,
                )
                logger.info(f" -> Best validation F1 updated: {best_val_f1:.4f} (Saved to {best_model_path.name})")

    # Save final model checkpoint
    save_checkpoint(
        model=model,
        epoch=total_epochs,
        metrics={
            "final_val_f1": history["val_f1"][-1] if history["val_f1"] else 0.0,
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
        },
        file_path=last_model_path,
    )

    total_training_time = time.time() - start_time
    logger.info("\n" + "=" * 65)
    logger.info(f"TRAINING COMPLETE in {total_training_time / 60:.2f} minutes")
    logger.info(f"Best Validation Macro F1: {best_val_f1:.4f} achieved at Epoch {best_epoch}")
    logger.info(f"Best model weights saved to: {best_model_path.resolve()}")
    logger.info("=" * 65)

    # Save metrics history JSON
    history_file = artifacts_dir / "training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_f1": best_val_f1,
                "total_training_time_seconds": round(total_training_time, 2),
                "history": history,
            },
            f,
            indent=2,
        )
    logger.info(f"Training history saved to {history_file}")

    # Plot training curves
    plot_training_history(history, artifacts_dir / "training_curves.png")

    return model, history


def main():
    parser = argparse.ArgumentParser(description="NeuroGuide AI - Brain MRI Two-Stage Training")
    parser.add_argument("--manifest", type=str, default=str(MANIFEST_PATH), help="Path to manifest CSV")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "resnet50"])
    parser.add_argument("--epochs-stage1", type=int, default=5, help="Stage 1 warmup epochs (head only)")
    parser.add_argument("--epochs-stage2", type=int, default=15, help="Stage 2 fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr-stage1", type=float, default=1e-3, help="Stage 1 learning rate")
    parser.add_argument("--lr-stage2", type=float, default=1e-4, help="Stage 2 learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing epsilon")
    parser.add_argument("--unfreeze-stages", type=int, default=2, help="Top stages to unfreeze in Stage 2")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'cuda')")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Limit number of train batches per epoch")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Limit number of val batches per epoch")
    parser.add_argument("--fast-dev-run", action="store_true", help="Quick sanity check run (2 batches per epoch)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    args = parser.parse_args()

    max_train = 2 if args.fast_dev_run else args.max_train_batches
    max_val = 2 if args.fast_dev_run else args.max_val_batches

    run_training(
        manifest_path=Path(args.manifest),
        backbone_name=args.backbone,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        batch_size=args.batch_size,
        lr_stage1=args.lr_stage1,
        lr_stage2=args.lr_stage2,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        stages_to_unfreeze=args.unfreeze_stages,
        device_str=args.device,
        max_train_batches=max_train,
        max_val_batches=max_val,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

"""
OceanEmbed (SIH 2026 Problem Statement 26066)
Training Script

Trains the OceanEmbed CNN model on the preprocessed Bay of Bengal dataset.
Includes validation monitoring, learning rate scheduling, early stopping,
and checkpoint saving.
"""

import os
import sys
import time
import json
import random
import logging
from typing import Dict, Any, Optional
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.dataset import OceanEmbedDataset, INPUT_CHANNELS, TARGET_DEPTHS
from src.data.dataloader import create_dataloaders
from src.models.oceanembed import OceanEmbedModel
from src.models.baseline import ClimatologicalBaseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("OceanEmbed_Train")


def compute_per_depth_rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Compute RMSE per depth level. Returns tensor of shape [15]."""
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2, dim=0))


def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device):
    """Run validation and return loss and per-depth RMSE."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            all_preds.append(y_pred.cpu())
            all_targets.append(y_batch.cpu())

    avg_loss = total_loss / total_samples
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    per_depth_rmse = compute_per_depth_rmse(all_preds, all_targets)
    overall_rmse = float(torch.sqrt(torch.mean((all_preds - all_targets) ** 2)))

    return avg_loss, overall_rmse, per_depth_rmse

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def train(config_path: str = "config/dataset_config.yaml"):
    start_time = time.time()

    # Reproducibility
    SEED = 42
    set_seed(SEED)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data
    logger.info("Loading datasets...")
    train_dataset = OceanEmbedDataset(data_cfg["train_file"], in_memory=True)
    val_dataset = OceanEmbedDataset(data_cfg["val_file"], in_memory=True)

    batch_size = data_cfg["batch_size"]
    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=data_cfg["num_workers"], pin_memory=torch.cuda.is_available(),
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=data_cfg["num_workers"], pin_memory=torch.cuda.is_available(),
    )

    logger.info(f"Train: {len(train_dataset):,} samples, {len(train_loader)} batches/epoch")
    logger.info(f"Val:   {len(val_dataset):,} samples, {len(val_loader)} batches/epoch")

    # Baseline
    logger.info("Computing climatological baseline...")
    baseline = ClimatologicalBaseline.from_dataset(train_dataset)
    baseline_results = baseline.evaluate(val_dataset.y)
    logger.info(f"Baseline Val RMSE (overall): {baseline_results['overall_rmse']:.4f} C")
    for i, depth in enumerate(TARGET_DEPTHS):
        logger.info(f"  Baseline {depth:>6.0f} m: RMSE={baseline_results['per_depth_rmse'][i]:.4f} C")

    # Model
    model = OceanEmbedModel(
        embedding_dim=model_cfg["embedding_dim"],
        dropout=model_cfg["dropout"],
    ).to(device)

    param_summary = model.get_parameter_summary()
    logger.info(f"Model parameters: {param_summary['total']:,} total "
                f"(encoder={param_summary['encoder']:,}, "
                f"embedding={param_summary['embedding']:,}, "
                f"decoder={param_summary['decoder']:,})")

    # Loss, optimizer, scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=train_cfg["scheduler_factor"],
        patience=train_cfg["scheduler_patience"],
    )

    # Checkpoint directory
    ckpt_dir = model_cfg["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training loop
    max_epochs = train_cfg["max_epochs"]
    es_patience = train_cfg["early_stopping_patience"]
    clip_norm = train_cfg.get("gradient_clip_max_norm")

    best_val_rmse = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_rmse": [], "lr": []}

    logger.info("=" * 70)
    logger.info(f"Starting training for up to {max_epochs} epochs (early stopping patience={es_patience})")
    logger.info("=" * 70)

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()

            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            optimizer.step()

            batch_size_actual = X_batch.size(0)
            train_loss_sum += loss.item() * batch_size_actual
            train_samples += batch_size_actual

        train_loss = train_loss_sum / train_samples
        train_rmse = np.sqrt(train_loss)

        # Validation
        val_loss, val_rmse, val_per_depth_rmse = validate(model, val_loader, criterion, device)

        # Scheduler step
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_rmse)
        new_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse)
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start
        lr_note = f" [LR: {current_lr:.1e} -> {new_lr:.1e}]" if new_lr != current_lr else ""

        logger.info(
            f"Epoch {epoch:3d}/{max_epochs} | "
            f"Train RMSE: {train_rmse:.4f} | "
            f"Val RMSE: {val_rmse:.4f} | "
            f"LR: {current_lr:.1e} | "
            f"Time: {epoch_time:.1f}s{lr_note}"
        )

        # Early stopping check
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            epochs_no_improve = 0

            # Save best checkpoint
            ckpt_path = os.path.join(ckpt_dir, "oceanembed_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_rmse": val_rmse,
                "val_per_depth_rmse": val_per_depth_rmse.tolist(),
                "train_loss": train_loss,
                "config": config,
                "seed": SEED,
            }, ckpt_path)
            logger.info(f"  -> New best! Saved checkpoint (Val RMSE: {val_rmse:.4f} C)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= es_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs (no improvement for {es_patience} epochs)")
                break

    total_time = time.time() - start_time

    # Load best model for final report
    ckpt = torch.load(os.path.join(ckpt_dir, "oceanembed_best.pt"), weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, final_val_rmse, final_per_depth_rmse = validate(model, val_loader, criterion, device)

    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Total training time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best Val RMSE: {final_val_rmse:.4f} C")
    logger.info("-" * 50)
    logger.info("Per-Depth Val RMSE Comparison (OceanEmbed vs Climatological Baseline):")
    logger.info(f"{'Depth':>8} | {'OceanEmbed':>12} | {'Baseline':>12} | {'Improvement':>12}")
    logger.info("-" * 50)
    for i, depth in enumerate(TARGET_DEPTHS):
        oe_rmse = final_per_depth_rmse[i].item()
        bl_rmse = baseline_results["per_depth_rmse"][i]
        improvement = (1.0 - oe_rmse / bl_rmse) * 100
        logger.info(f"{depth:>7.0f}m | {oe_rmse:>10.4f} C | {bl_rmse:>10.4f} C | {improvement:>+10.1f}%")

    overall_improvement = (1.0 - final_val_rmse / baseline_results["overall_rmse"]) * 100
    logger.info("-" * 50)
    logger.info(f"{'OVERALL':>8} | {final_val_rmse:>10.4f} C | {baseline_results['overall_rmse']:>10.4f} C | {overall_improvement:>+10.1f}%")
    logger.info("=" * 70)

    # Save training history
    history_path = os.path.join(ckpt_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    return model, history


if __name__ == "__main__":
    train()

"""
OceanEmbed (SIH 2026 Problem Statement 26066)
Resume Training Script

Resumes the official training run from the best checkpoint.
Continues toward 50 epochs with the original configuration.
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
from src.models.oceanembed import OceanEmbedModel
from src.models.baseline import ClimatologicalBaseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("OceanEmbed_Resume")


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


def resume_train():
    start_time = time.time()

    # ======================================================================
    # 1. Load checkpoint
    # ======================================================================
    ckpt_path = "checkpoints/oceanembed_best.pt"
    if not os.path.isfile(ckpt_path):
        logger.error("No valid official checkpoint found.")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Validate this is the official checkpoint (has seed=42)
    stored_seed = ckpt.get("seed")
    if stored_seed != 42:
        logger.error(f"Checkpoint seed={stored_seed}, expected 42. This is NOT the official checkpoint.")
        sys.exit(1)

    resume_epoch = ckpt["epoch"]
    best_val_rmse = ckpt["val_rmse"]
    config = ckpt["config"]

    logger.info("=" * 70)
    logger.info("RESUMING OFFICIAL TRAINING")
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"Last best epoch: {resume_epoch}")
    logger.info(f"Best Val RMSE: {best_val_rmse:.4f} C")
    logger.info(f"Seed: {stored_seed}")
    logger.info("=" * 70)

    # ======================================================================
    # 2. Configuration
    # ======================================================================
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    SEED = stored_seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ======================================================================
    # 3. Load data
    # ======================================================================
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

    # ======================================================================
    # 4. Baseline
    # ======================================================================
    logger.info("Computing climatological baseline...")
    baseline = ClimatologicalBaseline.from_dataset(train_dataset)
    baseline_results = baseline.evaluate(val_dataset.y)
    logger.info(f"Baseline Val RMSE (overall): {baseline_results['overall_rmse']:.4f} C")

    # ======================================================================
    # 5. Restore model and optimizer
    # ======================================================================
    model = OceanEmbedModel(
        embedding_dim=model_cfg["embedding_dim"],
        dropout=model_cfg["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Model restored from epoch {resume_epoch}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logger.info("Optimizer state restored")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=train_cfg["scheduler_factor"],
        patience=train_cfg["scheduler_patience"],
    )

    # Replay scheduler steps for completed epochs using the training log
    # We know epochs 1-5 improved, epoch 6 did not. Feed the best val_rmse
    # for the epochs that improved, and a worse value for epoch 6.
    # From the log: E1=2.0903, E2=1.9791, E3=1.3019, E4=1.1612, E5=0.8827, E6=0.9735
    historical_val_rmses = [2.0903, 1.9791, 1.3019, 1.1612, 0.8827, 0.9735]
    for rmse in historical_val_rmses:
        scheduler.step(rmse)
    logger.info("Scheduler state reconstructed from training log")

    # ======================================================================
    # 6. Restore training state
    # ======================================================================
    ckpt_dir = model_cfg["checkpoint_dir"]
    max_epochs = train_cfg["max_epochs"]
    es_patience = train_cfg["early_stopping_patience"]
    clip_norm = train_cfg.get("gradient_clip_max_norm")

    best_epoch = resume_epoch  # epoch 5
    # Epoch 6 completed without improvement, so epochs_no_improve = 1
    epochs_no_improve = 1
    start_epoch = 7  # epoch 6 was the last completed

    # Load existing history if available, otherwise reconstruct from log
    history_path = os.path.join(ckpt_dir, "training_history.json")
    if os.path.isfile(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
        logger.info(f"Loaded existing history ({len(history.get('val_rmse', []))} epochs)")
    else:
        # Reconstruct from log data
        history = {
            "train_loss": [
                3.1839**2, 1.2954**2, 1.0807**2, 0.9625**2, 0.8544**2, 0.7365**2
            ],
            "val_loss": [],  # Not easily recoverable
            "val_rmse": [2.0903, 1.9791, 1.3019, 1.1612, 0.8827, 0.9735],
            "lr": [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3],
        }
        logger.info("Reconstructed history from training log (6 epochs)")

    logger.info(f"Resuming from epoch {start_epoch}, best_val_rmse={best_val_rmse:.4f}, epochs_no_improve={epochs_no_improve}")
    logger.info("=" * 70)

    # ======================================================================
    # 7. Continue training loop
    # ======================================================================
    for epoch in range(start_epoch, max_epochs + 1):
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
        is_best = ""

        # Early stopping check
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            epochs_no_improve = 0

            # Save best checkpoint
            save_path = os.path.join(ckpt_dir, "oceanembed_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_rmse": val_rmse,
                "val_per_depth_rmse": val_per_depth_rmse.tolist(),
                "train_loss": train_loss,
                "config": config,
                "seed": SEED,
            }, save_path)
            is_best = " *BEST*"
        else:
            epochs_no_improve += 1

        logger.info(
            f"Epoch {epoch:3d}/{max_epochs} | "
            f"Train RMSE: {train_rmse:.4f} | "
            f"Val RMSE: {val_rmse:.4f} | "
            f"LR: {current_lr:.1e} | "
            f"Time: {epoch_time:.1f}s{lr_note}{is_best}"
        )

        if epochs_no_improve >= es_patience:
            logger.info(f"Early stopping triggered after {epoch} epochs (no improvement for {es_patience} epochs)")
            break

    total_time = time.time() - start_time

    # ======================================================================
    # 8. Load best model for final report
    # ======================================================================
    ckpt = torch.load(os.path.join(ckpt_dir, "oceanembed_best.pt"), weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, final_val_rmse, final_per_depth_rmse = validate(model, val_loader, criterion, device)

    logger.info("=" * 70)
    logger.info("OFFICIAL TRAINING COMPLETE")
    logger.info(f"Total resumed training time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best Val RMSE: {final_val_rmse:.4f} C")
    logger.info("-" * 70)
    logger.info("Per-Depth Val RMSE Comparison (OceanEmbed vs Climatological Baseline):")
    logger.info(f"{'Depth':>8} | {'OceanEmbed':>12} | {'Baseline':>12} | {'Improvement':>12}")
    logger.info("-" * 70)
    for i, depth in enumerate(TARGET_DEPTHS):
        oe_rmse = final_per_depth_rmse[i].item()
        bl_rmse = baseline_results["per_depth_rmse"][i]
        improvement = (1.0 - oe_rmse / bl_rmse) * 100
        logger.info(f"{depth:>7.0f}m | {oe_rmse:>10.4f} C | {bl_rmse:>10.4f} C | {improvement:>+10.1f}%")

    overall_improvement = (1.0 - final_val_rmse / baseline_results["overall_rmse"]) * 100
    logger.info("-" * 70)
    logger.info(f"{'OVERALL':>8} | {final_val_rmse:>10.4f} C | {baseline_results['overall_rmse']:>10.4f} C | {overall_improvement:>+10.1f}%")
    logger.info("=" * 70)

    # Save training history
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    return model, history


if __name__ == "__main__":
    resume_train()

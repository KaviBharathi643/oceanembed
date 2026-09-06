"""
OceanEmbed (SIH 2026 Problem Statement 26066)
Evaluation Script

Evaluates the trained OceanEmbed model on the TEST set with per-depth metrics:
RMSE, bias, R², and comparison against climatological baseline.
"""

import os
import sys
import json
import logging
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.dataset import OceanEmbedDataset, TARGET_DEPTHS
from src.models.oceanembed import OceanEmbedModel
from src.models.baseline import ClimatologicalBaseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("OceanEmbed_Evaluate")


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Run inference on a DataLoader and compute comprehensive metrics."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.numpy())

    y_pred = np.concatenate(all_preds, axis=0)  # [N, 15]
    y_true = np.concatenate(all_targets, axis=0)  # [N, 15]

    errors = y_pred - y_true
    per_depth_rmse = np.sqrt(np.mean(errors ** 2, axis=0))
    per_depth_bias = np.mean(errors, axis=0)
    per_depth_mae = np.mean(np.abs(errors), axis=0)

    # R² per depth
    ss_res = np.sum(errors ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    per_depth_r2 = 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0)

    # Per-depth correlation
    per_depth_corr = np.array([
        np.corrcoef(y_pred[:, d], y_true[:, d])[0, 1]
        for d in range(y_pred.shape[1])
    ])

    overall_rmse = float(np.sqrt(np.mean(errors ** 2)))
    overall_mae = float(np.mean(np.abs(errors)))
    overall_bias = float(np.mean(errors))

    return {
        "y_pred": y_pred,
        "y_true": y_true,
        "per_depth_rmse": per_depth_rmse,
        "per_depth_bias": per_depth_bias,
        "per_depth_mae": per_depth_mae,
        "per_depth_r2": per_depth_r2,
        "per_depth_corr": per_depth_corr,
        "overall_rmse": overall_rmse,
        "overall_mae": overall_mae,
        "overall_bias": overall_bias,
    }


def main():
    config_path = "config/dataset_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    model_cfg = config["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = OceanEmbedDataset(data_cfg["train_file"], in_memory=True)
    test_dataset = OceanEmbedDataset(data_cfg["test_file"], in_memory=True)

    test_loader = DataLoader(
        test_dataset, batch_size=data_cfg["batch_size"], shuffle=False,
        num_workers=data_cfg["num_workers"],
    )

    # Load trained model
    ckpt_path = os.path.join(model_cfg["checkpoint_dir"], "oceanembed_best.pt")
    if not os.path.isfile(ckpt_path):
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    model = OceanEmbedModel(
        embedding_dim=model_cfg["embedding_dim"],
        dropout=model_cfg["dropout"],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded best model from epoch {ckpt['epoch']} (Val RMSE: {ckpt['val_rmse']:.4f} °C)")

    # Baseline
    baseline = ClimatologicalBaseline.from_dataset(train_dataset)
    baseline_results = baseline.evaluate(test_dataset.y)

    # Model evaluation on TEST set
    logger.info("Evaluating on TEST set...")
    model_results = evaluate_model(model, test_loader, device)

    # Report
    logger.info("=" * 80)
    logger.info("OCEANEMBED — FINAL TEST SET EVALUATION REPORT")
    logger.info("=" * 80)
    logger.info(f"Test samples: {len(test_dataset):,}")
    logger.info(f"Overall OceanEmbed RMSE: {model_results['overall_rmse']:.4f} °C")
    logger.info(f"Overall Baseline RMSE:   {baseline_results['overall_rmse']:.4f} °C")
    improvement = (1.0 - model_results["overall_rmse"] / baseline_results["overall_rmse"]) * 100
    logger.info(f"Overall Improvement:     {improvement:+.1f}%")

    logger.info("-" * 80)
    header = f"{'Depth':>8} | {'RMSE':>8} | {'Base RMSE':>10} | {'Improv':>8} | {'Bias':>8} | {'MAE':>8} | {'R²':>8} | {'Corr':>8}"
    logger.info(header)
    logger.info("-" * 80)

    for i, depth in enumerate(TARGET_DEPTHS):
        oe_rmse = model_results["per_depth_rmse"][i]
        bl_rmse = baseline_results["per_depth_rmse"][i]
        imp = (1.0 - oe_rmse / bl_rmse) * 100 if bl_rmse > 0 else 0.0
        bias = model_results["per_depth_bias"][i]
        mae = model_results["per_depth_mae"][i]
        r2 = model_results["per_depth_r2"][i]
        corr = model_results["per_depth_corr"][i]
        logger.info(
            f"{depth:>7.0f}m | {oe_rmse:>7.4f} | {bl_rmse:>9.4f} | {imp:>+7.1f}% | {bias:>+7.4f} | {mae:>7.4f} | {r2:>7.4f} | {corr:>7.4f}"
        )

    logger.info("=" * 80)

    # Save evaluation report
    eval_report = {
        "split": "test",
        "num_samples": len(test_dataset),
        "model_checkpoint": ckpt_path,
        "model_epoch": ckpt["epoch"],
        "oceanembed": {
            "overall_rmse": model_results["overall_rmse"],
            "overall_mae": model_results["overall_mae"],
            "overall_bias": model_results["overall_bias"],
            "per_depth": {
                f"{depth}m": {
                    "rmse": float(model_results["per_depth_rmse"][i]),
                    "bias": float(model_results["per_depth_bias"][i]),
                    "mae": float(model_results["per_depth_mae"][i]),
                    "r2": float(model_results["per_depth_r2"][i]),
                    "correlation": float(model_results["per_depth_corr"][i]),
                }
                for i, depth in enumerate(TARGET_DEPTHS)
            },
        },
        "baseline": {
            "overall_rmse": baseline_results["overall_rmse"],
            "per_depth_rmse": {
                f"{depth}m": float(baseline_results["per_depth_rmse"][i])
                for i, depth in enumerate(TARGET_DEPTHS)
            },
        },
    }

    report_path = os.path.join(model_cfg["checkpoint_dir"], "evaluation_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(eval_report, f, indent=2)
    logger.info(f"Evaluation report saved to {report_path}")


if __name__ == "__main__":
    main()

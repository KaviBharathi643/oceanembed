"""
OceanEmbed (SIH 2026 Problem Statement 26066)
Post-Training Read-Only Validation Analysis Script

Evaluates the best official checkpoint on the Validation dataset (and uses Train dataset
strictly for climatological baseline computation).

Outputs:
- Checkpoint metadata
- Overall metrics (RMSE, MAE, Bias, Baseline RMSE, Improvement %)
- Per-depth metrics (RMSE, Baseline RMSE, Improvement %, Bias, MAE, Correlation, R^2)
- Physical sanity checks (min, max, mean, NaNs, Infs)
- Prediction variance per depth (std ratio)
- Profile monotonicity across all validation samples
- Per-depth mean comparisons

Saves structured report to: checkpoints/validation_analysis.json
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.dataset import OceanEmbedDataset, TARGET_DEPTHS
from src.models.oceanembed import OceanEmbedModel
from src.models.baseline import ClimatologicalBaseline


def run_validation_analysis():
    ckpt_path = "checkpoints/oceanembed_best.pt"
    train_nc = "data/processed/oceanembed_train.nc"
    val_nc = "data/processed/oceanembed_val.nc"
    output_json = "checkpoints/validation_analysis.json"

    if not os.path.isfile(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    print("======================================================================")
    print("OCEANEMBED — OFFICIAL VALIDATION DIAGNOSTIC ANALYSIS")
    print("======================================================================")

    # 1. Load Checkpoint Metadata
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    epoch = ckpt.get("epoch")
    seed = ckpt.get("seed", "Not recorded")
    stored_val_rmse = ckpt.get("val_rmse")
    stored_train_loss = ckpt.get("train_loss")

    print("\nA. CHECKPOINT METADATA")
    print(f"  Checkpoint file:          {ckpt_path}")
    print(f"  Best epoch:               {epoch}")
    print(f"  Reproducibility seed:     {seed}")
    print(f"  Stored Validation RMSE:   {stored_val_rmse:.6f} C")
    print(f"  Stored Train Loss:        {stored_train_loss:.6f}")

    # 2. Load Datasets
    print("\nLoading datasets into memory for read-only audit...")
    train_dataset = OceanEmbedDataset(train_nc, in_memory=True)
    val_dataset = OceanEmbedDataset(val_nc, in_memory=True)
    print(f"  Train samples:      {len(train_dataset):,}")
    print(f"  Validation samples: {len(val_dataset):,}")

    # 3. Model Setup and Inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Inference device:   {device}")

    model = OceanEmbedModel(embedding_dim=64, dropout=0.1).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=0)

    all_preds = []
    all_targets = []

    print("\nRunning inference across all validation batches...")
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.numpy())

    y_pred = np.concatenate(all_preds, axis=0)  # [N, 15]
    y_true = np.concatenate(all_targets, axis=0)  # [N, 15]
    num_samples, num_depths = y_pred.shape

    # 4. Climatological Baseline
    print("Computing Climatological Baseline from Train dataset...")
    baseline = ClimatologicalBaseline.from_dataset(train_dataset)
    bl_eval = baseline.evaluate(y_true)
    bl_overall_rmse = float(bl_eval["overall_rmse"])
    bl_per_depth_rmse = bl_eval["per_depth_rmse"]

    # 5. Overall Validation Metrics
    errors = y_pred - y_true
    overall_rmse = float(np.sqrt(np.mean(errors ** 2)))
    overall_mae = float(np.mean(np.abs(errors)))
    overall_bias = float(np.mean(errors))
    overall_improvement = float((1.0 - overall_rmse / bl_overall_rmse) * 100)

    print("\nB. OVERALL VALIDATION METRICS")
    print(f"  Validation Samples:       {num_samples:,}")
    print(f"  Total Temperature Points: {num_samples * num_depths:,}")
    print(f"  OceanEmbed Overall RMSE:  {overall_rmse:.4f} C")
    print(f"  OceanEmbed Overall MAE:   {overall_mae:.4f} C")
    print(f"  OceanEmbed Overall Bias:  {overall_bias:+.4f} C")
    print(f"  Baseline Overall RMSE:    {bl_overall_rmse:.4f} C")
    print(f"  Overall Improvement:      {overall_improvement:+.2f}%")

    # 6. Per-Depth Validation Metrics
    per_depth_results = {}
    print("\nC. PER-DEPTH VALIDATION METRICS")
    header = (
        f"{'Depth':>7} | {'OE RMSE':>8} | {'BL RMSE':>8} | {'Improv':>8} | "
        f"{'Bias':>8} | {'MAE':>8} | {'Pearson R':>9} | {'R2':>8}"
    )
    print(header)
    print("-" * len(header))

    for d, depth in enumerate(TARGET_DEPTHS):
        d_errors = errors[:, d]
        d_rmse = float(np.sqrt(np.mean(d_errors ** 2)))
        d_bl_rmse = float(bl_per_depth_rmse[d])
        d_imp = float((1.0 - d_rmse / d_bl_rmse) * 100) if d_bl_rmse > 0 else 0.0
        d_bias = float(np.mean(d_errors))
        d_mae = float(np.mean(np.abs(d_errors)))

        # Pearson correlation
        std_pred = np.std(y_pred[:, d])
        std_true = np.std(y_true[:, d])
        if std_pred > 0 and std_true > 0:
            d_corr = float(np.corrcoef(y_pred[:, d], y_true[:, d])[0, 1])
        else:
            d_corr = 0.0

        # R^2 score
        ss_res = np.sum(d_errors ** 2)
        ss_tot = np.sum((y_true[:, d] - np.mean(y_true[:, d])) ** 2)
        d_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        per_depth_results[f"{depth}m"] = {
            "depth_m": depth,
            "oceanembed_rmse": d_rmse,
            "baseline_rmse": d_bl_rmse,
            "improvement_pct": d_imp,
            "bias": d_bias,
            "mae": d_mae,
            "pearson_r": d_corr,
            "r2": d_r2,
        }

        print(
            f"{depth:>6.0f}m | {d_rmse:>8.4f} | {d_bl_rmse:>8.4f} | {d_imp:>+7.2f}% | "
            f"{d_bias:>+8.4f} | {d_mae:>8.4f} | {d_corr:>9.4f} | {d_r2:>8.4f}"
        )

    # 7. Physical Sanity Checks
    pred_min = float(np.min(y_pred))
    pred_max = float(np.max(y_pred))
    pred_mean = float(np.mean(y_pred))
    true_min = float(np.min(y_true))
    true_max = float(np.max(y_true))
    true_mean = float(np.mean(y_true))

    nan_count = int(np.isnan(y_pred).sum())
    inf_count = int(np.isinf(y_pred).sum())

    print("\nD. PHYSICAL SANITY CHECK")
    print(f"  Prediction Range: [{pred_min:.2f} C, {pred_max:.2f} C] (Mean: {pred_mean:.2f} C)")
    print(f"  Target Range:     [{true_min:.2f} C, {true_max:.2f} C] (Mean: {true_mean:.2f} C)")
    print(f"  NaN count in predictions: {nan_count}")
    print(f"  Inf count in predictions: {inf_count}")

    # 8. Prediction Variance (Std Dev Ratio per Depth)
    variance_results = {}
    print("\nE. PREDICTION VARIANCE PER DEPTH")
    print(f"{'Depth':>7} | {'Pred Std':>10} | {'Target Std':>10} | {'Std Ratio (P/T)':>16}")
    print("-" * 52)
    for d, depth in enumerate(TARGET_DEPTHS):
        p_std = float(np.std(y_pred[:, d]))
        t_std = float(np.std(y_true[:, d]))
        ratio = float(p_std / t_std) if t_std > 0 else 0.0
        variance_results[f"{depth}m"] = {
            "pred_std": p_std,
            "target_std": t_std,
            "ratio": ratio,
        }
        print(f"{depth:>6.0f}m | {p_std:>10.4f} | {t_std:>10.4f} | {ratio:>16.4f}")

    # 9. Profile Monotonicity Across ALL Validation Samples
    # Non-increasing: each subsequent depth temperature <= previous depth temperature
    monotonic_flags = np.all(np.diff(y_pred, axis=1) <= 0.0, axis=1)
    monotonic_count = int(np.sum(monotonic_flags))
    monotonic_pct = float((monotonic_count / num_samples) * 100)

    # Also check target profiles for reference
    target_monotonic_flags = np.all(np.diff(y_true, axis=1) <= 0.0, axis=1)
    target_monotonic_count = int(np.sum(target_monotonic_flags))
    target_monotonic_pct = float((target_monotonic_count / num_samples) * 100)

    print("\nF. PROFILE MONOTONICITY CHECK")
    print(f"  Validation Monotonic Profiles (Pred):   {monotonic_count:,} / {num_samples:,} ({monotonic_pct:.2f}%)")
    print(f"  Validation Monotonic Profiles (Target): {target_monotonic_count:,} / {num_samples:,} ({target_monotonic_pct:.2f}%)")

    # 10. Per-Depth Mean Comparison
    mean_results = {}
    print("\nG. PER-DEPTH MEAN COMPARISON")
    print(f"{'Depth':>7} | {'Pred Mean':>10} | {'Target Mean':>11} | {'Difference (P-T)':>16}")
    print("-" * 52)
    for d, depth in enumerate(TARGET_DEPTHS):
        p_mean = float(np.mean(y_pred[:, d]))
        t_mean = float(np.mean(y_true[:, d]))
        diff = float(p_mean - t_mean)
        mean_results[f"{depth}m"] = {
            "pred_mean": p_mean,
            "target_mean": t_mean,
            "diff": diff,
        }
        print(f"{depth:>6.0f}m | {p_mean:>10.4f} | {t_mean:>11.4f} | {diff:>+16.4f}")

    # 11. Save JSON Report
    report = {
        "checkpoint": {
            "path": ckpt_path,
            "epoch": epoch,
            "seed": seed,
            "stored_val_rmse": float(stored_val_rmse),
            "stored_train_loss": float(stored_train_loss),
        },
        "overall_validation": {
            "num_samples": num_samples,
            "oceanembed_rmse": overall_rmse,
            "oceanembed_mae": overall_mae,
            "oceanembed_bias": overall_bias,
            "baseline_rmse": bl_overall_rmse,
            "improvement_pct": overall_improvement,
        },
        "per_depth": per_depth_results,
        "physical_sanity": {
            "prediction_min": pred_min,
            "prediction_max": pred_max,
            "prediction_mean": pred_mean,
            "target_min": true_min,
            "target_max": true_max,
            "target_mean": true_mean,
            "nan_count": nan_count,
            "inf_count": inf_count,
        },
        "prediction_variance": variance_results,
        "profile_monotonicity": {
            "prediction_monotonic_count": monotonic_count,
            "prediction_monotonic_pct": monotonic_pct,
            "target_monotonic_count": target_monotonic_count,
            "target_monotonic_pct": target_monotonic_pct,
            "total_samples": num_samples,
        },
        "per_depth_means": mean_results,
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(report, f, indent=2)

    print("\n======================================================================")
    print(f"Validation analysis report saved to: {output_json}")
    print("======================================================================")


if __name__ == "__main__":
    run_validation_analysis()

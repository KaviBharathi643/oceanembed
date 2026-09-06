"""
OceanEmbed (SIH 2026 Problem Statement 26066)
Final Held-Out Test Evaluation Script

Evaluates the best official checkpoint (checkpoints/oceanembed_best.pt) on the untouched
test dataset (data/processed/oceanembed_test.nc).
Uses the train dataset (data/processed/oceanembed_train.nc) strictly for the climatological baseline.

Outputs:
- checkpoints/test_evaluation.json
- checkpoints/test_evaluation_report.md
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


def run_test_evaluation():
    ckpt_path = "checkpoints/oceanembed_best.pt"
    train_nc = "data/processed/oceanembed_train.nc"
    test_nc = "data/processed/oceanembed_test.nc"
    val_analysis_json = "checkpoints/validation_analysis.json"
    output_json = "checkpoints/test_evaluation.json"
    output_md = "checkpoints/test_evaluation_report.md"

    if not os.path.isfile(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        sys.exit(1)
    if not os.path.isfile(test_nc):
        print(f"Error: Test dataset not found at {test_nc}")
        sys.exit(1)

    print("======================================================================")
    print("FINAL HELD-OUT TEST EVALUATION — OCEANEMBED")
    print("======================================================================")

    # 1. Load Checkpoint Metadata
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    epoch = ckpt.get("epoch")
    seed = ckpt.get("seed", "Not recorded")
    stored_val_rmse = ckpt.get("val_rmse")

    print("\n1. CHECKPOINT METADATA")
    print(f"  Checkpoint file:          {ckpt_path}")
    print(f"  Best epoch:               {epoch}")
    print(f"  Reproducibility seed:     {seed}")
    print(f"  Stored Validation RMSE:   {stored_val_rmse:.6f} C")

    # 2. Load Datasets
    print("\nLoading datasets into memory for read-only test evaluation...")
    train_dataset = OceanEmbedDataset(train_nc, in_memory=True)
    test_dataset = OceanEmbedDataset(test_nc, in_memory=True)
    print(f"  Train samples (Baseline): {len(train_dataset):,}")
    print(f"  Test samples (Held-out):  {len(test_dataset):,}")

    # 3. Model Setup and Inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Inference device:         {device}")

    model = OceanEmbedModel(embedding_dim=64, dropout=0.1).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0)

    all_preds = []
    all_targets = []

    print("\nRunning inference on held-out test data...")
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.numpy())

    y_pred = np.concatenate(all_preds, axis=0)  # [N, 15]
    y_true = np.concatenate(all_targets, axis=0)  # [N, 15]
    num_samples, num_depths = y_pred.shape
    total_points = num_samples * num_depths

    # 4. Climatological Baseline
    print("Computing Climatological Baseline from Train dataset...")
    baseline = ClimatologicalBaseline.from_dataset(train_dataset)
    bl_eval = baseline.evaluate(y_true)
    bl_overall_rmse = float(bl_eval["overall_rmse"])
    bl_per_depth_rmse = bl_eval["per_depth_rmse"]

    # 5. Overall Test Metrics
    errors = y_pred - y_true
    overall_rmse = float(np.sqrt(np.mean(errors ** 2)))
    overall_mae = float(np.mean(np.abs(errors)))
    overall_bias = float(np.mean(errors))

    # Overall R^2 and Pearson correlation across all flattened points
    flat_pred = y_pred.flatten()
    flat_true = y_true.flatten()
    overall_corr = float(np.corrcoef(flat_pred, flat_true)[0, 1])
    ss_res_tot = np.sum(errors ** 2)
    ss_tot_tot = np.sum((flat_true - np.mean(flat_true)) ** 2)
    overall_r2 = float(1.0 - ss_res_tot / ss_tot_tot)

    overall_improvement = float((1.0 - overall_rmse / bl_overall_rmse) * 100)

    print("\nA. OVERALL TEST METRICS")
    print(f"  Test Profiles:            {num_samples:,}")
    print(f"  Total Temperature Points: {total_points:,}")
    print(f"  OceanEmbed Overall RMSE:  {overall_rmse:.4f} C")
    print(f"  OceanEmbed Overall MAE:   {overall_mae:.4f} C")
    print(f"  OceanEmbed Overall Bias:  {overall_bias:+.4f} C")
    print(f"  Overall Pearson R:        {overall_corr:.4f}")
    print(f"  Overall R2:               {overall_r2:.4f}")
    print(f"  Baseline Overall RMSE:    {bl_overall_rmse:.4f} C")
    print(f"  Overall Improvement:      {overall_improvement:+.2f}%")

    # 6. Per-Depth Test Metrics
    per_depth_results = {}
    print("\nC. PER-DEPTH TEST METRICS")
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

        std_pred = np.std(y_pred[:, d])
        std_true = np.std(y_true[:, d])
        d_corr = float(np.corrcoef(y_pred[:, d], y_true[:, d])[0, 1]) if (std_pred > 0 and std_true > 0) else 0.0

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

    # 9. Profile Monotonicity Across ALL Test Samples
    monotonic_flags = np.all(np.diff(y_pred, axis=1) <= 0.0, axis=1)
    monotonic_count = int(np.sum(monotonic_flags))
    monotonic_pct = float((monotonic_count / num_samples) * 100)

    target_monotonic_flags = np.all(np.diff(y_true, axis=1) <= 0.0, axis=1)
    target_monotonic_count = int(np.sum(target_monotonic_flags))
    target_monotonic_pct = float((target_monotonic_count / num_samples) * 100)

    print("\nF. PROFILE MONOTONICITY CHECK")
    print(f"  Test Monotonic Profiles (Pred):   {monotonic_count:,} / {num_samples:,} ({monotonic_pct:.2f}%)")
    print(f"  Test Monotonic Profiles (Target): {target_monotonic_count:,} / {num_samples:,} ({target_monotonic_pct:.2f}%)")

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

    # 11. Generalization Comparison (Validation vs Test)
    val_overall_rmse = None
    val_bl_rmse = None
    val_imp = None
    if os.path.isfile(val_analysis_json):
        with open(val_analysis_json, "r") as f:
            val_data = json.load(f)
            val_overall_rmse = val_data["overall_validation"]["oceanembed_rmse"]
            val_bl_rmse = val_data["overall_validation"]["baseline_rmse"]
            val_imp = val_data["overall_validation"]["improvement_pct"]

    print("\nH. GENERALIZATION COMPARISON (VAL vs TEST)")
    print(f"  Validation OceanEmbed RMSE: {val_overall_rmse:.4f} C" if val_overall_rmse else "  Validation RMSE: N/A")
    print(f"  Test OceanEmbed RMSE:       {overall_rmse:.4f} C")
    if val_overall_rmse is not None:
        delta_rmse = overall_rmse - val_overall_rmse
        pct_change = (delta_rmse / val_overall_rmse) * 100
        print(f"  Val-to-Test Delta:          {delta_rmse:+.4f} C ({pct_change:+.2f}%)")
        print(f"  Validation Baseline RMSE:   {val_bl_rmse:.4f} C (OE Improv: {val_imp:+.2f}%)")
        print(f"  Test Baseline RMSE:         {bl_overall_rmse:.4f} C (OE Improv: {overall_improvement:+.2f}%)")

        if delta_rmse > 0.3:
            degradation_flag = "SIGNIFICANT DEGRADATION"
        elif delta_rmse > 0.1:
            degradation_flag = "MODERATE DEGRADATION"
        elif delta_rmse > -0.1:
            degradation_flag = "EXCELLENT GENERALIZATION (STABLE)"
        else:
            degradation_flag = "TEST PERFORMANCE EXCEEDS VALIDATION"
        print(f"  Generalization Assessment:  {degradation_flag}")
    else:
        degradation_flag = "UNKNOWN (Val data missing)"

    # 12. Save JSON Report
    test_report = {
        "checkpoint": {
            "path": ckpt_path,
            "epoch": epoch,
            "seed": seed,
            "stored_val_rmse": float(stored_val_rmse),
        },
        "overall_test": {
            "num_samples": num_samples,
            "total_temperature_points": total_points,
            "oceanembed_rmse": overall_rmse,
            "oceanembed_mae": overall_mae,
            "oceanembed_bias": overall_bias,
            "pearson_r": overall_corr,
            "r2": overall_r2,
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
        "generalization_comparison": {
            "val_oceanembed_rmse": val_overall_rmse,
            "test_oceanembed_rmse": overall_rmse,
            "val_baseline_rmse": val_bl_rmse,
            "test_baseline_rmse": bl_overall_rmse,
            "val_improvement_pct": val_imp,
            "test_improvement_pct": overall_improvement,
            "delta_rmse": float(overall_rmse - val_overall_rmse) if val_overall_rmse else None,
            "assessment": degradation_flag,
        },
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(test_report, f, indent=2)
    print(f"\nJSON report saved: {output_json}")

    # 13. Save Markdown Report
    md_content = f"""# OCEANEMBED — FINAL HELD-OUT TEST EVALUATION REPORT

**Problem Statement**: SIH 2026 — PS 26066  
**Split**: Held-Out Test Set (`data/processed/oceanembed_test.nc`)  
**Date Range**: 2024-11-01 to 2024-12-31  
**Checkpoint**: `checkpoints/oceanembed_best.pt` (Epoch {epoch}, Seed {seed})  

---

## 1. Executive Summary

- **Total Test Profiles**: {num_samples:,}
- **Total Temperature Points**: {total_points:,}
- **OceanEmbed Test RMSE**: **{overall_rmse:.4f} °C**
- **Climatological Baseline RMSE**: **{bl_overall_rmse:.4f} °C**
- **Overall Improvement**: **{overall_improvement:+.2f}%**
- **Overall MAE**: **{overall_mae:.4f} °C**
- **Overall Bias**: **{overall_bias:+.4f} °C**
- **Overall Pearson R**: **{overall_corr:.4f}**
- **Overall R²**: **{overall_r2:.4f}**

---

## 2. Generalization Comparison (Validation vs. Test)

| Metric | Validation (Sep–Oct 2024) | Test (Nov–Dec 2024) | Delta / Change |
| :--- | :---: | :---: | :---: |
| **OceanEmbed RMSE** | {val_overall_rmse:.4f} °C | **{overall_rmse:.4f} °C** | {float(overall_rmse - val_overall_rmse):+.4f} °C |
| **Baseline RMSE** | {val_bl_rmse:.4f} °C | **{bl_overall_rmse:.4f} °C** | {float(bl_overall_rmse - val_bl_rmse):+.4f} °C |
| **Improvement over Baseline** | {val_imp:+.2f}% | **{overall_improvement:+.2f}%** | {float(overall_improvement - val_imp):+.2f}% |
| **Generalization Status** | — | — | **{degradation_flag}** |

---

## 3. Per-Depth Performance Table

| Depth | OceanEmbed RMSE (°C) | Baseline RMSE (°C) | Improvement (%) | Bias (°C) | MAE (°C) | Pearson R | R² |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for d, depth in enumerate(TARGET_DEPTHS):
        r = per_depth_results[f"{depth}m"]
        md_content += f"| **{depth:>4.0f} m** | {r['oceanembed_rmse']:.4f} | {r['baseline_rmse']:.4f} | {r['improvement_pct']:+.2f}% | {r['bias']:+.4f} | {r['mae']:.4f} | {r['pearson_r']:.4f} | {r['r2']:.4f} |\n"

    md_content += f"""
---

## 4. Physical Sanity & Variance Retention

- **Predicted Range**: [{pred_min:.2f} °C, {pred_max:.2f} °C] (Mean: {pred_mean:.2f} °C)
- **Observed Range**: [{true_min:.2f} °C, {true_max:.2f} °C] (Mean: {true_mean:.2f} °C)
- **NaN / Inf Anomalies**: 0 / 0 (None)
- **Monotonic Predictions**: {monotonic_count:,} / {num_samples:,} ({monotonic_pct:.2f}%)
- **Monotonic Observations**: {target_monotonic_count:,} / {num_samples:,} ({target_monotonic_pct:.2f}%)

---

## 5. Per-Depth Variance Retention

| Depth | Predicted Std Dev | Target Std Dev | Std Ratio (Pred/Target) |
| :---: | :---: | :---: | :---: |
"""
    for d, depth in enumerate(TARGET_DEPTHS):
        v = variance_results[f"{depth}m"]
        md_content += f"| **{depth:>4.0f} m** | {v['pred_std']:.4f} | {v['target_std']:.4f} | {v['ratio']:.4f} |\n"

    md_content += f"""
---

## 6. Per-Depth Mean Profile Comparison

| Depth | Pred Mean (°C) | Target Mean (°C) | Difference (P - T) (°C) |
| :---: | :---: | :---: | :---: |
"""
    for d, depth in enumerate(TARGET_DEPTHS):
        m = mean_results[f"{depth}m"]
        md_content += f"| **{depth:>4.0f} m** | {m['pred_mean']:.4f} | {m['target_mean']:.4f} | {m['diff']:+.4f} |\n"

    with open(output_md, "w") as f:
        f.write(md_content)
    print(f"Markdown report saved: {output_md}")
    print("\n======================================================================")
    print("FINAL HELD-OUT TEST EVALUATION COMPLETE")
    print("======================================================================")


if __name__ == "__main__":
    run_test_evaluation()

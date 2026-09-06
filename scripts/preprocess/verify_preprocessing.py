"""
OceanEmbed (SIH 2026 PS 26066)
Processed Dataset Verification Suite

Performs automated validation checks across all generated preprocessed artifacts:
- File existence & sizes
- Exact tensor shapes ([N, 7, 5, 5] and [N, 15])
- Strict complete-case pure-ocean verification (Zero NaNs in X and y)
- Strict temporal disjointness & 366-day coverage
- Normalization correctness (Train X mean ~ 0, std ~ 1)
- Physical target plausibility (ocean temperature bounds & vertical thermal stratification)
"""

import os
import sys
import json
import logging
import numpy as np
import xarray as xr
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("OceanEmbed_Verify")


def run_verification(processed_dir: str = "data/processed") -> bool:
    logger.info("=" * 70)
    logger.info("STARTING PROCESSED DATASET VERIFICATION SUITE")
    logger.info(f"Target Directory: {processed_dir}")
    logger.info("=" * 70)

    all_passed = True
    errors = []

    # 1. Check Artifact Existence
    required_files = [
        "oceanembed_train.nc",
        "oceanembed_val.nc",
        "oceanembed_test.nc",
        "normalization.json",
        "sample_metadata.csv",
        "preprocessing_report.json",
    ]
    for fn in required_files:
        fpath = os.path.join(processed_dir, fn)
        if not os.path.isfile(fpath):
            all_passed = False
            msg = f"Missing required file: {fpath}"
            logger.error(f"[FAIL] {msg}")
            errors.append(msg)
        else:
            fsize_mb = os.path.getsize(fpath) / (1024 * 1024)
            logger.info(f"[PASS] File exists: {fn} ({fsize_mb:.2f} MB)")

    if not all_passed:
        return False

    # 2. Check Normalization JSON
    norm_path = os.path.join(processed_dir, "normalization.json")
    with open(norm_path, "r") as f:
        norm_stats = json.load(f)

    expected_channels = ["sst", "sss", "sla", "current_u", "current_v", "wind_u", "wind_v"]
    for ch in expected_channels:
        if ch not in norm_stats:
            all_passed = False
            msg = f"Channel {ch} missing from normalization.json"
            logger.error(f"[FAIL] {msg}")
            errors.append(msg)
        else:
            m = norm_stats[ch]["mean"]
            s = norm_stats[ch]["std"]
            logger.info(f"[PASS] Normalization stats for {ch:<10}: mean={m:+.4f}, std={s:.4f}")

    # 3. Check NetCDF Splits
    splits = ["train", "val", "test"]
    split_datasets = {}
    split_dates = {}
    total_samples = 0

    expected_depths = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0]

    for s_name in splits:
        nc_path = os.path.join(processed_dir, f"oceanembed_{s_name}.nc")
        ds = xr.open_dataset(nc_path)
        split_datasets[s_name] = ds
        
        num_samples = len(ds["sample"])
        total_samples += num_samples
        logger.info("-" * 50)
        logger.info(f"Inspecting Split: {s_name.upper()} ({num_samples:,} samples)")

        # Verify dimensions
        X = ds["X"].values
        y = ds["y"].values
        dates = ds["date"].values.tolist()
        split_dates[s_name] = set(dates)

        if X.shape != (num_samples, 7, 5, 5):
            all_passed = False
            msg = f"{s_name} X shape mismatch: expected ({num_samples}, 7, 5, 5), got {X.shape}"
            logger.error(f"[FAIL] {msg}")
            errors.append(msg)
        else:
            logger.info(f"[PASS] X tensor shape matches: {X.shape}")

        if y.shape != (num_samples, 15):
            all_passed = False
            msg = f"{s_name} y shape mismatch: expected ({num_samples}, 15), got {y.shape}"
            logger.error(f"[FAIL] {msg}")
            errors.append(msg)
        else:
            logger.info(f"[PASS] y target profile shape matches: {y.shape}")

        # Depth coordinates
        depth_coords = ds["depth"].values.tolist()
        if depth_coords != expected_depths:
            all_passed = False
            msg = f"{s_name} depth coordinates mismatch: {depth_coords}"
            logger.error(f"[FAIL] {msg}")
            errors.append(msg)
        else:
            logger.info(f"[PASS] 15 Target depth levels verified: {depth_coords}")

        # Complete-case NaN verification
        nan_X = np.isnan(X).sum()
        nan_y = np.isnan(y).sum()
        if nan_X > 0 or nan_y > 0:
            all_passed = False
            msg = f"{s_name} contains NaNs! nan_X={nan_X}, nan_y={nan_y}"
            logger.error(f"[FAIL] {msg}")
            errors.append(msg)
        else:
            logger.info(f"[PASS] Zero NaNs confirmed (nan_X=0, nan_y=0) across all {num_samples:,} samples")

        # Target physics verification (temperatures in C)
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        y_surf_mean = float(np.mean(y[:, 0]))
        y_deep_mean = float(np.mean(y[:, -1]))

        if y_min < 1.0 or y_max > 38.0:
            all_passed = False
            msg = f"{s_name} y target temperatures outside plausible ocean bounds: [{y_min:.2f}, {y_max:.2f}] C"
            logger.error(f"[FAIL] {msg}")
            errors.append(msg)
        else:
            logger.info(f"[PASS] Physical thermal range verified: min={y_min:.2f} C, max={y_max:.2f} C")

        if y_surf_mean <= y_deep_mean:
            all_passed = False
            msg = f"{s_name} thermal inversion at bulk scale: surf_mean ({y_surf_mean:.2f}) <= deep_mean ({y_deep_mean:.2f})"
            logger.error(f"[FAIL] {msg}")
            errors.append(msg)
        else:
            logger.info(f"[PASS] Vertical stratification verified: surface mean={y_surf_mean:.2f} C -> 1000m deep mean={y_deep_mean:.2f} C")

        # Standardization verification on Train split
        if s_name == "train":
            ch_means = [float(np.mean(X[:, c, :, :])) for c in range(7)]
            ch_stds = [float(np.std(X[:, c, :, :])) for c in range(7)]
            logger.info(f"[PASS] Train X standardized channel means: {[round(m, 4) for m in ch_means]}")
            logger.info(f"[PASS] Train X standardized channel stds:  {[round(s, 4) for s in ch_stds]}")
            for c, (m, s) in enumerate(zip(ch_means, ch_stds)):
                if abs(m) > 0.02 or abs(s - 1.0) > 0.02:
                    all_passed = False
                    msg = f"Train channel {c} standardization inaccurate: mean={m}, std={s}"
                    logger.error(f"[FAIL] {msg}")
                    errors.append(msg)

    # 4. Temporal Disjointness and Coverage Check
    logger.info("-" * 50)
    logger.info("Checking Temporal Disjointness & Coverage...")
    all_unique_dates = set.union(*split_dates.values())
    train_val_overlap = split_dates["train"].intersection(split_dates["val"])
    train_test_overlap = split_dates["train"].intersection(split_dates["test"])
    val_test_overlap = split_dates["val"].intersection(split_dates["test"])

    if train_val_overlap or train_test_overlap or val_test_overlap:
        all_passed = False
        msg = f"Data leak detected between splits! Overlaps: TV={train_val_overlap}, TT={train_test_overlap}, VT={val_test_overlap}"
        logger.error(f"[FAIL] {msg}")
        errors.append(msg)
    else:
        logger.info("[PASS] Strict temporal split disjointness verified (Zero date overlap between Train, Val, and Test)")

    if len(all_unique_dates) != 366:
        all_passed = False
        msg = f"Expected 366 days covered across 2024 leap year, but found {len(all_unique_dates)} days"
        logger.error(f"[FAIL] {msg}")
        errors.append(msg)
    else:
        logger.info(f"[PASS] Full 366-day temporal coverage verified ({min(all_unique_dates)} to {max(all_unique_dates)})")

    # 5. Metadata CSV Check
    csv_path = os.path.join(processed_dir, "sample_metadata.csv")
    df_meta = pd.read_csv(csv_path)
    if len(df_meta) != total_samples:
        all_passed = False
        msg = f"sample_metadata.csv rows ({len(df_meta):,}) do not match total samples ({total_samples:,})"
        logger.error(f"[FAIL] {msg}")
        errors.append(msg)
    else:
        logger.info(f"[PASS] sample_metadata.csv matches total sample count: {len(df_meta):,} rows")

    logger.info("=" * 70)
    if all_passed:
        logger.info("ALL VERIFICATION SUITE TESTS PASSED PERFECTLY (100% GREEN)!")
        logger.info(f"Total verified dataset: {total_samples:,} pure-ocean samples.")
    else:
        logger.error(f"VERIFICATION FAILED with {len(errors)} errors:")
        for err in errors:
            logger.error(f"  - {err}")
    logger.info("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)

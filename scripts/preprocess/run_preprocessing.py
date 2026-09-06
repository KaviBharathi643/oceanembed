"""
OceanEmbed (SIH 2026 PS 26066)
Complete Preprocessing Pipeline Execution Script

Iterates through all 12 months of 2024 (366 days), regrids 7 surface variables
and GLORYS target profiles to 0.25 deg grid, extracts Strategy A pure-ocean 5x5 patches,
computes train-set normalization stats, standardizes inputs, and saves processed splits.
"""

import os
import sys
import time
import calendar
import logging
from typing import Dict, List, Any
import numpy as np
import yaml
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.preprocessing import (
    build_target_grid,
    load_monthly_oisst,
    load_monthly_sla,
    load_monthly_oscar,
    load_monthly_ccmp_daily_mean,
    load_monthly_smap_daily,
    load_monthly_glorys_target,
    extract_strategy_a_patches,
    compute_channel_statistics,
    apply_channel_standardization,
    save_split_dataset_netcdf,
    save_normalization_json,
    save_sample_metadata_csv,
    save_preprocessing_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("OceanEmbed_Preprocess")


def main():
    start_total_time = time.time()
    config_path = "config/dataset_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    prep_cfg = config["preprocessing"]
    grid_cfg = prep_cfg["target_grid"]
    target_lats, target_lons = build_target_grid(
        lat_min=grid_cfg["lat_min"],
        lat_max=grid_cfg["lat_max"],
        lat_step=grid_cfg["lat_step"],
        lon_min=grid_cfg["lon_min"],
        lon_max=grid_cfg["lon_max"],
        lon_step=grid_cfg["lon_step"],
    )
    target_depths = np.array(prep_cfg["target_depths_m"], dtype=np.float32)
    patch_radius = prep_cfg["patch"]["radius"]
    channel_names = prep_cfg["input_channels"]
    target_channel = prep_cfg["target_channel"]
    output_dir = prep_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    splits_cfg = prep_cfg["splits"]
    train_end = splits_cfg["train"]["end"]
    val_end = splits_cfg["val"]["end"]

    logger.info("=" * 70)
    logger.info("OCEANEMBED PREPROCESSING PIPELINE (Strategy A: Pure-Ocean Complete-Case)")
    logger.info(f"Target Grid: {len(target_lats)} lats ({target_lats[0]:.2f} to {target_lats[-1]:.2f} deg) x {len(target_lons)} lons ({target_lons[0]:.2f} to {target_lons[-1]:.2f} deg)")
    logger.info(f"Target Depths (15 levels): {target_depths.tolist()}")
    logger.info(f"Patch Size: {2 * patch_radius + 1}x{2 * patch_radius + 1} (radius={patch_radius})")
    logger.info(f"Input Channels (7): {channel_names}")
    logger.info(f"Target Channel: {target_channel}")
    logger.info("=" * 70)

    # Accumulators for raw patches per split
    split_data = {
        "train": {"X": [], "y": [], "dates": [], "lats": [], "lons": [], "grid_i": [], "grid_j": []},
        "val": {"X": [], "y": [], "dates": [], "lats": [], "lons": [], "grid_i": [], "grid_j": []},
        "test": {"X": [], "y": [], "dates": [], "lats": [], "lons": [], "grid_i": [], "grid_j": []},
    }

    daily_sample_counts = {}
    total_days_processed = 0

    # Process months 1 through 12
    for month in range(1, 13):
        m_start_time = time.time()
        _, last_day = calendar.monthrange(2024, month)
        date_range_str = f"2024{month:02d}01_2024{month:02d}{last_day:02d}"

        sst_path = f"data/raw/sst/sst_sst_bay_of_bengal_{date_range_str}.nc"
        sla_path = f"data/raw/sla/sla_sla_bay_of_bengal_{date_range_str}.nc"
        oscar_path = f"data/raw/currents/oscar_current_u_current_v_bay_of_bengal_{date_range_str}.nc"
        ccmp_path = f"data/raw/winds/ccmp_wind_u_wind_v_bay_of_bengal_{date_range_str}.nc"
        smap_path = f"data/raw/sss/sss_sss_smap_bay_of_bengal_{date_range_str}.nc"
        glorys_path = f"data/raw/glorys/glorys_thetao_bay_of_bengal_{date_range_str}.nc"

        logger.info(f"Loading Month {month:02d}/2024 ({date_range_str})...")

        d_sst, sst_cube = load_monthly_oisst(sst_path, target_lats, target_lons)
        d_sla, sla_cube = load_monthly_sla(sla_path, target_lats, target_lons)
        d_osc, u_cur_cube, v_cur_cube = load_monthly_oscar(oscar_path, target_lats, target_lons)
        d_ccmp, u_wnd_cube, v_wnd_cube = load_monthly_ccmp_daily_mean(ccmp_path, target_lats, target_lons)
        d_smap, sss_cube = load_monthly_smap_daily(smap_path, target_lats, target_lons, 2024, month)
        d_glo, thetao_cube = load_monthly_glorys_target(glorys_path, target_lats, target_lons, target_depths)

        num_days = len(d_sst)
        m_samples = 0

        for d_idx in range(num_days):
            cur_date = d_sst[d_idx]
            
            # Stack 7 surface channels: [7, n_lat, n_lon]
            surf_cube_7ch = np.stack(
                [
                    sst_cube[d_idx],
                    sss_cube[d_idx],
                    sla_cube[d_idx],
                    u_cur_cube[d_idx],
                    v_cur_cube[d_idx],
                    u_wnd_cube[d_idx],
                    v_wnd_cube[d_idx],
                ],
                axis=0,
            )
            # Target profile cube: [15, n_lat, n_lon]
            tgt_cube_15d = thetao_cube[d_idx]

            # Extract Strategy A patches
            patches_dict = extract_strategy_a_patches(
                surf_cube_7ch, tgt_cube_15d, target_lats, target_lons, cur_date, patch_radius
            )
            n_valid = len(patches_dict["dates"])
            daily_sample_counts[cur_date] = n_valid
            m_samples += n_valid

            # Determine split
            if cur_date <= train_end:
                split_key = "train"
            elif cur_date <= val_end:
                split_key = "val"
            else:
                split_key = "test"

            if n_valid > 0:
                split_data[split_key]["X"].append(patches_dict["X"])
                split_data[split_key]["y"].append(patches_dict["y"])
                split_data[split_key]["dates"].extend(patches_dict["dates"])
                split_data[split_key]["lats"].append(patches_dict["lats"])
                split_data[split_key]["lons"].append(patches_dict["lons"])
                split_data[split_key]["grid_i"].append(patches_dict["grid_i"])
                split_data[split_key]["grid_j"].append(patches_dict["grid_j"])

        total_days_processed += num_days
        logger.info(
            f"Month {month:02d} completed in {time.time() - m_start_time:.1f}s: "
            f"{num_days} days, {m_samples:,} pure-ocean samples extracted."
        )

    logger.info("=" * 70)
    logger.info(f"All 12 months processed! Total days: {total_days_processed}/366")

    # Concatenate per-split arrays
    processed_splits = {}
    for s_name in ["train", "val", "test"]:
        if len(split_data[s_name]["dates"]) > 0:
            X_concat = np.concatenate(split_data[s_name]["X"], axis=0)
            y_concat = np.concatenate(split_data[s_name]["y"], axis=0)
            lats_concat = np.concatenate(split_data[s_name]["lats"], axis=0)
            lons_concat = np.concatenate(split_data[s_name]["lons"], axis=0)
            grid_i_concat = np.concatenate(split_data[s_name]["grid_i"], axis=0)
            grid_j_concat = np.concatenate(split_data[s_name]["grid_j"], axis=0)
            dates_list = split_data[s_name]["dates"]
        else:
            X_concat = np.empty((0, 7, 5, 5), dtype=np.float32)
            y_concat = np.empty((0, 15), dtype=np.float32)
            lats_concat = np.empty((0,), dtype=np.float32)
            lons_concat = np.empty((0,), dtype=np.float32)
            grid_i_concat = np.empty((0,), dtype=np.int32)
            grid_j_concat = np.empty((0,), dtype=np.int32)
            dates_list = []

        processed_splits[s_name] = {
            "X": X_concat,
            "y": y_concat,
            "lats": lats_concat,
            "lons": lons_concat,
            "grid_i": grid_i_concat,
            "grid_j": grid_j_concat,
            "dates": dates_list,
        }
        logger.info(f"Split {s_name.upper()}: {len(dates_list):,} samples, X={X_concat.shape}, y={y_concat.shape}")

    # Compute normalization statistics strictly from training split
    logger.info("Computing channel normalization stats from TRAIN split only...")
    norm_stats = compute_channel_statistics(processed_splits["train"]["X"], channel_names)
    for ch, stats in norm_stats.items():
        logger.info(f"  {ch:<12}: mean={stats['mean']:+.4f}, std={stats['std']:.4f}, min={stats['min']:+.4f}, max={stats['max']:+.4f}")

    # Save normalization statistics JSON
    norm_json_path = os.path.join(output_dir, "normalization.json")
    save_normalization_json(norm_json_path, norm_stats)
    logger.info(f"Saved normalization stats to {norm_json_path}")

    # Apply standardization to inputs for all splits (targets y remain in physical deg C)
    for s_name in ["train", "val", "test"]:
        logger.info(f"Applying standardization to {s_name.upper()} inputs...")
        X_std = apply_channel_standardization(processed_splits[s_name]["X"], norm_stats, channel_names)
        processed_splits[s_name]["X_std"] = X_std

        # Save to NetCDF
        split_nc_path = os.path.join(output_dir, f"oceanembed_{s_name}.nc")
        save_split_dataset_netcdf(
            output_nc_path=split_nc_path,
            X=X_std,
            y=processed_splits[s_name]["y"],
            dates=processed_splits[s_name]["dates"],
            lats=processed_splits[s_name]["lats"],
            lons=processed_splits[s_name]["lons"],
            grid_i=processed_splits[s_name]["grid_i"],
            grid_j=processed_splits[s_name]["grid_j"],
            channel_names=channel_names,
            target_depths=target_depths,
            split_name=s_name,
        )

    # Save aggregated sample metadata CSV
    all_metadata_rows = []
    for s_name in ["train", "val", "test"]:
        n = len(processed_splits[s_name]["dates"])
        for idx in range(n):
            all_metadata_rows.append({
                "sample_id": f"{s_name}_{idx:07d}",
                "split": s_name,
                "date": processed_splits[s_name]["dates"][idx],
                "lat": float(processed_splits[s_name]["lats"][idx]),
                "lon": float(processed_splits[s_name]["lons"][idx]),
                "grid_i": int(processed_splits[s_name]["grid_i"][idx]),
                "grid_j": int(processed_splits[s_name]["grid_j"][idx]),
            })
            
    metadata_csv_path = os.path.join(output_dir, "sample_metadata.csv")
    save_sample_metadata_csv(metadata_csv_path, all_metadata_rows)
    logger.info(f"Saved sample metadata ({len(all_metadata_rows):,} rows) to {metadata_csv_path}")

    # Save comprehensive preprocessing report
    total_samples = sum(len(processed_splits[s]["dates"]) for s in ["train", "val", "test"])
    report = {
        "project": "OceanEmbed",
        "domain": "Bay of Bengal (5.0N to 25.0N, 80.0E to 100.0E)",
        "period": "2024-01-01 to 2024-12-31 (366 days)",
        "total_days_processed": total_days_processed,
        "grid": {
            "num_lats": len(target_lats),
            "num_lons": len(target_lons),
            "lat_range": [float(target_lats[0]), float(target_lats[-1])],
            "lon_range": [float(target_lons[0]), float(target_lons[-1])],
            "resolution_deg": 0.25,
            "max_interior_centers": (len(target_lats) - 4) * (len(target_lons) - 4),
        },
        "target_depths_m": target_depths.tolist(),
        "input_channels": channel_names,
        "target_channel": target_channel,
        "missing_data_strategy": "Strategy A: Complete-case pure-ocean (bathymetry >= 1000m, no padding)",
        "sample_counts": {
            "train": len(processed_splits["train"]["dates"]),
            "val": len(processed_splits["val"]["dates"]),
            "test": len(processed_splits["test"]["dates"]),
            "total": total_samples,
        },
        "tensor_shapes": {
            "X": [7, 5, 5],
            "y": [15],
        },
        "normalization_stats": norm_stats,
        "daily_sample_counts_summary": {
            "min_per_day": int(min(daily_sample_counts.values())),
            "max_per_day": int(max(daily_sample_counts.values())),
            "mean_per_day": float(np.mean(list(daily_sample_counts.values()))),
        },
        "execution_time_seconds": round(time.time() - start_total_time, 2),
    }

    report_json_path = os.path.join(output_dir, "preprocessing_report.json")
    save_preprocessing_report(report_json_path, report)
    logger.info(f"Saved preprocessing report to {report_json_path}")
    logger.info("=" * 70)
    logger.info(f"PREPROCESSING COMPLETED SUCCESSFULLY in {time.time() - start_total_time:.1f}s!")
    logger.info(f"Total pure-ocean samples: {total_samples:,} across 366 days.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

"""
OceanEmbed (SIH 2026 Problem Statement 26066)
Preprocessing Pipeline Module

Provides data loaders, spatial regridders, temporal aggregators,
vertical interpolators, 5x5 spatial patch extractors, and channel standardizers.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import xarray as xr
import pandas as pd
from scipy.interpolate import RegularGridInterpolator, interp1d

logger = logging.getLogger(__name__)


def build_target_grid(
    lat_min: float = 5.0,
    lat_max: float = 25.0,
    lat_step: float = 0.25,
    lon_min: float = 80.0,
    lon_max: float = 100.0,
    lon_step: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct the regular 2D target grid.
    Default: 81 lats from 5.0 to 25.0, 81 lons from 80.0 to 100.0 (spacing 0.25 deg).
    """
    num_lats = int(round((lat_max - lat_min) / lat_step)) + 1
    num_lons = int(round((lon_max - lon_min) / lon_step)) + 1
    lats = np.linspace(lat_min, lat_max, num_lats, dtype=np.float32)
    lons = np.linspace(lon_min, lon_max, num_lons, dtype=np.float32)
    return lats, lons


def regrid_2d_field(
    data: np.ndarray,
    src_lats: np.ndarray,
    src_lons: np.ndarray,
    dst_lats: np.ndarray,
    dst_lons: np.ndarray,
    method: str = "linear",
    bounds_error: bool = False,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Bilinearly interpolate a 2D field (lat, lon) from source coordinates to destination coordinates.
    Handles ascending/descending coordinates automatically.
    """
    # Ensure source coordinates are strictly ascending
    lat_flip = False
    lon_flip = False
    
    if src_lats[1] < src_lats[0]:
        src_lats = src_lats[::-1]
        data = data[::-1, :]
        lat_flip = True
        
    if src_lons[1] < src_lons[0]:
        src_lons = src_lons[::-1]
        data = data[:, ::-1]
        lon_flip = True

    interp = RegularGridInterpolator(
        (src_lats, src_lons),
        data,
        method=method,
        bounds_error=bounds_error,
        fill_value=fill_value,
    )
    
    # Meshgrid of target coordinates
    grid_lat, grid_lon = np.meshgrid(dst_lats, dst_lons, indexing="ij")
    pts = np.column_stack([grid_lat.ravel(), grid_lon.ravel()])
    regridded = interp(pts).reshape((len(dst_lats), len(dst_lons))).astype(np.float32)
    return regridded


def load_monthly_oisst(
    file_path: str,
    dst_lats: np.ndarray,
    dst_lons: np.ndarray,
) -> Tuple[List[str], np.ndarray]:
    """
    Load monthly NOAA OISST NetCDF file and regrid daily SST fields to target grid.
    Returns (date_strings, array of shape [num_days, len(dst_lats), len(dst_lons)]).
    """
    with xr.open_dataset(file_path) as ds:
        # time, lat, lon
        src_lats = ds["lat"].values.astype(np.float32)
        src_lons = ds["lon"].values.astype(np.float32)
        sst_raw = ds["sst"].values.astype(np.float32)  # [time, 1, lat, lon] or [time, lat, lon]
        if sst_raw.ndim == 4 and sst_raw.shape[1] == 1:
            sst_raw = sst_raw[:, 0, :, :]
            
        time_vals = ds["time"].values
        dates = [str(t)[:10] for t in time_vals]
        
    num_days = len(dates)
    regridded_all = np.empty((num_days, len(dst_lats), len(dst_lons)), dtype=np.float32)
    for d in range(num_days):
        regridded_all[d] = regrid_2d_field(
            sst_raw[d], src_lats, src_lons, dst_lats, dst_lons
        )
        
    return dates, regridded_all


def load_monthly_sla(
    file_path: str,
    dst_lats: np.ndarray,
    dst_lons: np.ndarray,
) -> Tuple[List[str], np.ndarray]:
    """
    Load monthly Copernicus DUACS SLA NetCDF file and regrid daily SLA fields to target grid.
    Returns (date_strings, array of shape [num_days, len(dst_lats), len(dst_lons)]).
    """
    with xr.open_dataset(file_path) as ds:
        lat_key = "latitude" if "latitude" in ds else "lat"
        lon_key = "longitude" if "longitude" in ds else "lon"
        src_lats = ds[lat_key].values.astype(np.float32)
        src_lons = ds[lon_key].values.astype(np.float32)
        sla_raw = ds["sla"].values.astype(np.float32)  # [time, lat, lon]
        time_vals = ds["time"].values
        dates = [str(t)[:10] for t in time_vals]

    num_days = len(dates)
    regridded_all = np.empty((num_days, len(dst_lats), len(dst_lons)), dtype=np.float32)
    for d in range(num_days):
        regridded_all[d] = regrid_2d_field(
            sla_raw[d], src_lats, src_lons, dst_lats, dst_lons
        )

    return dates, regridded_all


def load_monthly_oscar(
    file_path: str,
    dst_lats: np.ndarray,
    dst_lons: np.ndarray,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Load monthly NASA OSCAR surface currents NetCDF file and regrid daily current_u, current_v to target grid.
    OSCAR native shape is (time, lon, lat) or (time, lat, lon). Transposes to (time, lat, lon).
    Returns (date_strings, u_regridded, v_regridded).
    """
    with xr.open_dataset(file_path) as ds:
        src_lats = ds["lat"].values.astype(np.float32)
        src_lons = ds["lon"].values.astype(np.float32)
        u_var = "current_u" if "current_u" in ds else "u"
        v_var = "current_v" if "current_v" in ds else "v"
        u_raw = ds[u_var].values.astype(np.float32)
        v_raw = ds[v_var].values.astype(np.float32)
        
        # Check dimension ordering: if ('time', 'lon', 'lat'), transpose last two axes
        if ds[u_var].dims == ("time", "lon", "lat"):
            u_raw = np.transpose(u_raw, (0, 2, 1))
            v_raw = np.transpose(v_raw, (0, 2, 1))
            
        time_vals = ds["time"].values
        dates = [str(t)[:10] for t in time_vals]

    num_days = len(dates)
    u_all = np.empty((num_days, len(dst_lats), len(dst_lons)), dtype=np.float32)
    v_all = np.empty((num_days, len(dst_lats), len(dst_lons)), dtype=np.float32)
    
    for d in range(num_days):
        u_all[d] = regrid_2d_field(u_raw[d], src_lats, src_lons, dst_lats, dst_lons)
        v_all[d] = regrid_2d_field(v_raw[d], src_lats, src_lons, dst_lats, dst_lons)

    return dates, u_all, v_all


def load_monthly_ccmp_daily_mean(
    file_path: str,
    dst_lats: np.ndarray,
    dst_lons: np.ndarray,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Load monthly CCMP 6-hourly winds NetCDF file, aggregate to daily vector mean (u, v),
    and regrid to target grid.
    Returns (date_strings, u_regridded, v_regridded).
    """
    with xr.open_dataset(file_path) as ds:
        lat_key = "latitude" if "latitude" in ds else "lat"
        lon_key = "longitude" if "longitude" in ds else "lon"
        src_lats = ds[lat_key].values.astype(np.float32)
        src_lons = ds[lon_key].values.astype(np.float32)
        
        time_vals = ds["time"].values
        date_strs = np.array([str(t)[:10] for t in time_vals])
        u_var = "wind_u" if "wind_u" in ds else "uwnd"
        v_var = "wind_v" if "wind_v" in ds else "vwnd"
        uwnd = ds[u_var].values.astype(np.float32)
        vwnd = ds[v_var].values.astype(np.float32)

    df_time = pd.DataFrame({"date": date_strs})
    unique_dates = sorted(df_time["date"].unique().tolist())
    
    num_days = len(unique_dates)
    u_all = np.empty((num_days, len(dst_lats), len(dst_lons)), dtype=np.float32)
    v_all = np.empty((num_days, len(dst_lats), len(dst_lons)), dtype=np.float32)

    for d_idx, d_str in enumerate(unique_dates):
        indices = np.where(df_time["date"].values == d_str)[0]
        # Daily arithmetic mean of 4 observations (00, 06, 12, 18 UTC)
        u_day = np.nanmean(uwnd[indices], axis=0)
        v_day = np.nanmean(vwnd[indices], axis=0)
        
        u_all[d_idx] = regrid_2d_field(u_day, src_lats, src_lons, dst_lats, dst_lons)
        v_all[d_idx] = regrid_2d_field(v_day, src_lats, src_lons, dst_lats, dst_lons)

    return unique_dates, u_all, v_all


def load_monthly_smap_daily(
    file_path: str,
    dst_lats: np.ndarray,
    dst_lons: np.ndarray,
    year: int,
    month: int,
) -> Tuple[List[str], np.ndarray]:
    """
    Load monthly NASA SMAP SSS 8-day running mean NetCDF file.
    Uses center_day_of_observation (DOY) to select the exact representative slice
    for each calendar day of the month (slice index k = 4 + d - 1, starting from Day 1).
    Regrids to target grid.
    Returns (date_strings, array of shape [num_days, len(dst_lats), len(dst_lons)]).
    """
    import calendar
    _, num_days_in_month = calendar.monthrange(year, month)
    expected_dates = [f"{year:04d}-{month:02d}-{d:02d}" for d in range(1, num_days_in_month + 1)]
    
    with xr.open_dataset(file_path) as ds:
        src_lats = ds["lat"].values.astype(np.float32)
        src_lons = ds["lon"].values.astype(np.float32)
        sss_var = "sss_smap" if "sss_smap" in ds else "sss"
        sss_raw = ds[sss_var].values.astype(np.float32)  # [times, lat, lon]
        
        # Determine mapping from center_day_of_observation
        if "center_day_of_observation" in ds:
            center_days = ds["center_day_of_observation"].values.astype(int)
            # Find matching slice index for each calendar date's day of year
            slice_indices = []
            for d_str in expected_dates:
                doy = pd.to_datetime(d_str).dayofyear
                # Match doy with center_day_of_observation
                matches = np.where(center_days == doy)[0]
                if len(matches) > 0:
                    slice_indices.append(matches[0])
                else:
                    # Fallback to index formula: index 4 is Day 1
                    day_num = int(d_str.split("-")[2])
                    slice_indices.append(4 + day_num - 1)
        else:
            slice_indices = [4 + d - 1 for d in range(1, num_days_in_month + 1)]

    regridded_all = np.empty((len(expected_dates), len(dst_lats), len(dst_lons)), dtype=np.float32)
    for d_idx, s_idx in enumerate(slice_indices):
        regridded_all[d_idx] = regrid_2d_field(
            sss_raw[s_idx], src_lats, src_lons, dst_lats, dst_lons
        )

    return expected_dates, regridded_all


def interpolate_glorys_vertical_profiles(
    thetao_36d: np.ndarray,
    native_depths: np.ndarray,
    target_depths: np.ndarray,
) -> np.ndarray:
    """
    Vertically interpolate GLORYS thetao from 36 native depths to 15 target depths.
    Rules:
      - thetao(0m) = thetao(0.494m) (surface mixed layer approximation).
      - Interior depths (5m to 1000m) are linearly interpolated along depth axis.
      - If native level 35 (1062.44m) is NaN (bathymetry < 1000m), target profile at 1000m is NaN.
      - If any required depth bracket is NaN, target at that depth becomes NaN.
    Input: thetao_36d: shape [36, lat, lon]
    Output: thetao_15d: shape [15, lat, lon]
    """
    n_depths, n_lat, n_lon = thetao_36d.shape
    num_targets = len(target_depths)
    out = np.full((num_targets, n_lat, n_lon), np.nan, dtype=np.float32)

    # Augmented depths with depth 0.0m prepended
    aug_depths = np.concatenate([[0.0], native_depths])

    # Flatten spatial dimensions for vectorized 1D interpolation
    flat_data = thetao_36d.reshape(n_depths, -1)  # [36, N]
    flat_surface = flat_data[0:1, :]               # [1, N] at 0.494m
    aug_data = np.vstack([flat_surface, flat_data])  # [37, N]

    # Find valid vertical profiles where depth 0.494m is not NaN
    valid_mask = ~np.isnan(aug_data[0, :])
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) > 0:
        sub_data = aug_data[:, valid_indices]  # [37, M]
        
        # Vectorized linear interpolation along depth axis
        # For each target depth, find bracketing native depths
        for t_idx, td in enumerate(target_depths):
            if td == 0.0:
                out[t_idx].flat[valid_indices] = sub_data[0, :]
            else:
                # Find lower and upper index in aug_depths
                # aug_depths is monotonic increasing
                k = np.searchsorted(aug_depths, td)
                if k == 0:
                    out[t_idx].flat[valid_indices] = sub_data[0, :]
                elif k >= len(aug_depths):
                    # Extrapolation beyond max native depth -> NaN
                    continue
                else:
                    z0 = aug_depths[k - 1]
                    z1 = aug_depths[k]
                    v0 = sub_data[k - 1, :]
                    v1 = sub_data[k, :]
                    # Linear weight
                    w = (td - z0) / (z1 - z0)
                    interpolated = (1.0 - w) * v0 + w * v1
                    out[t_idx].flat[valid_indices] = interpolated

    return out


def load_monthly_glorys_target(
    file_path: str,
    dst_lats: np.ndarray,
    dst_lons: np.ndarray,
    target_depths: np.ndarray,
) -> Tuple[List[str], np.ndarray]:
    """
    Load monthly GLORYS12V1 thetao NetCDF file, regrid each native vertical level
    to target grid, and interpolate vertically to 15 target depths.
    Returns (date_strings, array of shape [num_days, 15, len(dst_lats), len(dst_lons)]).
    """
    with xr.open_dataset(file_path) as ds:
        lat_key = "latitude" if "latitude" in ds else "lat"
        lon_key = "longitude" if "longitude" in ds else "lon"
        src_lats = ds[lat_key].values.astype(np.float32)
        src_lons = ds[lon_key].values.astype(np.float32)
        native_depths = ds["depth"].values.astype(np.float32)
        
        time_vals = ds["time"].values
        dates = [str(t)[:10] for t in time_vals]
        
        thetao_raw = ds["thetao"].values.astype(np.float32)  # [time, depth, lat, lon]

    num_days = len(dates)
    num_native_depths = len(native_depths)
    num_target_depths = len(target_depths)
    
    out_target_all = np.empty(
        (num_days, num_target_depths, len(dst_lats), len(dst_lons)), dtype=np.float32
    )

    for d in range(num_days):
        # Regrid 36 native depth levels onto target grid
        regridded_36d = np.empty(
            (num_native_depths, len(dst_lats), len(dst_lons)), dtype=np.float32
        )
        for z in range(num_native_depths):
            regridded_36d[z] = regrid_2d_field(
                thetao_raw[d, z], src_lats, src_lons, dst_lats, dst_lons
            )
            
        # Vertically interpolate to 15 target depths
        out_target_all[d] = interpolate_glorys_vertical_profiles(
            regridded_36d, native_depths, target_depths
        )

    return dates, out_target_all


def extract_strategy_a_patches(
    surface_cube_7ch: np.ndarray,
    target_cube_15d: np.ndarray,
    dst_lats: np.ndarray,
    dst_lons: np.ndarray,
    date_str: str,
    patch_radius: int = 2,
) -> Dict[str, np.ndarray]:
    """
    Extract Strategy A (complete-case pure-ocean) 5x5 spatial patches.
    Discard any sample where:
      - Any cell in the 5x5 patch for ANY of the 7 surface input channels is NaN.
      - Any depth in the 15-depth target profile at the center cell is NaN (e.g. shallow bathymetry < 1000m).
      - Center cell is within patch_radius (2) of the grid boundary.
    Inputs:
      surface_cube_7ch: [7, n_lat, n_lon]
      target_cube_15d:  [15, n_lat, n_lon]
    Returns dictionary with:
      'X': [num_valid, 7, 5, 5]
      'y': [num_valid, 15]
      'dates': list of date_str
      'lats': [num_valid]
      'lons': [num_valid]
      'grid_i': [num_valid]
      'grid_j': [num_valid]
    """
    n_lat = len(dst_lats)
    n_lon = len(dst_lons)
    patch_size = 2 * patch_radius + 1
    
    # Pre-check center mask: target profile clean at all 15 depths
    target_clean = ~np.isnan(target_cube_15d).any(axis=0)  # [n_lat, n_lon]
    
    # Surface clean at all 7 channels
    surface_clean = ~np.isnan(surface_cube_7ch).any(axis=0)  # [n_lat, n_lon]

    # Interior bounds
    i_min, i_max = patch_radius, n_lat - patch_radius
    j_min, j_max = patch_radius, n_lon - patch_radius

    valid_X = []
    valid_y = []
    valid_lats = []
    valid_lons = []
    valid_i = []
    valid_j = []

    for i in range(i_min, i_max):
        for j in range(j_min, j_max):
            # Fast check: center cell target profile must be clean
            if not target_clean[i, j]:
                continue
                
            # Check 5x5 spatial patch for all 7 surface channels
            surf_patch = surface_cube_7ch[:, i - patch_radius : i + patch_radius + 1, j - patch_radius : j + patch_radius + 1]
            if np.isnan(surf_patch).any():
                continue
                
            # If all clean, accept Strategy A pure-ocean sample
            valid_X.append(surf_patch)
            valid_y.append(target_cube_15d[:, i, j])
            valid_lats.append(dst_lats[i])
            valid_lons.append(dst_lons[j])
            valid_i.append(i)
            valid_j.append(j)

    num_samples = len(valid_X)
    if num_samples > 0:
        return {
            "X": np.array(valid_X, dtype=np.float32),
            "y": np.array(valid_y, dtype=np.float32),
            "dates": [date_str] * num_samples,
            "lats": np.array(valid_lats, dtype=np.float32),
            "lons": np.array(valid_lons, dtype=np.float32),
            "grid_i": np.array(valid_i, dtype=np.int32),
            "grid_j": np.array(valid_j, dtype=np.int32),
        }
    else:
        return {
            "X": np.empty((0, 7, patch_size, patch_size), dtype=np.float32),
            "y": np.empty((0, 15), dtype=np.float32),
            "dates": [],
            "lats": np.empty((0,), dtype=np.float32),
            "lons": np.empty((0,), dtype=np.float32),
            "grid_i": np.empty((0,), dtype=np.int32),
            "grid_j": np.empty((0,), dtype=np.int32),
        }


def compute_channel_statistics(
    X_train: np.ndarray,
    channel_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute mean and standard deviation per surface input channel on the training set only.
    Input: X_train of shape [N, 7, 5, 5]
    Returns dictionary with channel statistics.
    """
    stats = {}
    num_channels = X_train.shape[1]
    for c in range(num_channels):
        c_name = channel_names[c] if c < len(channel_names) else f"channel_{c}"
        ch_vals = X_train[:, c, :, :]
        c_mean = float(np.mean(ch_vals))
        c_std = float(np.std(ch_vals))
        c_min = float(np.min(ch_vals))
        c_max = float(np.max(ch_vals))
        # Guard against zero division
        if c_std < 1e-7:
            c_std = 1.0
            
        stats[c_name] = {
            "mean": c_mean,
            "std": c_std,
            "min": c_min,
            "max": c_max,
        }
    return stats


def apply_channel_standardization(
    X: np.ndarray,
    stats: Dict[str, Dict[str, float]],
    channel_names: List[str],
) -> np.ndarray:
    """
    Apply train-computed channel standardization (x - mean) / std to input tensor X.
    Input: X of shape [N, 7, 5, 5]
    Output: Standardized X of same shape.
    """
    X_norm = np.empty_like(X, dtype=np.float32)
    for c, c_name in enumerate(channel_names):
        c_mean = stats[c_name]["mean"]
        c_std = stats[c_name]["std"]
        X_norm[:, c, :, :] = (X[:, c, :, :] - c_mean) / c_std
    return X_norm


def save_split_dataset_netcdf(
    output_nc_path: str,
    X: np.ndarray,
    y: np.ndarray,
    dates: List[str],
    lats: np.ndarray,
    lons: np.ndarray,
    grid_i: np.ndarray,
    grid_j: np.ndarray,
    channel_names: List[str],
    target_depths: np.ndarray,
    split_name: str,
) -> None:
    """
    Save preprocessed split dataset (X, y, coordinates, metadata) to NetCDF4 with compression.
    """
    os.makedirs(os.path.dirname(output_nc_path), exist_ok=True)
    num_samples = len(dates)
    patch_y_dim = X.shape[2] if num_samples > 0 else 5
    patch_x_dim = X.shape[3] if num_samples > 0 else 5
    
    ds = xr.Dataset(
        data_vars={
            "X": (["sample", "channel", "patch_y", "patch_x"], X),
            "y": (["sample", "depth"], y),
            "lat": (["sample"], lats.astype(np.float32)),
            "lon": (["sample"], lons.astype(np.float32)),
            "grid_i": (["sample"], grid_i.astype(np.int32)),
            "grid_j": (["sample"], grid_j.astype(np.int32)),
            "date": (["sample"], np.array(dates, dtype=str)),
        },
        coords={
            "sample": np.arange(num_samples, dtype=np.int32),
            "channel": np.array(channel_names, dtype=str),
            "patch_y": np.arange(patch_y_dim, dtype=np.int32),
            "patch_x": np.arange(patch_x_dim, dtype=np.int32),
            "depth": np.array(target_depths, dtype=np.float32),
        },
        attrs={
            "project": "OceanEmbed",
            "problem_statement": "SIH 2026 PS 26066",
            "split": split_name,
            "region": "Bay of Bengal (5.0N to 25.0N, 80.0E to 100.0E)",
            "grid_resolution_deg": 0.25,
            "patch_size": f"{patch_y_dim}x{patch_x_dim}",
            "patch_radius": 2,
            "missing_data_strategy": "Strategy A: Complete-case pure-ocean (bathymetry >= 1000m)",
            "num_samples": int(num_samples),
            "input_standardized": "true",
            "target_unit": "degrees_Celsius",
        },
    )

    encoding = {
        "X": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "y": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "lat": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "lon": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "grid_i": {"zlib": True, "complevel": 4, "dtype": "int32"},
        "grid_j": {"zlib": True, "complevel": 4, "dtype": "int32"},
    }

    ds.to_netcdf(output_nc_path, engine="netcdf4", encoding=encoding)
    logger.info(f"Saved {split_name} split ({num_samples} samples) to {output_nc_path}")


def save_normalization_json(filepath: str, stats: Dict[str, Dict[str, float]]) -> None:
    """Save training normalization statistics to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(stats, f, indent=2)


def save_sample_metadata_csv(
    filepath: str,
    all_splits_metadata: List[Dict[str, Any]],
) -> None:
    """Save aggregated sample metadata CSV across all splits."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame(all_splits_metadata)
    df.to_csv(filepath, index=False)


def save_preprocessing_report(filepath: str, report: Dict[str, Any]) -> None:
    """Save comprehensive preprocessing verification and execution report."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(report, f, indent=2)


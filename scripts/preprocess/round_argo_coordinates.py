"""
OceanEmbed (SIH 2026 Problem Statement 26066)
Reproducible Argo Coordinate Rounding & Audit Preservation Script

Transformation Rule:
  rounded_value = round(value / 0.25) * 0.25

Scientific Audit Principle:
- Raw Argo in-situ observations in data/raw/argo/*.nc remain 100% UNTOUCHED.
- Processed Argo coordinates are mapped to the nearest 0.25° model grid for integration.
- For auditability, both physical raw coordinates (raw_lat, raw_lon) and 0.25° grid coordinates
  (grid_lat, grid_lon, also mirrored in lat, lon) are recorded in the processed dataset.
- All observational fields (temps, depths, QC, dates, platforms, cycles) remain strictly unchanged.
"""

import copy
import glob
import json
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr

def round_coord(v: float) -> float:
    # rounded_value = round(value / 0.25) * 0.25
    return round(round(float(v) / 0.25) * 0.25, 2)

def is_quarter_mult(val: float, tol: float = 1e-5) -> bool:
    # Numerical precision tolerance check for exact 0.25 grid multiples
    grid_idx = round(val / 0.25)
    return abs(val - grid_idx * 0.25) < tol

def main():
    print("=" * 70)
    print("OCEANEMBED — ARGO COORDINATE ROUNDING & AUDIT PRESERVATION")
    print("=" * 70)

    # 1. Read Raw Truth from data/raw/argo/ (Read-only, untouched)
    raw_files = sorted(glob.glob("data/raw/argo/argo_TEMP_PRES_bay_of_bengal_2024*.nc"))
    raw_files = [f for f in raw_files if "_test.nc" not in f]
    print(f"Reading {len(raw_files)} raw Argo NetCDF files for ground-truth coordinates...")

    raw_dfs = []
    for f in raw_files:
        with xr.open_dataset(f) as ds:
            raw_dfs.append(pd.DataFrame({
                "time": ds["time"].values,
                "latitude": ds["latitude"].values,
                "longitude": ds["longitude"].values,
                "pres": ds["pres"].values,
                "temp": ds["temp"].values,
                "pres_qc": ds["pres_qc"].values.astype(str),
                "temp_qc": ds["temp_qc"].values.astype(str),
                "platform_number": ds["platform_number"].values.astype(str),
                "cycle_number": ds["cycle_number"].values,
            }))
    df_raw = pd.concat(raw_dfs, ignore_index=True)
    qc_mask = (
        (df_raw["temp_qc"] == "1") &
        (df_raw["pres_qc"] == "1") &
        (df_raw["pres"] >= 0.0) &
        (df_raw["temp"] > -2.0) &
        (df_raw["temp"] < 40.0) &
        df_raw["time"].notna() &
        df_raw["latitude"].notna() &
        df_raw["longitude"].notna()
    )
    df_qc = df_raw[qc_mask].copy()
    df_qc["date"] = pd.to_datetime(df_qc["time"]).dt.strftime("%Y-%m-%d")
    raw_grouped = df_qc.groupby(["platform_number", "cycle_number", "date"])

    raw_lookup = {}
    for (plat, cycle, date), prof in raw_grouped:
        raw_lat = float(prof["latitude"].mean())
        raw_lon = float(prof["longitude"].mean())
        raw_lookup[(str(plat), int(cycle), str(date))] = (raw_lat, raw_lon)

    assert len(raw_lookup) == 852, f"Expected 852 raw profiles, got {len(raw_lookup)}"
    print(f"Extracted {len(raw_lookup)} unique profiles from raw observations.")

    # 2. Update JSON
    json_path = Path("data/processed/argo_profiles.json")
    with open(json_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    assert len(profiles) == 852, f"Expected 852 profiles in JSON, got {len(profiles)}"

    new_profiles = []
    for p in profiles:
        plat = str(p["platform"])
        cycle = int(p["cycle"])
        date = str(p["date"])
        key = (plat, cycle, date)
        assert key in raw_lookup, f"Profile {key} not found in raw lookup!"

        raw_lat, raw_lon = raw_lookup[key]
        grid_lat = round_coord(raw_lat)
        grid_lon = round_coord(raw_lon)

        item = {
            "platform": plat,
            "cycle": cycle,
            "date": date,
            "lat": grid_lat,        # 0.25° model grid latitude (for backward compatibility)
            "lon": grid_lon,        # 0.25° model grid longitude (for backward compatibility)
            "grid_lat": grid_lat,   # Explicit 0.25° grid coordinate
            "grid_lon": grid_lon,   # Explicit 0.25° grid coordinate
            "raw_lat": round(raw_lat, 4),  # Preserved raw in-situ observation latitude
            "raw_lon": round(raw_lon, 4),  # Preserved raw in-situ observation longitude
            "z_min": p["z_min"],
            "z_max": p["z_max"],
            "temps": p["temps"],
        }
        new_profiles.append(item)

    # Verify JSON coordinates
    assert all(is_quarter_mult(p["lat"]) for p in new_profiles)
    assert all(is_quarter_mult(p["lon"]) for p in new_profiles)
    assert all(is_quarter_mult(p["grid_lat"]) for p in new_profiles)
    assert all(is_quarter_mult(p["grid_lon"]) for p in new_profiles)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(new_profiles, f, indent=2)
    print(f"Successfully wrote {len(new_profiles)} profiles to {json_path}")

    # 3. Update CSV
    csv_path = Path("data/processed/argo_profiles_all_852.csv")
    csv_rows = []
    for p in new_profiles:
        csv_rows.append({
            "Date": p["date"],
            "Latitude (°N)": p["lat"],
            "Longitude (°E)": p["lon"],
            "Grid Latitude (°N)": p["grid_lat"],
            "Grid Longitude (°E)": p["grid_lon"],
            "Raw Latitude (°N)": p["raw_lat"],
            "Raw Longitude (°E)": p["raw_lon"],
            "Platform": p["platform"],
            "Cycle": p["cycle"],
            "Min Depth (m)": p["z_min"],
            "Max Depth (m)": p["z_max"],
            "Observed Depths Count": sum(1 for t in p["temps"] if t is not None),
        })
    df_csv = pd.DataFrame(csv_rows)
    df_csv.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Successfully wrote {len(df_csv)} rows to {csv_path}")

    # 4. Mirror to qa_package
    qa_json = Path("qa_package/data/processed/argo_profiles.json")
    qa_csv = Path("qa_package/data/processed/argo_profiles_all_852.csv")
    if qa_json.parent.exists():
        shutil.copy2(json_path, qa_json)
        print(f"Synced to {qa_json}")
    if qa_csv.parent.exists():
        shutil.copy2(csv_path, qa_csv)
        print(f"Synced to {qa_csv}")

    print("\n" + "=" * 70)
    print("COORDINATE ROUNDING & AUDIT PRESERVATION COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    main()

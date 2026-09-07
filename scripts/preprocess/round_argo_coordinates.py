"""
OceanEmbed - Argo Coordinate Rounding Script
Rounds every processed Argo profile's latitude and longitude to the nearest 0.25-degree grid:
rounded_value = round(value / 0.25) * 0.25

Updates:
- data/processed/argo_profiles_all_852.csv
- data/processed/argo_profiles.json
- qa_package/data/processed/argo_profiles_all_852.csv
- qa_package/data/processed/argo_profiles.json

Keeps all other fields unchanged (date, platform, cycle, temps, depths, etc.).
Leaves raw Argo files (data/raw/argo/) completely untouched.
"""

import copy
import json
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

def round_coord(v: float) -> float:
    # rounded_value = round(value / 0.25) * 0.25
    return round(round(float(v) / 0.25) * 0.25, 2)

def is_quarter_degree(v: float) -> bool:
    # A number is a multiple of 0.25 if v * 4 is an integer
    return bool(np.isclose(v * 4.0, np.round(v * 4.0), atol=1e-5))

def main():
    json_path = Path("data/processed/argo_profiles.json")
    csv_path = Path("data/processed/argo_profiles_all_852.csv")

    assert json_path.exists(), f"File not found: {json_path}"
    assert csv_path.exists(), f"File not found: {csv_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        profiles_before = json.load(f)

    df_csv_before = pd.read_csv(csv_path, encoding="utf-8")

    n_json_before = len(profiles_before)
    n_csv_before = len(df_csv_before)
    assert n_json_before == 852, f"Expected 852 JSON profiles, got {n_json_before}"
    assert n_csv_before == 852, f"Expected 852 CSV rows, got {n_csv_before}"

    print(f"[1] Loaded {n_json_before} JSON profiles and {n_csv_before} CSV rows.")

    # 1. Process JSON
    profiles_after = []
    lat_before_list = []
    lon_before_list = []
    lat_after_list = []
    lon_after_list = []

    for p in profiles_before:
        new_p = copy.deepcopy(p)
        orig_lat = p["lat"]
        orig_lon = p["lon"]
        r_lat = round_coord(orig_lat)
        r_lon = round_coord(orig_lon)

        new_p["lat"] = r_lat
        new_p["lon"] = r_lon

        # Verification: all other keys must be identical
        for k in ["platform", "cycle", "date", "z_min", "z_max", "temps"]:
            assert new_p[k] == p[k], f"Field {k} altered for float {p['platform']} cycle {p['cycle']}"

        profiles_after.append(new_p)
        lat_before_list.append(orig_lat)
        lon_before_list.append(orig_lon)
        lat_after_list.append(r_lat)
        lon_after_list.append(r_lon)

    # 2. Process CSV
    df_csv_after = df_csv_before.copy()
    lat_col = [c for c in df_csv_after.columns if "Latitude" in c][0]
    lon_col = [c for c in df_csv_after.columns if "Longitude" in c][0]

    df_csv_after[lat_col] = df_csv_after[lat_col].apply(round_coord)
    df_csv_after[lon_col] = df_csv_after[lon_col].apply(round_coord)

    # Verification: check all other columns identical
    other_cols = [c for c in df_csv_before.columns if c not in (lat_col, lon_col)]
    for col in other_cols:
        assert (df_csv_before[col].values == df_csv_after[col].values).all(), f"Column {col} altered in CSV!"

    # 3. Check multiples of 0.25
    all_lat_mult_json = all(is_quarter_degree(l) for l in lat_after_list)
    all_lon_mult_json = all(is_quarter_degree(l) for l in lon_after_list)
    all_lat_mult_csv = all(is_quarter_degree(l) for l in df_csv_after[lat_col].values)
    all_lon_mult_csv = all(is_quarter_degree(l) for l in df_csv_after[lon_col].values)

    assert all_lat_mult_json, "Not all JSON latitudes are multiples of 0.25!"
    assert all_lon_mult_json, "Not all JSON longitudes are multiples of 0.25!"
    assert all_lat_mult_csv, "Not all CSV latitudes are multiples of 0.25!"
    assert all_lon_mult_csv, "Not all CSV longitudes are multiples of 0.25!"

    # 4. Save updated files
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(profiles_after, f, indent=2)
    print(f"[2] Successfully saved updated {json_path}")

    df_csv_after.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[3] Successfully saved updated {csv_path}")

    # Copy to qa_package if exists
    qa_json = Path("qa_package/data/processed/argo_profiles.json")
    qa_csv = Path("qa_package/data/processed/argo_profiles_all_852.csv")
    if qa_json.parent.exists():
        shutil.copy2(json_path, qa_json)
        print(f"[4] Synced to {qa_json}")
    if qa_csv.parent.exists():
        shutil.copy2(csv_path, qa_csv)
        print(f"[5] Synced to {qa_csv}")

    # Print summary statistics
    print("\n" + "=" * 65)
    print("COORDINATE ROUNDING AUDIT REPORT")
    print("=" * 65)
    print(f"Profiles count before:            {n_json_before}")
    print(f"Profiles count after:             {len(profiles_after)}")
    print(f"Latitude range before:            [{min(lat_before_list):.4f}°N, {max(lat_before_list):.4f}°N]")
    print(f"Latitude range after:             [{min(lat_after_list):.2f}°N, {max(lat_after_list):.2f}°N]")
    print(f"Longitude range before:           [{min(lon_before_list):.4f}°E, {max(lon_before_list):.4f}°E]")
    print(f"Longitude range after:            [{min(lon_after_list):.2f}°E, {max(lon_after_list):.2f}°E]")
    print(f"All latitudes exact mult of 0.25: {all_lat_mult_json}")
    print(f"All longitudes exact mult of 0.25:{all_lon_mult_json}")
    print(f"All temperatures unchanged:       True")
    print(f"All depths unchanged:             True")
    print(f"Raw NetCDF files touched:         False (0 modified)")
    print("=" * 65)

    print("\nExamples of Coordinate Transformations:")
    examples_idx = [0, 1, 6, 9, 20, 50, 100, 200]
    for idx in examples_idx:
        p_b = profiles_before[idx]
        p_a = profiles_after[idx]
        print(f"  Float #{p_b['platform']} (Cycle {p_b['cycle']}) [{p_b['date']}]: "
              f"({p_b['lat']:.4f}°N, {p_b['lon']:.4f}°E) -> ({p_a['lat']:.2f}°N, {p_a['lon']:.2f}°E)")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
==============================================================================
OceanEmbed - Subsurface Ocean Temperature Reconstruction (SIH 26066)
Module: scripts/inspect/inspect_glorys.py
Description: Inspection tool for raw GLORYS NetCDF datasets using xarray.
             Displays dimensions, coordinates, depth levels, variable metadata,
             missing value counts, and physical sanity statistics.
==============================================================================
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

DEFAULT_RAW_DIR = Path("data/raw/glorys")


def get_project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent.parent


def find_latest_nc_file(raw_dir: Path) -> Optional[Path]:
    """Find the most recently modified NetCDF file in raw_dir."""
    if not raw_dir.exists():
        return None
    nc_files = list(raw_dir.glob("*.nc"))
    if not nc_files:
        return None
    # Sort by modification time descending
    nc_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return nc_files[0]


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect downloaded GLORYS NetCDF datasets for OceanEmbed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Path to specific NetCDF file. If not provided, automatically picks the latest file in data/raw/glorys/.",
    )
    return parser.parse_args()


def inspect_glorys_file(file_path: Path) -> None:
    """Open and inspect the GLORYS NetCDF file."""
    if not file_path.exists():
        print(f"[ERROR] File does not exist: {file_path}", file=sys.stderr)
        sys.exit(1)
    
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    
    try:
        import xarray as xr
        import numpy as np
    except ImportError as e:
        print(
            f"[ERROR] Missing required library: {e}\n"
            "Please ensure xarray and netCDF4 are installed: pip install xarray netCDF4",
            file=sys.stderr,
        )
        sys.exit(1)
    
    print("=" * 75)
    print("GLORYS NETCDF INSPECTION REPORT")
    print("=" * 75)
    print(f"File Path    : {file_path.resolve()}")
    print(f"File Size    : {file_size_mb:.2f} MB\n")
    
    try:
        ds = xr.open_dataset(file_path)
    except Exception as e:
        print(f"[ERROR] Failed to open NetCDF file with xarray: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 1. Dimensions
    print("-" * 75)
    print("1. DIMENSIONS")
    print("-" * 75)
    for dim_name, size in ds.dims.items():
        print(f"  • {dim_name:<15}: {size}")
    
    # 2. Coordinates
    print("\n" + "-" * 75)
    print("2. COORDINATES & BOUNDS")
    print("-" * 75)
    
    # Time coordinate
    time_coord_name = None
    for cand in ["time", "time_counter", "valid_time"]:
        if cand in ds.coords:
            time_coord_name = cand
            break
    
    if time_coord_name:
        t_vals = ds[time_coord_name].values
        t_min = str(t_vals[0])[:19]
        t_max = str(t_vals[-1])[:19]
        print(f"  • Time Coord [{time_coord_name}]: {len(t_vals)} steps")
        print(f"    - Range: {t_min}  -->  {t_max}")
    else:
        print("  • Time Coord: Not found in coords")
        
    # Latitude coordinate
    lat_coord_name = None
    for cand in ["latitude", "lat", "nav_lat"]:
        if cand in ds.coords:
            lat_coord_name = cand
            break
            
    if lat_coord_name:
        lat_vals = ds[lat_coord_name].values
        print(f"  • Latitude [{lat_coord_name}]: {len(lat_vals)} points")
        print(f"    - Range: [{float(lat_vals.min()):.3f}°N, {float(lat_vals.max()):.3f}°N]")
        if len(lat_vals) > 1:
            lat_step = abs(float(lat_vals[1] - lat_vals[0]))
            print(f"    - Approx Resolution: ~{lat_step:.4f}° (~{lat_step*111:.1f} km)")
    
    # Longitude coordinate
    lon_coord_name = None
    for cand in ["longitude", "lon", "nav_lon"]:
        if cand in ds.coords:
            lon_coord_name = cand
            break
            
    if lon_coord_name:
        lon_vals = ds[lon_coord_name].values
        print(f"  • Longitude [{lon_coord_name}]: {len(lon_vals)} points")
        print(f"    - Range: [{float(lon_vals.min()):.3f}°E, {float(lon_vals.max()):.3f}°E]")
        if len(lon_vals) > 1:
            lon_step = abs(float(lon_vals[1] - lon_vals[0]))
            print(f"    - Approx Resolution: ~{lon_step:.4f}° (~{lon_step*111:.1f} km)")
            
    # Depth coordinate
    depth_coord_name = None
    for cand in ["depth", "deptht", "lev", "level"]:
        if cand in ds.coords:
            depth_coord_name = cand
            break
            
    if depth_coord_name:
        depth_vals = ds[depth_coord_name].values
        print(f"  • Depth [{depth_coord_name}]: {len(depth_vals)} levels")
        print(f"    - Range: [{float(depth_vals.min()):.2f} m, {float(depth_vals.max()):.2f} m]")
        print("    - Levels (m): " + ", ".join(f"{float(d):.1f}" for d in depth_vals))
    else:
        print("  • Depth: Not found in coords")
        
    # 3. Data Variables
    print("\n" + "-" * 75)
    print("3. DATA VARIABLES")
    print("-" * 75)
    for var_name, data_array in ds.data_vars.items():
        dims_str = ", ".join(f"{d}:{s}" for d, s in zip(data_array.dims, data_array.shape))
        units = data_array.attrs.get("units", "N/A")
        long_name = data_array.attrs.get("long_name", "N/A")
        print(f"  • {var_name}")
        print(f"    - Dimensions : ({dims_str})")
        print(f"    - Dtype      : {data_array.dtype}")
        print(f"    - Units      : {units}")
        print(f"    - Description: {long_name}")
        
    # 4. Target Variable Analysis (thetao)
    target_var = "thetao" if "thetao" in ds.data_vars else list(ds.data_vars.keys())[0] if len(ds.data_vars) > 0 else None
    
    if target_var:
        print("\n" + "-" * 75)
        print(f"4. TARGET VARIABLE PROFILE: '{target_var}'")
        print("-" * 75)
        var_data = ds[target_var]
        
        # Total elements and missing values
        total_elements = var_data.size
        nan_count = int(np.isnan(var_data.values).sum())
        valid_count = total_elements - nan_count
        nan_pct = (nan_count / total_elements) * 100.0 if total_elements > 0 else 0.0
        
        print(f"  • Total Grid Points : {total_elements:,}")
        print(f"  • Valid Ocean Points: {valid_count:,} ({100.0 - nan_pct:.1f}%)")
        print(f"  • Land / NaN Mask   : {nan_count:,} ({nan_pct:.1f}%)")
        
        # Physical values summary
        if valid_count > 0:
            val_min = float(np.nanmin(var_data.values))
            val_max = float(np.nanmax(var_data.values))
            val_mean = float(np.nanmean(var_data.values))
            val_std = float(np.nanstd(var_data.values))
            
            print(f"  • Min Value         : {val_min:.3f} °C")
            print(f"  • Max Value         : {val_max:.3f} °C")
            print(f"  • Mean Value        : {val_mean:.3f} °C")
            print(f"  • Std Deviation     : {val_std:.3f} °C")
            
            # Physical Sanity Check (Surface vs Bottom)
            if depth_coord_name and len(ds[depth_coord_name]) > 1:
                surface_idx = 0
                deep_idx = len(ds[depth_coord_name]) - 1
                surf_depth = float(ds[depth_coord_name].values[surface_idx])
                deep_depth = float(ds[depth_coord_name].values[deep_idx])
                
                surf_slice = var_data.isel({depth_coord_name: surface_idx}).values
                deep_slice = var_data.isel({depth_coord_name: deep_idx}).values
                
                surf_mean = float(np.nanmean(surf_slice))
                deep_mean = float(np.nanmean(deep_slice))
                
                print("\n  [Physical Sanity Check]")
                print(f"    - Surface Layer (~{surf_depth:.1f} m) Mean Temp: {surf_mean:.2f} °C")
                print(f"    - Deep Layer (~{deep_depth:.1f} m) Mean Temp   : {deep_mean:.2f} °C")
                print(f"    - Vertical Thermal Gradient (Surf - Deep)   : {surf_mean - deep_mean:.2f} °C")
                if surf_mean > deep_mean:
                    print("    ✓ Physical profile verified: Surface is warmer than deep ocean.")
                else:
                    print("    ! Note: Check thermal stratification behavior.")
    
    # 5. Global Metadata Attributes
    if ds.attrs:
        print("\n" + "-" * 75)
        print("5. GLOBAL ATTRIBUTES (Selected)")
        print("-" * 75)
        for k in ["title", "institution", "source", "references", "comment", "Conventions"]:
            if k in ds.attrs:
                print(f"  • {k}: {ds.attrs[k]}")
    
    ds.close()
    print("\n" + "=" * 75)
    print("[SUCCESS] Inspection completed successfully.")
    print("=" * 75)


def main() -> None:
    """Main execution function."""
    args = parse_arguments()
    project_root = get_project_root()
    
    if args.file:
        file_path = args.file
    else:
        raw_dir = project_root / DEFAULT_RAW_DIR
        print(f"[*] Searching for NetCDF files in {raw_dir}...")
        latest_file = find_latest_nc_file(raw_dir)
        if not latest_file:
            print(
                f"\n[ERROR] No NetCDF (*.nc) files found in {raw_dir.resolve()}.\n"
                "Please run the download script first:\n"
                "  python scripts/download/download_glorys.py --test\n"
                "Or provide a specific file with --file <path_to_nc>",
                file=sys.stderr,
            )
            sys.exit(1)
        file_path = latest_file
        print(f"[+] Found latest file: {file_path.name}")
        
    inspect_glorys_file(file_path)


if __name__ == "__main__":
    main()

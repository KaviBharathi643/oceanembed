"""
OceanEmbed (SIH 2026 Problem Statement 26066)
PyTorch Dataset Implementation for Ocean Subsurface Temperature Profile Estimation

Provides:
- OceanEmbedDataset: PyTorch Dataset loading standardized 7-channel 5x5 surface patches
  and 15-depth GLORYS target profiles from processed NetCDF files.
- Metadata constants for channel ordering and target depths.
"""

import os
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

# Exact input channel ordering (7 surface variables)
INPUT_CHANNELS: List[str] = [
    "sst",        # 0: Sea Surface Temperature (NOAA OISST v2.1)
    "sss",        # 1: Sea Surface Salinity (NASA SMAP RSS L3)
    "sla",        # 2: Sea Level Anomaly (CMEMS DUACS)
    "current_u",  # 3: Zonal Surface Current (NASA OSCAR v2.0)
    "current_v",  # 4: Meridional Surface Current (NASA OSCAR v2.0)
    "wind_u",     # 5: Zonal 10m Wind (NASA CCMP v3.1 daily vector mean)
    "wind_v",     # 6: Meridional 10m Wind (NASA CCMP v3.1 daily vector mean)
]

# Exact target depth levels (15 depths in meters)
TARGET_DEPTHS: List[float] = [
    0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0,
    125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0
]

NUM_INPUT_CHANNELS: int = len(INPUT_CHANNELS)  # 7
NUM_TARGET_DEPTHS: int = len(TARGET_DEPTHS)    # 15
PATCH_SHAPE: Tuple[int, int] = (5, 5)


class OceanEmbedDataset(Dataset):
    """
    PyTorch Dataset for OceanEmbed.

    Loads preprocessed inputs X of shape [7, 5, 5] (standardized)
    and target temperature profiles y of shape [15] (physical degrees Celsius).

    Attributes:
        nc_path (str): Path to the processed NetCDF file.
        in_memory (bool): If True, preloads arrays into RAM for fast indexed access.
        channels (List[str]): List of 7 input channel names in exact order.
        depths (List[float]): List of 15 target depth levels in meters.
    """

    def __init__(
        self,
        nc_path: str,
        in_memory: bool = True,
        load_coords: bool = False,
    ) -> None:
        super().__init__()
        if not os.path.isfile(nc_path):
            raise FileNotFoundError(f"Processed dataset file not found: {nc_path}")

        self.nc_path = nc_path
        self.in_memory = in_memory
        self.load_coords = load_coords
        self.channels = list(INPUT_CHANNELS)
        self.depths = list(TARGET_DEPTHS)

        # Open dataset to read dimensions and metadata
        with xr.open_dataset(self.nc_path) as ds:
            self._num_samples = int(ds.sizes["sample"])
            
            # Verify channel ordering in NetCDF file
            if "channel" in ds.coords:
                nc_channels = [str(c) for c in ds["channel"].values]
                if nc_channels != self.channels:
                    raise ValueError(
                        f"Channel mismatch in {nc_path}: expected {self.channels}, got {nc_channels}"
                    )
            
            # Verify depth ordering in NetCDF file
            if "depth" in ds.coords:
                nc_depths = [float(d) for d in ds["depth"].values]
                if nc_depths != self.depths:
                    raise ValueError(
                        f"Depth mismatch in {nc_path}: expected {self.depths}, got {nc_depths}"
                    )

            if self.in_memory:
                # Load X and y into contiguous numpy arrays
                self.X = ds["X"].values.astype(np.float32)  # [N, 7, 5, 5]
                self.y = ds["y"].values.astype(np.float32)  # [N, 15]
                if self.load_coords:
                    self.dates = ds["date"].values.astype(str)
                    self.lats = ds["lat"].values.astype(np.float32)
                    self.lons = ds["lon"].values.astype(np.float32)
                    self.grid_i = ds["grid_i"].values.astype(np.int32)
                    self.grid_j = ds["grid_j"].values.astype(np.int32)
            else:
                self.X = None
                self.y = None

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve sample at index `idx`.

        Returns:
            x (torch.FloatTensor): Input tensor of shape [7, 5, 5].
            y (torch.FloatTensor): Target profile tensor of shape [15] in physical deg C.
        """
        if self.in_memory:
            x_arr = self.X[idx]
            y_arr = self.y[idx]
        else:
            with xr.open_dataset(self.nc_path) as ds:
                x_arr = ds["X"].isel(sample=idx).values.astype(np.float32)
                y_arr = ds["y"].isel(sample=idx).values.astype(np.float32)

        x_tensor = torch.from_numpy(x_arr)
        y_tensor = torch.from_numpy(y_arr)
        return x_tensor, y_tensor

    def get_sample_metadata(self, idx: int) -> Dict[str, Any]:
        """Retrieve coordinate and date metadata for a given sample index."""
        if self.in_memory and self.load_coords:
            return {
                "date": str(self.dates[idx]),
                "lat": float(self.lats[idx]),
                "lon": float(self.lons[idx]),
                "grid_i": int(self.grid_i[idx]),
                "grid_j": int(self.grid_j[idx]),
            }
        with xr.open_dataset(self.nc_path) as ds:
            return {
                "date": str(ds["date"].isel(sample=idx).values),
                "lat": float(ds["lat"].isel(sample=idx).values),
                "lon": float(ds["lon"].isel(sample=idx).values),
                "grid_i": int(ds["grid_i"].isel(sample=idx).values),
                "grid_j": int(ds["grid_j"].isel(sample=idx).values),
            }

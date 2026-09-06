"""
OceanEmbed (SIH 2026 Problem Statement 26066)
PyTorch DataLoader Factory Module

Provides functions to construct train, validation, and test PyTorch DataLoaders.
"""

import os
from typing import Dict, Tuple, Optional, Any
import yaml
import torch
from torch.utils.data import DataLoader

from src.data.dataset import OceanEmbedDataset, INPUT_CHANNELS, TARGET_DEPTHS


def get_dataloader(
    nc_path: str,
    batch_size: int = 256,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    in_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Construct a PyTorch DataLoader for a specific processed NetCDF file.

    Args:
        nc_path (str): Path to the processed NetCDF file.
        batch_size (int): Number of samples per batch (default: 256).
        shuffle (bool): Whether to shuffle samples each epoch.
        num_workers (int): Number of worker subprocesses for data loading.
        pin_memory (bool, optional): If True, pin memory for faster GPU transfers.
                                     Defaults to True if CUDA is available, else False.
        in_memory (bool): If True, preloads arrays into RAM.
        drop_last (bool): Whether to drop the last incomplete batch.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    dataset = OceanEmbedDataset(nc_path=nc_path, in_memory=in_memory)
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return loader


def create_dataloaders(
    train_nc: str = "data/processed/oceanembed_train.nc",
    val_nc: str = "data/processed/oceanembed_val.nc",
    test_nc: str = "data/processed/oceanembed_test.nc",
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    in_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory function to create train, validation, and test DataLoaders.

    Rules:
      - Train loader: shuffle=True
      - Validation loader: shuffle=False
      - Test loader: shuffle=False

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
    """
    train_loader = get_dataloader(
        nc_path=train_nc,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        in_memory=in_memory,
        drop_last=False,
    )
    val_loader = get_dataloader(
        nc_path=val_nc,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        in_memory=in_memory,
        drop_last=False,
    )
    test_loader = get_dataloader(
        nc_path=test_nc,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        in_memory=in_memory,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader


def create_dataloaders_from_config(
    config_path: str = "config/dataset_config.yaml",
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Construct train, validation, and test DataLoaders using paths and settings from config YAML.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    train_nc = data_cfg.get("train_file", "data/processed/oceanembed_train.nc")
    val_nc = data_cfg.get("val_file", "data/processed/oceanembed_val.nc")
    test_nc = data_cfg.get("test_file", "data/processed/oceanembed_test.nc")

    bs = batch_size if batch_size is not None else data_cfg.get("batch_size", 256)
    nw = num_workers if num_workers is not None else data_cfg.get("num_workers", 0)

    return create_dataloaders(
        train_nc=train_nc,
        val_nc=val_nc,
        test_nc=test_nc,
        batch_size=bs,
        num_workers=nw,
    )

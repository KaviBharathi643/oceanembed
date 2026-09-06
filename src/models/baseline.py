"""
OceanEmbed (SIH 2026 Problem Statement 26066)
Baseline Model: Depth-Wise Climatological Mean

Predicts the training-set mean temperature at each of the 15 depths,
regardless of input features. Any useful model must substantially beat this baseline.
"""

import numpy as np
import torch
from typing import List, Dict


class ClimatologicalBaseline:
    """
    Baseline that predicts the per-depth training mean for every sample.

    Usage:
        baseline = ClimatologicalBaseline.from_dataset(train_dataset)
        predictions = baseline.predict(num_samples)
        rmse = baseline.evaluate(y_true)
    """

    def __init__(self, depth_means: np.ndarray, depths: List[float]):
        """
        Args:
            depth_means: Array of shape [15] with mean temperature at each depth.
            depths: List of 15 depth values in meters.
        """
        self.depth_means = depth_means.astype(np.float32)
        self.depths = depths

    @classmethod
    def from_dataset(cls, dataset) -> "ClimatologicalBaseline":
        """Compute depth-wise means from an OceanEmbedDataset."""
        y_all = dataset.y  # [N, 15]
        depth_means = np.mean(y_all, axis=0)  # [15]
        return cls(depth_means=depth_means, depths=dataset.depths)

    def predict(self, num_samples: int) -> np.ndarray:
        """Return constant predictions of shape [num_samples, 15]."""
        return np.tile(self.depth_means, (num_samples, 1))

    def predict_tensor(self, num_samples: int) -> torch.Tensor:
        """Return constant predictions as a torch.FloatTensor of shape [num_samples, 15]."""
        return torch.from_numpy(self.predict(num_samples))

    def evaluate(self, y_true: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Evaluate baseline RMSE, bias, and R² against true targets.

        Args:
            y_true: True temperatures of shape [N, 15]

        Returns:
            Dictionary with per-depth RMSE, bias, R², and overall RMSE.
        """
        n = y_true.shape[0]
        y_pred = self.predict(n)

        errors = y_pred - y_true
        per_depth_rmse = np.sqrt(np.mean(errors ** 2, axis=0))  # [15]
        per_depth_bias = np.mean(errors, axis=0)                # [15]

        # R² per depth
        ss_res = np.sum(errors ** 2, axis=0)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
        per_depth_r2 = 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0)

        overall_rmse = float(np.sqrt(np.mean(errors ** 2)))

        return {
            "per_depth_rmse": per_depth_rmse,
            "per_depth_bias": per_depth_bias,
            "per_depth_r2": per_depth_r2,
            "overall_rmse": overall_rmse,
            "depth_means": self.depth_means,
        }

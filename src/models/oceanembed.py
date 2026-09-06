"""
OceanEmbed (SIH 2026 Problem Statement 26066)
CNN Model Architecture

Architecture:
    7 surface variables × 5×5 spatial patch
        → CNN Encoder (3 conv blocks: 32→64→128, spatial 5×5→3×3→1×1)
        → Ocean Embedding (Linear 128→64, ReLU, Dropout)
        → Profile Decoder (MLP 64→128→64→15)
        → 15-depth subsurface temperature profile (°C)

Total parameters: ~120K
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any

from src.data.dataset import INPUT_CHANNELS, TARGET_DEPTHS, NUM_INPUT_CHANNELS, NUM_TARGET_DEPTHS


class OceanEmbedEncoder(nn.Module):
    """
    CNN Encoder that extracts spatial features from a 7-channel 5×5 patch.

    Three successive 3×3 convolutions naturally collapse the spatial extent:
      5×5 → 5×5 (pad=1) → 3×3 (pad=0) → 1×1 (pad=0)

    Every neuron in the final layer has a receptive field covering the entire 5×5 input.
    No explicit pooling is needed.
    """

    def __init__(self, in_channels: int = 7):
        super().__init__()
        # Conv Block 1: [B, 7, 5, 5] → [B, 32, 5, 5]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Conv Block 2: [B, 32, 5, 5] → [B, 64, 3, 3]
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Conv Block 3: [B, 64, 3, 3] → [B, 128, 1, 1]
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, 7, 5, 5]
        Returns:
            Flattened CNN features of shape [B, 128]
        """
        x = self.conv1(x)   # [B, 32, 5, 5]
        x = self.conv2(x)   # [B, 64, 3, 3]
        x = self.conv3(x)   # [B, 128, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # [B, 128]
        return x


class OceanEmbedding(nn.Module):
    """
    Information bottleneck that compresses the 128-dim CNN feature vector
    into a compact 64-dim latent representation of the local upper-ocean state.

    The embedding integrates:
      - Surface thermal state (SST)
      - Salinity / freshwater flux (SSS)
      - Thermocline depth proxy (SLA)
      - Surface circulation (current_u, current_v)
      - Wind forcing (wind_u, wind_v)
      - Local spatial structure (from 5×5 convolutions)
    """

    def __init__(self, input_dim: int = 128, embedding_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: CNN features of shape [B, 128]
        Returns:
            Ocean embedding of shape [B, 64]
        """
        return self.projection(x)


class ProfileDecoder(nn.Module):
    """
    MLP decoder that maps the 64-dim ocean embedding to a 15-depth
    subsurface temperature profile.

    All 15 depths are predicted jointly, preserving inter-depth correlations
    and ensuring vertical coherence.
    """

    def __init__(self, embedding_dim: int = 64, num_depths: int = 15, dropout: float = 0.1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_depths),  # No activation — output is physical °C
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding: Ocean embedding of shape [B, 64]
        Returns:
            Temperature profile of shape [B, 15] in physical °C
        """
        return self.decoder(embedding)


class OceanEmbedModel(nn.Module):
    """
    Complete OceanEmbed model: CNN Encoder → Ocean Embedding → Profile Decoder.

    Input:  [B, 7, 5, 5]  (standardized surface variables on 5×5 patch)
    Output: [B, 15]        (temperature at 15 depths in physical °C)

    Attributes:
        channels: List of 7 input channel names in order
        depths: List of 15 target depth levels in meters
        embedding_dim: Dimensionality of the ocean embedding (default: 64)
    """

    def __init__(
        self,
        in_channels: int = NUM_INPUT_CHANNELS,
        embedding_dim: int = 64,
        num_depths: int = NUM_TARGET_DEPTHS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channels = list(INPUT_CHANNELS)
        self.depths = list(TARGET_DEPTHS)
        self.embedding_dim = embedding_dim

        self.encoder = OceanEmbedEncoder(in_channels=in_channels)
        self.embedding = OceanEmbedding(
            input_dim=128, embedding_dim=embedding_dim, dropout=dropout
        )
        self.decoder = ProfileDecoder(
            embedding_dim=embedding_dim, num_depths=num_depths, dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: surface patch → temperature profile.

        Args:
            x: Input tensor of shape [B, 7, 5, 5]
        Returns:
            Predicted temperature profile of shape [B, 15] in physical °C
        """
        features = self.encoder(x)          # [B, 128]
        emb = self.embedding(features)      # [B, 64]
        profile = self.decoder(emb)         # [B, 15]
        return profile

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the ocean embedding without decoding.

        Args:
            x: Input tensor of shape [B, 7, 5, 5]
        Returns:
            Ocean embedding of shape [B, embedding_dim]
        """
        features = self.encoder(x)
        return self.embedding(features)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_summary(self) -> Dict[str, int]:
        """Return parameter count breakdown by component."""
        summary = {}
        for name, module in [
            ("encoder", self.encoder),
            ("embedding", self.embedding),
            ("decoder", self.decoder),
        ]:
            count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            summary[name] = count
        summary["total"] = self.count_parameters()
        return summary

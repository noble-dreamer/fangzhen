from __future__ import annotations

import torch
from torch import nn


class XMatrixEncoder(nn.Module):
    """Encode x_matrix with shape (B, Cx, F, TX, RX) into one embedding."""

    def __init__(
        self,
        *,
        x_channels: int = 7,
        frequency_count: int = 15,
        embedding_dim: int = 256,
        hidden_channels: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        in_channels = int(x_channels) * int(frequency_count)
        self.x_channels = int(x_channels)
        self.frequency_count = int(frequency_count)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, hidden_channels), num_channels=hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, hidden_channels), num_channels=hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=min(8, hidden_channels * 2), num_channels=hidden_channels * 2),
            nn.SiLU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, hidden_channels * 2), num_channels=hidden_channels * 2),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x_matrix: torch.Tensor) -> torch.Tensor:
        if x_matrix.ndim != 5:
            raise RuntimeError(f"x_matrix must be (B,C,F,TX,RX), got {tuple(x_matrix.shape)}")
        batch, channels, freq, tx, rx = x_matrix.shape
        if channels != self.x_channels:
            raise RuntimeError(f"Expected {self.x_channels} x channels, got {channels}")
        if freq != self.frequency_count:
            raise RuntimeError(f"Expected {self.frequency_count} frequencies, got {freq}")
        x = x_matrix.reshape(batch, channels * freq, tx, rx)
        return self.net(x)

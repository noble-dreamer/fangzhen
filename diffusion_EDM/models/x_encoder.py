from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .physical_encoding import (
    PhysicalEncodingConfig,
    PhysicalFrequencyEmbedding,
    PhysicalTxRxPositionEmbedding,
    sort_frequency_axis,
)


class DynamicFrequencyDecomposer(nn.Module):
    """Content-adaptive low/high decomposition along sorted frequencies."""

    def __init__(self, x_channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("frequency filter kernel_size must be an odd integer >= 3")
        self.x_channels = int(x_channels)
        self.kernel_size = int(kernel_size)
        hidden_channels = max(16, self.x_channels * 2)
        self.filter_generator = nn.Sequential(
            nn.Conv1d(2, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_channels, self.x_channels * self.kernel_size, kernel_size=1),
        )
        final = self.filter_generator[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, x_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x_matrix.ndim != 5:
            raise RuntimeError(f"x_matrix must be (B,C,F,TX,RX), got {tuple(x_matrix.shape)}")
        if x_matrix.shape[1] != self.x_channels:
            raise RuntimeError(f"Expected {self.x_channels} x channels, got {x_matrix.shape[1]}")

        stats_source = x_matrix.float()
        mean = stats_source.mean(dim=(1, 3, 4))
        std = stats_source.var(dim=(1, 3, 4), unbiased=False).add(1e-6).sqrt()
        stats = torch.stack([mean, std], dim=1).to(dtype=x_matrix.dtype)
        logits = self.filter_generator(stats).flatten(1)
        weights = logits.reshape(x_matrix.shape[0], self.x_channels, self.kernel_size)
        weights = torch.softmax(weights.float(), dim=-1).to(dtype=x_matrix.dtype)

        pad = self.kernel_size // 2
        padded = F.pad(x_matrix, (0, 0, 0, 0, pad, pad), mode="replicate")
        windows = padded.unfold(2, self.kernel_size, 1)
        low = (windows * weights[:, :, None, None, None, :]).sum(dim=-1)
        high = x_matrix - low
        return low, high, weights


class XMatrixEncoder(nn.Module):
    """Encode x_matrix into a global vector and physical low/high TX-RX tokens."""

    def __init__(
        self,
        *,
        x_channels: int = 7,
        frequency_count: int = 15,
        embedding_dim: int = 256,
        hidden_channels: int = 64,
        dropout: float = 0.0,
        token_dim: int | None = None,
        frequency_filter_kernel_size: int = 5,
        physical_encoding: PhysicalEncodingConfig | None = None,
    ) -> None:
        super().__init__()
        in_channels = int(x_channels) * int(frequency_count)
        self.x_channels = int(x_channels)
        self.frequency_count = int(frequency_count)
        self.token_dim = int(token_dim or embedding_dim)
        self.physical_encoding = physical_encoding or PhysicalEncodingConfig()
        self.frequency_decomposer = DynamicFrequencyDecomposer(
            self.x_channels,
            kernel_size=int(frequency_filter_kernel_size),
        )
        self.frequency_position_embedding = PhysicalFrequencyEmbedding(
            self.x_channels,
            reference_hz=self.physical_encoding.frequency_reference_hz,
        )
        self.tx_rx_position_embedding = PhysicalTxRxPositionEmbedding(
            self.token_dim,
            self.physical_encoding,
        )
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, hidden_channels), num_channels=hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, hidden_channels), num_channels=hidden_channels),
            nn.SiLU(),
        )
        self.global_net = nn.Sequential(
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
        self.token_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, self.token_dim, kernel_size=1),
            nn.GroupNorm(num_groups=min(8, self.token_dim), num_channels=self.token_dim),
            nn.SiLU(),
            nn.Conv2d(self.token_dim, self.token_dim, kernel_size=1),
        )

    def encode_features(self, x_matrix: torch.Tensor, frequency_hz: torch.Tensor) -> torch.Tensor:
        if x_matrix.ndim != 5:
            raise RuntimeError(f"x_matrix must be (B,C,F,TX,RX), got {tuple(x_matrix.shape)}")
        batch, channels, freq, tx, rx = x_matrix.shape
        if channels != self.x_channels:
            raise RuntimeError(f"Expected {self.x_channels} x channels, got {channels}")
        if freq != self.frequency_count:
            raise RuntimeError(f"Expected {self.frequency_count} frequencies, got {freq}")
        frequency_embedding = self.frequency_position_embedding(frequency_hz, dtype=x_matrix.dtype)
        x = (x_matrix + frequency_embedding).reshape(batch, channels * freq, tx, rx)
        return self.stem(x)

    def _project_tokens(self, features: torch.Tensor) -> torch.Tensor:
        token_map = self.token_proj(features)
        return token_map.flatten(2).transpose(1, 2).contiguous()

    def forward(
        self,
        x_matrix: torch.Tensor,
        *,
        frequency_hz: torch.Tensor,
        tx_indices: torch.Tensor,
        rx_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.physical_encoding
        if x_matrix.shape[-2:] != (cfg.tx_count, cfg.rx_count):
            raise RuntimeError(
                "x_matrix TX-RX shape must match physical encoding config: "
                f"expected {(cfg.tx_count, cfg.rx_count)}, got {tuple(x_matrix.shape[-2:])}"
            )
        sorted_x, sorted_frequency_hz, _ = sort_frequency_axis(x_matrix, frequency_hz)
        features = self.encode_features(sorted_x, sorted_frequency_hz)
        low, high, _ = self.frequency_decomposer(sorted_x)
        low_features = self.encode_features(low, sorted_frequency_hz)
        high_features = self.encode_features(high, sorted_frequency_hz)
        embedding = self.global_net(features)
        low_tokens = self._project_tokens(low_features)
        high_tokens = self._project_tokens(high_features)
        position = self.tx_rx_position_embedding(
            tx_indices,
            rx_indices,
            batch_size=x_matrix.shape[0],
            device=x_matrix.device,
            dtype=low_tokens.dtype,
        )
        return embedding, low_tokens + position, high_tokens + position

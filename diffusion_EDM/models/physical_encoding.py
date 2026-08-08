from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class PhysicalEncodingConfig:
    """Nominal Dataset-A geometry used to locate frequency and TX-RX data."""

    tx_count: int = 16
    rx_count: int = 16
    pipe_length_mm: float = 1000.0
    mid_radius_mm: float = 155.0
    tx_z_mm: float = 100.0
    rx_z_mm: float = 900.0
    frequency_reference_hz: float = 100000.0

    def __post_init__(self) -> None:
        if self.tx_count <= 0 or self.rx_count <= 0:
            raise ValueError("tx_count and rx_count must be positive")
        if self.pipe_length_mm <= 0.0 or self.mid_radius_mm <= 0.0:
            raise ValueError("pipe_length_mm and mid_radius_mm must be positive")
        if self.frequency_reference_hz <= 0.0:
            raise ValueError("frequency_reference_hz must be positive")


def as_batched_coordinate(
    values: torch.Tensor,
    *,
    name: str,
    batch_size: int,
    count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = torch.as_tensor(values, device=device)
    if values.ndim == 1:
        values = values.unsqueeze(0)
    if values.ndim != 2 or values.shape[1] != count:
        raise RuntimeError(f"{name} must be ({count},) or (B,{count}), got {tuple(values.shape)}")
    if values.shape[0] == 1 and batch_size > 1:
        values = values.expand(batch_size, -1)
    if values.shape[0] != batch_size:
        raise RuntimeError(f"{name} batch size must be 1 or {batch_size}, got {values.shape[0]}")
    values = values.to(dtype=dtype)
    if not torch.isfinite(values).all():
        raise RuntimeError(f"{name} must contain only finite values")
    return values


def sort_frequency_axis(
    x_matrix: torch.Tensor,
    frequency_hz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort each sample's physical frequency axis and return the gather order."""

    if x_matrix.ndim != 5:
        raise RuntimeError(f"x_matrix must be (B,C,F,TX,RX), got {tuple(x_matrix.shape)}")
    batch, _, frequency_count, _, _ = x_matrix.shape
    frequencies = as_batched_coordinate(
        frequency_hz,
        name="frequency_hz",
        batch_size=batch,
        count=frequency_count,
        device=x_matrix.device,
        dtype=torch.float32,
    )
    if (frequencies <= 0.0).any():
        raise RuntimeError("frequency_hz must contain positive values")
    order = torch.argsort(frequencies, dim=1)
    sorted_frequencies = torch.gather(frequencies, dim=1, index=order)
    if frequency_count > 1 and not (sorted_frequencies[:, 1:] > sorted_frequencies[:, :-1]).all():
        raise RuntimeError("frequency_hz must contain unique values in every sample")
    gather_index = order[:, None, :, None, None].expand_as(x_matrix)
    return torch.gather(x_matrix, dim=2, index=gather_index), sorted_frequencies, order


class PhysicalFrequencyEmbedding(nn.Module):
    """Encode real frequencies before the F axis is folded into CNN channels."""

    def __init__(self, x_channels: int, reference_hz: float) -> None:
        super().__init__()
        if reference_hz <= 0.0:
            raise ValueError("reference_hz must be positive")
        self.reference_hz = float(reference_hz)
        self.harmonics = (1.0, 2.0, 4.0)
        feature_dim = 2 + 2 * len(self.harmonics)
        hidden_dim = max(16, int(x_channels) * 4)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, int(x_channels)),
        )

    def forward(self, frequency_hz: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        normalized = frequency_hz.float() / self.reference_hz
        features = [normalized, torch.log1p(normalized)]
        for harmonic in self.harmonics:
            phase = 2.0 * math.pi * harmonic * normalized
            features.extend((torch.sin(phase), torch.cos(phase)))
        embedding = self.projection(torch.stack(features, dim=-1))
        return embedding.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1).to(dtype=dtype)


class PhysicalTxRxPositionEmbedding(nn.Module):
    """Linearly project minimal cylindrical geometry for each TX-RX cell."""

    feature_dim = 6

    def __init__(self, token_dim: int, config: PhysicalEncodingConfig) -> None:
        super().__init__()
        self.config = config
        self.projection = nn.Linear(self.feature_dim, int(token_dim), bias=False)

    def physical_features(
        self,
        tx_indices: torch.Tensor,
        rx_indices: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        cfg = self.config
        tx_ids = as_batched_coordinate(
            tx_indices,
            name="tx_indices",
            batch_size=batch_size,
            count=cfg.tx_count,
            device=device,
            dtype=torch.float32,
        )
        rx_ids = as_batched_coordinate(
            rx_indices,
            name="rx_indices",
            batch_size=batch_size,
            count=cfg.rx_count,
            device=device,
            dtype=torch.float32,
        )
        if (tx_ids < 1.0).any() or (rx_ids < 1.0).any():
            raise RuntimeError("tx_indices and rx_indices must use the generated dataset's 1-based IDs")

        tx_origin = tx_ids.min(dim=1, keepdim=True).values
        rx_origin = rx_ids.min(dim=1, keepdim=True).values
        tx_local = torch.remainder(tx_ids - tx_origin, float(cfg.tx_count))
        rx_local = torch.remainder(rx_ids - rx_origin, float(cfg.rx_count))
        tx_angle = 2.0 * math.pi * tx_local / float(cfg.tx_count)
        rx_angle = 2.0 * math.pi * rx_local / float(cfg.rx_count)

        tx_angle = tx_angle[:, :, None].expand(-1, -1, cfg.rx_count)
        rx_angle = rx_angle[:, None, :].expand(-1, cfg.tx_count, -1)
        angle_delta = torch.remainder(rx_angle - tx_angle + math.pi, 2.0 * math.pi) - math.pi

        length = float(cfg.pipe_length_mm)
        radius = float(cfg.mid_radius_mm)
        axial_delta = float(cfg.rx_z_mm - cfg.tx_z_mm)
        signed_arc = radius * angle_delta
        path_length = torch.sqrt(signed_arc.square() + axial_delta**2)
        features = torch.stack(
            [
                torch.sin(tx_angle),
                torch.cos(tx_angle),
                torch.sin(rx_angle),
                torch.cos(rx_angle),
                angle_delta / math.pi,
                path_length / length,
            ],
            dim=-1,
        )
        return features.flatten(1, 2)

    def forward(
        self,
        tx_indices: torch.Tensor,
        rx_indices: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        features = self.physical_features(
            tx_indices,
            rx_indices,
            batch_size=batch_size,
            device=device,
        )
        return self.projection(features).to(dtype=dtype)

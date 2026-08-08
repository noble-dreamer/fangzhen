from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RayGeometry:
    length_mm: float = 1000.0
    mid_radius_mm: float = 155.0
    tx_z_mm: float = 100.0
    rx_z_mm: float = 900.0
    count_per_ring: int = 16
    helical_orders: tuple[int, ...] = (-1, 0, 1)
    sigma_ray_mm: float = 25.0
    min_endpoint_distance_mm: float = 30.0
    kernel_sigma_cutoff: float = 3.0


def wrap_rad(value: torch.Tensor) -> torch.Tensor:
    return torch.remainder(value + math.pi, 2.0 * math.pi) - math.pi


def _distance_to_segment(
    x: torch.Tensor,
    z: torch.Tensor,
    x2: float,
    z1: float,
    z2: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    vx = float(x2)
    vz = float(z2 - z1)
    length_sq = vx * vx + vz * vz
    if length_sq <= 0.0:
        distance = torch.sqrt(x * x + (z - z1) ** 2)
        return distance, torch.zeros_like(distance), 0.0
    t = (x * vx + (z - z1) * vz) / length_sq
    t_clamped = t.clamp(0.0, 1.0)
    proj_x = t_clamped * vx
    proj_z = z1 + t_clamped * vz
    distance = torch.sqrt((x - proj_x) ** 2 + (z - proj_z) ** 2)
    return distance, t_clamped, math.sqrt(length_sq)


def build_ray_kernels(
    *,
    image_shape: tuple[int, int],
    geometry: RayGeometry,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    z_count, theta_count = int(image_shape[0]), int(image_shape[1])
    theta = torch.linspace(0.0, 2.0 * math.pi, theta_count + 1, device=device, dtype=dtype)[:-1][None, :]
    z = torch.linspace(0.0, geometry.length_mm, z_count, device=device, dtype=dtype)[:, None]
    kernels: list[torch.Tensor] = []
    step = 2.0 * math.pi / geometry.count_per_ring
    for tx_index in range(geometry.count_per_ring):
        tx_theta = tx_index * step
        for rx_index in range(geometry.count_per_ring):
            rx_theta = rx_index * step
            base_delta = float(((rx_theta - tx_theta + math.pi) % (2.0 * math.pi)) - math.pi)
            for order in geometry.helical_orders:
                path_delta = base_delta + 2.0 * math.pi * int(order)
                x2 = geometry.mid_radius_mm * path_delta
                base_pixel = wrap_rad(theta - tx_theta)
                distances = []
                ts = []
                for image_order in range(order - 2, order + 3):
                    x = geometry.mid_radius_mm * (base_pixel + 2.0 * math.pi * image_order)
                    distance, t, _ = _distance_to_segment(x, z, x2, geometry.tx_z_mm, geometry.rx_z_mm)
                    distances.append(distance)
                    ts.append(t)
                stacked = torch.stack(distances, dim=0)
                best = torch.argmin(stacked, dim=0, keepdim=True)
                distance = torch.gather(stacked, 0, best)[0]
                t = torch.gather(torch.stack(ts, dim=0), 0, best)[0]
                ray_length = math.hypot(x2, geometry.rx_z_mm - geometry.tx_z_mm)
                sigma = max(float(geometry.sigma_ray_mm), 1e-9)
                kernel = torch.exp(-0.5 * (distance / sigma) ** 2) / max(ray_length, 1e-9)
                cutoff = distance <= geometry.kernel_sigma_cutoff * sigma
                if geometry.min_endpoint_distance_mm > 0.0 and ray_length > 0.0:
                    margin = min(geometry.min_endpoint_distance_mm / ray_length, 0.49)
                    cutoff = cutoff & (t >= margin) & (t <= 1.0 - margin)
                kernel = torch.where(cutoff, kernel, torch.zeros_like(kernel))
                kernels.append(kernel)
    rays = torch.stack(kernels, dim=0)
    rays = rays / rays.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    return rays


class RayOperator(nn.Module):
    def __init__(
        self,
        *,
        image_shape: tuple[int, int] = (256, 256),
        geometry: RayGeometry | None = None,
    ) -> None:
        super().__init__()
        self.geometry = geometry or RayGeometry()
        kernels = build_ray_kernels(image_shape=image_shape, geometry=self.geometry)
        self.register_buffer("kernels", kernels)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise RuntimeError(f"image must be (B,C,H,W), got {tuple(image.shape)}")
        y = image[:, :1]
        return torch.einsum("bchw,rhw->br", y, self.kernels)

    def observed_frequency_from_x(self, x_matrix: torch.Tensor, feature_index: int = 1) -> torch.Tensor:
        if x_matrix.ndim != 5:
            raise RuntimeError(f"x_matrix must be (B,C,F,TX,RX), got {tuple(x_matrix.shape)}")
        obs = x_matrix[:, feature_index]
        obs = obs.reshape(obs.shape[0], obs.shape[1], -1)
        if len(self.geometry.helical_orders) > 1:
            obs = obs.repeat_interleave(len(self.geometry.helical_orders), dim=2)
        return obs

    @staticmethod
    def robust_normalize(values: torch.Tensor) -> torch.Tensor:
        flat = values.reshape(values.shape[0], -1)
        center = flat.median(dim=1, keepdim=True).values
        mad = (flat - center).abs().median(dim=1, keepdim=True).values.clamp_min(1e-6)
        normalized = ((flat - center) / mad).clamp(-8.0, 8.0)
        return normalized.reshape_as(values)

    def consistency_loss(self, image: torch.Tensor, x_matrix: torch.Tensor, feature_index: int = 1) -> torch.Tensor:
        pred = self.forward(image)
        obs = self.observed_frequency_from_x(x_matrix, feature_index=feature_index)
        pred = pred[:, None, :].expand(-1, obs.shape[1], -1)
        if pred.shape != obs.shape:
            min_dim = min(pred.shape[2], obs.shape[2])
            pred = pred[:, :, :min_dim]
            obs = obs[:, :, :min_dim]
        return F.smooth_l1_loss(self.robust_normalize(pred), self.robust_normalize(obs))

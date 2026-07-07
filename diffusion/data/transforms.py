from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_PIC_CHANNELS = (
    "ray_log_amp_loss",
    "ray_relative_delta",
    "ray_phase_change",
    "low_frequency_band_map",
    "mid_frequency_band_map",
    "high_frequency_band_map",
    "path_coverage",
    "reliability_mask",
)


def resize_label_nearest(label: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if label.shape == target_shape:
        return label.astype(np.float32, copy=False)
    if label.ndim != 2:
        raise RuntimeError(f"Label must be 2D, got shape {label.shape}")
    target_z, target_theta = target_shape
    source_z, source_theta = label.shape
    z_index = np.rint(np.linspace(0, source_z - 1, target_z)).astype(np.int64)
    theta_index = np.floor(np.arange(target_theta) * source_theta / target_theta).astype(np.int64)
    theta_index = np.clip(theta_index, 0, source_theta - 1)
    return label[np.ix_(z_index, theta_index)].astype(np.float32, copy=False)


def resize_image_tensor(image: torch.Tensor, target_shape: tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    if tuple(image.shape[-2:]) == target_shape:
        return image
    align_corners = False if mode in {"bilinear", "bicubic"} else None
    return F.interpolate(image, size=target_shape, mode=mode, align_corners=align_corners)


def robust_zscore(values: np.ndarray, valid_mask: np.ndarray | None = None, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if valid_mask is None:
        valid = np.isfinite(arr)
    else:
        valid = np.asarray(valid_mask).astype(bool) & np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr, dtype=np.float32)
    median = np.nanmedian(arr[valid])
    q25, q75 = np.nanpercentile(arr[valid], [25.0, 75.0])
    scale = max(float(q75 - q25) / 1.349, eps)
    normalized = (arr - float(median)) / scale
    normalized = np.clip(normalized, -8.0, 8.0)
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized.astype(np.float32, copy=False)


def normalize_x_matrix(x: np.ndarray, mode: str = "robust_per_sample") -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).copy()
    if mode in {"none", "identity", ""}:
        arr[~np.isfinite(arr)] = 0.0
        return arr
    valid = arr[-1] > 0.5 if arr.shape[0] > 0 else None
    for channel in range(arr.shape[0]):
        if channel == arr.shape[0] - 1:
            arr[channel] = (arr[channel] > 0.5).astype(np.float32)
            continue
        arr[channel] = robust_zscore(arr[channel], valid_mask=valid)
    return arr


@dataclass(frozen=True)
class RandomCircularRoll:
    enabled: bool = False
    max_fraction: float = 1.0

    def __call__(
        self,
        pic: np.ndarray,
        label: np.ndarray,
        x_matrix: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.enabled:
            return pic, label, x_matrix
        width = int(label.shape[-1])
        if width <= 1:
            return pic, label, x_matrix
        max_shift = max(1, int(round(width * float(self.max_fraction))))
        shift = int(rng.integers(-max_shift, max_shift + 1))
        if shift == 0:
            return pic, label, x_matrix
        pic = np.roll(pic, shift=shift, axis=-1)
        label = np.roll(label, shift=shift, axis=-1)
        # The tx/rx rings have 16 samples around theta. Map image-pixel roll to
        # the nearest circular channel roll so x_matrix stays roughly aligned.
        tx_rx_count = int(x_matrix.shape[-1])
        x_shift = int(round(shift * tx_rx_count / width))
        if x_shift:
            x_matrix = np.roll(x_matrix, shift=x_shift, axis=-1)
            x_matrix = np.roll(x_matrix, shift=x_shift, axis=-2)
        return pic, label, x_matrix

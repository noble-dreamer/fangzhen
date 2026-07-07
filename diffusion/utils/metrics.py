from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def tensor_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def tensor_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2).clamp_min(1e-12))


def tensor_nrmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scale = (target.amax(dim=(-2, -1), keepdim=True) - target.amin(dim=(-2, -1), keepdim=True)).clamp_min(1e-6)
    return torch.sqrt(torch.mean(((pred - target) / scale) ** 2))


def tensor_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    pred_mask = pred >= threshold
    target_mask = target >= threshold
    union = torch.logical_or(pred_mask, target_mask).float().sum()
    if float(union.detach().cpu()) <= 0.0:
        return pred.new_tensor(float("nan"))
    inter = torch.logical_and(pred_mask, target_mask).float().sum()
    return inter / union


def pearson_np(pred: np.ndarray, target: np.ndarray) -> float:
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    valid = np.isfinite(p) & np.isfinite(t)
    if not np.any(valid):
        return float("nan")
    p = p[valid]
    t = t[valid]
    if float(np.std(p)) <= 0.0 or float(np.std(t)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


def ssim_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    try:
        from pytorch_msssim import ssim

        return ssim(pred.clamp(0.0, 1.0), target.clamp(0.0, 1.0), data_range=1.0, size_average=True)
    except Exception:
        pooled_pred = F.avg_pool2d(pred, kernel_size=5, stride=1, padding=2)
        pooled_target = F.avg_pool2d(target, kernel_size=5, stride=1, padding=2)
        return 1.0 - torch.mean(torch.abs(pooled_pred - pooled_target)).clamp(0.0, 1.0)


def safe_float(value: float | np.floating | torch.Tensor) -> float:
    if isinstance(value, torch.Tensor):
        value = float(value.detach().cpu())
    result = float(value)
    if math.isfinite(result):
        return result
    return float("nan")

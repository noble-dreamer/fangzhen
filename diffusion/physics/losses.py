from __future__ import annotations

import torch
import torch.nn.functional as F


def total_variation(image: torch.Tensor) -> torch.Tensor:
    dz = torch.mean(torch.abs(image[..., 1:, :] - image[..., :-1, :]))
    dtheta = torch.mean(torch.abs(image[..., :, 1:] - image[..., :, :-1]))
    wrap = torch.mean(torch.abs(image[..., :, :1] - image[..., :, -1:]))
    return dz + dtheta + wrap


def range_penalty(image: torch.Tensor) -> torch.Tensor:
    return torch.mean(F.relu(-image) ** 2 + F.relu(image - 1.0) ** 2)


def coverage_weighted_l1(pred: torch.Tensor, target: torch.Tensor, pic: torch.Tensor, coverage_channel: int = -2) -> torch.Tensor:
    if pic.shape[1] < abs(coverage_channel):
        return F.l1_loss(pred, target)
    coverage = pic[:, coverage_channel : coverage_channel + 1 if coverage_channel >= 0 else coverage_channel + 1]
    if coverage.numel() == 0:
        coverage = pic[:, -2:-1]
    coverage = coverage.clamp(0.0, 1.0)
    return torch.sum(coverage * torch.abs(pred - target)) / coverage.sum().clamp_min(1.0)


def output_prior_losses(
    pred: torch.Tensor,
    target: torch.Tensor | None = None,
    pic: torch.Tensor | None = None,
    *,
    lambda_l1: float = 0.0,
    lambda_tv: float = 0.0,
    lambda_range: float = 0.0,
    lambda_coverage_l1: float = 0.0,
) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    zero = pred.new_tensor(0.0)
    losses["loss_l1"] = F.l1_loss(pred, target) if target is not None and lambda_l1 > 0.0 else zero
    losses["loss_tv"] = total_variation(pred) if lambda_tv > 0.0 else zero
    losses["loss_range"] = range_penalty(pred) if lambda_range > 0.0 else zero
    if target is not None and pic is not None and lambda_coverage_l1 > 0.0:
        losses["loss_coverage_l1"] = coverage_weighted_l1(pred, target, pic)
    else:
        losses["loss_coverage_l1"] = zero
    losses["loss_prior_total"] = (
        float(lambda_l1) * losses["loss_l1"]
        + float(lambda_tv) * losses["loss_tv"]
        + float(lambda_range) * losses["loss_range"]
        + float(lambda_coverage_l1) * losses["loss_coverage_l1"]
    )
    return losses

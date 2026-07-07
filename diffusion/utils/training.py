from __future__ import annotations

import math
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def autocast_context(device: torch.device, enabled: bool = True, dtype: str = "float16"):
    if not enabled:
        return torch.amp.autocast(device_type=device.type, enabled=False)
    if device.type == "cuda":
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        return torch.amp.autocast(device_type="cuda", dtype=torch_dtype, enabled=True)
    return torch.amp.autocast(device_type=device.type, enabled=False)


def make_scaler(device: torch.device, amp_enabled: bool = True, dtype: str = "float16") -> torch.amp.GradScaler:
    enabled = bool(amp_enabled and device.type == "cuda" and dtype != "bfloat16")
    return torch.amp.GradScaler("cuda", enabled=enabled)


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> AdamW:
    train_cfg = config.get("train", {})
    return AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        betas=tuple(train_cfg.get("betas", (0.9, 0.999))),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any], total_steps: int) -> LambdaLR:
    train_cfg = config.get("train", {})
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.05))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(float(step + 1) / float(warmup_steps), 1e-8)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().data.norm(2).item()
        total += value * value
    return math.sqrt(total)

from __future__ import annotations

import copy

import torch
from torch import nn


class ModelEma(nn.Module):
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        super().__init__()
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = model.state_dict()
        target = self.module.state_dict()
        for key, target_value in target.items():
            source_value = source[key].detach()
            if torch.is_floating_point(target_value):
                target_value.mul_(self.decay).add_(source_value, alpha=1.0 - self.decay)
            else:
                target_value.copy_(source_value)

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        return self.module.load_state_dict(state_dict, strict=strict)

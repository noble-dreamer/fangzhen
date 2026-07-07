from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def default_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        timesteps = timesteps.float()
        half_dim = self.dim // 2
        exponent = -math.log(10000.0) * torch.arange(half_dim, device=timesteps.device) / max(half_dim - 1, 1)
        freqs = torch.exp(exponent)
        args = timesteps[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(default_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(default_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.cond = nn.Linear(cond_dim, out_channels * 2)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.cond(cond).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(default_groups(channels), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        h = self.norm(x).reshape(batch, channels, height * width).transpose(1, 2)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.transpose(1, 2).reshape(batch, channels, height, width)
        return x + h


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int = 1,
        base_channels: int = 48,
        channel_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (32,),
        image_size: int = 256,
        cond_dim: int = 256,
        time_dim: int = 256,
        dropout: float = 0.05,
        use_time: bool = True,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.use_time = bool(use_time)
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(time_dim),
            nn.Linear(time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.null_time = nn.Parameter(torch.zeros(cond_dim))
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_channels: list[int] = []
        channels = base_channels
        resolution = self.image_size
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * int(mult)
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(channels, out_ch, cond_dim, dropout=dropout))
                channels = out_ch
                self.skip_channels.append(channels)
                if resolution in attention_resolutions:
                    blocks.append(AttentionBlock(channels))
            self.down_blocks.append(blocks)
            if level != len(channel_mult) - 1:
                self.downsamples.append(Downsample(channels))
                self.skip_channels.append(channels)
                resolution //= 2
            else:
                self.downsamples.append(nn.Identity())

        self.mid1 = ResBlock(channels, channels, cond_dim, dropout=dropout)
        self.mid_attn = AttentionBlock(channels)
        self.mid2 = ResBlock(channels, channels, cond_dim, dropout=dropout)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * int(mult)
            blocks = nn.ModuleList()
            block_count = num_res_blocks + (0 if level == len(channel_mult) - 1 else 1)
            for _ in range(block_count):
                skip_ch = self.skip_channels.pop()
                blocks.append(ResBlock(channels + skip_ch, out_ch, cond_dim, dropout=dropout))
                channels = out_ch
                if resolution in attention_resolutions:
                    blocks.append(AttentionBlock(channels))
            self.up_blocks.append(blocks)
            if level != 0:
                self.upsamples.append(Upsample(channels))
                resolution *= 2
            else:
                self.upsamples.append(nn.Identity())

        self.out_norm = nn.GroupNorm(default_groups(channels), channels)
        self.out_conv = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | None,
        data_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_time:
            if timesteps is None:
                raise RuntimeError("timesteps are required when use_time=True")
            cond = data_embedding + self.time_embed(timesteps)
        else:
            cond = data_embedding + self.null_time[None, :]
        h = self.input_conv(x)
        skips: list[torch.Tensor] = []
        for blocks, downsample in zip(self.down_blocks, self.downsamples):
            for block in blocks:
                if isinstance(block, ResBlock):
                    h = block(h, cond)
                    skips.append(h)
                else:
                    h = block(h)
            before_down = h
            h = downsample(h)
            if not isinstance(downsample, nn.Identity):
                skips.append(before_down)
        h = self.mid1(h, cond)
        h = self.mid_attn(h)
        h = self.mid2(h, cond)
        for blocks, upsample in zip(self.up_blocks, self.upsamples):
            for block in blocks:
                if isinstance(block, ResBlock):
                    skip = skips.pop()
                    if skip.shape[-2:] != h.shape[-2:]:
                        skip = F.interpolate(skip, size=h.shape[-2:], mode="nearest")
                    h = torch.cat([h, skip], dim=1)
                    h = block(h, cond)
                else:
                    h = block(h)
            h = upsample(h)
        return self.out_conv(F.silu(self.out_norm(h)))

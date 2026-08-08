from __future__ import annotations

import torch
from torch import nn

from .unet import ConditionalUNet
from .x_encoder import XMatrixEncoder


class ConditionalRegressor(nn.Module):
    def __init__(
        self,
        *,
        pic_channels: int = 8,
        x_channels: int = 7,
        frequency_count: int = 15,
        image_size: int = 256,
        base_channels: int = 48,
        channel_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (32,),
        cond_dim: int = 256,
        x_hidden_channels: int = 64,
        x_token_dim: int | None = None,
        cross_attention_resolutions: tuple[int, ...] = (),
        cross_attention_heads: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.x_encoder = XMatrixEncoder(
            x_channels=x_channels,
            frequency_count=frequency_count,
            embedding_dim=cond_dim,
            hidden_channels=x_hidden_channels,
            dropout=dropout,
            token_dim=x_token_dim or cond_dim,
        )
        self.unet = ConditionalUNet(
            in_channels=pic_channels,
            out_channels=1,
            base_channels=base_channels,
            channel_mult=tuple(channel_mult),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_resolutions),
            image_size=image_size,
            cond_dim=cond_dim,
            dropout=dropout,
            use_time=False,
            cross_attention_resolutions=tuple(cross_attention_resolutions),
            x_token_dim=x_token_dim or cond_dim,
            cross_attention_heads=cross_attention_heads,
            pic_condition_channels=pic_channels,
        )

    def forward(self, pic: torch.Tensor, x_matrix: torch.Tensor) -> torch.Tensor:
        embedding, tokens = self.x_encoder(x_matrix)
        logits = self.unet(pic, None, embedding, x_tokens=tokens, pic_condition=pic)
        return torch.sigmoid(logits)

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .physical_encoding import PhysicalEncodingConfig
from .unet import ConditionalUNet
from .x_encoder import XMatrixEncoder


def append_dims(value: torch.Tensor, target_ndim: int) -> torch.Tensor:
    return value.reshape(value.shape[0], *((1,) * (target_ndim - 1)))


class EDMDiffusion(nn.Module):
    """Karras EDM preconditioned conditional denoiser.

    The denoising backbone intentionally keeps the current diffusion architecture:
    x_matrix -> (global embedding, low/high frequency TX-RX tokens),
    pic -> PicAdapter features, and a ConditionalUNet that receives
    [noisy target, self-condition, pic].
    """

    def __init__(
        self,
        *,
        pic_channels: int = 8,
        x_channels: int = 7,
        frequency_count: int = 15,
        image_size: int = 256,
        target_channels: int = 1,
        base_channels: int = 48,
        channel_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (32,),
        cond_dim: int = 256,
        x_hidden_channels: int = 64,
        x_token_dim: int | None = None,
        frequency_filter_kernel_size: int = 5,
        physical_encoding: PhysicalEncodingConfig | None = None,
        fusion_resolutions: tuple[int, ...] = (),
        fusion_heads: int = 4,
        self_condition_prob: float = 0.5,
        dropout: float = 0.05,
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        rho: float = 7.0,
    ) -> None:
        super().__init__()
        self.pic_channels = int(pic_channels)
        self.target_channels = int(target_channels)
        self.self_condition_prob = float(self_condition_prob)
        self.sigma_data = float(sigma_data)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.rho = float(rho)
        token_dim = int(x_token_dim or cond_dim)
        self.x_encoder = XMatrixEncoder(
            x_channels=x_channels,
            frequency_count=frequency_count,
            embedding_dim=cond_dim,
            hidden_channels=x_hidden_channels,
            dropout=dropout,
            token_dim=token_dim,
            frequency_filter_kernel_size=frequency_filter_kernel_size,
            physical_encoding=physical_encoding,
        )
        self.unet = ConditionalUNet(
            in_channels=target_channels + target_channels + pic_channels,
            out_channels=target_channels,
            base_channels=base_channels,
            channel_mult=tuple(channel_mult),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_resolutions),
            image_size=image_size,
            cond_dim=cond_dim,
            dropout=dropout,
            use_time=True,
            fusion_resolutions=tuple(fusion_resolutions),
            x_token_dim=token_dim,
            fusion_heads=fusion_heads,
            pic_condition_channels=pic_channels,
        )

    def sample_sigma(self, batch_size: int, device: torch.device) -> torch.Tensor:
        sigma = torch.randn(batch_size, device=device) * self.p_std + self.p_mean
        return sigma.exp().clamp(self.sigma_min, self.sigma_max)

    def _coefficients(self, sigma: torch.Tensor, ndim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma = append_dims(sigma, ndim)
        sigma_data = self.sigma_data
        denom = sigma.square() + sigma_data**2
        c_skip = sigma_data**2 / denom
        c_out = sigma * sigma_data / torch.sqrt(denom)
        c_in = 1.0 / torch.sqrt(denom)
        return c_skip, c_out, c_in

    def model_output(
        self,
        noisy: torch.Tensor,
        sigma: torch.Tensor,
        pic: torch.Tensor,
        x_matrix: torch.Tensor,
        self_condition: torch.Tensor | None = None,
        *,
        frequency_hz: torch.Tensor,
        tx_indices: torch.Tensor,
        rx_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sigma.ndim == 0:
            sigma = sigma.expand(noisy.shape[0])
        if sigma.ndim == 1 and sigma.shape[0] == 1 and noisy.shape[0] > 1:
            sigma = sigma.expand(noisy.shape[0])
        embedding, low_frequency_tokens, high_frequency_tokens = self.x_encoder(
            x_matrix,
            frequency_hz=frequency_hz,
            tx_indices=tx_indices,
            rx_indices=rx_indices,
        )
        c_skip, c_out, c_in = self._coefficients(sigma, noisy.ndim)
        if self_condition is None:
            self_condition = torch.zeros_like(noisy)
        model_input = torch.cat([c_in * noisy, self_condition, pic], dim=1)
        noise_cond = sigma.clamp_min(1e-12).log() / 4.0
        raw = self.unet(
            model_input,
            noise_cond,
            embedding,
            low_frequency_tokens=low_frequency_tokens,
            high_frequency_tokens=high_frequency_tokens,
            pic_condition=pic,
        )
        denoised = c_skip * noisy + c_out * raw
        return denoised, raw

    def training_losses(
        self,
        x_start: torch.Tensor,
        pic: torch.Tensor,
        x_matrix: torch.Tensor,
        *,
        frequency_hz: torch.Tensor,
        tx_indices: torch.Tensor,
        rx_indices: torch.Tensor,
        sigma: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = x_start.shape[0]
        if sigma is None:
            sigma = self.sample_sigma(batch, x_start.device)
        if noise is None:
            noise = torch.randn_like(x_start)
        sigma_img = append_dims(sigma, x_start.ndim)
        noisy = x_start + sigma_img * noise
        self_cond = None
        if torch.rand((), device=x_start.device) < self.self_condition_prob:
            with torch.no_grad():
                pred_sc, _ = self.model_output(
                    noisy,
                    sigma,
                    pic,
                    x_matrix,
                    None,
                    frequency_hz=frequency_hz,
                    tx_indices=tx_indices,
                    rx_indices=rx_indices,
                )
                self_cond = pred_sc.clamp(0.0, 1.0).detach()
        denoised, raw = self.model_output(
            noisy,
            sigma,
            pic,
            x_matrix,
            self_cond,
            frequency_hz=frequency_hz,
            tx_indices=tx_indices,
            rx_indices=rx_indices,
        )
        weight = (sigma_img.square() + self.sigma_data**2) / ((sigma_img * self.sigma_data) ** 2).clamp_min(1e-12)
        loss = torch.mean(weight * (denoised - x_start).square())
        return {
            "loss_edm": loss,
            "x0_pred": denoised,
            "model_pred": raw,
            "noisy": noisy,
            "sigma": sigma,
        }

    def karras_sigmas(
        self,
        steps: int,
        *,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        rho: float | None = None,
        device: torch.device,
    ) -> torch.Tensor:
        sigma_min = float(self.sigma_min if sigma_min is None else sigma_min)
        sigma_max = float(self.sigma_max if sigma_max is None else sigma_max)
        rho = float(self.rho if rho is None else rho)
        ramp = torch.linspace(0.0, 1.0, int(steps), device=device)
        min_inv_rho = sigma_min ** (1.0 / rho)
        max_inv_rho = sigma_max ** (1.0 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return torch.cat([sigmas, torch.zeros(1, device=device)])

    def _physics_guided_x0(
        self,
        x0_pred: torch.Tensor,
        x_matrix: torch.Tensor,
        physics_operator: nn.Module,
        *,
        guidance_scale: float,
        feature_index: int,
    ) -> torch.Tensor:
        with torch.enable_grad():
            x0_var = x0_pred.detach().requires_grad_(True)
            loss = physics_operator.consistency_loss(x0_var, x_matrix, feature_index=feature_index)
            grad = torch.autograd.grad(loss, x0_var, only_inputs=True)[0]
            grad_scale = grad.flatten(1).norm(dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
            guided = x0_var - float(guidance_scale) * grad / grad_scale
        return guided.detach().clamp(0.0, 1.0)

    def sample(
        self,
        pic: torch.Tensor,
        x_matrix: torch.Tensor,
        *,
        frequency_hz: torch.Tensor,
        tx_indices: torch.Tensor,
        rx_indices: torch.Tensor,
        shape: tuple[int, int, int, int] | None = None,
        steps: int = 32,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        rho: float | None = None,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: torch.Generator | None = None,
        initial_noise: torch.Tensor | None = None,
        physics_operator: nn.Module | None = None,
        physics_guidance_scale: float = 0.0,
        physics_guidance_start_fraction: float = 0.5,
        physics_feature_index: int = 1,
    ) -> torch.Tensor:
        if shape is None:
            shape = (pic.shape[0], self.target_channels, pic.shape[-2], pic.shape[-1])
        sigmas = self.karras_sigmas(
            int(steps),
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            device=pic.device,
        )
        if initial_noise is None:
            initial_noise = torch.randn(
                shape,
                device=pic.device,
                dtype=pic.dtype,
                generator=generator,
            )
        else:
            if tuple(initial_noise.shape) != tuple(shape):
                raise RuntimeError(
                    f"initial_noise must have shape {shape}, got {tuple(initial_noise.shape)}"
                )
            initial_noise = initial_noise.to(device=pic.device, dtype=pic.dtype)
            if not torch.isfinite(initial_noise).all():
                raise RuntimeError("initial_noise contains non-finite values")
        image = initial_noise * sigmas[0]
        self_cond = None
        guidance_start_index = int(round((len(sigmas) - 1) * float(physics_guidance_start_fraction)))
        for index, (sigma_cur, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            gamma = 0.0
            if s_tmin <= float(sigma_cur) <= s_tmax and s_churn > 0.0:
                gamma = min(float(s_churn) / max(int(steps), 1), math.sqrt(2.0) - 1.0)
            sigma_hat = sigma_cur * (1.0 + gamma)
            image_hat = image
            if gamma > 0.0:
                eps = torch.randn(
                    image.shape,
                    device=image.device,
                    dtype=image.dtype,
                    generator=generator,
                ) * float(s_noise)
                image_hat = image_hat + eps * torch.sqrt((sigma_hat.square() - sigma_cur.square()).clamp_min(0.0))
            sigma_batch = sigma_hat.expand(shape[0])
            with torch.no_grad():
                denoised, _ = self.model_output(
                    image_hat,
                    sigma_batch,
                    pic,
                    x_matrix,
                    self_cond,
                    frequency_hz=frequency_hz,
                    tx_indices=tx_indices,
                    rx_indices=rx_indices,
                )
                denoised = denoised.clamp(0.0, 1.0)
            if physics_operator is not None and physics_guidance_scale > 0.0 and index >= guidance_start_index:
                denoised = self._physics_guided_x0(
                    denoised,
                    x_matrix,
                    physics_operator,
                    guidance_scale=physics_guidance_scale,
                    feature_index=physics_feature_index,
                )
            derivative = (image_hat - denoised) / sigma_hat.clamp_min(1e-12)
            image_next = image_hat + (sigma_next - sigma_hat) * derivative
            if sigma_next > 0:
                sigma_next_batch = sigma_next.expand(shape[0])
                with torch.no_grad():
                    denoised_next, _ = self.model_output(
                        image_next,
                        sigma_next_batch,
                        pic,
                        x_matrix,
                        denoised,
                        frequency_hz=frequency_hz,
                        tx_indices=tx_indices,
                        rx_indices=rx_indices,
                    )
                    denoised_next = denoised_next.clamp(0.0, 1.0)
                if physics_operator is not None and physics_guidance_scale > 0.0 and index >= guidance_start_index:
                    denoised_next = self._physics_guided_x0(
                        denoised_next,
                        x_matrix,
                        physics_operator,
                        guidance_scale=physics_guidance_scale,
                        feature_index=physics_feature_index,
                    )
                derivative_next = (image_next - denoised_next) / sigma_next.clamp_min(1e-12)
                image_next = image_hat + (sigma_next - sigma_hat) * (0.5 * derivative + 0.5 * derivative_next)
                self_cond = denoised_next.detach()
            else:
                self_cond = denoised.detach()
            image = image_next
        return self_cond.clamp(0.0, 1.0) if self_cond is not None else image.clamp(0.0, 1.0)

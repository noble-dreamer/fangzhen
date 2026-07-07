from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .unet import ConditionalUNet
from .x_encoder import XMatrixEncoder


def extract(coefficients: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    values = coefficients.gather(0, timesteps)
    return values.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
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
        dropout: float = 0.05,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        prediction_type: str = "v_prediction",
    ) -> None:
        super().__init__()
        self.pic_channels = int(pic_channels)
        self.target_channels = int(target_channels)
        self.num_timesteps = int(timesteps)
        self.prediction_type = str(prediction_type)
        self.x_encoder = XMatrixEncoder(
            x_channels=x_channels,
            frequency_count=frequency_count,
            embedding_dim=cond_dim,
            hidden_channels=x_hidden_channels,
            dropout=dropout,
        )
        self.unet = ConditionalUNet(
            in_channels=target_channels + pic_channels,
            out_channels=target_channels,
            base_channels=base_channels,
            channel_mult=tuple(channel_mult),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_resolutions),
            image_size=image_size,
            cond_dim=cond_dim,
            dropout=dropout,
            use_time=True,
        )
        betas = self._make_betas(self.num_timesteps, beta_schedule)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))
        self.register_buffer("sqrt_recip_alpha_cumprod", torch.sqrt(1.0 / alpha_cumprod))
        self.register_buffer("sqrt_recipm1_alpha_cumprod", torch.sqrt(1.0 / alpha_cumprod - 1.0))
        posterior_variance = betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(1e-20))
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp_min(1e-20)))
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alpha_cumprod_prev) / (1.0 - alpha_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alpha_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alpha_cumprod),
        )

    @staticmethod
    def _make_betas(timesteps: int, schedule: str) -> torch.Tensor:
        if schedule == "linear":
            return torch.linspace(1e-4, 0.02, timesteps, dtype=torch.float32)
        if schedule != "cosine":
            raise ValueError(f"Unsupported beta schedule: {schedule}")
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alpha_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
        alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
        betas = 1.0 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
        return betas.clamp(1e-5, 0.999).float()

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alpha_cumprod, timesteps, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alpha_cumprod, timesteps, x_start.shape) * noise
        )

    def predict_v(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_alpha_cumprod, timesteps, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alpha_cumprod, timesteps, x_start.shape) * x_start
        )

    def predict_start_from_noise(self, x_t: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alpha_cumprod, timesteps, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alpha_cumprod, timesteps, x_t.shape) * noise
        )

    def predict_start_from_v(self, x_t: torch.Tensor, timesteps: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_alpha_cumprod, timesteps, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alpha_cumprod, timesteps, x_t.shape) * v
        )

    def predict_noise_from_v(self, x_t: torch.Tensor, timesteps: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_alpha_cumprod, timesteps, x_t.shape) * v
            + extract(self.sqrt_one_minus_alpha_cumprod, timesteps, x_t.shape) * x_t
        )

    def model_output(self, x_t: torch.Tensor, timesteps: torch.Tensor, pic: torch.Tensor, x_matrix: torch.Tensor) -> torch.Tensor:
        embedding = self.x_encoder(x_matrix)
        model_input = torch.cat([x_t, pic], dim=1)
        return self.unet(model_input, timesteps, embedding)

    def training_losses(
        self,
        x_start: torch.Tensor,
        pic: torch.Tensor,
        x_matrix: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = x_start.shape[0]
        if timesteps is None:
            timesteps = torch.randint(0, self.num_timesteps, (batch,), device=x_start.device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, timesteps, noise)
        pred = self.model_output(x_t, timesteps, pic, x_matrix)
        if self.prediction_type == "epsilon":
            target = noise
            x0_pred = self.predict_start_from_noise(x_t, timesteps, pred)
        elif self.prediction_type == "v_prediction":
            target = self.predict_v(x_start, timesteps, noise)
            x0_pred = self.predict_start_from_v(x_t, timesteps, pred)
        else:
            raise ValueError(f"Unsupported prediction_type: {self.prediction_type}")
        loss = F.mse_loss(pred, target)
        return {
            "loss_diffusion": loss,
            "model_pred": pred,
            "target": target,
            "x0_pred": x0_pred,
            "timesteps": timesteps,
        }

    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        pic: torch.Tensor,
        x_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = self.model_output(x_t, timesteps, pic, x_matrix)
        if self.prediction_type == "epsilon":
            x0_pred = self.predict_start_from_noise(x_t, timesteps, pred)
        else:
            x0_pred = self.predict_start_from_v(x_t, timesteps, pred)
        x0_pred = x0_pred.clamp(0.0, 1.0)
        mean = (
            extract(self.posterior_mean_coef1, timesteps, x_t.shape) * x0_pred
            + extract(self.posterior_mean_coef2, timesteps, x_t.shape) * x_t
        )
        log_variance = extract(self.posterior_log_variance_clipped, timesteps, x_t.shape)
        return mean, log_variance, x0_pred

    @torch.no_grad()
    def sample(
        self,
        pic: torch.Tensor,
        x_matrix: torch.Tensor,
        *,
        shape: tuple[int, int, int, int] | None = None,
        steps: int | None = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        if shape is None:
            shape = (pic.shape[0], self.target_channels, pic.shape[-2], pic.shape[-1])
        if steps is None or steps >= self.num_timesteps:
            return self.p_sample_loop(pic, x_matrix, shape=shape)
        return self.ddim_sample(pic, x_matrix, shape=shape, steps=steps, eta=eta)

    @torch.no_grad()
    def p_sample_loop(self, pic: torch.Tensor, x_matrix: torch.Tensor, *, shape: tuple[int, int, int, int]) -> torch.Tensor:
        image = torch.randn(shape, device=pic.device)
        for time in reversed(range(self.num_timesteps)):
            timesteps = torch.full((shape[0],), time, device=pic.device, dtype=torch.long)
            mean, log_variance, _ = self.p_mean_variance(image, timesteps, pic, x_matrix)
            noise = torch.randn_like(image) if time > 0 else torch.zeros_like(image)
            image = mean + torch.exp(0.5 * log_variance) * noise
        return image.clamp(0.0, 1.0)

    @torch.no_grad()
    def ddim_sample(
        self,
        pic: torch.Tensor,
        x_matrix: torch.Tensor,
        *,
        shape: tuple[int, int, int, int],
        steps: int,
        eta: float = 0.0,
    ) -> torch.Tensor:
        image = torch.randn(shape, device=pic.device)
        times = torch.linspace(self.num_timesteps - 1, 0, steps, device=pic.device).long()
        for index, time in enumerate(times):
            timestep = torch.full((shape[0],), int(time.item()), device=pic.device, dtype=torch.long)
            pred = self.model_output(image, timestep, pic, x_matrix)
            if self.prediction_type == "epsilon":
                x0_pred = self.predict_start_from_noise(image, timestep, pred)
                eps = pred
            else:
                x0_pred = self.predict_start_from_v(image, timestep, pred)
                eps = self.predict_noise_from_v(image, timestep, pred)
            x0_pred = x0_pred.clamp(0.0, 1.0)
            if index == len(times) - 1:
                image = x0_pred
                continue
            next_time = torch.full((shape[0],), int(times[index + 1].item()), device=pic.device, dtype=torch.long)
            alpha = extract(self.alpha_cumprod, timestep, image.shape)
            alpha_next = extract(self.alpha_cumprod, next_time, image.shape)
            sigma = (
                eta
                * torch.sqrt((1.0 - alpha / alpha_next) * (1.0 - alpha_next) / (1.0 - alpha)).clamp_min(0.0)
            )
            c = torch.sqrt((1.0 - alpha_next - sigma ** 2).clamp_min(0.0))
            noise = torch.randn_like(image) if eta > 0.0 else torch.zeros_like(image)
            image = torch.sqrt(alpha_next) * x0_pred + c * eps + sigma * noise
        return image.clamp(0.0, 1.0)

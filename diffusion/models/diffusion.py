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
        x_token_dim: int | None = None,
        cross_attention_resolutions: tuple[int, ...] = (),
        cross_attention_heads: int = 4,
        self_condition_prob: float = 0.5,
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
        self.self_condition_prob = float(self_condition_prob)
        token_dim = int(x_token_dim or cond_dim)
        self.x_encoder = XMatrixEncoder(
            x_channels=x_channels,
            frequency_count=frequency_count,
            embedding_dim=cond_dim,
            hidden_channels=x_hidden_channels,
            dropout=dropout,
            token_dim=token_dim,
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
            cross_attention_resolutions=tuple(cross_attention_resolutions),
            x_token_dim=token_dim,
            cross_attention_heads=cross_attention_heads,
            pic_condition_channels=pic_channels,
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

    def model_output(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        pic: torch.Tensor,
        x_matrix: torch.Tensor,
        self_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedding, tokens = self.x_encoder(x_matrix)
        inputs = [x_t]
        if self_condition is None:
            self_condition = torch.zeros_like(x_t)
        inputs.append(self_condition)
        inputs.append(pic)
        model_input = torch.cat(inputs, dim=1)
        return self.unet(model_input, timesteps, embedding, x_tokens=tokens, pic_condition=pic)

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
        self_cond = None
        if torch.rand((), device=x_start.device) < self.self_condition_prob:
            with torch.no_grad():
                pred_sc = self.model_output(x_t, timesteps, pic, x_matrix, None)
                if self.prediction_type == "epsilon":
                    self_cond = self.predict_start_from_noise(x_t, timesteps, pred_sc)
                elif self.prediction_type == "v_prediction":
                    self_cond = self.predict_start_from_v(x_t, timesteps, pred_sc)
                else:
                    raise ValueError(f"Unsupported prediction_type: {self.prediction_type}")
                self_cond = self_cond.clamp(0.0, 1.0).detach()
        pred = self.model_output(x_t, timesteps, pic, x_matrix, self_cond)
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
        physics_operator: nn.Module | None = None,
        physics_guidance_scale: float = 0.0,
        physics_guidance_start_fraction: float = 0.5,
        physics_feature_index: int = 1,
    ) -> torch.Tensor:
        if shape is None:
            shape = (pic.shape[0], self.target_channels, pic.shape[-2], pic.shape[-1])
        if physics_operator is not None and physics_guidance_scale > 0.0:
            if steps is None:
                raise ValueError("Physics-guided sampling currently requires DDIM steps.")
            return self.ddim_sample_guided(
                pic,
                x_matrix,
                shape=shape,
                steps=steps,
                eta=eta,
                physics_operator=physics_operator,
                guidance_scale=physics_guidance_scale,
                guidance_start_fraction=physics_guidance_start_fraction,
                feature_index=physics_feature_index,
            )
        if steps is None or steps >= self.num_timesteps:
            return self.p_sample_loop(pic, x_matrix, shape=shape)
        return self.ddim_sample(pic, x_matrix, shape=shape, steps=steps, eta=eta)

    @torch.no_grad()
    def p_sample_loop(self, pic: torch.Tensor, x_matrix: torch.Tensor, *, shape: tuple[int, int, int, int]) -> torch.Tensor:
        image = torch.randn(shape, device=pic.device)
        self_cond = None
        for time in reversed(range(self.num_timesteps)):
            timesteps = torch.full((shape[0],), time, device=pic.device, dtype=torch.long)
            pred = self.model_output(image, timesteps, pic, x_matrix, self_cond)
            if self.prediction_type == "epsilon":
                x0_pred = self.predict_start_from_noise(image, timesteps, pred)
            else:
                x0_pred = self.predict_start_from_v(image, timesteps, pred)
            x0_pred = x0_pred.clamp(0.0, 1.0)
            self_cond = x0_pred
            mean = (
                extract(self.posterior_mean_coef1, timesteps, image.shape) * x0_pred
                + extract(self.posterior_mean_coef2, timesteps, image.shape) * image
            )
            log_variance = extract(self.posterior_log_variance_clipped, timesteps, image.shape)
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
        self_cond = None
        times = torch.linspace(self.num_timesteps - 1, 0, steps, device=pic.device).long()
        for index, time in enumerate(times):
            timestep = torch.full((shape[0],), int(time.item()), device=pic.device, dtype=torch.long)
            pred = self.model_output(image, timestep, pic, x_matrix, self_cond)
            if self.prediction_type == "epsilon":
                x0_pred = self.predict_start_from_noise(image, timestep, pred)
                eps = pred
            else:
                x0_pred = self.predict_start_from_v(image, timestep, pred)
                eps = self.predict_noise_from_v(image, timestep, pred)
            x0_pred = x0_pred.clamp(0.0, 1.0)
            self_cond = x0_pred
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
            scale = float(guidance_scale)
            grad_scale = grad.flatten(1).norm(dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
            guided = x0_var - scale * grad / grad_scale
        return guided.detach().clamp(0.0, 1.0)

    def _noise_from_x0(self, x_t: torch.Tensor, timesteps: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        return (
            x_t - extract(self.sqrt_alpha_cumprod, timesteps, x_t.shape) * x0
        ) / extract(self.sqrt_one_minus_alpha_cumprod, timesteps, x_t.shape).clamp_min(1e-8)

    def ddim_sample_guided(
        self,
        pic: torch.Tensor,
        x_matrix: torch.Tensor,
        *,
        shape: tuple[int, int, int, int],
        steps: int,
        eta: float,
        physics_operator: nn.Module,
        guidance_scale: float,
        guidance_start_fraction: float,
        feature_index: int,
    ) -> torch.Tensor:
        image = torch.randn(shape, device=pic.device)
        self_cond = None
        times = torch.linspace(self.num_timesteps - 1, 0, steps, device=pic.device).long()
        guidance_start_index = int(round(len(times) * float(guidance_start_fraction)))
        for index, time in enumerate(times):
            timestep = torch.full((shape[0],), int(time.item()), device=pic.device, dtype=torch.long)
            with torch.no_grad():
                pred = self.model_output(image, timestep, pic, x_matrix, self_cond)
                if self.prediction_type == "epsilon":
                    x0_pred = self.predict_start_from_noise(image, timestep, pred)
                    eps = pred
                else:
                    x0_pred = self.predict_start_from_v(image, timestep, pred)
                    eps = self.predict_noise_from_v(image, timestep, pred)
                x0_pred = x0_pred.clamp(0.0, 1.0)
            if index >= guidance_start_index:
                x0_pred = self._physics_guided_x0(
                    x0_pred,
                    x_matrix,
                    physics_operator,
                    guidance_scale=guidance_scale,
                    feature_index=feature_index,
                )
                eps = self._noise_from_x0(image, timestep, x0_pred)
            self_cond = x0_pred
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

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.edm import EDMDiffusion, append_dims, edm_preconditioned_target


def main() -> None:
    torch.manual_seed(17)
    x_start = torch.rand(4, 1, 8, 8, dtype=torch.float64)
    noise = torch.randn_like(x_start)
    sigma = torch.tensor([0.002, 0.02, 0.5, 80.0], dtype=torch.float64)
    sigma_data = 0.5
    raw = torch.randn_like(x_start, requires_grad=True)

    sigma_img = append_dims(sigma, x_start.ndim)
    denom = sigma_img.square() + sigma_data**2
    c_skip = sigma_data**2 / denom
    c_out = sigma_img * sigma_data / torch.sqrt(denom)
    denoised = c_skip * (x_start + sigma_img * noise) + c_out * raw
    weight = denom / (sigma_img * sigma_data).square()
    weighted_loss = torch.mean(weight * (denoised - x_start).square())

    target = edm_preconditioned_target(x_start, noise, sigma, sigma_data)
    stable_loss = F.mse_loss(raw, target)
    torch.testing.assert_close(stable_loss, weighted_loss, rtol=1e-10, atol=1e-10)

    stable_loss.float().backward()
    if raw.grad is None or not torch.isfinite(raw.grad).all():
        raise AssertionError("Stable EDM loss produced non-finite gradients")
    if not torch.isfinite(target).all():
        raise AssertionError("Stable EDM model target is non-finite")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EDMDiffusion(
        image_size=64,
        base_channels=16,
        channel_mult=(1, 2, 4, 4),
        attention_resolutions=(8,),
        cond_dim=64,
        x_hidden_channels=16,
        x_token_dim=64,
        fusion_resolutions=(8,),
        fusion_heads=4,
        self_condition_prob=0.0,
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    frequency_hz = torch.linspace(80_000.0, 220_000.0, 15, device=device)[None, :].expand(2, -1)
    tx_indices = torch.arange(1, 17, device=device)[None, :].expand(2, -1)
    rx_indices = torch.arange(1, 17, device=device)[None, :].expand(2, -1)
    sigma_cases = (
        torch.tensor([0.002, 80.0], device=device),
        torch.tensor([0.02, 2.0], device=device),
        torch.tensor([0.1, 0.5], device=device),
    )
    for step in range(6):
        optimizer.zero_grad(set_to_none=True)
        losses = model.training_losses(
            torch.rand(2, 1, 64, 64, device=device),
            torch.rand(2, 8, 64, 64, device=device),
            torch.randn(2, 7, 15, 16, 16, device=device),
            frequency_hz=frequency_hz,
            tx_indices=tx_indices,
            rx_indices=rx_indices,
            sigma=sigma_cases[step % len(sigma_cases)],
        )
        losses["loss_edm"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise AssertionError(f"Non-finite model parameter after step {step}")

    print(f"EDM stable-loss smoke test passed on {device}.")


if __name__ == "__main__":
    main()

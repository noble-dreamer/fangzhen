from __future__ import annotations

import json

import torch

from models import EDMDiffusion, PhysicalEncodingConfig, sort_frequency_axis


def main() -> None:
    torch.manual_seed(20260711)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    frequency_count = 5
    tx_count = 4
    rx_count = 4
    token_dim = 16
    image_size = 16

    geometry = PhysicalEncodingConfig(
        tx_count=tx_count,
        rx_count=rx_count,
        pipe_length_mm=1000.0,
        mid_radius_mm=155.0,
        tx_z_mm=100.0,
        rx_z_mm=900.0,
        frequency_reference_hz=100000.0,
    )
    model = EDMDiffusion(
        pic_channels=2,
        x_channels=7,
        frequency_count=frequency_count,
        image_size=image_size,
        base_channels=8,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(),
        cond_dim=token_dim,
        x_hidden_channels=8,
        x_token_dim=token_dim,
        frequency_filter_kernel_size=3,
        physical_encoding=geometry,
        fusion_resolutions=(8,),
        fusion_heads=4,
        self_condition_prob=0.0,
        dropout=0.0,
        sigma_min=0.01,
        sigma_max=1.0,
    ).to(device)

    x_matrix = torch.randn(batch_size, 7, frequency_count, tx_count, rx_count, device=device)
    frequency_hz = torch.tensor(
        [[50000.0, 20000.0, 95000.0, 32500.0, 70000.0]],
        device=device,
    ).expand(batch_size, -1)
    tx_indices = torch.arange(1, tx_count + 1, device=device).expand(batch_size, -1)
    rx_indices = torch.arange(tx_count + 1, tx_count + rx_count + 1, device=device).expand(batch_size, -1)
    pic = torch.randn(batch_size, 2, image_size, image_size, device=device)
    target = torch.rand(batch_size, 1, image_size, image_size, device=device)

    sorted_x, sorted_frequency_hz, order = sort_frequency_axis(x_matrix, frequency_hz)
    assert torch.all(sorted_frequency_hz[:, 1:] > sorted_frequency_hz[:, :-1])
    assert order[0].tolist() == [1, 3, 0, 4, 2]

    encoder = model.x_encoder
    frequency_embedding = encoder.frequency_position_embedding(sorted_frequency_hz, dtype=x_matrix.dtype)
    assert frequency_embedding.shape == (batch_size, 7, frequency_count, 1, 1)
    assert not torch.allclose(frequency_embedding[:, :, 0], frequency_embedding[:, :, -1])

    physical_features = encoder.tx_rx_position_embedding.physical_features(
        tx_indices,
        rx_indices,
        batch_size=batch_size,
        device=device,
    )
    assert physical_features.shape == (batch_size, tx_count * rx_count, 6)
    expected_same_angle = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 0.8], device=device)
    assert torch.allclose(physical_features[0, 0], expected_same_angle, atol=1e-6, rtol=1e-6)
    position_parameters = sum(
        parameter.numel() for parameter in encoder.tx_rx_position_embedding.parameters()
    )
    assert position_parameters == 6 * token_dim

    position = encoder.tx_rx_position_embedding(
        tx_indices,
        rx_indices,
        batch_size=batch_size,
        device=device,
        dtype=x_matrix.dtype,
    )
    assert position.shape == (batch_size, tx_count * rx_count, token_dim)
    assert not torch.allclose(position[:, 0], position[:, 1])
    assert not torch.allclose(position[:, 0], position[:, rx_count])

    embedding, low_tokens, high_tokens = encoder(
        x_matrix,
        frequency_hz=frequency_hz,
        tx_indices=tx_indices,
        rx_indices=rx_indices,
    )
    low, high, _ = encoder.frequency_decomposer(sorted_x)
    low_content = encoder._project_tokens(encoder.encode_features(low, sorted_frequency_hz))
    high_content = encoder._project_tokens(encoder.encode_features(high, sorted_frequency_hz))
    assert embedding.shape == (batch_size, token_dim)
    assert low_tokens.shape == high_tokens.shape == (batch_size, tx_count * rx_count, token_dim)
    assert torch.allclose(low_tokens - low_content, position, atol=1e-5, rtol=1e-5)
    assert torch.allclose(high_tokens - high_content, position, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(
        model.unet.mid_fusion.domain_embedding[0],
        model.unet.mid_fusion.domain_embedding[1],
    )

    model.train()
    losses = model.training_losses(
        target,
        pic,
        x_matrix,
        frequency_hz=frequency_hz,
        tx_indices=tx_indices,
        rx_indices=rx_indices,
        sigma=torch.full((batch_size,), 0.5, device=device),
    )
    losses["loss_edm"].backward()
    finite_gradients = [
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert finite_gradients and all(bool(value) for value in finite_gradients)

    model.eval()
    prediction = model.sample(
        pic,
        x_matrix,
        frequency_hz=frequency_hz,
        tx_indices=tx_indices,
        rx_indices=rx_indices,
        steps=2,
        sigma_min=0.01,
        sigma_max=1.0,
    )
    assert prediction.shape == target.shape
    assert torch.isfinite(prediction).all()

    print(
        json.dumps(
            {
                "status": "passed",
                "device": str(device),
                "sorted_frequency_hz": sorted_frequency_hz[0].tolist(),
                "physical_features": list(physical_features.shape),
                "position_tokens": list(position.shape),
                "position_parameters": position_parameters,
                "prediction": list(prediction.shape),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()

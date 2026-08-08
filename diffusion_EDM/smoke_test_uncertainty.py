from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from models import EDMDiffusion, PhysicalEncodingConfig
from sample_edm import (
    LabelDisplaySpec,
    load_label_display_spec,
    resolve_sample_output_directory,
    sample_directory_name,
    sample_output_directory,
    save_physical_posterior_outputs,
    save_uncertainty_preview,
)
from uncertainty import (
    load_posterior_outputs,
    posterior_metrics,
    sample_posterior,
    save_posterior_outputs,
    summarize_posterior,
)


def build_tiny_model(device: torch.device) -> EDMDiffusion:
    geometry = PhysicalEncodingConfig(
        tx_count=2,
        rx_count=2,
        pipe_length_mm=1000.0,
        mid_radius_mm=155.0,
        tx_z_mm=100.0,
        rx_z_mm=900.0,
        frequency_reference_hz=100000.0,
    )
    return EDMDiffusion(
        pic_channels=2,
        x_channels=7,
        frequency_count=3,
        image_size=16,
        target_channels=1,
        base_channels=8,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(),
        cond_dim=16,
        x_hidden_channels=8,
        x_token_dim=16,
        frequency_filter_kernel_size=3,
        physical_encoding=geometry,
        fusion_resolutions=(8,),
        fusion_heads=4,
        self_condition_prob=0.0,
        dropout=0.0,
        sigma_min=0.01,
        sigma_max=1.0,
    ).to(device).eval()


def assert_summary_finite(summary) -> None:
    tensors = (
        summary.mean,
        summary.std,
        summary.median,
        summary.lower,
        summary.upper,
        summary.defect_probability,
        summary.defect_entropy,
        summary.consensus_prediction,
    )
    assert all(tensor.shape == (1, 1, 16, 16) for tensor in tensors)
    assert all(bool(torch.isfinite(tensor).all()) for tensor in tensors)


def main() -> None:
    torch.manual_seed(20260712)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_tiny_model(device)
    pic = torch.randn(1, 2, 16, 16, device=device)
    x_matrix = torch.randn(1, 7, 3, 2, 2, device=device)
    frequency_hz = torch.tensor([[20000.0, 50000.0, 95000.0]], device=device)
    tx_indices = torch.tensor([[1, 2]], device=device)
    rx_indices = torch.tensor([[3, 4]], device=device)
    sample_kwargs = {
        "steps": 2,
        "sigma_min": 0.01,
        "sigma_max": 1.0,
        "s_churn": 0.1,
        "s_tmin": 0.0,
        "s_tmax": float("inf"),
        "s_noise": 1.0,
    }

    def sample_once(seed: int) -> torch.Tensor:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return model.sample(
            pic,
            x_matrix,
            frequency_hz=frequency_hz,
            tx_indices=tx_indices,
            rx_indices=rx_indices,
            generator=generator,
            **sample_kwargs,
        )

    reference = sample_once(4000)
    repeated = sample_once(4000)
    different = sample_once(4001)
    assert torch.equal(reference, repeated)
    assert not torch.equal(reference, different)

    posterior = sample_posterior(
        model,
        pic,
        x_matrix,
        frequency_hz=frequency_hz,
        tx_indices=tx_indices,
        rx_indices=rx_indices,
        num_samples=3,
        sample_seed=4000,
        lower_quantile=0.1,
        upper_quantile=0.9,
        defect_threshold=0.1,
        consensus_probability_threshold=0.5,
        sample_kwargs=sample_kwargs,
    )
    assert posterior.samples is not None
    assert posterior.samples.shape == (3, 1, 1, 16, 16)
    assert posterior.sample_count == 3
    assert torch.equal(posterior.samples[0], reference.cpu())
    assert torch.allclose(posterior.mean, posterior.samples.mean(dim=0))
    assert_summary_finite(posterior)

    single = summarize_posterior(posterior.samples[:1])
    assert torch.count_nonzero(single.std) == 0

    sparse_samples = torch.zeros(4, 1, 1, 16, 16)
    sparse_samples[0, 0, 0, 3, 4] = 0.8
    sparse_samples[:, 0, 0, 8, 9] = 0.4
    sparse = summarize_posterior(
        sparse_samples,
        defect_threshold=0.1,
        consensus_probability_threshold=0.5,
    )
    assert torch.isclose(sparse.defect_probability[0, 0, 3, 4], torch.tensor(0.25))
    assert sparse.mean[0, 0, 3, 4] > 0.0
    assert sparse.consensus_prediction[0, 0, 3, 4] == 0.0
    assert torch.isclose(sparse.consensus_prediction[0, 0, 8, 9], torch.tensor(0.4))

    with tempfile.TemporaryDirectory(prefix="edm_uncertainty_") as temp_dir:
        output_dir = Path(temp_dir)
        label_path = output_dir / "case_defect_depth_norm.npy"
        metadata_path = output_dir / "case_defect_label_metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "normalization_denominator_mm": 9.0,
                    "depth_limit_mm": 5.0,
                    "preview_max_mm": 2.3,
                    "theta_axis": {"range": [0.0, 360.0]},
                    "z_axis": {"range": [0.0, 1000.0]},
                }
            ),
            encoding="utf-8",
        )
        loaded_display = load_label_display_spec(
            label_path,
            np.asarray([[0.0, 2.3 / 9.0]], dtype=np.float32),
        )
        assert loaded_display.normalization_denominator_mm == 9.0
        assert loaded_display.depth_limit_mm == 5.0
        assert loaded_display.preview_vmax_mm == 2.3
        assert np.allclose(
            loaded_display.depth_mm(np.asarray([[0.0, 2.3 / 9.0]], dtype=np.float32)),
            np.asarray([[0.0, 2.3]], dtype=np.float32),
        )

        nested_root = output_dir / "samples"
        nested_id = "dataset_a_frequency_sample_0001"
        assert sample_directory_name(nested_id) == "sample0001"
        nested_dir = sample_output_directory(nested_root, nested_id)
        nested_dir.mkdir(parents=True)
        save_posterior_outputs(nested_dir, nested_id, posterior, save_all_samples=False)
        assert resolve_sample_output_directory(nested_root, nested_id) == nested_dir

        save_posterior_outputs(output_dir, "summary_only", posterior, save_all_samples=False)
        assert resolve_sample_output_directory(output_dir, "summary_only") == output_dir
        loaded_summary = load_posterior_outputs(output_dir, "summary_only")
        assert loaded_summary is not None
        assert loaded_summary.samples is None
        assert loaded_summary.sample_count == 3
        assert torch.equal(loaded_summary.mean, posterior.mean)
        assert torch.equal(loaded_summary.std, posterior.std)

        save_posterior_outputs(output_dir, "with_samples", posterior, save_all_samples=True)
        loaded_samples = load_posterior_outputs(output_dir, "with_samples")
        assert loaded_samples is not None and loaded_samples.samples is not None
        assert loaded_samples.sample_count == 3
        assert torch.equal(loaded_samples.samples, posterior.samples)

        display = LabelDisplaySpec(
            normalization_denominator_mm=9.0,
            depth_limit_mm=5.0,
            preview_vmax_mm=2.5,
            theta_range_deg=(0.0, 360.0),
            z_range_mm=(0.0, 1000.0),
        )
        physical_paths = save_physical_posterior_outputs(
            output_dir,
            "physical",
            posterior,
            display,
        )
        prediction_mm = np.load(physical_paths["prediction_mm"])
        uncertainty_mm = np.load(physical_paths["uncertainty_mm"])
        assert prediction_mm.shape == (16, 16)
        assert 0.0 <= float(prediction_mm.min()) <= float(prediction_mm.max()) <= 5.0
        expected_physical_samples = np.clip(
            posterior.samples[:, 0, 0].numpy() * display.normalization_denominator_mm,
            0.0,
            display.depth_limit_mm,
        )
        assert np.allclose(
            uncertainty_mm,
            expected_physical_samples.std(axis=0),
        )
        preview_path = output_dir / "physical_preview.png"
        save_uncertainty_preview(
            preview_path,
            posterior,
            posterior.mean[0, 0].numpy(),
            np.zeros((16, 16), dtype=np.float32),
            display,
        )
        assert preview_path.exists() and preview_path.stat().st_size > 0

        target = np.clip(
            posterior.mean[0, 0].numpy()
            + np.linspace(-0.02, 0.02, 16 * 16, dtype=np.float32).reshape(16, 16),
            0.0,
            1.0,
        )
        coverage = np.linspace(0.0, 1.0, 16 * 16, dtype=np.float32).reshape(16, 16)
        reliability = (coverage >= 0.5).astype(np.float32)
        metrics = posterior_metrics(
            loaded_samples,
            target,
            coverage=coverage,
            reliability=reliability,
        )
        required_metrics = (
            "posterior_sample_count",
            "posterior_crps",
            "interval_empirical_coverage",
            "interval_calibration_error",
            "uncertainty_error_spearman",
            "uncertainty_low_coverage_spearman",
            "defect_probability_brier",
        )
        assert all(name in metrics for name in required_metrics)
        assert all(np.isfinite(metrics[name]) for name in required_metrics)
        assert metrics["posterior_sample_count"] == 3.0

    print(
        json.dumps(
            {
                "status": "passed",
                "device": str(device),
                "samples": list(posterior.samples.shape),
                "sample_count_after_summary_only_load": loaded_summary.sample_count,
                "same_seed_exact": True,
                "different_seed_exact": False,
                "sparse_hallucination_probability": float(
                    sparse.defect_probability[0, 0, 3, 4]
                ),
                "sparse_hallucination_consensus": float(
                    sparse.consensus_prediction[0, 0, 3, 4]
                ),
                "posterior_crps": metrics["posterior_crps"],
                "physical_prediction_max_mm": float(prediction_mm.max()),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()

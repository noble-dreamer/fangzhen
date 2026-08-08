"""Fast deterministic checks for the distribution-matched channel-prior math."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from channel_common import ChannelPriorConfig
from fit_channel_prior import estimate_paths, fit_channel_model_huber


HERE = Path(__file__).resolve().parent


def main() -> None:
    config = ChannelPriorConfig.from_json(HERE / "configs" / "dataset_a_channel_prior.json")
    rng = np.random.default_rng(9201)
    sample_count, path_count, frequency_count = 24, 16, 5
    truth = rng.uniform(0.01, 0.45, size=(sample_count, path_count))
    amp_slope = rng.uniform(-1.1, 1.1, size=(path_count, frequency_count))
    phase_slope = rng.uniform(-0.9, 0.9, size=(path_count, frequency_count))
    amp_intercept = rng.normal(0.0, 0.03, size=(path_count, frequency_count))
    phase_intercept = rng.normal(0.0, 0.03, size=(path_count, frequency_count))
    amplitude = amp_intercept[None] + amp_slope[None] * truth[:, :, None]
    phase = phase_intercept[None] + phase_slope[None] * truth[:, :, None]
    amplitude += rng.normal(0.0, 0.01, size=amplitude.shape)
    phase += rng.normal(0.0, 0.01, size=phase.shape)
    amplitude[0, 0, 0] += 0.6  # Huber fit must reject this outlier.
    amp_model = fit_channel_model_huber(amplitude, truth)
    phase_model = fit_channel_model_huber(phase, truth)
    reliability = np.ones((path_count, frequency_count), dtype=np.float64)
    estimate, weights, retained_amp, retained_phase = estimate_paths(
        amplitude,
        phase,
        reliability,
        amp_model,
        phase_model,
        config,
    )
    rmse = float(np.sqrt(np.mean((estimate - truth) ** 2)))
    assert np.all(np.isfinite(estimate))
    assert np.all(np.isfinite(weights))
    assert np.all(estimate >= 0.0)
    assert rmse < 0.06, rmse
    assert retained_amp > 0.0 and retained_phase > 0.0
    print(
        "channel-prior smoke passed:",
        f"path_rmse_mm={rmse:.6g}",
        f"retained_amp={retained_amp:.3f}",
        f"retained_phase={retained_phase:.3f}",
    )


if __name__ == "__main__":
    main()

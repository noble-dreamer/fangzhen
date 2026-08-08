from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class PosteriorSummary:
    samples: torch.Tensor | None
    stored_sample_count: int
    mean: torch.Tensor
    std: torch.Tensor
    median: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    defect_probability: torch.Tensor
    defect_entropy: torch.Tensor
    consensus_prediction: torch.Tensor
    lower_quantile: float
    upper_quantile: float
    defect_threshold: float
    consensus_probability_threshold: float

    @property
    def interval_width(self) -> torch.Tensor:
        return self.upper - self.lower

    @property
    def sample_count(self) -> int:
        if self.samples is not None:
            return int(self.samples.shape[0])
        return int(self.stored_sample_count)


def summarize_posterior(
    samples: torch.Tensor,
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    defect_threshold: float = 0.1,
    consensus_probability_threshold: float = 0.5,
) -> PosteriorSummary:
    if samples.ndim != 5:
        raise RuntimeError(f"samples must be (K,B,C,H,W), got {tuple(samples.shape)}")
    if samples.shape[0] < 1:
        raise ValueError("At least one posterior sample is required")
    if not torch.isfinite(samples).all():
        raise RuntimeError("Posterior samples contain non-finite values")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("Posterior quantiles must satisfy 0 <= lower < upper <= 1")
    if not 0.0 <= defect_threshold <= 1.0:
        raise ValueError("defect_threshold must be in [0, 1]")
    if not 0.0 <= consensus_probability_threshold <= 1.0:
        raise ValueError("consensus_probability_threshold must be in [0, 1]")

    values = samples.float()
    mean = values.mean(dim=0)
    std = values.std(dim=0, correction=0)
    quantiles = torch.tensor(
        [lower_quantile, 0.5, upper_quantile],
        device=values.device,
        dtype=values.dtype,
    )
    lower, median, upper = torch.quantile(values, quantiles, dim=0)
    defect_probability = (values >= float(defect_threshold)).float().mean(dim=0)
    probability_safe = defect_probability.clamp(1e-7, 1.0 - 1e-7)
    defect_entropy = -(
        probability_safe * torch.log(probability_safe)
        + (1.0 - probability_safe) * torch.log(1.0 - probability_safe)
    ) / math.log(2.0)
    defect_entropy = torch.where(
        (defect_probability <= 0.0) | (defect_probability >= 1.0),
        torch.zeros_like(defect_entropy),
        defect_entropy,
    )
    consensus_prediction = torch.where(
        defect_probability >= float(consensus_probability_threshold),
        mean,
        torch.zeros_like(mean),
    )
    return PosteriorSummary(
        samples=values,
        stored_sample_count=int(values.shape[0]),
        mean=mean,
        std=std,
        median=median,
        lower=lower,
        upper=upper,
        defect_probability=defect_probability,
        defect_entropy=defect_entropy,
        consensus_prediction=consensus_prediction,
        lower_quantile=float(lower_quantile),
        upper_quantile=float(upper_quantile),
        defect_threshold=float(defect_threshold),
        consensus_probability_threshold=float(consensus_probability_threshold),
    )


def sample_posterior(
    model: Any,
    pic: torch.Tensor,
    x_matrix: torch.Tensor,
    *,
    frequency_hz: torch.Tensor,
    tx_indices: torch.Tensor,
    rx_indices: torch.Tensor,
    num_samples: int,
    sample_seed: int,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    defect_threshold: float = 0.1,
    consensus_probability_threshold: float = 0.5,
    sample_kwargs: dict[str, Any] | None = None,
) -> PosteriorSummary:
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    kwargs = dict(sample_kwargs or {})
    if "generator" in kwargs or "initial_noise" in kwargs:
        raise ValueError("sample_kwargs must not override generator or initial_noise")

    samples: list[torch.Tensor] = []
    for sample_index in range(int(num_samples)):
        generator = torch.Generator(device=pic.device)
        generator.manual_seed(int(sample_seed) + sample_index)
        prediction = model.sample(
            pic,
            x_matrix,
            frequency_hz=frequency_hz,
            tx_indices=tx_indices,
            rx_indices=rx_indices,
            generator=generator,
            **kwargs,
        )
        samples.append(prediction.detach().float().cpu())
    stacked = torch.stack(samples, dim=0)
    return summarize_posterior(
        stacked,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        defect_threshold=defect_threshold,
        consensus_probability_threshold=consensus_probability_threshold,
    )


def _single_image(value: torch.Tensor, name: str) -> np.ndarray:
    if value.ndim != 4 or value.shape[:2] != (1, 1):
        raise RuntimeError(f"{name} must be (1,1,H,W), got {tuple(value.shape)}")
    return value[0, 0].detach().cpu().numpy().astype(np.float32)


def save_posterior_outputs(
    output_dir: Path,
    sample_id: str,
    summary: PosteriorSummary,
    *,
    save_all_samples: bool = False,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "prediction": _single_image(summary.mean, "mean"),
        "uncertainty": _single_image(summary.std, "std"),
        "posterior_median": _single_image(summary.median, "median"),
        "posterior_lower": _single_image(summary.lower, "lower"),
        "posterior_upper": _single_image(summary.upper, "upper"),
        "defect_probability": _single_image(summary.defect_probability, "defect_probability"),
        "defect_entropy": _single_image(summary.defect_entropy, "defect_entropy"),
        "consensus_prediction": _single_image(summary.consensus_prediction, "consensus_prediction"),
    }
    paths: dict[str, Path] = {}
    for name, array in arrays.items():
        path = output_dir / f"{sample_id}_{name}.npy"
        np.save(path, array)
        paths[name] = path

    summary_path = output_dir / f"{sample_id}_posterior_summary.npz"
    np.savez_compressed(
        summary_path,
        **arrays,
        lower_quantile=np.asarray(summary.lower_quantile, dtype=np.float32),
        upper_quantile=np.asarray(summary.upper_quantile, dtype=np.float32),
        defect_threshold=np.asarray(summary.defect_threshold, dtype=np.float32),
        consensus_probability_threshold=np.asarray(
            summary.consensus_probability_threshold,
            dtype=np.float32,
        ),
        sample_count=np.asarray(summary.sample_count, dtype=np.int32),
    )
    paths["posterior_summary"] = summary_path

    if save_all_samples:
        if summary.samples is None:
            raise RuntimeError("Cannot save posterior samples because they are unavailable")
        if summary.samples.shape[1:3] != (1, 1):
            raise RuntimeError(f"samples must be (K,1,1,H,W), got {tuple(summary.samples.shape)}")
        samples_path = output_dir / f"{sample_id}_posterior_samples.npz"
        samples_np = summary.samples[:, 0, 0].numpy().astype(np.float32)
        np.savez_compressed(samples_path, samples=samples_np)
        paths["posterior_samples"] = samples_path
    return paths


def load_posterior_outputs(output_dir: Path, sample_id: str) -> PosteriorSummary | None:
    summary_path = output_dir / f"{sample_id}_posterior_summary.npz"
    if not summary_path.exists():
        return None
    with np.load(summary_path, allow_pickle=False) as data:
        required = (
            "prediction",
            "uncertainty",
            "posterior_median",
            "posterior_lower",
            "posterior_upper",
            "defect_probability",
            "defect_entropy",
            "consensus_prediction",
        )
        missing = [name for name in required if name not in data.files]
        if missing:
            raise RuntimeError(f"{summary_path} missing posterior arrays: {missing}")

        def tensor(name: str) -> torch.Tensor:
            array = np.asarray(data[name], dtype=np.float32).copy()
            if array.ndim != 2:
                raise RuntimeError(f"{summary_path}:{name} must be 2D, got {array.shape}")
            return torch.from_numpy(array)[None, None]

        mean = tensor("prediction")
        std = tensor("uncertainty")
        median = tensor("posterior_median")
        lower = tensor("posterior_lower")
        upper = tensor("posterior_upper")
        defect_probability = tensor("defect_probability")
        defect_entropy = tensor("defect_entropy")
        consensus_prediction = tensor("consensus_prediction")
        lower_quantile = float(np.asarray(data["lower_quantile"]).item())
        upper_quantile = float(np.asarray(data["upper_quantile"]).item())
        defect_threshold = float(np.asarray(data["defect_threshold"]).item())
        consensus_probability_threshold = float(
            np.asarray(data["consensus_probability_threshold"]).item()
        )
        stored_sample_count = (
            int(np.asarray(data["sample_count"]).item())
            if "sample_count" in data.files
            else 0
        )
        if stored_sample_count < 0:
            raise RuntimeError(
                f"{summary_path}:sample_count must be non-negative, got {stored_sample_count}"
            )

    samples = None
    samples_path = output_dir / f"{sample_id}_posterior_samples.npz"
    if samples_path.exists():
        with np.load(samples_path, allow_pickle=False) as data:
            values = np.asarray(data["samples"], dtype=np.float32).copy()
        if values.ndim != 3:
            raise RuntimeError(f"{samples_path}:samples must be (K,H,W), got {values.shape}")
        samples = torch.from_numpy(values)[:, None, None]
        if stored_sample_count not in (0, int(samples.shape[0])):
            raise RuntimeError(
                f"{samples_path}: sample count {samples.shape[0]} does not match "
                f"summary sample_count {stored_sample_count}"
            )
        stored_sample_count = int(samples.shape[0])
    return PosteriorSummary(
        samples=samples,
        stored_sample_count=stored_sample_count,
        mean=mean,
        std=std,
        median=median,
        lower=lower,
        upper=upper,
        defect_probability=defect_probability,
        defect_entropy=defect_entropy,
        consensus_prediction=consensus_prediction,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        defect_threshold=defect_threshold,
        consensus_probability_threshold=consensus_probability_threshold,
    )


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        sorted_ranks[start:stop] = 0.5 * (start + stop - 1)
        start = stop
    ranks = np.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


def spearman_np(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 2:
        return float("nan")
    first_rank = _rankdata(first[valid])
    second_rank = _rankdata(second[valid])
    if np.std(first_rank) <= 0.0 or np.std(second_rank) <= 0.0:
        return float("nan")
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def posterior_metrics(
    summary: PosteriorSummary,
    target: np.ndarray,
    *,
    coverage: np.ndarray | None = None,
    reliability: np.ndarray | None = None,
) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float32)
    mean = _single_image(summary.mean, "mean")
    std = _single_image(summary.std, "std")
    lower = _single_image(summary.lower, "lower")
    upper = _single_image(summary.upper, "upper")
    probability = _single_image(summary.defect_probability, "defect_probability")
    entropy = _single_image(summary.defect_entropy, "defect_entropy")
    consensus = _single_image(summary.consensus_prediction, "consensus_prediction")
    if target.shape != mean.shape:
        raise RuntimeError(f"target shape {target.shape} does not match posterior shape {mean.shape}")
    valid = np.isfinite(target) & np.isfinite(mean) & np.isfinite(std)
    if not np.any(valid):
        return {"posterior_sample_count": float(summary.sample_count)}

    absolute_error = np.abs(mean - target)
    interval_contains = (target >= lower) & (target <= upper)
    nominal_coverage = float(summary.upper_quantile - summary.lower_quantile)
    empirical_coverage = float(np.mean(interval_contains[valid]))
    target_defect = target >= summary.defect_threshold
    mean_defect = mean >= summary.defect_threshold
    consensus_defect = consensus >= summary.defect_threshold
    result = {
        "posterior_sample_count": float(summary.sample_count),
        "uncertainty_mean": float(np.mean(std[valid])),
        "uncertainty_p95": float(np.quantile(std[valid], 0.95)),
        "uncertainty_max": float(np.max(std[valid])),
        "defect_entropy_mean": float(np.mean(entropy[valid])),
        "defect_probability_mean": float(np.mean(probability[valid])),
        "interval_nominal_coverage": nominal_coverage,
        "interval_empirical_coverage": empirical_coverage,
        "interval_calibration_error": abs(empirical_coverage - nominal_coverage),
        "interval_mean_width": float(np.mean((upper - lower)[valid])),
        "uncertainty_error_spearman": spearman_np(std[valid], absolute_error[valid]),
        "defect_probability_brier": float(
            np.mean((probability[valid] - target_defect[valid].astype(np.float32)) ** 2)
        ),
        "posterior_mean_defect_fraction": float(np.mean(mean_defect[valid])),
        "consensus_defect_fraction": float(np.mean(consensus_defect[valid])),
    }
    if np.any(mean_defect & valid):
        removed = mean_defect & ~consensus_defect & valid
        result["consensus_removed_fraction"] = float(removed.sum() / (mean_defect & valid).sum())
    else:
        result["consensus_removed_fraction"] = 0.0
    for name, mask in (
        ("defect", target_defect & valid),
        ("background", ~target_defect & valid),
    ):
        result[f"interval_coverage_{name}"] = (
            float(np.mean(interval_contains[mask])) if np.any(mask) else float("nan")
        )
        result[f"uncertainty_mean_{name}"] = (
            float(np.mean(std[mask])) if np.any(mask) else float("nan")
        )

    if summary.samples is not None:
        samples = summary.samples[:, 0, 0].numpy().astype(np.float32)
        if samples.shape[1:] != target.shape:
            raise RuntimeError(f"sample shape {samples.shape} does not match target {target.shape}")
        sorted_samples = np.sort(samples, axis=0)
        count = samples.shape[0]
        weights = (2.0 * np.arange(1, count + 1) - count - 1.0).reshape(count, 1, 1)
        first_term = np.mean(np.abs(samples - target[None]), axis=0)
        second_term = np.sum(weights * sorted_samples, axis=0) / float(count * count)
        result["posterior_crps"] = float(np.mean((first_term - second_term)[valid]))

    if coverage is not None:
        coverage = np.asarray(coverage, dtype=np.float32)
        if coverage.shape != target.shape:
            raise RuntimeError(f"coverage shape {coverage.shape} does not match target {target.shape}")
        coverage_valid = valid & np.isfinite(coverage)
        if np.any(coverage_valid):
            result["uncertainty_low_coverage_spearman"] = spearman_np(
                std[coverage_valid],
                1.0 - coverage[coverage_valid],
            )
            q25, q75 = np.quantile(coverage[coverage_valid], [0.25, 0.75])
            low_mask = coverage_valid & (coverage <= q25)
            high_mask = coverage_valid & (coverage >= q75)
            result["uncertainty_low_coverage_mean"] = float(np.mean(std[low_mask]))
            result["uncertainty_high_coverage_mean"] = float(np.mean(std[high_mask]))

    if reliability is not None:
        reliability = np.asarray(reliability, dtype=np.float32)
        if reliability.shape != target.shape:
            raise RuntimeError(f"reliability shape {reliability.shape} does not match target {target.shape}")
        reliable = valid & np.isfinite(reliability) & (reliability >= 0.5)
        unreliable = valid & np.isfinite(reliability) & (reliability < 0.5)
        result["uncertainty_reliable_mean"] = (
            float(np.mean(std[reliable])) if np.any(reliable) else float("nan")
        )
        result["uncertainty_unreliable_mean"] = (
            float(np.mean(std[unreliable])) if np.any(unreliable) else float("nan")
        )
    return result

"""Fit a fixed-alpha complex Rytov channel prior from independent COMSOL cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from channel_common import (
    ChannelPriorConfig,
    canonical_sample_metadata,
    config_sha256,
    physical_sources_match,
    response_is_complete,
    source_fingerprints,
)
from common import (
    build_order_kernel_bank,
    canonical_json_sha256,
    load_geometry,
    model_semantic_sha256,
    sha256_array,
    sha256_file,
    subset_response,
    validate_requested_axes,
)
from simple.get_pic import coarse_map_common as cm
from simple.get_pic.physical_inversion.inversion import (
    EPS,
    ChannelModel,
    InversionConfig,
    PhysicalInverter,
    resize_label_nearest,
    safe_pearson,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs" / "dataset_a_channel_prior.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a simulation-only complex Rytov prior from the matched COMSOL corpus."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--corpus-root", type=Path, default=None)
    parser.add_argument("--output-metadata", type=Path, default=None)
    parser.add_argument("--output-model", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def top_k_weights(weights: np.ndarray, top_k: int) -> np.ndarray:
    if top_k >= weights.shape[-1]:
        return weights
    keep = np.argpartition(weights, -top_k, axis=-1)[..., -top_k:]
    mask = np.zeros_like(weights, dtype=bool)
    np.put_along_axis(mask, keep, True, axis=-1)
    return np.where(mask, weights, 0.0)


def channel_weights(
    model: ChannelModel,
    reliability: np.ndarray,
    config: ChannelPriorConfig,
    sample_count: int,
) -> np.ndarray:
    quality = np.abs(model.correlation)
    weights = quality * quality / (model.sigma_path_mm * model.sigma_path_mm + config.channel_noise_floor_mm**2)
    weights = np.where(quality >= config.channel_min_abs_correlation, weights, 0.0)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    weights = top_k_weights(weights, min(config.channel_top_k, weights.shape[-1]))
    weights /= np.maximum(np.max(weights, axis=-1, keepdims=True), EPS)
    weights = np.broadcast_to(weights, (sample_count, *weights.shape)).copy()
    if config.channel_use_healthy_reliability:
        weights *= reliability[None, :, :]
    return weights


def estimate_paths(
    amplitude: np.ndarray,
    phase: np.ndarray,
    reliability: np.ndarray,
    amplitude_model: ChannelModel,
    phase_model: ChannelModel,
    config: ChannelPriorConfig,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    sample_count = amplitude.shape[0]
    amp_denominator = np.where(np.abs(amplitude_model.slope) > 1e-12, amplitude_model.slope, np.nan)
    phase_denominator = np.where(np.abs(phase_model.slope) > 1e-12, phase_model.slope, np.nan)
    amp_estimate = np.nan_to_num(
        (amplitude - amplitude_model.intercept[None, :, :]) / amp_denominator[None, :, :],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    phase_estimate = np.nan_to_num(
        (phase - phase_model.intercept[None, :, :]) / phase_denominator[None, :, :],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    amp_weights = channel_weights(amplitude_model, reliability, config, sample_count)
    phase_weights = channel_weights(phase_model, reliability, config, sample_count)
    estimates = np.concatenate([amp_estimate, phase_estimate], axis=-1)
    weights = np.concatenate([amp_weights, phase_weights], axis=-1)
    weights = top_k_weights(weights, min(config.channel_top_k, weights.shape[-1]))
    estimates = np.clip(estimates, -config.defect_loss_limit_mm, 2.0 * config.defect_loss_limit_mm)
    numerator = np.sum(estimates * weights, axis=-1)
    denominator = np.sum(weights, axis=-1)
    path = np.clip(numerator / np.maximum(denominator, EPS), 0.0, config.defect_loss_limit_mm)
    path_weights = denominator / max(float(np.max(denominator)), EPS)
    return (
        path.astype(np.float64),
        path_weights.astype(np.float64),
        float(np.mean(amp_weights > 0.0)),
        float(np.mean(phase_weights > 0.0)),
    )


def simplex_grid(order_count: int, step: float) -> Iterable[np.ndarray]:
    scale = int(round(1.0 / step))
    if order_count != 3:
        raise ValueError("The matched prior currently supports exactly three helical orders")
    for first in range(scale + 1):
        for second in range(scale - first + 1):
            third = scale - first - second
            yield np.asarray([first, second, third], dtype=np.float64) / float(scale)


def fit_models(
    amplitude: np.ndarray,
    phase: np.ndarray,
    truth: np.ndarray,
) -> tuple[ChannelModel, ChannelModel]:
    return fit_channel_model_huber(amplitude, truth), fit_channel_model_huber(phase, truth)


def fit_channel_model_huber(
    response: np.ndarray,
    path_truth_mm: np.ndarray,
    *,
    delta: float = 1.5,
    iterations: int = 8,
) -> ChannelModel:
    """Independent robust affine fit for every offset/frequency channel."""

    x = np.asarray(path_truth_mm, dtype=np.float64)
    y = np.asarray(response, dtype=np.float64)
    if x.ndim != 2 or y.shape[:2] != x.shape:
        raise ValueError("response must have shape (sample, path, frequency)")
    paths, frequencies = x.shape[1], y.shape[2]
    intercept = np.zeros((paths, frequencies), dtype=np.float64)
    slope = np.zeros_like(intercept)
    correlation = np.zeros_like(intercept)
    sigma_path = np.full_like(intercept, np.inf)
    for path in range(paths):
        xp = x[:, path]
        x_centered = xp - np.mean(xp)
        sum_x2 = float(np.sum(x_centered * x_centered))
        if sum_x2 <= 1e-18:
            continue
        for frequency in range(frequencies):
            yp = y[:, path, frequency]
            if not np.all(np.isfinite(yp)):
                continue
            slope_value = float(np.sum(x_centered * (yp - np.mean(yp))) / sum_x2)
            intercept_value = float(np.mean(yp) - slope_value * np.mean(xp))
            weights = np.ones_like(xp)
            for _ in range(iterations):
                residual = yp - (intercept_value + slope_value * xp)
                scale = float(np.median(np.abs(residual - np.median(residual))) / 0.67448975)
                scale = max(scale, 1e-9)
                weights = np.minimum(1.0, delta * scale / np.maximum(np.abs(residual), 1e-12))
                total = float(np.sum(weights))
                x_mean = float(np.sum(weights * xp) / max(total, EPS))
                y_mean = float(np.sum(weights * yp) / max(total, EPS))
                denom = float(np.sum(weights * (xp - x_mean) ** 2))
                if denom <= 1e-18:
                    break
                slope_value = float(np.sum(weights * (xp - x_mean) * (yp - y_mean)) / denom)
                intercept_value = y_mean - slope_value * x_mean
            residual = yp - (intercept_value + slope_value * xp)
            y_centered = yp - np.mean(yp)
            sum_y2 = float(np.sum(y_centered * y_centered))
            corr = float(np.sum(x_centered * y_centered) / np.sqrt(max(sum_x2 * sum_y2, 1e-30)))
            sigma_response = float(np.sqrt(np.mean(residual * residual)))
            if not np.isfinite(slope_value) or not np.isfinite(corr) or abs(slope_value) <= 1e-12:
                continue
            intercept[path, frequency] = intercept_value
            slope[path, frequency] = slope_value
            correlation[path, frequency] = corr
            sigma_path[path, frequency] = sigma_response / abs(slope_value)
    return ChannelModel(
        intercept=intercept,
        slope=slope,
        correlation=correlation,
        sigma_path_mm=sigma_path,
    )


def alpha_cross_validation(
    projections: np.ndarray,
    amplitude: np.ndarray,
    phase: np.ndarray,
    reliability: np.ndarray,
    config: ChannelPriorConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    sample_count = projections.shape[0]
    positions = np.arange(sample_count)
    candidates: list[dict[str, Any]] = []
    for alpha in simplex_grid(projections.shape[-1], config.alpha_grid_step):
        truth = np.einsum("spo,o->sp", projections, alpha)
        prediction = np.zeros_like(truth)
        for fold in range(config.calibration_folds):
            held = positions[positions % config.calibration_folds == fold]
            fit = positions[positions % config.calibration_folds != fold]
            amplitude_model, phase_model = fit_models(amplitude[fit], phase[fit], truth[fit])
            prediction[held] = estimate_paths(
                amplitude[held],
                phase[held],
                reliability,
                amplitude_model,
                phase_model,
                config,
            )[0]
        rmse = float(np.sqrt(np.mean((prediction - truth) ** 2)))
        pearson = safe_pearson(prediction, truth)
        candidates.append(
            {
                "alpha": alpha,
                "path_rmse_mm": rmse,
                "path_pearson": pearson,
            }
        )
    candidates.sort(
        key=lambda item: (
            item["path_rmse_mm"],
            -np.nan_to_num(item["path_pearson"], nan=-1.0),
        )
    )
    selected = candidates[0]
    return selected["alpha"], {
        "candidate_count": len(candidates),
        "selection": "minimum_kfold_path_rmse_then_maximum_pearson",
        "selected_path_rmse_mm": selected["path_rmse_mm"],
        "selected_path_pearson": selected["path_pearson"],
        "top_candidates": [
            {
                "order_weights": item["alpha"].tolist(),
                "path_rmse_mm": item["path_rmse_mm"],
                "path_pearson": item["path_pearson"],
            }
            for item in candidates[:8]
        ],
    }


def validate_plan(plan: dict[str, Any], config: ChannelPriorConfig) -> None:
    if plan.get("artifact_kind") != "comsol_distribution_matched_channel_corpus_plan":
        raise RuntimeError("Unexpected channel corpus plan artifact")
    if plan.get("config_sha256") != config_sha256(config):
        raise RuntimeError("Channel corpus plan/config mismatch")
    if plan.get("healthy_npz_sha256") != sha256_file(config.healthy_path):
        raise RuntimeError("Channel corpus plan was built against a different healthy response")
    healthy_metadata = json.loads(config.healthy_metadata_path.read_text(encoding="utf-8"))
    if plan.get("healthy_model_semantic_sha256") != model_semantic_sha256(healthy_metadata):
        raise RuntimeError("Channel corpus formal model fingerprint mismatch")
    if not physical_sources_match(plan.get("source_fingerprints")):
        raise RuntimeError("Channel corpus physical-model fingerprint mismatch")


def load_corpus(
    config: ChannelPriorConfig,
    plan: dict[str, Any],
    plan_path: Path,
    corpus_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    healthy, geometry = load_geometry(config)
    validate_requested_axes(
        healthy,
        tx_indices=config.simulation_tx_indices,
        frequencies_hz=config.frequencies_hz,
    )
    frequencies = tuple(sorted(config.frequencies_hz))
    h0 = subset_response(
        healthy, config.simulation_tx_indices, healthy.rx_indices, frequencies
    ).reshape(-1, len(frequencies))
    all_h0 = subset_response(
        healthy, healthy.tx_indices, healthy.rx_indices, frequencies
    )
    floor = np.quantile(
        np.abs(all_h0).reshape(-1, len(frequencies)), config.healthy_floor_quantile, axis=0
    )
    reliability = np.abs(h0) ** 2 / (np.abs(h0) ** 2 + floor[None, :] ** 2 + EPS)
    bank = build_order_kernel_bank(config, healthy, geometry)
    amplitudes: list[np.ndarray] = []
    phases: list[np.ndarray] = []
    projections: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for sample in plan["samples"]:
        sample_id = str(sample["sample_id"])
        response_path = corpus_root / "frequency_response" / f"{sample_id}_H_complex.npz"
        metadata_path = corpus_root / "metadata" / f"{sample_id}.json"
        label_path = corpus_root / "labels" / f"{sample_id}_defect_depth_mm.npy"
        expected = canonical_sample_metadata(sample, plan_path, config)
        if not response_is_complete(
            response_path,
            metadata_path,
            tx_indices=config.simulation_tx_indices,
            rx_indices=tuple(int(value) for value in healthy.rx_indices),
            frequencies_hz=config.frequencies_hz,
            expected_sample=expected,
        ):
            raise RuntimeError(f"Corpus response is incomplete or incompatible: {sample_id}")
        if not label_path.exists():
            raise FileNotFoundError(label_path)
        response_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if model_semantic_sha256(response_metadata) != plan["healthy_model_semantic_sha256"]:
            raise RuntimeError(f"Corpus response model differs from formal healthy: {sample_id}")
        response = cm.load_frequency_response(response_path)
        validate_requested_axes(
            response,
            tx_indices=config.simulation_tx_indices,
            frequencies_hz=frequencies,
        )
        hd = subset_response(
            response, config.simulation_tx_indices, healthy.rx_indices, frequencies
        ).reshape(-1, len(frequencies))
        if not np.all(np.isfinite(hd.real)) or not np.all(np.isfinite(hd.imag)):
            raise RuntimeError(f"Non-finite complex corpus response: {sample_id}")
        z = 1.0 + (hd - h0) * np.conj(h0) / (np.abs(h0) ** 2 + floor[None, :] ** 2)
        amplitudes.append(np.log(np.abs(z) + EPS))
        phases.append(np.angle(z))
        label = np.asarray(np.load(label_path), dtype=np.float32)
        label = resize_label_nearest(label, (config.inversion_grid_size, config.inversion_grid_size))
        if not np.all(np.isfinite(label)) or float(np.max(label)) <= 0.0:
            raise RuntimeError(f"Invalid physical label generated for corpus sample {sample_id}")
        projections.append(
            np.einsum("poj,j->po", bank.matrix.astype(np.float64), label.reshape(-1))
        )
        records.append(
            {
                "sample_id": sample_id,
                "split": sample["split"],
                "response_npz": str(response_path),
                "response_npz_sha256": sha256_file(response_path),
                "metadata_sha256": sha256_file(metadata_path),
                "label_depth_mm_sha256": sha256_file(label_path),
                "label_peak_mm": float(np.max(label)),
                "label_mean_mm": float(np.mean(label)),
            }
        )
    return (
        np.stack(amplitudes),
        np.stack(phases),
        reliability,
        np.stack(projections),
        np.asarray(frequencies, dtype=np.float64),
        records,
    )


def main() -> None:
    args = parse_args()
    config = ChannelPriorConfig.from_json(args.config.resolve())
    plan_path = args.plan.resolve() if args.plan else config.plan_path
    corpus_root = args.corpus_root.resolve() if args.corpus_root else config.corpus_output_path
    metadata_path = args.output_metadata.resolve() if args.output_metadata else config.prior_metadata_path
    model_path = args.output_model.resolve() if args.output_model else config.prior_model_path
    if plan_path.resolve() != config.plan_path.resolve() or corpus_root.resolve() != config.corpus_output_path.resolve():
        raise RuntimeError("Formal channel-prior fitting requires the configured plan and corpus roots")
    if (metadata_path.exists() or model_path.exists()) and not args.force:
        raise FileExistsError("Refusing to overwrite channel prior; pass --force")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan, config)
    amplitude, phase, reliability, projections, frequencies, records = load_corpus(
        config, plan, plan_path, corpus_root
    )
    fit_mask = np.asarray([item["split"] == "fit" for item in plan["samples"]], dtype=bool)
    validation_mask = ~fit_mask
    if int(np.sum(fit_mask)) != config.corpus_fit_count or int(np.sum(validation_mask)) < 4:
        raise RuntimeError("Channel corpus split does not match the configured fit/validation counts")
    phase_branch_fraction = np.mean(np.abs(phase[fit_mask]) > (np.pi / 2.0), axis=0)
    invalid_phase = phase_branch_fraction > 0.05
    phase_fit = phase[fit_mask].copy()
    # Branch-contaminated phase channels cannot guide alpha selection either.
    phase_fit[:, invalid_phase] = 0.0
    alpha, alpha_selection = alpha_cross_validation(
        projections[fit_mask], amplitude[fit_mask], phase_fit, reliability, config
    )
    fit_truth = np.einsum("spo,o->sp", projections[fit_mask], alpha)
    amplitude_model, phase_model = fit_models(amplitude[fit_mask], phase_fit, fit_truth)
    validation_truth = np.einsum("spo,o->sp", projections[validation_mask], alpha)
    validation_prediction, validation_weights, retained_amp, retained_phase = estimate_paths(
        amplitude[validation_mask],
        phase[validation_mask],
        reliability,
        amplitude_model,
        phase_model,
        config,
    )
    validation = {
        "sample_count": int(np.sum(validation_mask)),
        "path_observation_count": int(validation_truth.size),
        "path_depth_rmse_mm": float(np.sqrt(np.mean((validation_prediction - validation_truth) ** 2))),
        "path_depth_mae_mm": float(np.mean(np.abs(validation_prediction - validation_truth))),
        "path_depth_bias_mm": float(np.mean(validation_prediction - validation_truth)),
        "path_depth_pearson": safe_pearson(validation_prediction, validation_truth),
        "mean_path_weight": float(np.mean(validation_weights)),
        "retained_amplitude_channel_fraction": retained_amp,
        "retained_phase_channel_fraction": retained_phase,
        "acceptance_thresholds": {
            "maximum_path_depth_rmse_mm": config.maximum_heldout_path_rmse_mm,
            "minimum_path_depth_pearson": config.minimum_heldout_path_pearson,
        },
    }
    failures: list[str] = []
    if validation["path_depth_rmse_mm"] > config.maximum_heldout_path_rmse_mm:
        failures.append(
            f"path_depth_rmse_mm={validation['path_depth_rmse_mm']:.6g} exceeds "
            f"{config.maximum_heldout_path_rmse_mm:.6g}"
        )
    if not np.isfinite(validation["path_depth_pearson"]) or validation["path_depth_pearson"] < config.minimum_heldout_path_pearson:
        failures.append(
            f"path_depth_pearson={validation['path_depth_pearson']:.6g} is below "
            f"{config.minimum_heldout_path_pearson:.6g}"
        )
    validation["accepted"] = not failures
    validation["failures"] = failures
    if failures:
        rejected = metadata_path.with_name("channel_prior_fit_rejected.json")
        rejected.parent.mkdir(parents=True, exist_ok=True)
        rejected.write_text(
            json.dumps(
                {
                    "artifact_kind": "rejected_comsol_distribution_matched_channel_prior",
                    "real_formal_sample_labels_used": False,
                    "held_out_simulation_validation": validation,
                    "alpha_selection": alpha_selection,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raise RuntimeError(f"Held-out synthetic corpus validation failed: {failures}; see {rejected}")

    inversion_config = InversionConfig.from_json(config.inversion_config_path)
    if (
        inversion_config.helical_orders != config.helical_orders
        or inversion_config.inversion_grid_size != config.inversion_grid_size
        or not np.isclose(inversion_config.sigma_ray_mm, config.sigma_ray_mm)
        or not np.isclose(inversion_config.min_endpoint_distance_mm, config.min_endpoint_distance_mm)
        or not np.isclose(inversion_config.kernel_sigma_cutoff, config.kernel_sigma_cutoff)
    ):
        raise RuntimeError("Channel prior and physical inversion configs define different K geometry")
    fixed_inverter = PhysicalInverter(inversion_config, helical_order_weights=alpha)
    fixed_operator = np.asarray(fixed_inverter.operator.matrix, dtype=np.float32)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_path,
        helical_orders=np.asarray(config.helical_orders, dtype=np.int32),
        order_weights=alpha,
        frequency_hz=frequencies,
        relative_offset=np.arange(len(fixed_inverter.healthy.rx_indices), dtype=np.int32),
        amplitude_intercept=amplitude_model.intercept,
        amplitude_slope=amplitude_model.slope,
        amplitude_correlation=amplitude_model.correlation,
        amplitude_sigma_path_mm=amplitude_model.sigma_path_mm,
        phase_intercept=phase_model.intercept,
        phase_slope=phase_model.slope,
        phase_correlation=phase_model.correlation,
        phase_sigma_path_mm=phase_model.sigma_path_mm,
        phase_branch_fraction=phase_branch_fraction,
        ring_tx_indices=fixed_inverter.healthy.tx_indices,
        ring_rx_indices=fixed_inverter.healthy.rx_indices,
        fixed_operator_matrix=fixed_operator,
    )
    data_contract = {
        "healthy_npz_sha256": sha256_file(fixed_inverter.healthy_path),
        "healthy_metadata_sha256": sha256_file(fixed_inverter.healthy_metadata_path),
        "inversion_config": str(config.inversion_config_path),
        "inversion_config_sha256": canonical_json_sha256(
            json.loads(config.inversion_config_path.read_text(encoding="utf-8"))
        ),
        "channel_prior_config_path": str(args.config.resolve()),
        "channel_prior_config_sha256": config_sha256(config),
        "frequency_hz_sorted": frequencies.tolist(),
        "ring_tx_indices": [int(value) for value in fixed_inverter.healthy.tx_indices],
        "ring_rx_indices": [int(value) for value in fixed_inverter.healthy.rx_indices],
        "healthy_floor_quantile": config.healthy_floor_quantile,
        "channel_corpus_plan": str(plan_path),
        "channel_corpus_plan_sha256": sha256_file(plan_path),
        "corpus_root": str(corpus_root),
        "corpus_records": records,
        "source_fingerprints": source_fingerprints(),
        "kernel": {
            "helical_orders": list(inversion_config.helical_orders),
            "inversion_grid_size": inversion_config.inversion_grid_size,
            "sigma_ray_mm": inversion_config.sigma_ray_mm,
            "min_endpoint_distance_mm": inversion_config.min_endpoint_distance_mm,
            "kernel_sigma_cutoff": inversion_config.kernel_sigma_cutoff,
            "fixed_operator_shape": list(fixed_operator.shape),
            "fixed_operator_dtype": fixed_operator.dtype.str,
            "fixed_operator_sha256": sha256_array(fixed_operator),
        },
    }
    metadata = {
        "schema_version": 1,
        "artifact_kind": "comsol_distribution_matched_complex_rytov_channel_prior",
        "method": "independent_comsol_corpus_kfold_alpha_selected_complex_rytov_channel_calibration",
        "response_features": ["regularized_complex_rytov_log_amplitude", "regularized_complex_rytov_phase"],
        "forward_model": (
            "response[path,f] = intercept[offset(path),f] + slope[offset(path),f] "
            "* <K_fixed[path], depth_mm>, with K_fixed=sum_o alpha_o K_o"
        ),
        "model_npz": str(model_path),
        "model_npz_sha256": sha256_file(model_path),
        "real_formal_sample_labels_used": False,
        "synthetic_comsol_labels_used_for_calibration": True,
        "formal_data_used_for_inversion_only": True,
        "helical_orders": list(config.helical_orders),
        "order_weights": alpha.tolist(),
        "frequency_hz_sorted": frequencies.tolist(),
        "physical_scale": {
            "depth_limit_mm": config.defect_loss_limit_mm,
            "normalization_denominator_mm": config.normalization_denominator_mm,
            "minimum_remaining_wall_mm": config.minimum_remaining_wall_mm,
        },
        "channel_weighting": {
            "min_abs_correlation": config.channel_min_abs_correlation,
            "top_k": config.channel_top_k,
            "noise_floor_mm": config.channel_noise_floor_mm,
            "use_healthy_reliability": config.channel_use_healthy_reliability,
        },
        "phase_branch_diagnostics": {
            "gate_abs_phase_rad": float(np.pi / 2.0),
            "maximum_branch_fraction_for_use": 0.05,
            "median_branch_fraction": float(np.median(phase_branch_fraction)),
            "maximum_branch_fraction": float(np.max(phase_branch_fraction)),
            "excluded_channel_count": int(np.sum(invalid_phase)),
        },
        "alpha_selection": alpha_selection,
        "held_out_simulation_validation": validation,
        "data_contract": data_contract,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Channel prior metadata: {metadata_path}")
    print(f"Channel prior model: {model_path}")
    print("alpha=" + ", ".join(f"{value:.6f}" for value in alpha))
    print(
        f"held-out path RMSE={validation['path_depth_rmse_mm']:.6g} mm, "
        f"Pearson={validation['path_depth_pearson']:.6g}"
    )


if __name__ == "__main__":
    main()

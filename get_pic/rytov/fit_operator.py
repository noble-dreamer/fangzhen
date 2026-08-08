"""Fit a full-wave linearized Rytov Jacobian from weak COMSOL probes."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from .basis import build_basis_grid, expand_rotational_response
    from .common import (
        EPS,
        FrequencyResponse,
        assert_axes,
        complex_rytov,
        frequency_floor,
        healthy_rotational_relative_l2,
        linearized_rytov_factor,
        load_response,
        reliability_weights,
        sha256_file,
        sha256_json,
        write_json,
    )
    from .config import HERE, RytovConfig
    from .build_training_plan import validate_plan_for_config
    from .rytov_operator import (
        ARTIFACT_KIND,
        SCHEMA_VERSION,
        FullWaveRytovOperator,
        operator_array_hashes,
        save_operator_npz,
    )
except ImportError:  # Direct script execution from the rytov directory.
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from simple.get_pic.rytov.basis import (  # type: ignore
        build_basis_grid,
        expand_rotational_response,
    )
    from simple.get_pic.rytov.common import (  # type: ignore
        EPS,
        FrequencyResponse,
        assert_axes,
        complex_rytov,
        frequency_floor,
        healthy_rotational_relative_l2,
        linearized_rytov_factor,
        load_response,
        reliability_weights,
        sha256_file,
        sha256_json,
        write_json,
    )
    from simple.get_pic.rytov.config import HERE, RytovConfig  # type: ignore
    from simple.get_pic.rytov.build_training_plan import (  # type: ignore
        validate_plan_for_config,
    )
    from simple.get_pic.rytov.rytov_operator import (  # type: ignore
        ARTIFACT_KIND,
        SCHEMA_VERSION,
        FullWaveRytovOperator,
        operator_array_hashes,
        save_operator_npz,
    )


DEFAULT_CONFIG = HERE / "configs" / "dataset_a_fullwave_rytov.json"


@dataclass(frozen=True)
class ProbeRecord:
    basis_index: int
    z_index: int
    theta_index: int
    depth_mm: float
    sample_id: str
    response_path: Path


def _positions(values: np.ndarray, requested: Iterable[float | int], *, atol: float) -> list[int]:
    array = np.asarray(values)
    positions: list[int] = []
    for value in requested:
        matches = np.flatnonzero(
            np.isclose(array.astype(np.float64), float(value), rtol=0.0, atol=atol)
        )
        if matches.size != 1:
            raise RuntimeError(f"Expected one axis match for {value}, got {matches.size}")
        positions.append(int(matches[0]))
    return positions


def _subset_formal_healthy(
    response: FrequencyResponse,
    frequencies_hz: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if response.tx_indices.size != 16 or response.rx_indices.size != 16:
        raise RuntimeError("Formal healthy response must contain complete 16 by 16 rings")
    frequency_positions = _positions(response.frequencies_hz, frequencies_hz, atol=1.0e-6)
    h = response.h[:, :, frequency_positions]
    return (
        np.asarray(h, dtype=np.complex128),
        response.tx_indices.astype(np.int32),
        response.rx_indices.astype(np.int32),
        np.asarray(frequencies_hz, dtype=np.float64),
    )


def _sample_response_path(config: RytovConfig, item: dict[str, Any], sample_id: str) -> Path:
    for key in (
        "response_npz",
        "response_path",
        "frequency_response_npz",
        "output_npz",
    ):
        value = item.get(key)
        if isinstance(value, str) and value:
            path = Path(value)
            if not path.is_absolute():
                candidates = (
                    config.output_path / path,
                    config.training_path / path,
                    config.plan_path.parent / path,
                )
                for candidate in candidates:
                    if candidate.exists():
                        return candidate.resolve()
                return candidates[0].resolve()
            return path.resolve()
    return (
        config.training_path / "frequency_response" / f"{sample_id}_H_complex.npz"
    ).resolve()


def _probe_records(
    config: RytovConfig,
    plan: dict[str, Any],
) -> dict[int, list[ProbeRecord]]:
    samples = plan.get("samples")
    if not isinstance(samples, list) or not samples:
        raise RuntimeError("Training plan must contain a non-empty samples list")
    result: dict[int, list[ProbeRecord]] = {}
    seen: set[tuple[int, float]] = set()
    for item in samples:
        if not isinstance(item, dict):
            raise RuntimeError("Every training-plan sample must be an object")
        z_index = int(item["z_index"])
        theta_index = int(item["theta_index"])
        expected_index = z_index * config.theta_basis_count + theta_index
        basis_index = int(item.get("basis_index", expected_index))
        if basis_index != expected_index:
            raise RuntimeError(
                f"Basis index {basis_index} violates z-major/theta-minor order {expected_index}"
            )
        if not 0 <= z_index < config.z_basis_count:
            raise RuntimeError(f"Invalid z_index={z_index}")
        if not 0 <= theta_index < config.theta_basis_count:
            raise RuntimeError(f"Invalid theta_index={theta_index}")
        depth_mm = float(item["depth_mm"])
        if not any(
            np.isclose(depth_mm, expected, rtol=0.0, atol=1.0e-12)
            for expected in config.perturbation_depths_mm
        ):
            raise RuntimeError(f"Unconfigured perturbation depth {depth_mm:g} mm")
        key = (basis_index, depth_mm)
        if key in seen:
            raise RuntimeError(f"Duplicate probe for basis={basis_index}, depth={depth_mm:g}")
        seen.add(key)
        sample_id = str(item["sample_id"])
        record = ProbeRecord(
            basis_index=basis_index,
            z_index=z_index,
            theta_index=theta_index,
            depth_mm=depth_mm,
            sample_id=sample_id,
            response_path=_sample_response_path(config, item, sample_id),
        )
        result.setdefault(basis_index, []).append(record)

    expected_depths = np.sort(np.asarray(config.perturbation_depths_mm, dtype=np.float64))
    expected_basis_count = config.z_basis_count * config.theta_basis_count
    if set(result) != set(range(expected_basis_count)):
        missing = sorted(set(range(expected_basis_count)).difference(result))
        raise RuntimeError(f"Training plan lacks basis probes: {missing}")
    for basis_index, records in result.items():
        actual = np.sort(np.asarray([record.depth_mm for record in records]))
        if not np.allclose(actual, expected_depths, rtol=0.0, atol=1.0e-12):
            raise RuntimeError(f"Basis {basis_index} does not contain every configured depth")
        records.sort(key=lambda record: record.depth_mm)
    return result


def _weighted_relative_l2(
    actual: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
) -> float:
    residual = np.asarray(actual) - np.asarray(predicted)
    weight = np.broadcast_to(np.asarray(weights, dtype=np.float64), residual.shape)
    numerator = float(np.sum(weight * np.abs(residual) ** 2))
    denominator = float(np.sum(weight * np.abs(actual) ** 2))
    if denominator <= EPS:
        return float("inf") if numerator > EPS else 0.0
    return float(np.sqrt(numerator / denominator))


def _fit_probe_slopes(
    config: RytovConfig,
    records: dict[int, list[ProbeRecord]],
    training_healthy: FrequencyResponse,
    floor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    shape = training_healthy.h.shape
    coefficient_count = config.z_basis_count * config.theta_basis_count
    raw_slopes = np.empty((*shape, coefficient_count), dtype=np.complex128)
    raw_error = np.empty(coefficient_count, dtype=np.float64)
    rytov_error = np.empty(coefficient_count, dtype=np.float64)
    training_weights = reliability_weights(
        training_healthy.h,
        floor,
        config.minimum_data_weight,
    )
    factor = linearized_rytov_factor(training_healthy.h, floor)

    for basis_index in range(coefficient_count):
        probe_records = records[basis_index]
        depths = np.asarray([record.depth_mm for record in probe_records], dtype=np.float64)
        responses: list[np.ndarray] = []
        for record in probe_records:
            response = load_response(record.response_path)
            assert_axes(
                response,
                tx_indices=training_healthy.tx_indices,
                rx_indices=training_healthy.rx_indices,
                frequencies_hz=training_healthy.frequencies_hz,
            )
            responses.append(response.h)
        response_stack = np.stack(responses, axis=0)
        delta_stack = response_stack - training_healthy.h[None, ...]
        depth_energy = float(np.dot(depths, depths))
        slope = np.tensordot(depths, delta_stack, axes=(0, 0)) / depth_energy
        raw_slopes[..., basis_index] = slope

        raw_prediction = depths.reshape((-1,) + (1,) * len(shape)) * slope[None, ...]
        repeated_weights = np.broadcast_to(training_weights, delta_stack.shape)
        raw_error[basis_index] = _weighted_relative_l2(
            delta_stack,
            raw_prediction,
            repeated_weights,
        )
        actual_rytov = np.stack(
            [
                complex_rytov(
                    response_stack[index],
                    training_healthy.h,
                    floor,
                    training_healthy.frequencies_hz,
                )
                for index in range(depths.size)
            ],
            axis=0,
        )
        predicted_rytov = (
            depths.reshape((-1,) + (1,) * len(shape))
            * (factor * slope)[None, ...]
        )
        rytov_error[basis_index] = _weighted_relative_l2(
            actual_rytov,
            predicted_rytov,
            repeated_weights,
        )

    training_linearity = np.maximum(raw_error, rytov_error)
    diagnostics = {
        "slope_fit": "complex_dH_per_mm_forced_through_healthy_origin",
        "perturbation_depths_mm": [float(value) for value in config.perturbation_depths_mm],
        "independent_raw_slope_linearity_test": len(config.perturbation_depths_mm) > 1,
        "raw_dh_relative_l2_mean": float(np.mean(raw_error)),
        "raw_dh_relative_l2_max": float(np.max(raw_error)),
        "linearized_rytov_relative_l2_mean": float(np.mean(rytov_error)),
        "linearized_rytov_relative_l2_max": float(np.max(rytov_error)),
        "combined_relative_l2_mean": float(np.mean(training_linearity)),
        "combined_relative_l2_max": float(np.max(training_linearity)),
        "acceptance_threshold": float(config.maximum_training_linearity_error),
        "accepted": bool(
            np.all(np.isfinite(training_linearity))
            and np.max(training_linearity) <= config.maximum_training_linearity_error
        ),
    }
    return raw_slopes, training_linearity, raw_error, diagnostics


def _assemble_full_raw_slope(
    config: RytovConfig,
    training_slopes: np.ndarray,
    formal_h0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if config.jacobian_assembly_mode == "all_tx":
        if training_slopes.shape[:3] != formal_h0.shape:
            raise RuntimeError("All-TX training slopes do not match formal healthy axes")
        correction = np.ones(formal_h0.shape, dtype=np.float64)
        return training_slopes, correction, {
            "method": "direct_all_tx_comsol_finite_difference",
            "virtual_tx_used": False,
            "healthy_amplitude_correction_used": False,
        }

    if training_slopes.shape[0] != 1:
        raise RuntimeError("Rotational assembly requires exactly one training TX")
    rx_count, frequency_count = training_slopes.shape[1:3]
    reference = training_slopes[0].reshape(
        rx_count,
        frequency_count,
        config.z_basis_count,
        config.theta_basis_count,
    ).transpose(2, 3, 0, 1)
    expanded = expand_rotational_response(reference)
    correction = np.empty(formal_h0.shape, dtype=np.float64)
    for tx_slot in range(formal_h0.shape[0]):
        for rx_slot in range(formal_h0.shape[1]):
            relative_rx = (rx_slot - tx_slot) % formal_h0.shape[1]
            reference_magnitude = np.abs(formal_h0[0, relative_rx])
            target_magnitude = np.abs(formal_h0[tx_slot, rx_slot])
            correction[tx_slot, rx_slot] = target_magnitude / np.maximum(
                reference_magnitude,
                np.finfo(np.float64).tiny,
            )
    expanded *= correction[..., None]
    return expanded, correction, {
        "method": "tx1_rotation_with_formal_healthy_amplitude_correction",
        "virtual_tx_used": True,
        "rotation_index_formula": (
            "J[t,r,f,z,q]=Jref[(r-t)%16,f,z,(q-t)%16] before amplitude correction"
        ),
        "complex_conjugation_used_for_rotation": False,
        "healthy_amplitude_correction_used": True,
        "amplitude_correction_min": float(np.min(correction)),
        "amplitude_correction_max": float(np.max(correction)),
        "amplitude_correction_median": float(np.median(correction)),
    }


def build_operator(
    config: RytovConfig,
    *,
    config_path: Path | None = None,
    force: bool = False,
) -> FullWaveRytovOperator:
    config.validate()
    if config.operator_metadata_path.exists() and not force:
        return FullWaveRytovOperator.load(config.operator_metadata_path, config=config)
    if not config.plan_path.exists():
        raise FileNotFoundError(config.plan_path)
    if not config.training_healthy_path.exists():
        raise FileNotFoundError(config.training_healthy_path)
    if not config.healthy_metadata_path.exists():
        raise FileNotFoundError(config.healthy_metadata_path)

    plan = json.loads(config.plan_path.read_text(encoding="utf-8"))
    validate_plan_for_config(plan, config)
    if plan.get("real_formal_sample_labels_used", False):
        raise RuntimeError("Rytov training plan must not use formal labels")
    config_sha = sha256_json(config.to_dict())
    plan_config_sha = plan.get("config_sha256")
    if plan_config_sha is not None and plan_config_sha != config_sha:
        raise RuntimeError("Training plan was built from a different Rytov config")

    formal_response = load_response(config.healthy_path)
    formal_h0, tx_indices, rx_indices, frequencies_hz = _subset_formal_healthy(
        formal_response,
        config.frequencies_hz,
    )
    if not np.array_equal(tx_indices, np.arange(1, 17, dtype=np.int32)):
        raise RuntimeError("Formal TX IDs must be 1 through 16 in stored ring order")
    if not np.array_equal(rx_indices, np.arange(17, 33, dtype=np.int32)):
        raise RuntimeError("Formal RX IDs must be 17 through 32 in stored ring order")

    training_healthy = load_response(config.training_healthy_path)
    expected_training_tx = np.asarray(config.training_tx_indices, dtype=np.int32)
    assert_axes(
        training_healthy,
        tx_indices=expected_training_tx,
        rx_indices=rx_indices,
        frequencies_hz=frequencies_hz,
    )
    formal_tx_positions = _positions(tx_indices, expected_training_tx, atol=0.0)
    formal_training_subset = formal_h0[formal_tx_positions]
    closure_error = float(
        np.linalg.norm(training_healthy.h - formal_training_subset)
        / max(np.linalg.norm(formal_training_subset), EPS)
    )
    closure_accepted = closure_error <= config.maximum_healthy_closure_relative_l2
    if not closure_accepted:
        raise RuntimeError(
            "Training healthy response does not close against formal healthy: "
            f"relative_l2={closure_error:.6g}, threshold="
            f"{config.maximum_healthy_closure_relative_l2:.6g}"
        )

    symmetry_error = healthy_rotational_relative_l2(formal_h0)
    symmetry_accepted = symmetry_error <= config.maximum_healthy_symmetry_relative_l2
    if config.jacobian_assembly_mode == "rotational_tx1" and not symmetry_accepted:
        raise RuntimeError(
            "Formal healthy response is not rotationally symmetric enough for virtual TX: "
            f"relative_l2={symmetry_error:.6g}"
        )

    floor = frequency_floor(formal_h0, config.healthy_floor_quantile)
    records = _probe_records(config, plan)
    training_slopes, training_linearity, _raw_error, linearity_diagnostics = (
        _fit_probe_slopes(config, records, training_healthy, floor)
    )
    if not linearity_diagnostics["accepted"]:
        raise RuntimeError(
            "Weak COMSOL probes fail the configured linearity threshold: "
            f"max={linearity_diagnostics['combined_relative_l2_max']:.6g}"
        )

    raw_full, amplitude_correction, assembly_diagnostics = _assemble_full_raw_slope(
        config,
        training_slopes,
        formal_h0,
    )
    factor = linearized_rytov_factor(formal_h0, floor)
    jacobian = factor[..., None] * raw_full
    data_weights = reliability_weights(formal_h0, floor, config.minimum_data_weight)
    if not np.all(np.isfinite(jacobian.real)) or not np.all(np.isfinite(jacobian.imag)):
        raise RuntimeError("Fitted Rytov Jacobian contains non-finite values")

    grid = build_basis_grid(config)
    config.operator_npz_path.parent.mkdir(parents=True, exist_ok=True)
    save_operator_npz(
        config.operator_npz_path,
        jacobian=jacobian,
        data_weights=data_weights,
        healthy_h=formal_h0,
        frequency_floor=floor,
        tx_indices=tx_indices,
        rx_indices=rx_indices,
        frequencies_hz=frequencies_hz,
        theta_centers_deg=grid.theta_centers_deg,
        z_centers_mm=grid.z_centers_mm,
        basis_radius_theta_mm=grid.radius_theta_mm,
        basis_radius_z_mm=grid.radius_z_mm,
        training_linearity=training_linearity,
        raw_dh_slope=raw_full,
        amplitude_correction=amplitude_correction,
    )

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "purpose": (
            "physics-only full-wave linearized complex-Rytov wall-loss inversion"
        ),
        "real_formal_sample_labels_used": False,
        "model_npz": config.operator_npz_path.name,
        "model_npz_sha256": sha256_file(config.operator_npz_path),
        "config": config.to_dict(),
        "config_sha256": config_sha,
        "config_path": None if config_path is None else str(Path(config_path).resolve()),
        "jacobian_assembly_mode": config.jacobian_assembly_mode,
        "axes": {
            "jacobian": ["tx", "rx", "frequency", "coefficient"],
            "coefficient_order": "z_major_theta_minor",
            "tx_indices": tx_indices.tolist(),
            "rx_indices": rx_indices.tolist(),
            "frequencies_hz_ranked_order": frequencies_hz.tolist(),
            "theta_centers_deg": grid.theta_centers_deg.tolist(),
            "z_centers_mm": grid.z_centers_mm.tolist(),
        },
        "shape": {
            "jacobian": list(jacobian.shape),
            "healthy_h": list(formal_h0.shape),
            "coefficient_count": int(grid.coefficient_count),
            "complex_measurement_count": int(np.prod(formal_h0.shape)),
        },
        "basis": {
            "type": "periodic_theta_z_super_gaussian_wall_loss",
            "radius_theta_mm": grid.radius_theta_mm,
            "radius_z_mm": grid.radius_z_mm,
            "window_power": 3,
            "coefficient_units": "mm peak wall loss",
        },
        "linearization": {
            "raw_response_model": "dH = (dH/dd_mm) * coefficient_mm",
            "rytov_model": "psi approximately J * coefficient_mm",
            "regularization": (
                "psi=principal_log(1+(H-H0)*conj(H0)/(|H0|^2+floor_f^2))"
            ),
            "jacobian_units": "1/mm",
            "blind_frequency_phase_unwrap_used": False,
            **assembly_diagnostics,
        },
        "rytov_stability": {
            "healthy_floor_quantile": config.healthy_floor_quantile,
            "minimum_data_weight": config.minimum_data_weight,
            "minimum_ratio_magnitude": config.minimum_ratio_magnitude,
            "phase_branch_margin_rad": config.phase_branch_margin_rad,
        },
        "physical_scale": {
            "coefficient_units": "mm",
            "depth_limit_mm": config.depth_limit_mm,
            "normalization_denominator_mm": config.normalization_denominator_mm,
            "millimetre_source": "known COMSOL weak-perturbation depth; no label calibration",
        },
        "healthy_closure": {
            "relative_l2": closure_error,
            "threshold": config.maximum_healthy_closure_relative_l2,
            "accepted": closure_accepted,
        },
        "healthy_rotational_symmetry": {
            "relative_l2": symmetry_error,
            "threshold": config.maximum_healthy_symmetry_relative_l2,
            "required_for_acceptance": config.jacobian_assembly_mode == "rotational_tx1",
            "accepted": symmetry_accepted,
        },
        "training_linearity": linearity_diagnostics,
        "data_contract": {
            "formal_healthy_npz": str(config.healthy_path.resolve()),
            "formal_healthy_npz_sha256": sha256_file(config.healthy_path),
            "formal_healthy_metadata": str(config.healthy_metadata_path.resolve()),
            "formal_healthy_metadata_sha256": sha256_file(config.healthy_metadata_path),
            "training_healthy_npz": str(config.training_healthy_path.resolve()),
            "training_healthy_npz_sha256": sha256_file(config.training_healthy_path),
            "training_plan": str(config.plan_path.resolve()),
            "training_plan_sha256": sha256_file(config.plan_path),
        },
        "array_sha256": operator_array_hashes(
            jacobian=jacobian,
            data_weights=data_weights,
            healthy_h=formal_h0,
            frequency_floor=floor,
            tx_indices=tx_indices,
            rx_indices=rx_indices,
            frequencies_hz=frequencies_hz,
            theta_centers_deg=grid.theta_centers_deg,
            z_centers_mm=grid.z_centers_mm,
            training_linearity=training_linearity,
        ),
    }
    write_json(config.operator_metadata_path, metadata)
    return FullWaveRytovOperator.load(config.operator_metadata_path, config=config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit the COMSOL full-wave linearized complex-Rytov operator."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.config.resolve()
    config = RytovConfig.from_json(config_path)
    operator = build_operator(
        config,
        config_path=config_path,
        force=bool(args.force),
    )
    print(
        json.dumps(
            {
                "operator_metadata": str(operator.metadata_path),
                "operator_npz": str(operator.model_path),
                "assembly_mode": operator.assembly_mode,
                "jacobian_shape": list(operator.jacobian.shape),
                "training_linearity_max": float(np.max(operator.training_linearity)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

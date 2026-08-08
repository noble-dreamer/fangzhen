"""Bounded complex-data inversion for the full-wave Rytov operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear

try:
    from .basis import (
        MID_RADIUS_MM,
        BasisGrid,
        coefficient_map,
        difference_operator,
    )
    from .common import EPS, FrequencyResponse
    from .config import RytovConfig
    from .rytov_operator import FullWaveRytovOperator
except ImportError:  # Direct execution of sibling entry points.
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from simple.get_pic.rytov.basis import (  # type: ignore
        MID_RADIUS_MM,
        BasisGrid,
        coefficient_map,
        difference_operator,
    )
    from simple.get_pic.rytov.common import EPS, FrequencyResponse  # type: ignore
    from simple.get_pic.rytov.config import RytovConfig  # type: ignore
    from simple.get_pic.rytov.rytov_operator import (  # type: ignore
        FullWaveRytovOperator,
    )


@dataclass(frozen=True)
class RytovInversionResult:
    coefficients_mm: np.ndarray
    image_mm: np.ndarray
    image_norm: np.ndarray
    theta_deg: np.ndarray
    z_mm: np.ndarray
    observation: np.ndarray
    predicted_rytov: np.ndarray
    data_weights: np.ndarray
    diagnostics: dict[str, Any]


def _real_stacked_system(
    jacobian: np.ndarray,
    observation: np.ndarray,
    data_weights: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    coefficient_count = jacobian.shape[-1]
    matrix = np.asarray(jacobian, dtype=np.complex128).reshape(-1, coefficient_count)
    target = np.asarray(observation, dtype=np.complex128).reshape(-1)
    weights = np.asarray(data_weights, dtype=np.float64).reshape(-1)
    valid = (
        np.isfinite(target.real)
        & np.isfinite(target.imag)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if not np.any(valid):
        raise RuntimeError("No finite positive-weight Rytov measurements")
    matrix = matrix[valid]
    target = target[valid]
    weights = weights[valid]
    root_weight = np.sqrt(weights)
    real_matrix = np.vstack(
        (
            root_weight[:, None] * matrix.real,
            root_weight[:, None] * matrix.imag,
        )
    )
    real_target = np.concatenate(
        (root_weight * target.real, root_weight * target.imag)
    )
    return sparse.csr_matrix(real_matrix), real_target, valid


def _objective_terms(
    operator: FullWaveRytovOperator,
    coefficients_mm: np.ndarray,
    observation: np.ndarray,
    data_weights: np.ndarray,
    differences: sparse.csr_matrix,
    config: RytovConfig,
) -> dict[str, float]:
    predicted = operator.predict(coefficients_mm)
    residual = predicted - observation
    weights = np.asarray(data_weights, dtype=np.float64)
    data_term = float(np.sum(weights * np.abs(residual) ** 2))
    ridge_term = float(config.ridge_weight * np.dot(coefficients_mm, coefficients_mm))
    gradient = differences @ coefficients_mm
    tv_term = float(
        config.tv_weight
        * np.sum(np.sqrt(gradient * gradient + config.tv_epsilon_mm**2))
    )
    return {
        "weighted_data_term": data_term,
        "ridge_term": ridge_term,
        "smoothed_tv_term": tv_term,
        "total": data_term + ridge_term + tv_term,
    }


def invert_observation(
    *,
    config: RytovConfig,
    operator: FullWaveRytovOperator,
    observation: np.ndarray,
    data_weights: np.ndarray,
    observation_diagnostics: dict[str, Any] | None = None,
) -> RytovInversionResult:
    """Invert a complex Rytov observation into nonnegative wall loss in mm."""

    operator.assert_compatible(config)
    observed = np.asarray(observation, dtype=np.complex128)
    weights = np.asarray(data_weights, dtype=np.float64)
    if observed.shape != operator.measurement_shape:
        raise ValueError(
            f"Observation shape {observed.shape} differs from {operator.measurement_shape}"
        )
    if weights.shape != operator.measurement_shape:
        raise ValueError(f"Data-weight shape {weights.shape} differs from operator axes")
    if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("Data weights must be finite and nonnegative")

    data_matrix, real_target, valid_rows = _real_stacked_system(
        operator.jacobian,
        observed,
        weights,
    )
    theta_spacing_mm = 2.0 * np.pi * MID_RADIUS_MM / operator.theta_centers_deg.size
    z_spacing_mm = float(np.mean(np.diff(operator.z_centers_mm)))
    differences = difference_operator(
        operator.z_centers_mm.size,
        operator.theta_centers_deg.size,
        theta_spacing_mm=theta_spacing_mm,
        z_spacing_mm=z_spacing_mm,
    )
    coefficient_count = operator.coefficient_count
    ridge = sparse.eye(coefficient_count, dtype=np.float64, format="csr")
    coefficients = np.zeros(coefficient_count, dtype=np.float64)
    trace: list[dict[str, Any]] = []

    for outer_iteration in range(1, config.irls_iterations + 1):
        gradient = differences @ coefficients
        irls_weight = 1.0 / np.sqrt(
            gradient * gradient + config.tv_epsilon_mm**2
        )
        blocks: list[sparse.spmatrix] = [data_matrix]
        targets = [real_target]
        if config.ridge_weight > 0.0:
            blocks.append(np.sqrt(config.ridge_weight) * ridge)
            targets.append(np.zeros(coefficient_count, dtype=np.float64))
        if config.tv_weight > 0.0:
            tv_scale = np.sqrt(config.tv_weight * irls_weight)
            blocks.append(differences.multiply(tv_scale[:, None]))
            targets.append(np.zeros(differences.shape[0], dtype=np.float64))
        augmented = sparse.vstack(blocks, format="csr")
        augmented_target = np.concatenate(targets)
        solve = lsq_linear(
            augmented,
            augmented_target,
            bounds=(0.0, config.depth_limit_mm),
            method="trf",
            tol=config.lsq_tolerance,
            lsq_solver="lsmr",
            lsmr_tol="auto",
            max_iter=config.lsq_max_iterations,
            verbose=0,
        )
        if not np.all(np.isfinite(solve.x)):
            raise RuntimeError("Bounded Rytov least-squares returned non-finite coefficients")
        coefficients = np.clip(
            np.asarray(solve.x, dtype=np.float64),
            0.0,
            config.depth_limit_mm,
        )
        terms = _objective_terms(
            operator,
            coefficients,
            observed,
            weights,
            differences,
            config,
        )
        trace.append(
            {
                "outer_iteration": outer_iteration,
                "least_squares_status": int(solve.status),
                "least_squares_success": bool(solve.success),
                "least_squares_iterations": int(solve.nit),
                "optimality": float(solve.optimality),
                **terms,
            }
        )

    predicted = operator.predict(coefficients)
    residual = predicted - observed
    nonzero_weight = weights > 0.0
    weighted_denominator = float(np.sum(weights))
    weighted_complex_rmse = float(
        np.sqrt(np.sum(weights * np.abs(residual) ** 2) / max(weighted_denominator, EPS))
    )
    unweighted_complex_rmse = float(
        np.sqrt(np.mean(np.abs(residual[nonzero_weight]) ** 2))
    )

    grid = BasisGrid(
        theta_centers_deg=operator.theta_centers_deg,
        z_centers_mm=operator.z_centers_mm,
        radius_theta_mm=operator.basis_radius_theta_mm,
        radius_z_mm=operator.basis_radius_z_mm,
    )
    image_mm, theta_deg, z_mm = coefficient_map(
        coefficients,
        grid,
        image_size=config.image_size,
        depth_limit_mm=config.depth_limit_mm,
    )
    image_norm = np.clip(
        image_mm / config.normalization_denominator_mm,
        0.0,
        1.0,
    ).astype(np.float32)
    diagnostics: dict[str, Any] = {
        "solver": "bounded_nonnegative_complex_real_stack_irls_periodic_tv",
        "units": "mm wall loss",
        "jacobian_assembly_mode": operator.assembly_mode,
        "coefficient_count": coefficient_count,
        "complex_measurement_count": int(observed.size),
        "effective_complex_measurement_count": int(np.count_nonzero(valid_rows)),
        "real_stacked_equation_count": int(data_matrix.shape[0]),
        "theta_difference_periodic": True,
        "theta_spacing_mm": float(theta_spacing_mm),
        "z_spacing_mm": float(z_spacing_mm),
        "ridge_weight": float(config.ridge_weight),
        "tv_weight": float(config.tv_weight),
        "tv_epsilon_mm": float(config.tv_epsilon_mm),
        "irls_iterations": int(config.irls_iterations),
        "weighted_complex_rmse": weighted_complex_rmse,
        "unweighted_complex_rmse": unweighted_complex_rmse,
        "weighted_real_rmse": float(
            np.sqrt(
                np.sum(weights * residual.real**2) / max(weighted_denominator, EPS)
            )
        ),
        "weighted_imag_rmse": float(
            np.sqrt(
                np.sum(weights * residual.imag**2) / max(weighted_denominator, EPS)
            )
        ),
        "coefficient_min_mm": float(np.min(coefficients)),
        "coefficient_max_mm": float(np.max(coefficients)),
        "coefficient_mean_mm": float(np.mean(coefficients)),
        "coefficient_at_upper_bound_fraction": float(
            np.mean(coefficients >= config.depth_limit_mm * (1.0 - 1.0e-6))
        ),
        "prediction_min_mm": float(np.min(image_mm)),
        "prediction_max_mm": float(np.max(image_mm)),
        "prediction_mean_mm": float(np.mean(image_mm)),
        "normalization_denominator_mm": float(config.normalization_denominator_mm),
        "trace": trace,
        "observation": dict(observation_diagnostics or {}),
    }
    return RytovInversionResult(
        coefficients_mm=coefficients.astype(np.float32),
        image_mm=image_mm,
        image_norm=image_norm,
        theta_deg=theta_deg,
        z_mm=z_mm,
        observation=observed.astype(np.complex64),
        predicted_rytov=predicted.astype(np.complex64),
        data_weights=weights.astype(np.float32),
        diagnostics=diagnostics,
    )


def invert_response(
    *,
    config: RytovConfig,
    operator: FullWaveRytovOperator,
    damaged_response: FrequencyResponse,
) -> RytovInversionResult:
    """Create the regularized complex-Rytov observation and invert it."""

    observation, weights, observation_diagnostics = operator.observation(
        damaged_response
    )
    return invert_observation(
        config=config,
        operator=operator,
        observation=observation,
        data_weights=weights,
        observation_diagnostics=observation_diagnostics,
    )

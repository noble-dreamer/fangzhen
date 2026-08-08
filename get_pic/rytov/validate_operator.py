"""Validate a frozen COMSOL-derived full-wave Rytov operator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simple.get_pic.rytov.common import load_response, sha256_file, sha256_json, write_json  # noqa: E402
from simple.get_pic.rytov.config import RytovConfig, project_path  # noqa: E402
from simple.get_pic.rytov.rytov_operator import FullWaveRytovOperator  # noqa: E402


DEFAULT_CONFIG = HERE / "configs" / "dataset_a_fullwave_rytov.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a full-wave Rytov operator artifact.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--operator", type=Path, default=None, help="Operator metadata JSON.")
    parser.add_argument(
        "--sample-id",
        type=int,
        default=None,
        help="Formal response used for a label-free observation smoke check.",
    )
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def _safe_rytov_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(HERE.resolve())
    except ValueError as error:
        raise ValueError(f"Artifact/report must stay under {HERE}: {resolved}") from error
    return resolved


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sample_name(sample_id: int) -> str:
    return f"dataset_a_frequency_sample_{sample_id:04d}"


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = RytovConfig.from_json(config_path)
    metadata_path = _safe_rytov_path(
        args.operator.resolve() if args.operator else config.operator_metadata_path
    )
    operator = FullWaveRytovOperator.load(metadata_path, config=config)
    operator.assert_compatible(config)
    _safe_rytov_path(operator.model_path)

    expected_shape = (
        16,
        16,
        len(config.frequencies_hz),
        config.theta_basis_count * config.z_basis_count,
    )
    jacobian = np.asarray(operator.jacobian, dtype=np.complex128)
    weights = np.asarray(operator.data_weights, dtype=np.float64)
    healthy = np.asarray(operator.healthy_h, dtype=np.complex128)
    floor = np.asarray(operator.frequency_floor, dtype=np.float64)
    if jacobian.shape != expected_shape:
        raise RuntimeError(f"Jacobian shape {jacobian.shape} != {expected_shape}")
    if weights.shape != expected_shape[:3] or healthy.shape != expected_shape[:3]:
        raise RuntimeError("Operator weights/healthy response do not match measurement axes")
    if floor.shape != (expected_shape[2],):
        raise RuntimeError(f"Frequency-floor shape {floor.shape} is invalid")
    for name, value in (
        ("jacobian.real", jacobian.real),
        ("jacobian.imag", jacobian.imag),
        ("data_weights", weights),
        ("healthy.real", healthy.real),
        ("healthy.imag", healthy.imag),
        ("frequency_floor", floor),
    ):
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"Operator field {name} contains non-finite values")
    if np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise RuntimeError("Operator data weights must contain positive finite entries")
    if np.any(floor <= 0.0):
        raise RuntimeError("Frequency floor must be strictly positive")

    expected_tx = np.arange(1, 17, dtype=np.int32)
    expected_rx = np.arange(17, 33, dtype=np.int32)
    if not np.array_equal(operator.tx_indices, expected_tx):
        raise RuntimeError(f"Unexpected operator TX axis: {operator.tx_indices}")
    if not np.array_equal(operator.rx_indices, expected_rx):
        raise RuntimeError(f"Unexpected operator RX axis: {operator.rx_indices}")
    if not np.allclose(
        operator.frequencies_hz,
        np.asarray(config.frequencies_hz),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise RuntimeError("Operator frequency axis differs from the configured ranked top-15 axis")
    if operator.assembly_mode != config.jacobian_assembly_mode:
        raise RuntimeError("Operator assembly mode differs from config")
    if operator.model_sha256 != sha256_file(operator.model_path):
        raise RuntimeError("Operator model SHA does not match the NPZ artifact")

    flattened = jacobian.reshape(-1, jacobian.shape[-1])
    row_scale = np.sqrt(weights.reshape(-1))[:, None]
    real_system = np.vstack((row_scale * flattened.real, row_scale * flattened.imag))
    column_norms = np.linalg.norm(real_system, axis=0)
    if np.any(column_norms <= 1.0e-14):
        missing = np.flatnonzero(column_norms <= 1.0e-14).tolist()
        raise RuntimeError(f"Jacobian has zero weighted columns: {missing}")
    singular_values = np.linalg.svd(real_system, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        raise RuntimeError("Jacobian has no measurable singular directions")
    tolerance = singular_values[0] * max(real_system.shape) * np.finfo(np.float64).eps
    numerical_rank = int(np.sum(singular_values > tolerance))
    power = singular_values**2
    effective_rank = float(np.sum(power) ** 2 / max(float(np.sum(power**2)), 1.0e-30))
    condition_number = (
        float(singular_values[0] / singular_values[numerical_rank - 1])
        if numerical_rank > 0
        else float("inf")
    )

    sample_id = args.sample_id
    if sample_id is None:
        sample_id = config.validation_sample_ids[0] if config.validation_sample_ids else config.formal_sample_ids[0]
    name = _sample_name(sample_id)
    damaged = load_response(
        config.dataset_path / "frequency_response" / f"{name}_H_complex.npz"
    )
    observation, observation_weights, observation_diagnostics = operator.observation(damaged)
    observation = np.asarray(observation, dtype=np.complex128)
    observation_weights = np.asarray(observation_weights, dtype=np.float64)
    if observation.shape != expected_shape[:3] or observation_weights.shape != observation.shape:
        raise RuntimeError("Operator observation API returned incompatible arrays")
    if not np.all(np.isfinite(observation.real)) or not np.all(np.isfinite(observation.imag)):
        raise RuntimeError("Formal observation smoke check produced non-finite values")
    if np.any(observation_weights < 0.0) or not np.any(observation_weights > 0.0):
        raise RuntimeError("Formal observation smoke check retained no measurements")

    report = {
        "schema_version": 1,
        "artifact_kind": "fullwave_rytov_operator_validation",
        "config_sha256": sha256_json(config.to_dict()),
        "operator_metadata": str(operator.metadata_path),
        "operator_model": str(operator.model_path),
        "operator_model_sha256": operator.model_sha256,
        "assembly_mode": operator.assembly_mode,
        "jacobian_shape": list(jacobian.shape),
        "measurement_count_complex": int(np.prod(jacobian.shape[:3])),
        "coefficient_count": int(jacobian.shape[-1]),
        "positive_weight_fraction": float(np.mean(weights > 0.0)),
        "column_norm_min": float(np.min(column_norms)),
        "column_norm_max": float(np.max(column_norms)),
        "numerical_rank": numerical_rank,
        "effective_rank": effective_rank,
        "condition_number_nonzero": condition_number,
        "largest_singular_value": float(singular_values[0]),
        "smallest_nonzero_singular_value": float(singular_values[numerical_rank - 1]),
        "training_linearity": _json_safe(operator.training_linearity),
        "formal_response_smoke_sample": name,
        "formal_response_positive_weight_fraction": float(np.mean(observation_weights > 0.0)),
        "formal_response_diagnostics": _json_safe(observation_diagnostics),
        "formal_labels_read": False,
    }
    if args.report is not None:
        report_path = _safe_rytov_path(project_path(args.report))
        write_json(report_path, _json_safe(report))
        print(f"Validation report: {report_path}")
    print(
        "Operator valid: "
        f"shape={jacobian.shape}, rank={numerical_rank}/{jacobian.shape[-1]}, "
        f"effective_rank={effective_rank:.2f}, positive_weights={np.mean(weights > 0.0):.3f}"
    )
    print(
        f"Formal response smoke: {name}, retained={np.mean(observation_weights > 0.0):.3f}; "
        "formal labels were not read"
    )


if __name__ == "__main__":
    main()

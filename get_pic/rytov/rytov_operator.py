"""Strict artifact contract for the full-wave linearized Rytov operator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .common import (
        EPS,
        FrequencyResponse,
        assert_axes,
        complex_rytov,
        regularized_ratio,
        rytov_validity_weights,
        sha256_array,
        sha256_file,
        sha256_json,
    )
    from .config import RytovConfig
except ImportError:  # Direct execution of sibling entry points.
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from simple.get_pic.rytov.common import (  # type: ignore
        EPS,
        FrequencyResponse,
        assert_axes,
        complex_rytov,
        regularized_ratio,
        rytov_validity_weights,
        sha256_array,
        sha256_file,
        sha256_json,
    )
    from simple.get_pic.rytov.config import RytovConfig  # type: ignore


ARTIFACT_KIND = "comsol_fullwave_linearized_complex_rytov_operator"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FullWaveRytovOperator:
    """A COMSOL finite-difference Jacobian linearized about the healthy pipe."""

    metadata_path: Path
    model_path: Path
    model_sha256: str
    assembly_mode: str
    jacobian: np.ndarray
    data_weights: np.ndarray
    healthy_h: np.ndarray
    frequency_floor: np.ndarray
    tx_indices: np.ndarray
    rx_indices: np.ndarray
    frequencies_hz: np.ndarray
    theta_centers_deg: np.ndarray
    z_centers_mm: np.ndarray
    basis_radius_theta_mm: float
    basis_radius_z_mm: float
    depth_limit_mm: float
    normalization_denominator_mm: float
    training_linearity: np.ndarray
    metadata: dict[str, Any]

    @property
    def coefficient_count(self) -> int:
        return int(self.theta_centers_deg.size * self.z_centers_mm.size)

    @property
    def measurement_shape(self) -> tuple[int, int, int]:
        return (
            int(self.tx_indices.size),
            int(self.rx_indices.size),
            int(self.frequencies_hz.size),
        )

    @classmethod
    def load(
        cls,
        metadata_path: Path,
        *,
        config: RytovConfig | None = None,
    ) -> "FullWaveRytovOperator":
        metadata_path = Path(metadata_path).resolve()
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("Unsupported full-wave Rytov operator schema")
        if metadata.get("artifact_kind") != ARTIFACT_KIND:
            raise RuntimeError(f"Expected {ARTIFACT_KIND}")
        if metadata.get("real_formal_sample_labels_used") is not False:
            raise RuntimeError("A physics-only Rytov operator cannot use formal labels")

        model_value = metadata.get("model_npz")
        if not isinstance(model_value, str) or not model_value:
            raise RuntimeError("Operator metadata does not name its NPZ model")
        model_path = Path(model_value)
        if not model_path.is_absolute():
            model_path = metadata_path.parent / model_path
        model_path = model_path.resolve()
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        model_sha = sha256_file(model_path)
        if model_sha != metadata.get("model_npz_sha256"):
            raise RuntimeError("Rytov operator NPZ SHA-256 mismatch")

        with np.load(model_path, allow_pickle=False) as arrays:
            required = {
                "jacobian_real",
                "jacobian_imag",
                "data_weights",
                "healthy_real",
                "healthy_imag",
                "frequency_floor",
                "tx_indices",
                "rx_indices",
                "frequencies_hz",
                "theta_centers_deg",
                "z_centers_mm",
                "basis_radius_theta_mm",
                "basis_radius_z_mm",
                "training_linearity",
            }
            missing = sorted(required.difference(arrays.files))
            if missing:
                raise RuntimeError(f"Operator NPZ is missing arrays: {missing}")
            operator = cls(
                metadata_path=metadata_path,
                model_path=model_path,
                model_sha256=model_sha,
                assembly_mode=str(metadata["jacobian_assembly_mode"]),
                jacobian=(
                    np.asarray(arrays["jacobian_real"], dtype=np.float64)
                    + 1j * np.asarray(arrays["jacobian_imag"], dtype=np.float64)
                ),
                data_weights=np.asarray(arrays["data_weights"], dtype=np.float64),
                healthy_h=(
                    np.asarray(arrays["healthy_real"], dtype=np.float64)
                    + 1j * np.asarray(arrays["healthy_imag"], dtype=np.float64)
                ),
                frequency_floor=np.asarray(arrays["frequency_floor"], dtype=np.float64),
                tx_indices=np.asarray(arrays["tx_indices"], dtype=np.int32),
                rx_indices=np.asarray(arrays["rx_indices"], dtype=np.int32),
                frequencies_hz=np.asarray(arrays["frequencies_hz"], dtype=np.float64),
                theta_centers_deg=np.asarray(arrays["theta_centers_deg"], dtype=np.float64),
                z_centers_mm=np.asarray(arrays["z_centers_mm"], dtype=np.float64),
                basis_radius_theta_mm=float(np.asarray(arrays["basis_radius_theta_mm"]).item()),
                basis_radius_z_mm=float(np.asarray(arrays["basis_radius_z_mm"]).item()),
                depth_limit_mm=float(metadata["physical_scale"]["depth_limit_mm"]),
                normalization_denominator_mm=float(
                    metadata["physical_scale"]["normalization_denominator_mm"]
                ),
                training_linearity=np.asarray(arrays["training_linearity"], dtype=np.float64),
                metadata=metadata,
            )
        operator._validate_arrays()
        operator._validate_array_hashes()
        operator._validate_external_contract()
        if config is not None:
            operator.assert_compatible(config)
        return operator

    def _validate_arrays(self) -> None:
        expected_measurements = self.measurement_shape
        expected_jacobian = (*expected_measurements, self.coefficient_count)
        if self.jacobian.shape != expected_jacobian:
            raise RuntimeError(
                f"Jacobian shape {self.jacobian.shape} does not match {expected_jacobian}"
            )
        if self.healthy_h.shape != expected_measurements:
            raise RuntimeError("Stored healthy response shape differs from operator axes")
        if self.data_weights.shape != expected_measurements:
            raise RuntimeError("Stored data weights shape differs from operator axes")
        if self.frequency_floor.shape != (self.frequencies_hz.size,):
            raise RuntimeError("Frequency floor shape differs from frequency axis")
        if self.training_linearity.shape != (self.coefficient_count,):
            raise RuntimeError("Training-linearity vector differs from coefficient axis")
        for name, values in (
            ("jacobian real", self.jacobian.real),
            ("jacobian imag", self.jacobian.imag),
            ("healthy real", self.healthy_h.real),
            ("healthy imag", self.healthy_h.imag),
            ("data weights", self.data_weights),
            ("frequency floor", self.frequency_floor),
            ("training linearity", self.training_linearity),
        ):
            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"Operator {name} contains non-finite values")
        if np.any(self.data_weights < 0.0) or np.any(self.data_weights > 1.0):
            raise RuntimeError("Operator data weights must be in [0, 1]")
        if np.any(self.frequency_floor <= 0.0):
            raise RuntimeError("Operator frequency floors must be positive")
        if np.unique(self.tx_indices).size != self.tx_indices.size:
            raise RuntimeError("TX axis is not unique")
        if np.unique(self.rx_indices).size != self.rx_indices.size:
            raise RuntimeError("RX axis is not unique")
        if np.unique(self.frequencies_hz).size != self.frequencies_hz.size:
            raise RuntimeError("Frequency axis is not unique")
        if self.assembly_mode not in {"rotational_tx1", "all_tx"}:
            raise RuntimeError("Unknown Jacobian assembly mode")
        if self.depth_limit_mm <= 0.0 or self.normalization_denominator_mm <= 0.0:
            raise RuntimeError("Invalid physical scale in operator")

    def _validate_array_hashes(self) -> None:
        expected = self.metadata.get("array_sha256", {})
        actual = {
            "jacobian": sha256_array(self.jacobian.astype(np.complex64)),
            "data_weights": sha256_array(self.data_weights.astype(np.float32)),
            "healthy_h": sha256_array(self.healthy_h.astype(np.complex128)),
            "frequency_floor": sha256_array(self.frequency_floor.astype(np.float64)),
            "tx_indices": sha256_array(self.tx_indices.astype(np.int32)),
            "rx_indices": sha256_array(self.rx_indices.astype(np.int32)),
            "frequencies_hz": sha256_array(self.frequencies_hz.astype(np.float64)),
            "theta_centers_deg": sha256_array(self.theta_centers_deg.astype(np.float64)),
            "z_centers_mm": sha256_array(self.z_centers_mm.astype(np.float64)),
            "training_linearity": sha256_array(self.training_linearity.astype(np.float32)),
        }
        if expected != actual:
            raise RuntimeError("Operator array SHA-256 contract mismatch")

    def _validate_external_contract(self) -> None:
        contract = self.metadata.get("data_contract", {})
        for path_key, sha_key in (
            ("formal_healthy_npz", "formal_healthy_npz_sha256"),
            ("formal_healthy_metadata", "formal_healthy_metadata_sha256"),
            ("training_healthy_npz", "training_healthy_npz_sha256"),
            ("training_plan", "training_plan_sha256"),
        ):
            value = contract.get(path_key)
            expected_sha = contract.get(sha_key)
            if not isinstance(value, str) or not isinstance(expected_sha, str):
                raise RuntimeError(f"Operator data contract lacks {path_key}")
            path = Path(value)
            if not path.is_absolute():
                path = self.metadata_path.parent / path
            if not path.exists() or sha256_file(path) != expected_sha:
                raise RuntimeError(f"Operator external artifact mismatch: {path_key}")

    def assert_compatible(self, config: RytovConfig) -> None:
        config.validate()
        expected_config_sha = sha256_json(config.to_dict())
        if expected_config_sha != self.metadata.get("config_sha256"):
            raise RuntimeError("Rytov config differs from the fitted operator config")
        if config.jacobian_assembly_mode != self.assembly_mode:
            raise RuntimeError("Rytov assembly mode differs from fitted operator")
        expected_shape = (
            16,
            16,
            len(config.frequencies_hz),
            config.z_basis_count * config.theta_basis_count,
        )
        if self.jacobian.shape != expected_shape:
            raise RuntimeError(
                f"Configured operator shape should be {expected_shape}, got {self.jacobian.shape}"
            )
        if not np.allclose(
            self.frequencies_hz,
            np.asarray(config.frequencies_hz, dtype=np.float64),
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise RuntimeError("Configured frequency order differs from fitted operator")
        if not np.isclose(self.depth_limit_mm, config.depth_limit_mm, rtol=0.0, atol=1.0e-12):
            raise RuntimeError("Configured depth limit differs from fitted operator")
        if not np.isclose(
            self.normalization_denominator_mm,
            config.normalization_denominator_mm,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError("Configured normalization scale differs from fitted operator")

    def observation(
        self,
        damaged_response: FrequencyResponse,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        assert_axes(
            damaged_response,
            tx_indices=self.tx_indices,
            rx_indices=self.rx_indices,
            frequencies_hz=self.frequencies_hz,
        )
        settings = self.metadata["rytov_stability"]
        observation = complex_rytov(
            damaged_response.h,
            self.healthy_h,
            self.frequency_floor,
            self.frequencies_hz,
        )
        validity, diagnostics = rytov_validity_weights(
            damaged_response.h,
            self.healthy_h,
            self.frequency_floor,
            minimum_reliability=float(settings["minimum_data_weight"]),
            minimum_ratio_magnitude=float(settings["minimum_ratio_magnitude"]),
            phase_branch_margin_rad=float(settings["phase_branch_margin_rad"]),
        )
        # The stored weights bind the inversion to the exact healthy reference used
        # when the Jacobian was fitted. Validity only removes target-specific branch
        # and near-zero-ratio failures.
        weights = self.data_weights * (validity > 0.0)
        ratio = regularized_ratio(
            damaged_response.h - self.healthy_h,
            self.healthy_h,
            self.frequency_floor,
        )
        diagnostics.update(
            {
                "sample_id": damaged_response.sample_id,
                "response_feature": "principal_regularized_complex_rytov_log",
                "frequency_order_preserved": True,
                "blind_frequency_phase_unwrap_used": False,
                "effective_measurement_count": int(np.count_nonzero(weights)),
                "measurement_count": int(weights.size),
                "weight_min_nonzero": float(
                    np.min(weights[weights > 0.0]) if np.any(weights > 0.0) else 0.0
                ),
                "weight_max": float(np.max(weights)),
                "ratio_magnitude_median": float(np.median(np.abs(ratio))),
            }
        )
        if not np.any(weights > 0.0):
            raise RuntimeError(f"No stable Rytov measurements remain for {damaged_response.sample_id}")
        return observation, weights.astype(np.float64), diagnostics

    def predict(self, coefficients_mm: np.ndarray) -> np.ndarray:
        coefficients = np.asarray(coefficients_mm, dtype=np.float64).reshape(-1)
        if coefficients.shape != (self.coefficient_count,):
            raise ValueError(
                f"Expected {self.coefficient_count} coefficients, got {coefficients.shape}"
            )
        return np.einsum("trfc,c->trf", self.jacobian, coefficients, optimize=True)


def save_operator_npz(
    path: Path,
    *,
    jacobian: np.ndarray,
    data_weights: np.ndarray,
    healthy_h: np.ndarray,
    frequency_floor: np.ndarray,
    tx_indices: np.ndarray,
    rx_indices: np.ndarray,
    frequencies_hz: np.ndarray,
    theta_centers_deg: np.ndarray,
    z_centers_mm: np.ndarray,
    basis_radius_theta_mm: float,
    basis_radius_z_mm: float,
    training_linearity: np.ndarray,
    raw_dh_slope: np.ndarray,
    amplitude_correction: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(
            file,
            jacobian_real=np.asarray(jacobian.real, dtype=np.float32),
            jacobian_imag=np.asarray(jacobian.imag, dtype=np.float32),
            data_weights=np.asarray(data_weights, dtype=np.float32),
            healthy_real=np.asarray(healthy_h.real, dtype=np.float64),
            healthy_imag=np.asarray(healthy_h.imag, dtype=np.float64),
            frequency_floor=np.asarray(frequency_floor, dtype=np.float64),
            tx_indices=np.asarray(tx_indices, dtype=np.int32),
            rx_indices=np.asarray(rx_indices, dtype=np.int32),
            frequencies_hz=np.asarray(frequencies_hz, dtype=np.float64),
            theta_centers_deg=np.asarray(theta_centers_deg, dtype=np.float64),
            z_centers_mm=np.asarray(z_centers_mm, dtype=np.float64),
            basis_radius_theta_mm=np.asarray(basis_radius_theta_mm, dtype=np.float64),
            basis_radius_z_mm=np.asarray(basis_radius_z_mm, dtype=np.float64),
            training_linearity=np.asarray(training_linearity, dtype=np.float32),
            raw_dh_slope_real=np.asarray(raw_dh_slope.real, dtype=np.float32),
            raw_dh_slope_imag=np.asarray(raw_dh_slope.imag, dtype=np.float32),
            amplitude_correction=np.asarray(amplitude_correction, dtype=np.float32),
        )
    temporary.replace(path)


def operator_array_hashes(
    *,
    jacobian: np.ndarray,
    data_weights: np.ndarray,
    healthy_h: np.ndarray,
    frequency_floor: np.ndarray,
    tx_indices: np.ndarray,
    rx_indices: np.ndarray,
    frequencies_hz: np.ndarray,
    theta_centers_deg: np.ndarray,
    z_centers_mm: np.ndarray,
    training_linearity: np.ndarray,
) -> dict[str, str]:
    return {
        "jacobian": sha256_array(np.asarray(jacobian, dtype=np.complex64)),
        "data_weights": sha256_array(np.asarray(data_weights, dtype=np.float32)),
        "healthy_h": sha256_array(np.asarray(healthy_h, dtype=np.complex128)),
        "frequency_floor": sha256_array(np.asarray(frequency_floor, dtype=np.float64)),
        "tx_indices": sha256_array(np.asarray(tx_indices, dtype=np.int32)),
        "rx_indices": sha256_array(np.asarray(rx_indices, dtype=np.int32)),
        "frequencies_hz": sha256_array(np.asarray(frequencies_hz, dtype=np.float64)),
        "theta_centers_deg": sha256_array(np.asarray(theta_centers_deg, dtype=np.float64)),
        "z_centers_mm": sha256_array(np.asarray(z_centers_mm, dtype=np.float64)),
        "training_linearity": sha256_array(np.asarray(training_linearity, dtype=np.float32)),
    }

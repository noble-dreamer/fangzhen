"""Simulation-only complex-Rytov channel calibration with a fixed helical kernel."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .channel_common import ChannelPriorConfig, config_sha256
    from .common import (
        canonical_json_sha256,
        relative_ring_offsets,
        sha256_array,
        sha256_file,
    )
except ImportError:  # Direct script execution from simulation_prior/.
    from channel_common import ChannelPriorConfig, config_sha256  # type: ignore
    from common import (  # type: ignore
        canonical_json_sha256,
        relative_ring_offsets,
        sha256_array,
        sha256_file,
    )
from simple.get_pic import coarse_map_common as cm
from simple.get_pic.physical_inversion.inversion import (
    EPS,
    ChannelModel,
    PathDepthObservation,
    PhysicalInverter,
)


@dataclass(frozen=True)
class SimulationChannelPrior:
    """A fixed alpha K with simulated complex-response channel calibration."""

    metadata_path: Path
    model_path: Path
    model_sha256: str
    helical_orders: np.ndarray
    order_weights: np.ndarray
    frequencies_hz: np.ndarray
    relative_offsets: np.ndarray
    amplitude: ChannelModel
    phase: ChannelModel
    ring_tx_indices: np.ndarray
    ring_rx_indices: np.ndarray
    depth_limit_mm: float
    normalization_denominator_mm: float
    minimum_remaining_wall_mm: float
    min_abs_correlation: float
    top_k: int
    noise_floor_mm: float
    use_healthy_reliability: bool
    fixed_operator_matrix: np.ndarray
    held_out_validation: dict[str, Any]
    data_contract: dict[str, Any]

    @classmethod
    def load(cls, metadata_path: Path) -> "SimulationChannelPrior":
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("artifact_kind") != "comsol_distribution_matched_complex_rytov_channel_prior":
            raise RuntimeError("Expected a distribution-matched COMSOL channel prior")
        if metadata.get("real_formal_sample_labels_used") is not False:
            raise RuntimeError("Channel prior must not use formal sample labels")
        model_path = Path(metadata["model_npz"])
        if not model_path.is_absolute():
            model_path = metadata_path.parent / model_path
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        model_sha = sha256_file(model_path)
        if model_sha != metadata.get("model_npz_sha256"):
            raise RuntimeError("Channel prior model SHA-256 does not match metadata")
        with np.load(model_path, allow_pickle=False) as arrays:
            model = cls(
                metadata_path=metadata_path,
                model_path=model_path,
                model_sha256=model_sha,
                helical_orders=np.asarray(arrays["helical_orders"], dtype=np.int32),
                order_weights=np.asarray(arrays["order_weights"], dtype=np.float64),
                frequencies_hz=np.asarray(arrays["frequency_hz"], dtype=np.float64),
                relative_offsets=np.asarray(arrays["relative_offset"], dtype=np.int32),
                amplitude=ChannelModel(
                    intercept=np.asarray(arrays["amplitude_intercept"], dtype=np.float64),
                    slope=np.asarray(arrays["amplitude_slope"], dtype=np.float64),
                    correlation=np.asarray(arrays["amplitude_correlation"], dtype=np.float64),
                    sigma_path_mm=np.asarray(arrays["amplitude_sigma_path_mm"], dtype=np.float64),
                ),
                phase=ChannelModel(
                    intercept=np.asarray(arrays["phase_intercept"], dtype=np.float64),
                    slope=np.asarray(arrays["phase_slope"], dtype=np.float64),
                    correlation=np.asarray(arrays["phase_correlation"], dtype=np.float64),
                    sigma_path_mm=np.asarray(arrays["phase_sigma_path_mm"], dtype=np.float64),
                ),
                ring_tx_indices=np.asarray(arrays["ring_tx_indices"], dtype=np.int32),
                ring_rx_indices=np.asarray(arrays["ring_rx_indices"], dtype=np.int32),
                depth_limit_mm=float(metadata["physical_scale"]["depth_limit_mm"]),
                normalization_denominator_mm=float(
                    metadata["physical_scale"]["normalization_denominator_mm"]
                ),
                minimum_remaining_wall_mm=float(
                    metadata["physical_scale"]["minimum_remaining_wall_mm"]
                ),
                min_abs_correlation=float(metadata["channel_weighting"]["min_abs_correlation"]),
                top_k=int(metadata["channel_weighting"]["top_k"]),
                noise_floor_mm=float(metadata["channel_weighting"]["noise_floor_mm"]),
                use_healthy_reliability=bool(
                    metadata["channel_weighting"]["use_healthy_reliability"]
                ),
                fixed_operator_matrix=np.asarray(arrays["fixed_operator_matrix"], dtype=np.float32),
                held_out_validation=dict(metadata["held_out_simulation_validation"]),
                data_contract=dict(metadata["data_contract"]),
            )
        model.validate_arrays()
        return model

    def validate_arrays(self) -> None:
        order_count = self.helical_orders.size
        if order_count == 0 or self.order_weights.shape != (order_count,):
            raise RuntimeError("Invalid helical order weights")
        if not np.all(np.isfinite(self.order_weights)) or np.any(self.order_weights < 0.0):
            raise RuntimeError("Helical order weights must be finite and nonnegative")
        if not np.isclose(np.sum(self.order_weights), 1.0, rtol=0.0, atol=1e-7):
            raise RuntimeError("Helical order weights must sum to one")
        if self.frequencies_hz.ndim != 1 or self.frequencies_hz.size == 0:
            raise RuntimeError("Invalid frequency axis")
        if np.any(~np.isfinite(self.frequencies_hz)) or np.unique(self.frequencies_hz).size != self.frequencies_hz.size:
            raise RuntimeError("Frequency axis must be finite and unique")
        if not np.array_equal(self.relative_offsets, np.arange(self.relative_offsets.size)):
            raise RuntimeError("Relative offsets must be dense from zero")
        expected = (self.relative_offsets.size, self.frequencies_hz.size)
        for name, model in (("amplitude", self.amplitude), ("phase", self.phase)):
            for item_name, values in (
                ("intercept", model.intercept),
                ("slope", model.slope),
                ("correlation", model.correlation),
                ("sigma_path_mm", model.sigma_path_mm),
            ):
                if values.shape != expected:
                    raise RuntimeError(f"{name} {item_name} has shape {values.shape}, expected {expected}")
                if not np.all(np.isfinite(values) | np.isinf(values)):
                    raise RuntimeError(f"{name} {item_name} has non-finite values")
            if not np.all(np.isfinite(model.intercept)) or not np.all(np.isfinite(model.slope)):
                raise RuntimeError(f"{name} channel affine parameters must be finite")
        if self.ring_tx_indices.size != self.ring_rx_indices.size or self.ring_tx_indices.size != self.relative_offsets.size:
            raise RuntimeError("Ring axes do not match the relative-offset channel model")
        if self.top_k <= 0 or self.noise_floor_mm < 0.0:
            raise RuntimeError("Invalid channel weighting parameters")
        if self.depth_limit_mm <= 0.0 or self.normalization_denominator_mm <= 0.0:
            raise RuntimeError("Invalid physical scale")

    def assert_compatible(self, inverter: PhysicalInverter) -> None:
        config = ChannelPriorConfig.from_json(
            Path(self.data_contract["channel_prior_config_path"])
        )
        expected = {
            "healthy_npz_sha256": sha256_file(inverter.healthy_path),
            "healthy_metadata_sha256": sha256_file(inverter.healthy_metadata_path),
            "inversion_config": str(self.data_contract["inversion_config"]),
            "channel_prior_config_sha256": config_sha256(config),
            "frequency_hz_sorted": self.frequencies_hz.tolist(),
            "ring_tx_indices": [int(value) for value in inverter.healthy.tx_indices],
            "ring_rx_indices": [int(value) for value in inverter.healthy.rx_indices],
            "kernel": {
                "helical_orders": list(inverter.config.helical_orders),
                "inversion_grid_size": inverter.config.inversion_grid_size,
                "sigma_ray_mm": inverter.config.sigma_ray_mm,
                "min_endpoint_distance_mm": inverter.config.min_endpoint_distance_mm,
                "kernel_sigma_cutoff": inverter.config.kernel_sigma_cutoff,
                "fixed_operator_shape": list(inverter.operator.matrix.shape),
                "fixed_operator_dtype": inverter.operator.matrix.dtype.str,
                "fixed_operator_sha256": sha256_array(inverter.operator.matrix),
            },
        }
        actual = {
            key: self.data_contract.get(key)
            for key in expected
        }
        if actual != expected:
            raise RuntimeError("Channel prior data contract does not match this inversion geometry")
        if not np.array_equal(self.helical_orders, np.asarray(inverter.config.helical_orders, dtype=np.int32)):
            raise RuntimeError("Prior and inversion helical orders differ")
        if not np.array_equal(self.ring_tx_indices, inverter.healthy.tx_indices) or not np.array_equal(
            self.ring_rx_indices, inverter.healthy.rx_indices
        ):
            raise RuntimeError("Prior ring axes differ from formal healthy response")
        if not np.allclose(
            self.fixed_operator_matrix, inverter.operator.matrix, rtol=1e-6, atol=1e-8
        ):
            raise RuntimeError("Fixed K matrix differs from the inversion operator")
        if not self.held_out_validation.get("accepted", False):
            raise RuntimeError("The simulation channel prior did not pass held-out validation")

    @staticmethod
    def _top_k_weights(weights: np.ndarray, top_k: int) -> np.ndarray:
        if top_k >= weights.shape[-1]:
            return weights
        keep = np.argpartition(weights, -top_k, axis=-1)[:, -top_k:]
        mask = np.zeros_like(weights, dtype=bool)
        np.put_along_axis(mask, keep, True, axis=-1)
        return np.where(mask, weights, 0.0)

    def _weights(self, model: ChannelModel, reliability: np.ndarray) -> np.ndarray:
        quality = np.abs(model.correlation)
        weights = quality * quality / (model.sigma_path_mm * model.sigma_path_mm + self.noise_floor_mm**2)
        weights = np.where(quality >= self.min_abs_correlation, weights, 0.0)
        weights = np.where(np.isfinite(weights), weights, 0.0)
        weights = self._top_k_weights(weights, min(self.top_k, weights.shape[-1]))
        weights /= np.maximum(np.max(weights, axis=-1, keepdims=True), EPS)
        if self.use_healthy_reliability:
            weights *= reliability
        return weights

    @staticmethod
    def _invert(response: np.ndarray, model: ChannelModel) -> np.ndarray:
        denominator = np.where(np.abs(model.slope) > 1e-12, model.slope, np.nan)
        estimate = (np.asarray(response, dtype=np.float64) - model.intercept) / denominator
        return np.nan_to_num(estimate, nan=0.0, posinf=0.0, neginf=0.0)

    def estimate_path_depth(
        self,
        inverter: PhysicalInverter,
        damaged: cm.FrequencyResponse,
    ) -> tuple[PathDepthObservation, dict[str, Any]]:
        self.assert_compatible(inverter)
        cm.assert_compatible(inverter.healthy, damaged)
        positions = []
        available = np.asarray(inverter.healthy.frequencies_hz, dtype=np.float64)
        for frequency in self.frequencies_hz:
            matches = np.flatnonzero(np.isclose(available, frequency, rtol=0.0, atol=1e-6))
            if matches.size != 1:
                raise RuntimeError(f"Frequency {frequency:g} Hz is absent from the formal response")
            positions.append(int(matches[0]))
        h0 = inverter.healthy.h[..., positions]
        hd = damaged.h[..., positions]
        floor = np.quantile(
            np.abs(h0).reshape(-1, self.frequencies_hz.size),
            self.data_contract.get("healthy_floor_quantile", 0.10),
            axis=0,
        )
        z = 1.0 + (hd - h0) * np.conj(h0) / (np.abs(h0) ** 2 + floor[None, None, :] ** 2)
        amplitude = np.log(np.abs(z) + EPS).reshape(-1, self.frequencies_hz.size)
        phase = np.angle(z).reshape(-1, self.frequencies_hz.size)
        reliability = (
            np.abs(h0) ** 2 / (np.abs(h0) ** 2 + floor[None, None, :] ** 2 + EPS)
        ).reshape(-1, self.frequencies_hz.size)
        if not np.all(np.isfinite(amplitude)) or not np.all(np.isfinite(phase)):
            raise RuntimeError(f"Non-finite complex Rytov feature for {damaged.sample_id}")
        if amplitude.shape[1] != self.frequencies_hz.size:
            raise RuntimeError("Prior frequency axis does not match complex Rytov channels")
        offsets = relative_ring_offsets(
            inverter.operator.tx_indices,
            inverter.operator.rx_indices,
            self.ring_tx_indices,
            self.ring_rx_indices,
        )
        amplitude_model = ChannelModel(
            intercept=self.amplitude.intercept[offsets],
            slope=self.amplitude.slope[offsets],
            correlation=self.amplitude.correlation[offsets],
            sigma_path_mm=self.amplitude.sigma_path_mm[offsets],
        )
        phase_model = ChannelModel(
            intercept=self.phase.intercept[offsets],
            slope=self.phase.slope[offsets],
            correlation=self.phase.correlation[offsets],
            sigma_path_mm=self.phase.sigma_path_mm[offsets],
        )
        amplitude_estimate = self._invert(amplitude, amplitude_model)
        phase_estimate = self._invert(phase, phase_model)
        amplitude_weights = self._weights(amplitude_model, reliability)
        phase_weights = self._weights(phase_model, reliability)
        estimates = np.concatenate([amplitude_estimate, phase_estimate], axis=-1)
        weights = np.concatenate([amplitude_weights, phase_weights], axis=-1)
        weights = self._top_k_weights(weights, min(self.top_k, weights.shape[-1]))
        estimates = np.clip(estimates, -self.depth_limit_mm, 2.0 * self.depth_limit_mm)
        numerator = np.sum(estimates * weights, axis=-1)
        denominator = np.sum(weights, axis=-1)
        values = np.clip(numerator / np.maximum(denominator, EPS), 0.0, self.depth_limit_mm)
        path_weights = denominator / max(float(np.max(denominator)), EPS)
        observation = PathDepthObservation(
            values_mm=values.astype(np.float32),
            weights=path_weights.astype(np.float32),
            frequencies_hz=self.frequencies_hz.astype(np.float64),
            retained_amplitude_fraction=float(np.mean(amplitude_weights > 0.0)),
            retained_phase_fraction=float(np.mean(phase_weights > 0.0)),
        )
        diagnostics = {
            "response_feature": "regularized_complex_rytov_log_amplitude_and_phase",
            "real_formal_sample_labels_used": False,
            "frequency_count": int(self.frequencies_hz.size),
            "retained_amplitude_channel_fraction": observation.retained_amplitude_fraction,
            "retained_phase_channel_fraction": observation.retained_phase_fraction,
            "path_depth_min_mm": float(np.min(values)),
            "path_depth_max_mm": float(np.max(values)),
            "path_depth_mean_mm": float(np.mean(values)),
            "prior_model_sha256": self.model_sha256,
        }
        return observation, diagnostics

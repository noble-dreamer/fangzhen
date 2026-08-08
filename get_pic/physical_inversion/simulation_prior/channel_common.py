"""Contracts shared by the v2 distribution-matched COMSOL channel prior."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .common import (
        MODEL_SOURCE_RELATIVE_PATHS,
        canonical_json_sha256,
        project_path,
        sha256_file,
        sha256_source_file,
    )
except ImportError:  # Direct script execution from simulation_prior/.
    from common import (  # type: ignore
        MODEL_SOURCE_RELATIVE_PATHS,
        canonical_json_sha256,
        project_path,
        sha256_file,
        sha256_source_file,
    )


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]


CHANNEL_SOURCE_RELATIVE_PATHS = (
    *MODEL_SOURCE_RELATIVE_PATHS,
    "simple/simple_defect_common.py",
    "simple/get_pic/physical_inversion/simulation_prior/channel_common.py",
    "simple/get_pic/physical_inversion/simulation_prior/channel_prior.py",
    "simple/get_pic/physical_inversion/simulation_prior/build_channel_corpus_plan.py",
    "simple/get_pic/physical_inversion/simulation_prior/solve_channel_corpus.py",
    "simple/get_pic/physical_inversion/simulation_prior/fit_channel_prior.py",
)

PHYSICAL_SOURCE_RELATIVE_PATHS = (
    "simple/f_domain/frequency_domain_common.py",
    "simple/simple_shell_common.py",
    "simple/streaming_export_common.py",
    "simple/defect_label_common.py",
    "simple/get_pic/coarse_map_common.py",
    "simple/simple_defect_common.py",
)


@dataclass(frozen=True)
class ChannelPriorConfig:
    dataset_root: str = "simple/f_domain/output_dataset/streaming_dataset_a_frequency_shell"
    output_root: str = "simple/get_pic/physical_inversion/simulation_prior/output_matched_corpus"
    inversion_config: str = "simple/get_pic/physical_inversion/configs/dataset_a_rytov_tv.json"
    healthy_sample_id: str = "dataset_a_frequency_healthy"
    frequencies_hz: tuple[float, ...] = ()
    simulation_tx_indices: tuple[int, ...] = (1,)
    helical_orders: tuple[int, ...] = (-1, 0, 1)
    inversion_grid_size: int = 64
    sigma_ray_mm: float = 25.0
    min_endpoint_distance_mm: float = 30.0
    kernel_sigma_cutoff: float = 3.0
    healthy_floor_quantile: float = 0.10
    corpus_sample_count: int = 48
    corpus_fit_count: int = 36
    corpus_seed0: int = 930000
    corpus_sample_id_prefix: str = "simprior_c"
    alpha_grid_step: float = 0.05
    calibration_folds: int = 6
    channel_min_abs_correlation: float = 0.20
    channel_top_k: int = 15
    channel_noise_floor_mm: float = 0.01
    channel_use_healthy_reliability: bool = False
    minimum_remaining_wall_mm: float = 1.0
    defect_loss_limit_mm: float = 5.0
    normalization_denominator_mm: float = 9.0
    maximum_heldout_path_rmse_mm: float = 0.20
    minimum_heldout_path_pearson: float = 0.50

    @classmethod
    def from_json(cls, path: Path) -> "ChannelPriorConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(raw).difference(allowed))
        if unknown:
            raise ValueError(f"Unknown channel-prior config keys: {unknown}")
        tuple_int = ("simulation_tx_indices", "helical_orders")
        tuple_float = ("frequencies_hz",)
        for key in tuple_int:
            if key in raw:
                raw[key] = tuple(int(value) for value in raw[key])
        for key in tuple_float:
            if key in raw:
                raw[key] = tuple(float(value) for value in raw[key])
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if not self.inversion_config_path.exists():
            raise FileNotFoundError(self.inversion_config_path)
        if not self.frequencies_hz or len(set(self.frequencies_hz)) != len(self.frequencies_hz):
            raise ValueError("frequencies_hz must be a non-empty unique sequence")
        if len(self.simulation_tx_indices) != 1:
            raise ValueError("The matched corpus currently requires exactly one simulated TX")
        if not self.helical_orders:
            raise ValueError("helical_orders cannot be empty")
        if self.inversion_grid_size < 8:
            raise ValueError("inversion_grid_size must be at least 8")
        if self.corpus_sample_count < 8 or not 4 <= self.corpus_fit_count < self.corpus_sample_count:
            raise ValueError("corpus_fit_count must leave at least four held-out synthetic samples")
        if not self.corpus_sample_id_prefix or not self.corpus_sample_id_prefix.isascii():
            raise ValueError("corpus_sample_id_prefix must be non-empty ASCII")
        inverse_step = 1.0 / self.alpha_grid_step if self.alpha_grid_step > 0.0 else 0.0
        if not np.isclose(inverse_step, round(inverse_step), rtol=0.0, atol=1e-9):
            raise ValueError("alpha_grid_step must divide one exactly")
        if self.calibration_folds < 2 or self.calibration_folds > self.corpus_fit_count:
            raise ValueError("calibration_folds must be in [2, corpus_fit_count]")
        if not 0.0 <= self.healthy_floor_quantile <= 1.0:
            raise ValueError("healthy_floor_quantile must be in [0, 1]")
        if not 0.0 <= self.channel_min_abs_correlation <= 1.0:
            raise ValueError("channel_min_abs_correlation must be in [0, 1]")
        if self.channel_top_k <= 0 or self.channel_noise_floor_mm < 0.0:
            raise ValueError("invalid channel weighting parameters")
        if self.minimum_remaining_wall_mm <= 0.0 or self.defect_loss_limit_mm <= 0.0:
            raise ValueError("physical depth limits must be positive")
        if self.normalization_denominator_mm <= 0.0:
            raise ValueError("normalization_denominator_mm must be positive")
        if self.maximum_heldout_path_rmse_mm <= 0.0 or not 0.0 <= self.minimum_heldout_path_pearson <= 1.0:
            raise ValueError("invalid held-out acceptance thresholds")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key, value in list(result.items()):
            if isinstance(value, tuple):
                result[key] = list(value)
        return result

    @property
    def dataset_path(self) -> Path:
        return project_path(self.dataset_root)

    @property
    def output_path(self) -> Path:
        return project_path(self.output_root)

    @property
    def inversion_config_path(self) -> Path:
        return project_path(self.inversion_config)

    @property
    def healthy_path(self) -> Path:
        return self.dataset_path / "frequency_response" / f"{self.healthy_sample_id}_H_complex.npz"

    @property
    def healthy_metadata_path(self) -> Path:
        return self.dataset_path / "metadata" / f"{self.healthy_sample_id}.json"

    @property
    def plan_path(self) -> Path:
        return self.output_path / "channel_corpus_plan.json"

    @property
    def corpus_output_path(self) -> Path:
        return self.output_path / "channel_corpus"

    @property
    def prior_metadata_path(self) -> Path:
        return self.output_path / "channel_prior.json"

    @property
    def prior_model_path(self) -> Path:
        return self.output_path / "channel_prior.npz"


def source_fingerprints() -> dict[str, str]:
    return {
        relative: sha256_source_file(PROJECT_ROOT / relative)
        for relative in CHANNEL_SOURCE_RELATIVE_PATHS
    }


def physical_source_fingerprints() -> dict[str, str]:
    """Fingerprint only files that alter a COMSOL response or synthetic geometry."""

    return {
        relative: sha256_source_file(PROJECT_ROOT / relative)
        for relative in PHYSICAL_SOURCE_RELATIVE_PATHS
    }


def physical_sources_match(stored: dict[str, str] | None) -> bool:
    if not isinstance(stored, dict):
        return False
    expected = physical_source_fingerprints()
    return all(stored.get(key) == value for key, value in expected.items())


def config_sha256(config: ChannelPriorConfig) -> str:
    return canonical_json_sha256(config.to_dict())


def plan_sha256(path: Path) -> str:
    return sha256_file(path)


def model_sha256(path: Path) -> str:
    return sha256_file(path)


def canonical_sample_metadata(sample: dict[str, Any], plan_path: Path, config: ChannelPriorConfig) -> dict[str, Any]:
    return {
        **dict(sample["sample_metadata"]),
        "simulation_prior_channel_plan_sha256": plan_sha256(plan_path),
        "simulation_prior_channel_config_sha256": config_sha256(config),
        "formal_healthy_npz_sha256": sha256_file(config.healthy_path),
        "channel_prior_source_sha256": source_fingerprints(),
    }


def response_is_complete(
    path: Path,
    metadata_path: Path,
    *,
    tx_indices: tuple[int, ...],
    rx_indices: tuple[int, ...],
    frequencies_hz: tuple[float, ...],
    expected_sample: dict[str, Any],
) -> bool:
    if not path.exists() or not metadata_path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            existing_tx = tuple(int(value) for value in data["tx_indices"].tolist())
            existing_rx = tuple(int(value) for value in data["rx_indices"].tolist())
            existing_frequency = tuple(float(value) for value in data["frequencies_hz"].tolist())
            completed = np.asarray(data["completed_mask"], dtype=bool)
            h_real = np.asarray(data["H_real"])
            h_imag = np.asarray(data["H_imag"])
            expected_shape = (len(tx_indices), len(rx_indices), len(frequencies_hz))
            axes_complete = (
                existing_tx == tx_indices
                and existing_rx == rx_indices
                and len(existing_frequency) == len(frequencies_hz)
                and np.allclose(existing_frequency, frequencies_hz, rtol=0.0, atol=1e-6)
                and completed.shape == (len(tx_indices), len(frequencies_hz))
                and bool(np.all(completed))
                and h_real.shape == expected_shape
                and h_imag.shape == expected_shape
                and bool(np.all(np.isfinite(h_real)))
                and bool(np.all(np.isfinite(h_imag)))
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        actual_sample = dict(metadata.get("sample", {}))
        expected_sample = dict(expected_sample)
        # Pipeline-source hashes do not alter a solved COMSOL response. The physical
        # source subset is checked independently by the plan validation.
        actual_sample.pop("channel_prior_source_sha256", None)
        expected_sample.pop("channel_prior_source_sha256", None)
        return axes_complete and actual_sample == expected_sample
    except Exception:
        return False

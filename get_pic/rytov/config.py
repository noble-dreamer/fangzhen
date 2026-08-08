"""Configuration contract for the standalone full-wave Rytov pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class RytovConfig:
    dataset_root: str = "simple/f_domain/output_dataset/streaming_dataset_a_frequency_shell"
    output_root: str = "simple/get_pic/rytov/output"
    healthy_sample_id: str = "dataset_a_frequency_healthy"
    training_healthy_sample_id: str = "rytov_training_healthy"
    frequencies_hz: tuple[float, ...] = ()
    jacobian_assembly_mode: str = "rotational_tx1"
    training_tx_indices: tuple[int, ...] = (1,)
    theta_basis_count: int = 16
    z_basis_count: int = 8
    z_min_mm: float = 260.0
    z_max_mm: float = 740.0
    basis_radius_theta_mm: float = 45.0
    basis_radius_z_mm: float = 50.0
    perturbation_depths_mm: tuple[float, ...] = (0.25,)
    image_size: int = 256
    depth_limit_mm: float = 5.0
    normalization_denominator_mm: float = 9.0
    healthy_floor_quantile: float = 0.10
    minimum_data_weight: float = 0.05
    minimum_ratio_magnitude: float = 0.10
    phase_branch_margin_rad: float = 0.15
    maximum_healthy_symmetry_relative_l2: float = 0.07
    maximum_healthy_closure_relative_l2: float = 5.0e-4
    ridge_weight: float = 2.0e-4
    tv_weight: float = 3.0e-3
    tv_epsilon_mm: float = 0.05
    irls_iterations: int = 5
    lsq_tolerance: float = 1.0e-6
    lsq_max_iterations: int = 300
    formal_sample_ids: tuple[int, ...] = ()
    validation_sample_ids: tuple[int, ...] = ()
    defect_threshold_mm: float = 0.10
    maximum_training_linearity_error: float = 0.35

    @classmethod
    def from_json(cls, path: Path) -> "RytovConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(raw).difference(allowed))
        if unknown:
            raise ValueError(f"Unknown Rytov config keys: {unknown}")
        for key in ("frequencies_hz", "perturbation_depths_mm"):
            if key in raw:
                raw[key] = tuple(float(value) for value in raw[key])
        for key in ("training_tx_indices", "formal_sample_ids", "validation_sample_ids"):
            if key in raw:
                raw[key] = tuple(int(value) for value in raw[key])
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if not self.frequencies_hz or len(set(self.frequencies_hz)) != len(self.frequencies_hz):
            raise ValueError("frequencies_hz must be a non-empty unique sequence")
        if self.jacobian_assembly_mode not in {"rotational_tx1", "all_tx"}:
            raise ValueError("jacobian_assembly_mode must be rotational_tx1 or all_tx")
        expected_tx = (1,) if self.jacobian_assembly_mode == "rotational_tx1" else tuple(range(1, 17))
        if self.training_tx_indices != expected_tx:
            raise ValueError(
                f"training_tx_indices must be {expected_tx} for {self.jacobian_assembly_mode}"
            )
        if self.theta_basis_count != 16:
            raise ValueError("Rotational expansion requires theta_basis_count=16")
        if self.z_basis_count < 2 or not 0.0 <= self.z_min_mm < self.z_max_mm <= 1000.0:
            raise ValueError("Invalid axial basis grid")
        if self.basis_radius_theta_mm <= 0.0 or self.basis_radius_z_mm <= 0.0:
            raise ValueError("Basis radii must be positive")
        if not self.perturbation_depths_mm:
            raise ValueError("At least one perturbation depth is required")
        depths = tuple(float(value) for value in self.perturbation_depths_mm)
        if any(value <= 0.0 or value > 0.5 for value in depths):
            raise ValueError("Linearization depths must be in (0, 0.5] mm")
        if len(set(depths)) != len(depths):
            raise ValueError("perturbation_depths_mm must be unique")
        if self.image_size < 16:
            raise ValueError("image_size must be at least 16")
        if self.depth_limit_mm <= 0.0 or self.normalization_denominator_mm <= 0.0:
            raise ValueError("Physical depth scales must be positive")
        if not 0.0 <= self.healthy_floor_quantile <= 1.0:
            raise ValueError("healthy_floor_quantile must be in [0, 1]")
        if not 0.0 < self.minimum_data_weight <= 1.0:
            raise ValueError("minimum_data_weight must be in (0, 1]")
        if self.minimum_ratio_magnitude <= 0.0:
            raise ValueError("minimum_ratio_magnitude must be positive")
        if not 0.0 < self.phase_branch_margin_rad < 1.0:
            raise ValueError("phase_branch_margin_rad must be in (0, 1) rad")
        if self.maximum_healthy_symmetry_relative_l2 <= 0.0:
            raise ValueError("maximum_healthy_symmetry_relative_l2 must be positive")
        if self.maximum_healthy_closure_relative_l2 <= 0.0:
            raise ValueError("maximum_healthy_closure_relative_l2 must be positive")
        if self.ridge_weight < 0.0 or self.tv_weight < 0.0 or self.tv_epsilon_mm <= 0.0:
            raise ValueError("Invalid regularization parameters")
        if self.irls_iterations <= 0 or self.lsq_tolerance <= 0.0 or self.lsq_max_iterations <= 0:
            raise ValueError("Invalid bounded least-squares settings")
        if not self.formal_sample_ids or len(set(self.formal_sample_ids)) != len(self.formal_sample_ids):
            raise ValueError("formal_sample_ids must be a non-empty unique sequence")
        if not set(self.validation_sample_ids).issubset(self.formal_sample_ids):
            raise ValueError("validation_sample_ids must be a subset of formal_sample_ids")
        if self.defect_threshold_mm <= 0.0:
            raise ValueError("defect_threshold_mm must be positive")
        if self.maximum_training_linearity_error <= 0.0:
            raise ValueError("maximum_training_linearity_error must be positive")
        output_path = project_path(self.output_root).resolve()
        try:
            output_path.relative_to(HERE.resolve())
        except ValueError as error:
            raise ValueError(f"output_root must stay under {HERE}: {output_path}") from error

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
    def healthy_path(self) -> Path:
        return self.dataset_path / "frequency_response" / f"{self.healthy_sample_id}_H_complex.npz"

    @property
    def healthy_metadata_path(self) -> Path:
        return self.dataset_path / "metadata" / f"{self.healthy_sample_id}.json"

    @property
    def plan_path(self) -> Path:
        return self.output_path / "training_plan.json"

    @property
    def training_path(self) -> Path:
        return self.output_path / "training_corpus"

    @property
    def training_healthy_path(self) -> Path:
        return (
            self.training_path
            / "frequency_response"
            / f"{self.training_healthy_sample_id}_H_complex.npz"
        )

    @property
    def operator_metadata_path(self) -> Path:
        return self.output_path / "fullwave_rytov_operator.json"

    @property
    def operator_npz_path(self) -> Path:
        return self.output_path / "fullwave_rytov_operator.npz"

    @property
    def formal_output_path(self) -> Path:
        return self.output_path / "output_dataset"

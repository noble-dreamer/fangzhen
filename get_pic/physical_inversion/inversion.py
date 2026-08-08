"""V2 fixed-kernel physical inversion primitives.

The response-to-path calibration belongs exclusively to the independent COMSOL
matched corpus in ``simulation_prior/channel_prior.py``. This module owns the
shared geometry, fixed helical ray operator, bounded SIRT, and millimetre I/O.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import resize


HERE = Path(__file__).resolve().parent
GET_PIC_ROOT = HERE.parent
PROJECT_ROOT = HERE.parents[2]
if str(GET_PIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GET_PIC_ROOT))

import coarse_map_common as cm  # noqa: E402


EPS = 1e-12


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class InversionConfig:
    dataset_root: str = "simple/f_domain/output_dataset/streaming_dataset_a_frequency_shell"
    output_root: str = (
        "simple/get_pic/physical_inversion/simulation_prior/output_dataset_matched_corpus"
    )
    image_size: int = 256
    inversion_grid_size: int = 64
    helical_orders: tuple[int, ...] = (-1, 0, 1)
    sigma_ray_mm: float = 25.0
    min_endpoint_distance_mm: float = 30.0
    kernel_sigma_cutoff: float = 3.0
    sirt_iterations: int = 64
    sirt_relaxation: float = 0.85
    tv_weight_mm: float = 0.20
    tv_inner_iterations: int = 60
    coverage_threshold: float = 0.01
    formal_sample_ids: tuple[int, ...] = ()
    validation_sample_ids: tuple[int, ...] = ()
    defect_threshold_mm: float = 0.1

    @classmethod
    def from_json(cls, path: Path) -> "InversionConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(raw).difference(allowed))
        if unknown:
            raise ValueError(f"Unknown v2 inversion config keys: {unknown}")
        for key in ("helical_orders", "formal_sample_ids", "validation_sample_ids"):
            if key in raw:
                raw[key] = tuple(int(value) for value in raw[key])
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if self.image_size <= 0 or self.inversion_grid_size <= 0:
            raise ValueError("image_size and inversion_grid_size must be positive")
        if not self.helical_orders:
            raise ValueError("helical_orders cannot be empty")
        if self.sigma_ray_mm <= 0.0:
            raise ValueError("sigma_ray_mm must be positive")
        if not 0.0 <= self.coverage_threshold <= 1.0:
            raise ValueError("coverage_threshold must be in [0, 1]")
        if self.sirt_iterations <= 0 or not 0.0 < self.sirt_relaxation <= 2.0:
            raise ValueError("Invalid SIRT parameters")
        if not self.formal_sample_ids or len(set(self.formal_sample_ids)) != len(self.formal_sample_ids):
            raise ValueError("formal_sample_ids must be a non-empty unique sequence")
        if not set(self.validation_sample_ids).issubset(self.formal_sample_ids):
            raise ValueError("validation_sample_ids must be a subset of formal_sample_ids")
        if self.defect_threshold_mm <= 0.0:
            raise ValueError("defect_threshold_mm must be positive")

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        for key, value in list(output.items()):
            if isinstance(value, tuple):
                output[key] = list(value)
        return output

    @property
    def dataset_path(self) -> Path:
        return project_path(self.dataset_root)

    @property
    def output_path(self) -> Path:
        return project_path(self.output_root)


@dataclass(frozen=True)
class RayOperator:
    matrix: np.ndarray
    theta_deg: np.ndarray
    z_mm: np.ndarray
    tx_indices: np.ndarray
    rx_indices: np.ndarray
    coverage: np.ndarray
    support_mask: np.ndarray


@dataclass(frozen=True)
class ChannelModel:
    intercept: np.ndarray
    slope: np.ndarray
    correlation: np.ndarray
    sigma_path_mm: np.ndarray


@dataclass(frozen=True)
class PathDepthObservation:
    values_mm: np.ndarray
    weights: np.ndarray
    frequencies_hz: np.ndarray
    retained_amplitude_fraction: float
    retained_phase_fraction: float


def sample_name(sample_id: int | str) -> str:
    if isinstance(sample_id, str) and sample_id.startswith("dataset_a_frequency_sample_"):
        return sample_id
    return f"dataset_a_frequency_sample_{int(sample_id):04d}"


def parse_sample_ids(values: Iterable[str | int] | str | int) -> list[int]:
    if isinstance(values, (str, int)):
        values = [values]
    output: list[int] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if not token:
                continue
            if token.startswith("dataset_a_frequency_sample_"):
                token = token.rsplit("_", 1)[-1]
            if "-" in token:
                start_text, stop_text = token.split("-", 1)
                start, stop = int(start_text), int(stop_text)
                if stop < start:
                    raise ValueError(f"Invalid sample range: {token}")
                output.extend(range(start, stop + 1))
            else:
                output.append(int(token))
    return list(dict.fromkeys(output))


def load_label_metadata(label_dir: Path, sample_id: int | str) -> dict[str, Any]:
    path = label_dir / f"{sample_name(sample_id)}_defect_label_metadata.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def physical_limits(metadata: dict[str, Any]) -> tuple[float, float]:
    denominator = float(metadata.get("normalization_denominator_mm", 0.0))
    depth_limit = float(metadata.get("depth_limit_mm", 0.0))
    if denominator <= 0.0 or depth_limit <= 0.0:
        raise ValueError("Label metadata must define positive physical scales")
    return denominator, depth_limit


def resize_field(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(field, dtype=np.float32)
    if array.shape == shape:
        return array.copy()
    return resize(
        array,
        shape,
        order=1,
        mode="reflect",
        anti_aliasing=array.shape[0] > shape[0] or array.shape[1] > shape[1],
        preserve_range=True,
    ).astype(np.float32)


def resize_label_nearest(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(field, dtype=np.float32)
    if array.shape == shape:
        return array.copy()
    if array.ndim != 2:
        raise ValueError(f"Label must be 2D, got {array.shape}")
    target_z, target_theta = shape
    source_z, source_theta = array.shape
    z_index = np.rint(np.linspace(0, source_z - 1, target_z)).astype(np.int64)
    theta_index = np.floor(np.arange(target_theta) * source_theta / target_theta).astype(np.int64)
    theta_index = np.clip(theta_index, 0, source_theta - 1)
    return array[np.ix_(z_index, theta_index)].astype(np.float32, copy=False)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    if np.std(a) <= EPS or np.std(b) <= EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


class PhysicalInverter:
    """Fixed ``K=sum(alpha_o K_o)`` geometry plus bounded SIRT reconstruction."""

    def __init__(
        self,
        config: InversionConfig,
        *,
        helical_order_weights: Iterable[float] | None = None,
    ):
        self.config = config
        if helical_order_weights is None:
            self.helical_order_weights = np.full(
                len(config.helical_orders), 1.0 / len(config.helical_orders), dtype=np.float64
            )
        else:
            weights = np.asarray(tuple(helical_order_weights), dtype=np.float64)
            if weights.shape != (len(config.helical_orders),):
                raise ValueError("helical_order_weights must have one value per configured helical order")
            if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
                raise ValueError("helical_order_weights must be finite and nonnegative")
            total = float(np.sum(weights))
            if total <= 0.0:
                raise ValueError("helical_order_weights must contain a positive value")
            self.helical_order_weights = weights / total
        self.response_dir = config.dataset_path / "frequency_response"
        self.metadata_dir = config.dataset_path / "metadata"
        self.label_dir = config.dataset_path / "labels"
        self.healthy_path = self.response_dir / "dataset_a_frequency_healthy_H_complex.npz"
        self.healthy_metadata_path = self.metadata_dir / "dataset_a_frequency_healthy.json"
        self.healthy = cm.load_frequency_response(self.healthy_path)
        self.geometry = cm.geometry_from_metadata(cm.read_json(self.healthy_metadata_path))
        self.operator = self._build_operator()

    def damaged_path(self, sample_id: int | str) -> Path:
        return self.response_dir / f"{sample_name(sample_id)}_H_complex.npz"

    def label_path(self, sample_id: int | str) -> Path:
        return self.label_dir / f"{sample_name(sample_id)}_defect_depth_mm.npy"

    def _build_operator(self) -> RayOperator:
        ray_config = cm.CoarseMapConfig(
            theta_count=self.config.inversion_grid_size,
            z_count=self.config.inversion_grid_size,
            helical_orders=self.config.helical_orders,
            sigma_ray_mm=self.config.sigma_ray_mm,
            min_endpoint_distance_mm=self.config.min_endpoint_distance_mm,
            kernel_sigma_cutoff=self.config.kernel_sigma_cutoff,
        )
        theta_deg, z_mm = cm.label_grid(
            self.geometry,
            theta_count=ray_config.theta_count,
            z_count=ray_config.z_count,
        )
        rows: list[np.ndarray] = []
        tx_case: list[int] = []
        rx_case: list[int] = []
        for tx_id in self.healthy.tx_indices:
            tx = self.geometry.tx_positions[tx_id]
            for rx_id in self.healthy.rx_indices:
                rx = self.geometry.rx_positions[rx_id]
                combined = np.zeros((ray_config.z_count, ray_config.theta_count), dtype=np.float64)
                for order_index, order in enumerate(ray_config.helical_orders):
                    kernel, _tube, _length = cm.ray_kernel(
                        theta_deg, z_mm, self.geometry, tx, rx, order, ray_config
                    )
                    kernel = np.asarray(kernel, dtype=np.float64)
                    kernel_sum = float(np.sum(kernel))
                    if kernel_sum > 0.0:
                        combined += self.helical_order_weights[order_index] * kernel / kernel_sum
                combined_sum = float(np.sum(combined))
                if combined_sum <= 0.0 or not np.isfinite(combined_sum):
                    raise RuntimeError(f"Zero ray sensitivity for tx={tx_id}, rx={rx_id}")
                rows.append((combined / combined_sum).reshape(-1).astype(np.float32))
                tx_case.append(int(tx_id))
                rx_case.append(int(rx_id))
        matrix = np.stack(rows, axis=0)
        coverage = np.sum(matrix, axis=0).reshape(ray_config.z_count, ray_config.theta_count)
        support = coverage >= float(np.max(coverage)) * self.config.coverage_threshold
        return RayOperator(
            matrix=matrix,
            theta_deg=theta_deg.astype(np.float32),
            z_mm=z_mm.astype(np.float32),
            tx_indices=np.asarray(tx_case, dtype=np.int32),
            rx_indices=np.asarray(rx_case, dtype=np.int32),
            coverage=coverage.astype(np.float32),
            support_mask=support,
        )

    def load_label_mm(self, sample_id: int | str, *, grid_size: int | None = None) -> np.ndarray:
        path = self.label_path(sample_id)
        if not path.exists():
            raise FileNotFoundError(path)
        label = np.asarray(np.load(path), dtype=np.float32)
        size = self.config.image_size if grid_size is None else int(grid_size)
        return resize_label_nearest(label, (size, size))

    def _periodic_tv(self, image: np.ndarray) -> np.ndarray:
        if self.config.tv_weight_mm <= 0.0:
            return image
        pad = max(2, min(8, image.shape[1] // 8))
        wrapped = np.concatenate([image[:, -pad:], image, image[:, :pad]], axis=1)
        denoised = denoise_tv_chambolle(
            wrapped,
            weight=self.config.tv_weight_mm,
            channel_axis=None,
            max_num_iter=self.config.tv_inner_iterations,
        )
        return np.asarray(denoised[:, pad:-pad], dtype=np.float64)

    def invert_path_depth(
        self,
        observation: PathDepthObservation,
        *,
        depth_limit_mm: float,
        iterations: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        matrix = self.operator.matrix.astype(np.float64, copy=False)
        target = observation.values_mm.astype(np.float64)
        count = self.config.sirt_iterations if iterations is None else int(iterations)
        if count <= 0:
            raise ValueError("iterations must be positive")
        path_weights = np.clip(observation.weights.astype(np.float64), 0.05, 1.0)
        row_scale = 1.0 / np.maximum(np.sum(matrix, axis=1), EPS)
        col_scale = 1.0 / np.maximum(matrix.T @ path_weights, EPS)
        x = np.zeros(matrix.shape[1], dtype=np.float64)
        residual_trace: list[float] = []
        for iteration in range(1, count + 1):
            residual = target - matrix @ x
            update = matrix.T @ (path_weights * residual * row_scale)
            x = np.clip(
                x + self.config.sirt_relaxation * update * col_scale,
                0.0,
                depth_limit_mm,
            )
            if iteration == 1 or iteration % 8 == 0 or iteration == count:
                residual_trace.append(float(np.sqrt(np.mean(residual * residual))))
        shape = (self.config.inversion_grid_size, self.config.inversion_grid_size)
        image = self._periodic_tv(x.reshape(shape))
        image = np.clip(image, 0.0, depth_limit_mm)
        image[~self.operator.support_mask] = 0.0
        final_residual = matrix @ image.reshape(-1) - target
        diagnostics = {
            "units": "mm",
            "solver": "nonnegative_sirt_then_periodic_tv",
            "iterations": count,
            "relaxation": self.config.sirt_relaxation,
            "tv_weight_mm": self.config.tv_weight_mm,
            "path_weight_min": float(np.min(path_weights)),
            "path_weight_max": float(np.max(path_weights)),
            "path_rmse_mm": float(np.sqrt(np.mean(final_residual * final_residual))),
            "residual_trace_mm": residual_trace,
            "prediction_min_mm": float(np.min(image)),
            "prediction_max_mm": float(np.max(image)),
            "prediction_mean_mm": float(np.mean(image)),
            "nonzero_fraction": float(np.mean(image > 0.0)),
        }
        return image.astype(np.float32), diagnostics

    def output_grid(self, image: np.ndarray) -> np.ndarray:
        return resize_field(image, (self.config.image_size, self.config.image_size))

    def grid_axes(self) -> tuple[np.ndarray, np.ndarray]:
        theta = np.linspace(0.0, 360.0, self.config.image_size, endpoint=False, dtype=np.float32)
        z = np.linspace(0.0, self.geometry.length_mm, self.config.image_size, dtype=np.float32)
        return theta, z

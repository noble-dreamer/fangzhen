"""Shared V1 coarse-map helpers.

Run scripts in this folder with:
    conda run -n get_pic python <script>.py
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent
SIMPLE_ROOT = ROOT.parent
DEFAULT_FREQUENCY_DATASET_ROOT = (
    SIMPLE_ROOT / "f_domain" / "output" / "streaming_dataset_a_frequency_shell"
)
DEFAULT_RESPONSE_DIR = DEFAULT_FREQUENCY_DATASET_ROOT / "frequency_response"
DEFAULT_METADATA_DIR = DEFAULT_FREQUENCY_DATASET_ROOT / "metadata"
DEFAULT_LABEL_DIR = DEFAULT_FREQUENCY_DATASET_ROOT / "labels"
DEFAULT_HEALTHY_ID = "dataset_a_frequency_healthy"
DEFAULT_HEALTHY_RESPONSE = DEFAULT_RESPONSE_DIR / f"{DEFAULT_HEALTHY_ID}_H_complex.npz"
DEFAULT_HEALTHY_METADATA = DEFAULT_METADATA_DIR / f"{DEFAULT_HEALTHY_ID}.json"
DEFAULT_OUTPUT_ROOT = ROOT / "output"
DEFAULT_SELECTION_TXT = (
    SIMPLE_ROOT
    / "f_domain"
    / "output"
    / "frequency_selection"
    / "frequency_sensitivity_top15_frequencies.txt"
)


EPS = 1e-30


@dataclass(frozen=True)
class CoarseMapConfig:
    theta_count: int = 512
    z_count: int = 512
    helical_orders: tuple[int, ...] = (-1, 0, 1)
    sigma_ray_mm: float = 25.0
    min_endpoint_distance_mm: float = 30.0
    kernel_sigma_cutoff: float = 3.0
    reliability_threshold: float = 0.05
    positive_log_amp_loss: bool = True
    robust_percentiles: tuple[float, float] = (1.0, 99.0)
    frequency_tolerance_hz: float = 1e-6
    low_band_hz: tuple[float, float] = (0.0, 40_000.0)
    mid_band_hz: tuple[float, float] = (40_000.0, 60_000.0)
    high_band_hz: tuple[float, float] = (60_000.0, float("inf"))

    @classmethod
    def from_json(cls, path: Path | None) -> "CoarseMapConfig":
        if path is None:
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        values = dict(data)
        for key in (
            "helical_orders",
            "robust_percentiles",
            "low_band_hz",
            "mid_band_hz",
            "high_band_hz",
        ):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, tuple):
                data[key] = list(value)
        return data


@dataclass(frozen=True)
class FrequencyResponse:
    path: Path
    h: np.ndarray
    completed_mask: np.ndarray
    tx_indices: tuple[int, ...]
    rx_indices: tuple[int, ...]
    frequencies_hz: tuple[float, ...]
    sample_id: str
    dataset: str
    defect_state: str


@dataclass(frozen=True)
class Position:
    index: int
    ring: str
    theta_deg: float
    z_mm: float


@dataclass(frozen=True)
class Geometry:
    length_mm: float
    mid_radius_mm: float
    tx_positions: dict[int, Position]
    rx_positions: dict[int, Position]


@dataclass
class ProjectionProducts:
    pic_raw: np.ndarray
    pic: np.ndarray
    channel_names: list[str]
    theta_deg: np.ndarray
    z_mm: np.ndarray
    coverage: np.ndarray
    valid_case_count: np.ndarray
    reliability_mask: np.ndarray
    x_matrix: np.ndarray
    x_feature_names: list[str]
    selected_frequency_mask: np.ndarray
    frequency_weights: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def scalar_string(data: np.lib.npyio.NpzFile, key: str, fallback: str) -> str:
    if key not in data.files:
        return fallback
    value = np.asarray(data[key])
    if value.shape == ():
        return str(value.item())
    return str(value.tolist())


def load_frequency_response(path: Path) -> FrequencyResponse:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    required = {"H_real", "H_imag", "tx_indices", "rx_indices", "frequencies_hz"}
    missing = required.difference(data.files)
    if missing:
        raise RuntimeError(f"{path} missing required fields: {sorted(missing)}")

    h_real = np.asarray(data["H_real"], dtype=np.float64)
    h_imag = np.asarray(data["H_imag"], dtype=np.float64)
    if h_real.shape != h_imag.shape or h_real.ndim != 3:
        raise RuntimeError(f"{path} H_real/H_imag must have same 3D shape")
    h = h_real + 1j * h_imag
    if "completed_mask" in data.files:
        completed = np.asarray(data["completed_mask"], dtype=bool)
    else:
        completed = np.ones((h.shape[0], h.shape[2]), dtype=bool)
    if completed.shape != (h.shape[0], h.shape[2]):
        raise RuntimeError(
            f"{path} completed_mask shape {completed.shape} does not match H {h.shape}"
        )
    return FrequencyResponse(
        path=path,
        h=h,
        completed_mask=completed,
        tx_indices=tuple(int(item) for item in np.asarray(data["tx_indices"]).tolist()),
        rx_indices=tuple(int(item) for item in np.asarray(data["rx_indices"]).tolist()),
        frequencies_hz=tuple(float(item) for item in np.asarray(data["frequencies_hz"]).tolist()),
        sample_id=scalar_string(data, "sample_id", path.stem.replace("_H_complex", "")),
        dataset=scalar_string(data, "dataset", ""),
        defect_state=scalar_string(data, "defect_state", ""),
    )


def assert_compatible(healthy: FrequencyResponse, damaged: FrequencyResponse) -> None:
    checks = {
        "H shape": healthy.h.shape == damaged.h.shape,
        "tx_indices": healthy.tx_indices == damaged.tx_indices,
        "rx_indices": healthy.rx_indices == damaged.rx_indices,
        "frequencies_hz": healthy.frequencies_hz == damaged.frequencies_hz,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(
            f"{damaged.path} is not compatible with {healthy.path}; failed: {failed}"
        )


def standard_metadata_path(sample_id: str, metadata_dir: Path = DEFAULT_METADATA_DIR) -> Path:
    return metadata_dir / f"{sample_id}.json"


def sample_id_from_response_path(path: Path) -> str:
    name = path.name
    if name.endswith("_H_complex.npz"):
        return name[: -len("_H_complex.npz")]
    return path.stem


def perturbation_for(metadata: dict[str, Any], index: int) -> dict[str, float]:
    perturbations = metadata.get("model", {}).get("position_perturbations", {})
    item = perturbations.get(str(index), perturbations.get(index, {}))
    return item if isinstance(item, dict) else {}


def geometry_from_metadata(metadata: dict[str, Any] | None = None) -> Geometry:
    metadata = metadata or {}
    model = metadata.get("model", {})
    pipe = model.get("pipe", {})
    transducer = model.get("transducer", {})
    length_mm = float(pipe.get("length_mm", 1000.0))
    outer_radius_mm = float(pipe.get("outer_radius_mm", 160.0))
    inner_radius_mm = float(pipe.get("inner_radius_mm", 150.0))
    mid_radius_mm = float(pipe.get("mid_radius_mm", 0.5 * (outer_radius_mm + inner_radius_mm)))
    count = int(transducer.get("count_per_ring", 16))
    tx_z_mm = float(transducer.get("tx_z_mm", 100.0))
    rx_z_mm = float(transducer.get("rx_z_mm", 900.0))
    step = 360.0 / count

    tx_positions: dict[int, Position] = {}
    rx_positions: dict[int, Position] = {}
    for index in range(1, 2 * count + 1):
        if index <= count:
            ring = "tx"
            n = index - 1
            base_z = tx_z_mm
        else:
            ring = "rx"
            n = index - count - 1
            base_z = rx_z_mm
        perturb = perturbation_for(metadata, index)
        position = Position(
            index=index,
            ring=ring,
            theta_deg=(n * step + float(perturb.get("dtheta_deg", 0.0))) % 360.0,
            z_mm=base_z + float(perturb.get("dz_mm", 0.0)),
        )
        if ring == "tx":
            tx_positions[index] = position
        else:
            rx_positions[index] = position
    return Geometry(
        length_mm=length_mm,
        mid_radius_mm=mid_radius_mm,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
    )


def label_grid(
    geometry: Geometry,
    *,
    theta_count: int,
    z_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    theta_deg = np.linspace(0.0, 360.0, theta_count, endpoint=False, dtype=np.float64)
    z_mm = np.linspace(0.0, geometry.length_mm, z_count, dtype=np.float64)
    return theta_deg, z_mm


def parse_frequency_values(text: str) -> list[float]:
    values: list[float] = []
    for token in text.replace("\n", ",").replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values


def load_selected_frequencies(path: Path | None) -> tuple[set[float], dict[float, float]]:
    if path is None or not path.exists():
        return set(), {}
    if path.suffix.lower() == ".csv":
        selected: set[float] = set()
        weights: dict[float, float] = {}
        with path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or "frequency_hz" not in reader.fieldnames:
                raise RuntimeError(f"{path} CSV must contain frequency_hz column")
            for row in reader:
                frequency = float(row["frequency_hz"])
                selected.add(frequency)
                for key in ("sensitivity_mean", "weight", "frequency_weight"):
                    if key in row and row[key] not in ("", None):
                        weights[frequency] = float(row[key])
                        break
        return selected, weights
    return set(parse_frequency_values(path.read_text(encoding="utf-8"))), {}


def frequency_selection_mask(
    frequencies_hz: tuple[float, ...],
    selected: set[float],
    tolerance_hz: float,
) -> np.ndarray:
    if not selected:
        return np.ones(len(frequencies_hz), dtype=bool)
    mask = np.zeros(len(frequencies_hz), dtype=bool)
    selected_values = np.asarray(sorted(selected), dtype=np.float64)
    for index, frequency in enumerate(frequencies_hz):
        mask[index] = bool(np.any(np.abs(selected_values - frequency) <= tolerance_hz))
    return mask


def frequency_weights(
    frequencies_hz: tuple[float, ...],
    selected_weights: dict[float, float],
    tolerance_hz: float,
) -> np.ndarray:
    weights = np.ones(len(frequencies_hz), dtype=np.float64)
    if not selected_weights:
        return weights
    keys = np.asarray(list(selected_weights.keys()), dtype=np.float64)
    values = np.asarray([selected_weights[float(key)] for key in keys], dtype=np.float64)
    for index, frequency in enumerate(frequencies_hz):
        matches = np.abs(keys - frequency) <= tolerance_hz
        if np.any(matches):
            weights[index] = float(np.nanmean(values[matches]))
    weights[~np.isfinite(weights)] = 1.0
    max_weight = float(np.nanmax(weights)) if weights.size else 1.0
    if max_weight > 0.0:
        weights = weights / max_weight
    return weights


def finite_valid_mask(healthy: FrequencyResponse, damaged: FrequencyResponse) -> np.ndarray:
    completed = healthy.completed_mask[:, None, :] & damaged.completed_mask[:, None, :]
    finite = np.isfinite(np.real(healthy.h)) & np.isfinite(np.imag(healthy.h))
    finite &= np.isfinite(np.real(damaged.h)) & np.isfinite(np.imag(damaged.h))
    return completed & finite


def feature_arrays(
    healthy: FrequencyResponse,
    damaged: FrequencyResponse,
    *,
    eps: float = EPS,
    positive_log_amp_loss: bool = True,
) -> dict[str, np.ndarray]:
    h0 = healthy.h
    hd = damaged.h
    abs_h0 = np.abs(h0)
    abs_hd = np.abs(hd)
    delta = hd - h0
    delta_abs = np.abs(delta)
    rel_delta = delta_abs / (abs_h0 + eps)
    log_amp_loss = np.log(abs_h0 + eps) - np.log(abs_hd + eps)
    if positive_log_amp_loss:
        log_amp_loss = np.maximum(log_amp_loss, 0.0)
    phase_diff = np.abs(np.angle(hd * np.conj(h0)))
    return {
        "ray_log_amp_loss": log_amp_loss,
        "ray_relative_delta": rel_delta,
        "ray_phase_change": phase_diff,
        "ray_delta_abs": delta_abs,
        "healthy_log_abs": np.log(abs_h0 + eps),
        "damaged_log_abs": np.log(abs_hd + eps),
        "log_abs_delta": np.log(delta_abs + eps),
        "log_abs_reldelta": np.log(rel_delta + eps),
        "phase_cos": np.cos(np.angle(hd * np.conj(h0))),
        "phase_sin": np.sin(np.angle(hd * np.conj(h0))),
    }


def band_masks(frequencies_hz: tuple[float, ...], config: CoarseMapConfig) -> dict[str, np.ndarray]:
    f = np.asarray(frequencies_hz, dtype=np.float64)
    low_min, low_max = config.low_band_hz
    mid_min, mid_max = config.mid_band_hz
    high_min, high_max = config.high_band_hz
    return {
        "low_frequency_band_map": (f >= low_min) & (f <= low_max),
        "mid_frequency_band_map": (f > mid_min) & (f <= mid_max),
        "high_frequency_band_map": (f > high_min) & (f <= high_max),
    }


def wrap_rad(angle_rad: np.ndarray | float) -> np.ndarray | float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def distance_to_segment(
    x: np.ndarray,
    y: np.ndarray,
    x2: float,
    y1: float,
    y2: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    vx = x2
    vy = y2 - y1
    length_sq = vx * vx + vy * vy
    if length_sq <= 0.0:
        distance = np.hypot(x, y - y1)
        return distance, np.zeros_like(distance), 0.0
    t = (x * vx + (y - y1) * vy) / length_sq
    t_clamped = np.clip(t, 0.0, 1.0)
    proj_x = t_clamped * vx
    proj_y = y1 + t_clamped * vy
    return np.hypot(x - proj_x, y - proj_y), t_clamped, math.sqrt(length_sq)


def ray_kernel(
    theta_grid_deg: np.ndarray,
    z_grid_mm: np.ndarray,
    geometry: Geometry,
    tx: Position,
    rx: Position,
    order: int,
    config: CoarseMapConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    theta_rad = np.deg2rad(theta_grid_deg)[None, :]
    z = z_grid_mm[:, None]
    tx_theta = math.radians(tx.theta_deg)
    rx_theta = math.radians(rx.theta_deg)
    base_delta = float(wrap_rad(rx_theta - tx_theta))
    path_delta = base_delta + 2.0 * math.pi * order
    x2 = geometry.mid_radius_mm * path_delta
    y1 = tx.z_mm
    y2 = rx.z_mm

    base_pixel = wrap_rad(theta_rad - tx_theta)
    distances: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    for image_order in range(order - 2, order + 3):
        x = geometry.mid_radius_mm * (base_pixel + 2.0 * math.pi * image_order)
        distance, t, _length = distance_to_segment(x, z, x2, y1, y2)
        distances.append(distance)
        ts.append(t)
    stacked = np.stack(distances, axis=0)
    best_index = np.argmin(stacked, axis=0)
    distance = np.take_along_axis(stacked, best_index[None, :, :], axis=0)[0]
    t = np.take_along_axis(np.stack(ts, axis=0), best_index[None, :, :], axis=0)[0]

    ray_length = math.hypot(x2, y2 - y1)
    sigma = max(float(config.sigma_ray_mm), 1e-9)
    kernel = np.exp(-0.5 * (distance / sigma) ** 2) / max(ray_length, 1e-9)
    cutoff = distance <= config.kernel_sigma_cutoff * sigma
    if config.min_endpoint_distance_mm > 0.0 and ray_length > 0.0:
        margin = min(config.min_endpoint_distance_mm / ray_length, 0.49)
        cutoff &= (t >= margin) & (t <= 1.0 - margin)
    kernel = np.where(cutoff, kernel, 0.0).astype(np.float32)
    return kernel, cutoff, ray_length


def robust_normalize(channel: np.ndarray, percentiles: tuple[float, float]) -> tuple[np.ndarray, dict[str, float]]:
    values = np.asarray(channel, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values, dtype=np.float32), {"low": float("nan"), "high": float("nan")}
    low, high = np.nanpercentile(values[finite], percentiles)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(values, dtype=np.float32), {"low": float(low), "high": float(high)}
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    normalized[~finite] = 0.0
    return normalized.astype(np.float32), {"low": float(low), "high": float(high)}


def build_x_matrix(
    features: dict[str, np.ndarray],
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    names = [
        "log_abs_delta",
        "log_abs_reldelta",
        "phase_cos",
        "phase_sin",
        "healthy_log_abs",
        "damaged_log_abs",
        "valid_mask",
    ]
    arrays = []
    for name in names:
        if name == "valid_mask":
            arr = valid_mask.astype(np.float32)
        else:
            arr = np.asarray(features[name], dtype=np.float32)
            arr = np.where(np.isfinite(arr), arr, 0.0)
        arrays.append(np.transpose(arr, (2, 0, 1)))
    return np.stack(arrays, axis=0).astype(np.float32), names


def aggregate_scalar(
    values: np.ndarray,
    valid_f: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    if not np.any(valid_f):
        return 0.0, 0.0
    local_weights = weights[valid_f]
    local_values = values[valid_f]
    finite = np.isfinite(local_values) & np.isfinite(local_weights)
    if not np.any(finite):
        return 0.0, 0.0
    local_weights = local_weights[finite]
    local_values = local_values[finite]
    return float(np.sum(local_weights * local_values)), float(np.sum(local_weights))


def make_v1_coarse_map(
    healthy: FrequencyResponse,
    damaged: FrequencyResponse,
    *,
    healthy_metadata: dict[str, Any] | None = None,
    damaged_metadata: dict[str, Any] | None = None,
    selected_frequency_file: Path | None = None,
    config: CoarseMapConfig | None = None,
) -> ProjectionProducts:
    config = config or CoarseMapConfig()
    assert_compatible(healthy, damaged)
    metadata_for_geometry = damaged_metadata or healthy_metadata or {}
    geometry = geometry_from_metadata(metadata_for_geometry)
    theta_deg, z_mm = label_grid(
        geometry,
        theta_count=config.theta_count,
        z_count=config.z_count,
    )
    selected_frequencies, selected_weights = load_selected_frequencies(selected_frequency_file)
    selected_mask = frequency_selection_mask(
        healthy.frequencies_hz,
        selected_frequencies,
        config.frequency_tolerance_hz,
    )
    freq_weights = frequency_weights(
        healthy.frequencies_hz,
        selected_weights,
        config.frequency_tolerance_hz,
    )
    valid = finite_valid_mask(healthy, damaged)
    valid &= selected_mask[None, None, :]
    if not np.any(valid):
        raise RuntimeError(
            f"No valid tx-rx-frequency cases after completed_mask/finite/selection filtering for {damaged.path}"
        )

    features = feature_arrays(
        healthy,
        damaged,
        eps=EPS,
        positive_log_amp_loss=config.positive_log_amp_loss,
    )
    band = band_masks(healthy.frequencies_hz, config)
    channel_names = [
        "ray_log_amp_loss",
        "ray_relative_delta",
        "ray_phase_change",
        "ray_delta_abs",
        "low_frequency_band_map",
        "mid_frequency_band_map",
        "high_frequency_band_map",
        "path_coverage",
        "valid_case_count",
        "reliability_mask",
    ]
    accum = np.zeros((len(channel_names), config.z_count, config.theta_count), dtype=np.float64)
    channel_coverage = np.zeros((7, config.z_count, config.theta_count), dtype=np.float64)
    coverage = np.zeros((config.z_count, config.theta_count), dtype=np.float64)
    valid_case_count = np.zeros((config.z_count, config.theta_count), dtype=np.float64)

    tx_lookup = {value: index for index, value in enumerate(healthy.tx_indices)}
    rx_lookup = {value: index for index, value in enumerate(healthy.rx_indices)}
    for tx_index in healthy.tx_indices:
        if tx_index not in geometry.tx_positions:
            raise RuntimeError(f"Missing tx geometry for PZT {tx_index}")
        t_index = tx_lookup[tx_index]
        tx = geometry.tx_positions[tx_index]
        for rx_index in healthy.rx_indices:
            if rx_index not in geometry.rx_positions:
                raise RuntimeError(f"Missing rx geometry for PZT {rx_index}")
            r_index = rx_lookup[rx_index]
            rx = geometry.rx_positions[rx_index]
            valid_f = valid[t_index, r_index, :]
            if not np.any(valid_f):
                continue
            scalars: dict[str, float] = {}
            scalar_weights: dict[str, float] = {}
            for name in channel_names[:4]:
                scalar, weight_sum = aggregate_scalar(
                    features[name][t_index, r_index, :],
                    valid_f,
                    freq_weights,
                )
                scalars[name] = scalar
                scalar_weights[name] = weight_sum
            for name, mask in band.items():
                scalar, weight_sum = aggregate_scalar(
                    features["ray_relative_delta"][t_index, r_index, :],
                    valid_f & mask,
                    freq_weights,
                )
                scalars[name] = scalar
                scalar_weights[name] = weight_sum

            coverage_weight = float(np.sum(freq_weights[valid_f]))
            case_count = float(np.sum(valid_f))
            for order in config.helical_orders:
                kernel, tube_mask, _ray_length = ray_kernel(theta_deg, z_mm, geometry, tx, rx, order, config)
                if not np.any(kernel):
                    continue
                kernel64 = kernel.astype(np.float64, copy=False)
                for channel_index, name in enumerate(channel_names[:7]):
                    scalar_weight = scalar_weights.get(name, 0.0)
                    if scalar_weight <= 0.0:
                        continue
                    accum[channel_index] += kernel64 * scalars[name]
                    channel_coverage[channel_index] += kernel64 * scalar_weight
                coverage += kernel64 * coverage_weight
                valid_case_count += tube_mask.astype(np.float64) * case_count

    if not np.any(coverage > 0.0):
        raise RuntimeError("V1 projection produced zero path coverage")

    pic_raw = np.zeros_like(accum, dtype=np.float32)
    for index in range(7):
        pic_raw[index] = (accum[index] / (channel_coverage[index] + EPS)).astype(np.float32)
    coverage_p99 = float(np.nanpercentile(coverage[coverage > 0.0], 99.0))
    coverage_norm = np.clip(coverage / max(coverage_p99, EPS), 0.0, 1.0).astype(np.float32)
    count_norm = valid_case_count.astype(np.float32)
    reliability = (coverage_norm >= config.reliability_threshold).astype(np.float32)
    pic_raw[7] = coverage_norm
    pic_raw[8] = count_norm
    pic_raw[9] = reliability

    pic = np.zeros_like(pic_raw, dtype=np.float32)
    normalization: dict[str, Any] = {}
    for index, name in enumerate(channel_names[:7]):
        pic[index], normalization[name] = robust_normalize(pic_raw[index], config.robust_percentiles)
    pic[7] = coverage_norm
    max_count = float(np.nanmax(count_norm)) if np.size(count_norm) else 0.0
    pic[8] = np.clip(count_norm / max(max_count, EPS), 0.0, 1.0)
    pic[9] = reliability
    normalization["path_coverage"] = {"p99": coverage_p99}
    normalization["valid_case_count"] = {"max": max_count}
    normalization["reliability_mask"] = {"threshold": config.reliability_threshold}

    x_matrix, x_feature_names = build_x_matrix(features, valid)
    valid_count = int(np.sum(valid))
    metadata = {
        "algorithm": "v1_coverage_normalized_ray_tube_backprojection",
        "config": config.to_json_dict(),
        "normalization": normalization,
        "valid_tx_rx_frequency_count": valid_count,
        "selected_frequencies_hz": [
            float(f)
            for f, selected in zip(healthy.frequencies_hz, selected_mask, strict=True)
            if selected
        ],
        "used_frequencies_hz": [
            float(f)
            for index, f in enumerate(healthy.frequencies_hz)
            if selected_mask[index] and np.any(valid[:, :, index])
        ],
        "helical_orders": list(config.helical_orders),
        "geometry": {
            "length_mm": geometry.length_mm,
            "mid_radius_mm": geometry.mid_radius_mm,
        },
    }
    return ProjectionProducts(
        pic_raw=pic_raw,
        pic=pic,
        channel_names=channel_names,
        theta_deg=theta_deg.astype(np.float32),
        z_mm=z_mm.astype(np.float32),
        coverage=coverage_norm,
        valid_case_count=count_norm,
        reliability_mask=reliability.astype(np.uint8),
        x_matrix=x_matrix,
        x_feature_names=x_feature_names,
        selected_frequency_mask=selected_mask,
        frequency_weights=freq_weights.astype(np.float32),
        metadata=metadata,
    )


def write_projection_outputs(
    products: ProjectionProducts,
    *,
    output_root: Path,
    healthy: FrequencyResponse,
    damaged: FrequencyResponse,
    healthy_metadata_path: Path | None,
    damaged_metadata_path: Path | None,
) -> tuple[Path, Path]:
    sample_id = damaged.sample_id
    coarse_dir = output_root / "coarse_maps"
    x_dir = output_root / "x_matrix"
    coarse_dir.mkdir(parents=True, exist_ok=True)
    x_dir.mkdir(parents=True, exist_ok=True)
    coarse_path = coarse_dir / f"{sample_id}_coarse_maps.npz"
    x_path = x_dir / f"{sample_id}_x_matrix.npz"

    metadata = dict(products.metadata)
    metadata.update({
        "sample_id": sample_id,
        "source_healthy_npz": str(healthy.path),
        "source_damaged_npz": str(damaged.path),
        "source_healthy_metadata": None if healthy_metadata_path is None else str(healthy_metadata_path),
        "source_damaged_metadata": None if damaged_metadata_path is None else str(damaged_metadata_path),
        "coarse_map_npz": str(coarse_path),
        "x_matrix_npz": str(x_path),
    })
    np.savez_compressed(
        coarse_path,
        pic=products.pic,
        pic_raw=products.pic_raw,
        channel_names=np.asarray(products.channel_names),
        theta_deg=products.theta_deg,
        z_mm=products.z_mm,
        coverage=products.coverage,
        valid_case_count=products.valid_case_count,
        reliability_mask=products.reliability_mask,
        frequency_hz=np.asarray(healthy.frequencies_hz, dtype=np.float64),
        selected_frequency_mask=products.selected_frequency_mask,
        frequency_weights=products.frequency_weights,
        tx_indices=np.asarray(healthy.tx_indices, dtype=np.int32),
        rx_indices=np.asarray(healthy.rx_indices, dtype=np.int32),
        algorithm_config_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    np.savez_compressed(
        x_path,
        x=products.x_matrix,
        feature_names=np.asarray(products.x_feature_names),
        frequency_hz=np.asarray(healthy.frequencies_hz, dtype=np.float64),
        tx_indices=np.asarray(healthy.tx_indices, dtype=np.int32),
        rx_indices=np.asarray(healthy.rx_indices, dtype=np.int32),
        selected_frequency_mask=products.selected_frequency_mask,
        source_healthy_npz=np.asarray(str(healthy.path)),
        source_damaged_npz=np.asarray(str(damaged.path)),
    )
    return coarse_path, x_path


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_coarse_npz(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def pearson(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    true = np.asarray(target, dtype=np.float64).reshape(-1)
    valid = np.isfinite(pred) & np.isfinite(true)
    if not np.any(valid):
        return float("nan")
    pred = pred[valid]
    true = true[valid]
    if float(np.std(pred)) <= 0.0 or float(np.std(true)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(pred, true)[0, 1])


def nrmse(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=np.float64)
    true = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(true)
    if pred.shape != true.shape or not np.any(valid):
        return float("nan")
    scale = max(float(np.nanmax(true[valid]) - np.nanmin(true[valid])), EPS)
    return float(np.sqrt(np.nanmean((pred[valid] - true[valid]) ** 2)) / scale)


def mask_iou(prediction: np.ndarray, target: np.ndarray, threshold: float) -> float:
    pred_mask = np.asarray(prediction) >= threshold
    target_mask = np.asarray(target) >= threshold
    if pred_mask.shape != target_mask.shape:
        return float("nan")
    union = np.logical_or(pred_mask, target_mask)
    if not np.any(union):
        return float("nan")
    return float(np.logical_and(pred_mask, target_mask).sum() / union.sum())


def top_fraction_hit_rate(prediction: np.ndarray, target: np.ndarray, target_threshold: float, top_fraction: float = 0.05) -> float:
    pred = np.asarray(prediction, dtype=np.float64)
    true = np.asarray(target, dtype=np.float64)
    if pred.shape != true.shape:
        return float("nan")
    valid = np.isfinite(pred) & np.isfinite(true)
    if not np.any(valid):
        return float("nan")
    valid_scores = pred[valid]
    if valid_scores.size == 0:
        return float("nan")
    cutoff = float(np.nanquantile(valid_scores, max(0.0, min(1.0, 1.0 - top_fraction))))
    pred_top = valid & (pred >= cutoff)
    if not np.any(pred_top):
        return float("nan")
    target_mask = true >= target_threshold
    return float(np.logical_and(pred_top, target_mask).sum() / pred_top.sum())


def prediction_mass_in_target(prediction: np.ndarray, target: np.ndarray, target_threshold: float) -> float:
    pred = np.asarray(prediction, dtype=np.float64)
    true = np.asarray(target, dtype=np.float64)
    if pred.shape != true.shape:
        return float("nan")
    pred = np.where(np.isfinite(pred), np.maximum(pred, 0.0), 0.0)
    total = float(np.sum(pred))
    if total <= 0.0:
        return float("nan")
    target_mask = np.asarray(true) >= target_threshold
    return float(np.sum(pred[target_mask]) / total)


def centroid_error_mm(
    prediction: np.ndarray,
    target: np.ndarray,
    theta_deg: np.ndarray,
    z_mm: np.ndarray,
    mid_radius_mm: float = 155.0,
) -> float:
    pred = np.asarray(prediction, dtype=np.float64)
    true = np.asarray(target, dtype=np.float64)
    if pred.shape != true.shape:
        return float("nan")

    def centroid(values: np.ndarray) -> tuple[float, float] | None:
        values = np.maximum(values, 0.0)
        total = float(np.sum(values))
        if total <= 0.0:
            return None
        z_center = float(np.sum(values * z_mm[:, None]) / total)
        angles = np.deg2rad(theta_deg)[None, :]
        sin_mean = float(np.sum(values * np.sin(angles)) / total)
        cos_mean = float(np.sum(values * np.cos(angles)) / total)
        theta_center = math.atan2(sin_mean, cos_mean)
        return theta_center, z_center

    pred_center = centroid(pred)
    true_center = centroid(true)
    if pred_center is None or true_center is None:
        return float("nan")
    dtheta = float(wrap_rad(pred_center[0] - true_center[0]))
    dz = pred_center[1] - true_center[1]
    return float(math.hypot(mid_radius_mm * dtheta, dz))

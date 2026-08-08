"""Shared geometry and artifact helpers for the v2 matched COMSOL corpus."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
GET_PIC_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(GET_PIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GET_PIC_ROOT))

from simple.get_pic import coarse_map_common as cm  # noqa: E402


EPS = 1e-12
MODEL_FINGERPRINT_KEYS = (
    "model_family",
    "pipe",
    "transducer",
    "material",
    "absorbing_layer",
    "defect_model",
    "solver",
    "mesh",
    "receiver_indices",
    "receiver_model",
    "actuation_model",
    "create_receiver_datasets",
    "create_visual_marker_datasets",
    "position_perturbations",
    "amplitude_scale",
    "analysis_type",
    "frequency_domain",
)
MODEL_SOURCE_RELATIVE_PATHS = (
    "simple/f_domain/frequency_domain_common.py",
    "simple/simple_shell_common.py",
    "simple/streaming_export_common.py",
    "simple/defect_label_common.py",
    "simple/get_pic/coarse_map_common.py",
    "simple/get_pic/physical_inversion/simulation_prior/solve_channel_corpus.py",
)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def sha256_source_file(path: Path) -> str:
    text_value = path.read_text(encoding="utf-8-sig")
    normalized = text_value.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def model_semantic_contract(metadata: dict[str, Any]) -> dict[str, Any]:
    model = metadata.get("model", {})
    if not isinstance(model, dict):
        raise ValueError("metadata.model must be an object")
    missing = [key for key in MODEL_FINGERPRINT_KEYS if key not in model]
    if missing:
        raise ValueError(f"Model metadata lacks semantic fingerprint keys: {missing}")
    return {key: model[key] for key in MODEL_FINGERPRINT_KEYS}


def model_semantic_sha256(metadata: dict[str, Any]) -> str:
    return canonical_json_sha256(model_semantic_contract(metadata))


@dataclass(frozen=True)
class OrderKernelBank:
    matrix: np.ndarray
    theta_deg: np.ndarray
    z_mm: np.ndarray
    tx_indices: np.ndarray
    rx_indices: np.ndarray
    orders: np.ndarray


@dataclass(frozen=True)
class PriorGeometry:
    length_mm: float
    outer_radius_mm: float
    inner_radius_mm: float
    mid_radius_mm: float
    tx_positions: dict[int, cm.Position]
    rx_positions: dict[int, cm.Position]


def load_geometry(config: Any) -> tuple[cm.FrequencyResponse, PriorGeometry]:
    """Load formal healthy geometry and enforce the v2 millimetre scale."""

    healthy = cm.load_frequency_response(config.healthy_path)
    metadata = cm.read_json(config.healthy_metadata_path)
    base = cm.geometry_from_metadata(metadata)
    pipe = metadata.get("model", {}).get("pipe", {})
    outer_radius_mm = float(pipe.get("outer_radius_mm", 160.0))
    inner_radius_mm = float(pipe.get("inner_radius_mm", 150.0))
    geometry = PriorGeometry(
        length_mm=base.length_mm,
        outer_radius_mm=outer_radius_mm,
        inner_radius_mm=inner_radius_mm,
        mid_radius_mm=base.mid_radius_mm,
        tx_positions=base.tx_positions,
        rx_positions=base.rx_positions,
    )
    available_wall_loss_mm = (
        geometry.outer_radius_mm - geometry.inner_radius_mm - config.minimum_remaining_wall_mm
    )
    if not np.isclose(
        available_wall_loss_mm,
        config.normalization_denominator_mm,
        rtol=0.0,
        atol=1e-9,
    ):
        raise RuntimeError(
            "normalization_denominator_mm must equal wall thickness minus minimum remaining wall"
        )
    if config.defect_loss_limit_mm > available_wall_loss_mm:
        raise RuntimeError("defect_loss_limit_mm exceeds the physically available wall loss")
    return healthy, geometry


def validate_requested_axes(
    response: cm.FrequencyResponse,
    *,
    tx_indices: Iterable[int],
    frequencies_hz: Iterable[float],
) -> None:
    available_tx = {int(value) for value in response.tx_indices}
    missing_tx = sorted(set(int(value) for value in tx_indices).difference(available_tx))
    available_frequency = np.asarray(response.frequencies_hz, dtype=np.float64)
    missing_frequency = [
        float(value)
        for value in frequencies_hz
        if not np.any(np.isclose(available_frequency, float(value), rtol=0.0, atol=1e-6))
    ]
    if missing_tx or missing_frequency:
        raise RuntimeError(
            f"Healthy response does not contain requested axes: tx={missing_tx}, "
            f"frequency_hz={missing_frequency}"
        )
    if not np.all(response.completed_mask):
        raise RuntimeError("Healthy response is incomplete")


def response_index(values: np.ndarray, requested: float | int, *, atol: float = 1e-6) -> int:
    array = np.asarray(values)
    matches = np.flatnonzero(np.isclose(array.astype(float), float(requested), rtol=0.0, atol=atol))
    if matches.size != 1:
        raise RuntimeError(f"Expected one match for {requested}, got {matches.size} in {array.tolist()}")
    return int(matches[0])


def subset_response(
    response: cm.FrequencyResponse,
    tx_indices: Iterable[int],
    rx_indices: Iterable[int],
    frequencies_hz: Iterable[float],
) -> np.ndarray:
    tx_pos = [response_index(response.tx_indices, value) for value in tx_indices]
    rx_pos = [response_index(response.rx_indices, value) for value in rx_indices]
    frequency_pos = [response_index(response.frequencies_hz, value) for value in frequencies_hz]
    return response.h[np.ix_(tx_pos, rx_pos, frequency_pos)]


def build_order_kernel_bank(
    config: Any,
    healthy: cm.FrequencyResponse,
    geometry: PriorGeometry,
    *,
    tx_indices: Iterable[int] | None = None,
) -> OrderKernelBank:
    tx_values = tuple(int(value) for value in (tx_indices or config.simulation_tx_indices))
    ray_config = cm.CoarseMapConfig(
        theta_count=config.inversion_grid_size,
        z_count=config.inversion_grid_size,
        helical_orders=config.helical_orders,
        sigma_ray_mm=config.sigma_ray_mm,
        min_endpoint_distance_mm=config.min_endpoint_distance_mm,
        kernel_sigma_cutoff=config.kernel_sigma_cutoff,
    )
    theta_deg, z_mm = cm.label_grid(
        geometry,
        theta_count=ray_config.theta_count,
        z_count=ray_config.z_count,
    )
    rows: list[np.ndarray] = []
    path_tx: list[int] = []
    path_rx: list[int] = []
    for tx_id in tx_values:
        tx = geometry.tx_positions[tx_id]
        for rx_id in healthy.rx_indices:
            rx = geometry.rx_positions[int(rx_id)]
            order_rows: list[np.ndarray] = []
            for order in config.helical_orders:
                kernel, _tube, _length = cm.ray_kernel(
                    theta_deg, z_mm, geometry, tx, rx, int(order), ray_config
                )
                kernel = np.asarray(kernel, dtype=np.float64)
                total = float(np.sum(kernel))
                if not np.isfinite(total) or total <= 0.0:
                    raise RuntimeError(f"Zero order kernel for tx={tx_id}, rx={rx_id}, order={order}")
                order_rows.append((kernel / total).reshape(-1))
            rows.append(np.stack(order_rows, axis=0))
            path_tx.append(tx_id)
            path_rx.append(int(rx_id))
    return OrderKernelBank(
        matrix=np.stack(rows, axis=0).astype(np.float32),
        theta_deg=np.asarray(theta_deg, dtype=np.float32),
        z_mm=np.asarray(z_mm, dtype=np.float32),
        tx_indices=np.asarray(path_tx, dtype=np.int32),
        rx_indices=np.asarray(path_rx, dtype=np.int32),
        orders=np.asarray(config.helical_orders, dtype=np.int32),
    )


def relative_ring_offsets(
    path_tx_indices: Iterable[int],
    path_rx_indices: Iterable[int],
    ring_tx_indices: Iterable[int],
    ring_rx_indices: Iterable[int],
) -> np.ndarray:
    tx_ring = tuple(int(value) for value in ring_tx_indices)
    rx_ring = tuple(int(value) for value in ring_rx_indices)
    if len(tx_ring) != len(rx_ring) or not tx_ring:
        raise ValueError("TX and RX rings must have the same non-zero channel count")
    tx_slot = {value: index for index, value in enumerate(tx_ring)}
    rx_slot = {value: index for index, value in enumerate(rx_ring)}
    offsets = []
    for tx_id, rx_id in zip(path_tx_indices, path_rx_indices, strict=True):
        if int(tx_id) not in tx_slot or int(rx_id) not in rx_slot:
            raise ValueError(f"Path ({tx_id}, {rx_id}) is outside the configured transducer rings")
        offsets.append((rx_slot[int(rx_id)] - tx_slot[int(tx_id)]) % len(tx_ring))
    return np.asarray(offsets, dtype=np.int32)

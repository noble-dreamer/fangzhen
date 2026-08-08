"""Data, hashing, and complex-Rytov helpers shared by the new pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


EPS = 1.0e-12


@dataclass(frozen=True)
class FrequencyResponse:
    path: Path
    sample_id: str
    h: np.ndarray
    completed_mask: np.ndarray
    tx_indices: np.ndarray
    rx_indices: np.ndarray
    frequencies_hz: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_response(path: Path) -> FrequencyResponse:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        required = {
            "H_real",
            "H_imag",
            "completed_mask",
            "tx_indices",
            "rx_indices",
            "frequencies_hz",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise RuntimeError(f"Response {path} is missing fields: {missing}")
        h = np.asarray(data["H_real"], dtype=np.float64) + 1j * np.asarray(
            data["H_imag"], dtype=np.float64
        )
        completed = np.asarray(data["completed_mask"], dtype=bool)
        tx_indices = np.asarray(data["tx_indices"], dtype=np.int32)
        rx_indices = np.asarray(data["rx_indices"], dtype=np.int32)
        frequencies_hz = np.asarray(data["frequencies_hz"], dtype=np.float64)
        sample_value = data["sample_id"] if "sample_id" in data.files else path.stem
        sample_id = str(np.asarray(sample_value).reshape(-1)[0])
    expected_shape = (tx_indices.size, rx_indices.size, frequencies_hz.size)
    if h.shape != expected_shape:
        raise RuntimeError(f"Response {path} has shape {h.shape}, expected {expected_shape}")
    if completed.shape != (tx_indices.size, frequencies_hz.size) or not bool(np.all(completed)):
        raise RuntimeError(f"Response {path} is incomplete")
    if not np.all(np.isfinite(h.real)) or not np.all(np.isfinite(h.imag)):
        raise RuntimeError(f"Response {path} contains non-finite complex values")
    return FrequencyResponse(
        path=path,
        sample_id=sample_id,
        h=h,
        completed_mask=completed,
        tx_indices=tx_indices,
        rx_indices=rx_indices,
        frequencies_hz=frequencies_hz,
    )


def assert_axes(
    response: FrequencyResponse,
    *,
    tx_indices: np.ndarray,
    rx_indices: np.ndarray,
    frequencies_hz: np.ndarray,
) -> None:
    if not np.array_equal(response.tx_indices, np.asarray(tx_indices, dtype=np.int32)):
        raise RuntimeError(f"TX axis mismatch for {response.path}")
    if not np.array_equal(response.rx_indices, np.asarray(rx_indices, dtype=np.int32)):
        raise RuntimeError(f"RX axis mismatch for {response.path}")
    if not np.allclose(
        response.frequencies_hz,
        np.asarray(frequencies_hz, dtype=np.float64),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise RuntimeError(f"Frequency axis mismatch for {response.path}")


def frequency_floor(h0: np.ndarray, quantile: float) -> np.ndarray:
    array = np.asarray(h0, dtype=np.complex128)
    floor = np.quantile(np.abs(array).reshape(-1, array.shape[-1]), quantile, axis=0)
    return np.maximum(np.asarray(floor, dtype=np.float64), EPS)


def reliability_weights(h0: np.ndarray, floor: np.ndarray, minimum: float) -> np.ndarray:
    power = np.abs(np.asarray(h0, dtype=np.complex128)) ** 2
    denominator = power + np.asarray(floor, dtype=np.float64).reshape((1,) * (power.ndim - 1) + (-1,)) ** 2
    weights = power / np.maximum(denominator, EPS)
    return np.clip(weights, minimum, 1.0).astype(np.float64)


def regularized_ratio(delta_h: np.ndarray, h0: np.ndarray, floor: np.ndarray) -> np.ndarray:
    base = np.asarray(h0, dtype=np.complex128)
    delta = np.asarray(delta_h, dtype=np.complex128)
    floor_array = np.asarray(floor, dtype=np.float64).reshape(
        (1,) * (base.ndim - 1) + (-1,)
    )
    return 1.0 + delta * np.conj(base) / (np.abs(base) ** 2 + floor_array**2)


def complex_rytov(
    damaged_h: np.ndarray,
    healthy_h: np.ndarray,
    floor: np.ndarray,
    frequencies_hz: np.ndarray,
) -> np.ndarray:
    ratio = regularized_ratio(
        np.asarray(damaged_h, dtype=np.complex128) - np.asarray(healthy_h, dtype=np.complex128),
        healthy_h,
        floor,
    )
    amplitude = np.log(np.maximum(np.abs(ratio), EPS))
    if ratio.shape[-1] != np.asarray(frequencies_hz).size:
        raise ValueError("Rytov response and frequency axes differ")
    phase = np.angle(ratio)
    output = amplitude + 1j * phase
    if not np.all(np.isfinite(output.real)) or not np.all(np.isfinite(output.imag)):
        raise RuntimeError("Complex Rytov observation contains non-finite values")
    return output


def rytov_validity_weights(
    damaged_h: np.ndarray,
    healthy_h: np.ndarray,
    floor: np.ndarray,
    *,
    minimum_reliability: float,
    minimum_ratio_magnitude: float,
    phase_branch_margin_rad: float,
) -> tuple[np.ndarray, dict[str, float]]:
    ratio = regularized_ratio(
        np.asarray(damaged_h, dtype=np.complex128) - np.asarray(healthy_h, dtype=np.complex128),
        healthy_h,
        floor,
    )
    reliability = reliability_weights(healthy_h, floor, minimum_reliability)
    magnitude = np.abs(ratio)
    phase = np.abs(np.angle(ratio))
    valid_magnitude = magnitude >= minimum_ratio_magnitude
    valid_phase = phase <= np.pi - phase_branch_margin_rad
    weights = reliability * valid_magnitude * valid_phase
    diagnostics = {
        "retained_ratio_magnitude_fraction": float(np.mean(valid_magnitude)),
        "retained_phase_branch_fraction": float(np.mean(valid_phase)),
        "retained_combined_fraction": float(np.mean(weights > 0.0)),
        "minimum_ratio_magnitude": float(np.min(magnitude)),
        "maximum_abs_phase_rad": float(np.max(phase)),
    }
    return weights.astype(np.float64), diagnostics


def healthy_rotational_relative_l2(h0: np.ndarray) -> float:
    array = np.asarray(h0, dtype=np.complex128)
    if array.ndim != 3 or array.shape[0] != array.shape[1]:
        raise ValueError("Healthy response must have square TX/RX ring axes")
    reference = array[0]
    predicted = np.empty_like(array)
    for tx_slot in range(array.shape[0]):
        for rx_slot in range(array.shape[1]):
            predicted[tx_slot, rx_slot] = reference[(rx_slot - tx_slot) % array.shape[1]]
    return float(np.linalg.norm(array - predicted) / max(np.linalg.norm(array), EPS))


def linearized_rytov_factor(h0: np.ndarray, floor: np.ndarray) -> np.ndarray:
    base = np.asarray(h0, dtype=np.complex128)
    floor_array = np.asarray(floor, dtype=np.float64).reshape(
        (1,) * (base.ndim - 1) + (-1,)
    )
    return np.conj(base) / (np.abs(base) ** 2 + floor_array**2)

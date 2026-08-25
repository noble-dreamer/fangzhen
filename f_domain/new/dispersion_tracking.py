"""Track pipe eigenmodes and derive dispersion sensitivities with NumPy only."""

from __future__ import annotations

import numpy as np


def linear_sum_assignment(cost: np.ndarray) -> np.ndarray:
    """Return the minimum-cost column for every row using Hungarian assignment."""
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1] or cost.shape[0] == 0:
        raise ValueError("Assignment cost must be a nonempty square matrix")
    if not np.isfinite(cost).all():
        raise ValueError("Assignment cost must be finite")
    size = cost.shape[0]
    u = np.zeros(size + 1)
    v = np.zeros(size + 1)
    matched_row = np.zeros(size + 1, dtype=int)
    predecessor = np.zeros(size + 1, dtype=int)
    for row in range(1, size + 1):
        matched_row[0] = row
        min_cost = np.full(size + 1, np.inf)
        used = np.zeros(size + 1, dtype=bool)
        column = 0
        while True:
            used[column] = True
            active_row = matched_row[column]
            candidate = cost[active_row - 1] - u[active_row] - v[1:]
            unused = ~used[1:]
            improved = candidate < min_cost[1:]
            update = unused & improved
            min_cost[1:][update] = candidate[update]
            predecessor[1:][update] = column
            masked = np.where(unused, min_cost[1:], np.inf)
            next_column = int(np.argmin(masked)) + 1
            delta = min_cost[next_column]
            u[matched_row[used]] += delta
            v[used] -= delta
            min_cost[~used] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = predecessor[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break
    assignment = np.empty(size, dtype=int)
    for column in range(1, size + 1):
        assignment[matched_row[column] - 1] = column - 1
    return assignment


def modal_cost(
    previous_frequency: np.ndarray,
    current_frequency: np.ndarray,
    previous_polarization: np.ndarray,
    current_polarization: np.ndarray,
    previous_order: np.ndarray,
    current_order: np.ndarray,
    previous_signature: np.ndarray,
    current_signature: np.ndarray,
) -> np.ndarray:
    previous_valid = np.isfinite(previous_frequency)
    current_valid = np.isfinite(current_frequency)
    mac = np.abs(previous_signature.conj() @ current_signature.T) ** 2
    previous_log_frequency = np.log(np.maximum(np.abs(previous_frequency[:, None]), 1.0))
    current_log_frequency = np.log(np.maximum(np.abs(current_frequency[None, :]), 1.0))
    log_frequency_gap = np.abs(current_log_frequency - previous_log_frequency)
    polarization_gap = 0.5 * np.sum(
        np.abs(previous_polarization[:, None, :] - current_polarization[None, :, :]), axis=2
    )
    order_mismatch = previous_order[:, None] != current_order[None, :]
    cost = (
        1.0 - np.clip(mac, 0.0, 1.0)
        + 1.5 * log_frequency_gap
        + 0.25 * polarization_gap
        + 0.75 * order_mismatch
    )
    both_invalid = ~previous_valid[:, None] & ~current_valid[None, :]
    one_invalid = previous_valid[:, None] ^ current_valid[None, :]
    cost[both_invalid] = 0.0
    cost[one_invalid] = 1e6
    return np.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=1e6)


def _take_modes(array: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return array[indices]
    return array[indices, ...]


def track_modes(
    frequency_hz: np.ndarray,
    polarization: np.ndarray,
    circumferential_order: np.ndarray,
    observability: np.ndarray,
    signature: np.ndarray,
    *,
    frequency_imag_hz: np.ndarray | None = None,
    order_confidence: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    if frequency_hz.ndim != 3:
        raise ValueError("frequency_hz must have shape (thickness, k, mode)")
    thickness_count, k_count, mode_count = frequency_hz.shape
    arrays = {
        "frequency_hz": frequency_hz.copy(),
        "polarization": np.asarray(polarization, dtype=float).copy(),
        "circumferential_order": np.asarray(circumferential_order, dtype=np.int32).copy(),
        "observability": np.asarray(observability, dtype=float).copy(),
        "signature": np.asarray(signature, dtype=complex).copy(),
        "frequency_imag_hz": np.asarray(
            np.zeros_like(frequency_hz) if frequency_imag_hz is None else frequency_imag_hz,
            dtype=float,
        ).copy(),
        "order_confidence": np.asarray(
            np.full_like(frequency_hz, np.nan) if order_confidence is None else order_confidence,
            dtype=float,
        ).copy(),
    }
    residual = np.full_like(frequency_hz, np.nan)
    thickness_residual = np.full_like(frequency_hz, np.nan)
    for thickness_index in range(thickness_count):
        initial = np.argsort(np.nan_to_num(arrays["frequency_hz"][thickness_index, 0], nan=np.inf))
        for key in arrays:
            arrays[key][thickness_index, 0] = _take_modes(arrays[key][thickness_index, 0], initial)
        residual[thickness_index, 0] = np.where(
            np.isfinite(arrays["frequency_hz"][thickness_index, 0]), 0.0, np.nan
        )
        if thickness_index == 0:
            thickness_residual[0] = np.where(
                np.isfinite(arrays["frequency_hz"][0]), 0.0, np.nan
            )
        for k_index in range(1, k_count):
            cost = modal_cost(
                arrays["frequency_hz"][thickness_index, k_index - 1],
                arrays["frequency_hz"][thickness_index, k_index],
                arrays["polarization"][thickness_index, k_index - 1],
                arrays["polarization"][thickness_index, k_index],
                arrays["circumferential_order"][thickness_index, k_index - 1],
                arrays["circumferential_order"][thickness_index, k_index],
                arrays["signature"][thickness_index, k_index - 1],
                arrays["signature"][thickness_index, k_index],
            )
            assignment = linear_sum_assignment(cost)
            for key in arrays:
                arrays[key][thickness_index, k_index] = _take_modes(
                    arrays[key][thickness_index, k_index], assignment
                )
            residual[thickness_index, k_index] = cost[np.arange(mode_count), assignment]

    reference_k = k_count // 2
    for thickness_index in range(1, thickness_count):
        cost = modal_cost(
            arrays["frequency_hz"][thickness_index - 1, reference_k],
            arrays["frequency_hz"][thickness_index, reference_k],
            arrays["polarization"][thickness_index - 1, reference_k],
            arrays["polarization"][thickness_index, reference_k],
            arrays["circumferential_order"][thickness_index - 1, reference_k],
            arrays["circumferential_order"][thickness_index, reference_k],
            arrays["signature"][thickness_index - 1, reference_k],
            arrays["signature"][thickness_index, reference_k],
        )
        assignment = linear_sum_assignment(cost)
        for key in arrays:
            arrays[key][thickness_index] = arrays[key][thickness_index, :, assignment].swapaxes(0, 1)
        residual[thickness_index] = residual[thickness_index, :, assignment].T
        assigned_cost = cost[np.arange(mode_count), assignment]
        thickness_residual[thickness_index] = np.broadcast_to(assigned_cost, (k_count, mode_count))
    arrays["tracking_residual"] = residual
    arrays["thickness_tracking_residual"] = thickness_residual
    total_residual = np.nan_to_num(residual, nan=50.0) + np.nan_to_num(thickness_residual, nan=50.0)
    arrays["tracking_confidence"] = np.exp(-np.clip(total_residual, 0.0, 50.0))
    arrays["branch_id"] = np.broadcast_to(np.arange(mode_count), frequency_hz.shape).copy()
    return arrays


def derive_dispersion(
    tracked: dict[str, np.ndarray],
    thickness_mm: np.ndarray,
    kz_rad_m: np.ndarray,
    *,
    residual_threshold: float = 1.1,
) -> dict[str, np.ndarray]:
    frequency = tracked["frequency_hz"]
    thickness_mm = np.asarray(thickness_mm, dtype=float)
    kz_rad_m = np.asarray(kz_rad_m, dtype=float)
    if kz_rad_m.size < 2 or thickness_mm.size < 2:
        raise ValueError("At least two k values and two thickness values are required for derivatives")
    if np.any(np.diff(kz_rad_m) <= 0.0) or np.any(np.diff(thickness_mm) <= 0.0):
        raise ValueError("Thickness and wave-number axes must be strictly increasing")
    omega = 2.0 * np.pi * frequency
    k_order = 2 if kz_rad_m.size >= 3 else 1
    h_order = 2 if thickness_mm.size >= 3 else 1
    group_velocity = np.gradient(omega, kz_rad_m, axis=1, edge_order=k_order)
    df_dh = np.gradient(frequency, thickness_mm, axis=0, edge_order=h_order)
    k_link_ok = tracked["tracking_residual"][:, 1:] <= residual_threshold
    group_valid = np.zeros_like(frequency, dtype=bool)
    group_valid[:, 0] = k_link_ok[:, 0]
    group_valid[:, -1] = k_link_ok[:, -1]
    if kz_rad_m.size > 2:
        group_valid[:, 1:-1] = k_link_ok[:, :-1] & k_link_ok[:, 1:]
    h_link_ok = tracked["thickness_tracking_residual"][1:] <= residual_threshold
    thickness_valid = np.zeros_like(frequency, dtype=bool)
    thickness_valid[0] = h_link_ok[0]
    thickness_valid[-1] = h_link_ok[-1]
    if thickness_mm.size > 2:
        thickness_valid[1:-1] = h_link_ok[:-1] & h_link_ok[1:]
    group_velocity[~group_valid] = np.nan
    df_dh[~thickness_valid] = np.nan
    phase_velocity = np.full_like(frequency, np.nan)
    np.divide(omega, kz_rad_m[None, :, None], out=phase_velocity, where=kz_rad_m[None, :, None] != 0.0)
    dk_dh = np.full_like(frequency, np.nan)
    derivative_valid = group_valid & thickness_valid & np.isfinite(frequency)
    stable = derivative_valid & np.isfinite(group_velocity) & (np.abs(group_velocity) >= 1.0)
    np.divide(-2.0 * np.pi * df_dh, group_velocity, out=dk_dh, where=stable)
    return {
        **tracked,
        "phase_velocity_m_s": phase_velocity,
        "group_velocity_m_s": group_velocity,
        "df_dh_hz_per_mm": df_dh,
        "dk_dh_rad_m_per_mm": dk_dh,
        "group_velocity_valid_mask": group_valid,
        "thickness_derivative_valid_mask": thickness_valid,
        "derivative_valid_mask": derivative_valid,
    }

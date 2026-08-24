"""Extract physically interpretable features from Shell eigenmodes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import shell_dispersion_common as common


COMPONENTS = ("(x*u+y*v)/Rm", "(-y*u+x*v)/Rm", "w")
POLARIZATION_NAMES = ("radial", "circumferential", "axial")


@dataclass(frozen=True)
class ModePointFeatures:
    frequency_hz: np.ndarray
    polarization: np.ndarray
    circumferential_order: np.ndarray
    order_confidence: np.ndarray
    observability: np.ndarray
    signature: np.ndarray
    surface_area_m2: float
    edge_length_m: float


def _sinc(value: np.ndarray | float) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    output = np.ones_like(value)
    nonzero = np.abs(value) > 1e-12
    output[nonzero] = np.sin(value[nonzero]) / value[nonzero]
    return output


def stabilize_degenerate_subspaces(
    frequency_hz: np.ndarray,
    polarization: np.ndarray,
    signature: np.ndarray,
    kz_rad_m: float,
    *,
    max_order: int,
    frequency_tolerance_hz: float = 20.0,
    patch_width_mm: float = 6.0,
    patch_length_mm: float = 27.0,
    mid_radius_mm: float = 155.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Make Fourier features invariant to rotations inside degenerate eigenspaces."""
    count = frequency_hz.size
    orders = np.arange(-max_order, max_order + 1, dtype=int)
    power = np.abs(signature.reshape(count, orders.size, len(COMPONENTS))) ** 2
    stable_power = power.copy()
    stable_polarization = polarization.copy()
    start = 0
    while start < count:
        stop = start + 1
        while stop < count and abs(frequency_hz[stop].real - frequency_hz[stop - 1].real) <= frequency_tolerance_hz:
            stop += 1
        stable_power[start:stop] = power[start:stop].sum(axis=0)
        mean_polarization = polarization[start:stop].mean(axis=0)
        stable_polarization[start:stop] = mean_polarization / mean_polarization.sum()
        start = stop

    stable_signature = stable_power.reshape(count, -1)
    stable_signature /= np.maximum(np.linalg.norm(stable_signature, axis=1, keepdims=True), 1e-30)
    signed_power = stable_power.sum(axis=2)
    order_power = np.empty((count, max_order + 1), dtype=float)
    order_power[:, 0] = signed_power[:, max_order]
    for order in range(1, max_order + 1):
        order_power[:, order] = signed_power[:, max_order - order] + signed_power[:, max_order + order]
    circumferential_order = np.argmax(order_power, axis=1).astype(np.int32)
    order_confidence = np.max(order_power, axis=1) / np.maximum(order_power.sum(axis=1), 1e-30)
    theta_argument = circumferential_order * (patch_width_mm * 1e-3) / (2.0 * mid_radius_mm * 1e-3)
    axial_argument = kz_rad_m * patch_length_mm * 1e-3 / 2.0
    observability = stable_polarization[:, 2] * _sinc(theta_argument) ** 2
    observability *= float(_sinc(axial_argument) ** 2)
    return (
        stable_polarization,
        circumferential_order,
        order_confidence,
        np.clip(observability, 0.0, 1.0),
        stable_signature.astype(complex),
    )


def extract_mode_features(
    model,
    thickness_mm: float,
    kz_rad_m: float,
    *,
    max_order: int = 12,
    patch_width_mm: float = 6.0,
    patch_length_mm: float = 27.0,
    mid_radius_mm: float = 155.0,
) -> ModePointFeatures:
    if max_order < 0:
        raise ValueError("max_order must be nonnegative")
    model.parameter("h_cell", f"{thickness_mm:.12g}[mm]")
    model.parameter("kz", f"{kz_rad_m:.12g}[1/m]")
    model.solve(common.STUDY_NAME)
    frequency = np.atleast_1d(np.asarray(model.evaluate("freq", unit="Hz"), dtype=complex)).reshape(-1)
    valid = np.isfinite(frequency) & (frequency.real > 0.0)
    if not valid.all():
        raise RuntimeError("COMSOL returned invalid eigenfrequencies")
    count = frequency.size

    energy_expr = [f"intop_disp(abs({component})^2)" for component in COMPONENTS]
    orders = np.arange(-max_order, max_order + 1, dtype=int)
    fourier_expr = [
        f"intop_edge(({component})*exp(-1i*({order})*atan2(y,x)))/intop_edge(1)"
        for order in orders
        for component in COMPONENTS
    ]
    expressions = ["intop_disp(1)", "intop_edge(1)/1[m]", *energy_expr, *fourier_expr]
    evaluated = np.asarray(model.evaluate(expressions), dtype=complex)
    if evaluated.shape != (count, len(expressions)):
        raise RuntimeError(
            f"Mode evaluation shape must be {(count, len(expressions))}, got {evaluated.shape}"
        )
    if not np.isfinite(evaluated).all():
        raise RuntimeError("Mode evaluation contains non-finite values")
    area = evaluated[:, 0].real
    edge = evaluated[:, 1].real
    energy = evaluated[:, 2:5].real
    negative_tolerance = np.maximum(np.max(np.abs(energy), axis=1, keepdims=True), 1e-30) * 1e-10
    if np.any(energy < -negative_tolerance):
        raise RuntimeError("Displacement energy contains significant negative values")
    energy = np.maximum(energy, 0.0)
    total_energy = energy.sum(axis=1, keepdims=True)
    if np.any(total_energy <= 0.0):
        raise RuntimeError("Eigenmode displacement energy is zero")
    polarization = energy / total_energy

    coefficients = evaluated[:, 5:].reshape(count, orders.size, len(COMPONENTS))
    signature = coefficients.reshape(count, -1)
    signature_norm = np.linalg.norm(signature, axis=1, keepdims=True)
    if np.any(signature_norm <= 1e-30):
        raise RuntimeError("Eigenmode Fourier signature is zero")
    signature = signature / signature_norm

    sort_order = np.argsort(frequency.real)
    frequency = frequency[sort_order]
    polarization = polarization[sort_order]
    signature = signature[sort_order]
    polarization, circumferential_order, order_confidence, observability, signature = (
        stabilize_degenerate_subspaces(
            frequency,
            polarization,
            signature,
            kz_rad_m,
            max_order=max_order,
            patch_width_mm=patch_width_mm,
            patch_length_mm=patch_length_mm,
            mid_radius_mm=mid_radius_mm,
        )
    )
    return ModePointFeatures(
        frequency_hz=frequency,
        polarization=polarization,
        circumferential_order=circumferential_order,
        order_confidence=order_confidence,
        observability=observability,
        signature=signature,
        surface_area_m2=float(np.median(area)),
        edge_length_m=float(np.median(edge)),
    )

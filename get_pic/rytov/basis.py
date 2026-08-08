"""Periodic theta-z basis and rotational Jacobian expansion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .config import RytovConfig


MID_RADIUS_MM = 155.0
WINDOW_POWER = 3


@dataclass(frozen=True)
class BasisGrid:
    theta_centers_deg: np.ndarray
    z_centers_mm: np.ndarray
    radius_theta_mm: float
    radius_z_mm: float

    @property
    def coefficient_count(self) -> int:
        return int(self.theta_centers_deg.size * self.z_centers_mm.size)


def build_basis_grid(config: RytovConfig) -> BasisGrid:
    return BasisGrid(
        theta_centers_deg=np.linspace(
            0.0, 360.0, config.theta_basis_count, endpoint=False, dtype=np.float64
        ),
        z_centers_mm=np.linspace(
            config.z_min_mm, config.z_max_mm, config.z_basis_count, dtype=np.float64
        ),
        radius_theta_mm=float(config.basis_radius_theta_mm),
        radius_z_mm=float(config.basis_radius_z_mm),
    )


def circular_delta_deg(values: np.ndarray, center: float) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - center + 180.0) % 360.0 - 180.0


def basis_matrix(
    grid: BasisGrid,
    *,
    theta_deg: np.ndarray,
    z_mm: np.ndarray,
) -> np.ndarray:
    theta = np.asarray(theta_deg, dtype=np.float64)
    z = np.asarray(z_mm, dtype=np.float64)
    rows: list[np.ndarray] = []
    for z_center in grid.z_centers_mm:
        dz = z[:, None] - z_center
        for theta_center in grid.theta_centers_deg:
            ds = MID_RADIUS_MM * np.deg2rad(circular_delta_deg(theta[None, :], theta_center))
            component = np.exp(
                -(
                    (ds / grid.radius_theta_mm) ** 2
                    + (dz / grid.radius_z_mm) ** 2
                )
                ** WINDOW_POWER
            )
            rows.append(component.reshape(-1))
    return np.stack(rows, axis=1).astype(np.float32)


def output_axes(image_size: int, length_mm: float = 1000.0) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 360.0, image_size, endpoint=False, dtype=np.float32)
    z = np.linspace(0.0, length_mm, image_size, dtype=np.float32)
    return theta, z


def coefficient_map(
    coefficients_mm: np.ndarray,
    grid: BasisGrid,
    *,
    image_size: int,
    depth_limit_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta, z = output_axes(image_size)
    matrix = basis_matrix(grid, theta_deg=theta, z_mm=z)
    values = matrix @ np.asarray(coefficients_mm, dtype=np.float64)
    image = np.clip(values.reshape(image_size, image_size), 0.0, depth_limit_mm)
    return image.astype(np.float32), theta, z


def difference_operator(
    z_count: int,
    theta_count: int,
    *,
    theta_spacing_mm: float = 1.0,
    z_spacing_mm: float = 1.0,
) -> sparse.csr_matrix:
    if theta_spacing_mm <= 0.0 or z_spacing_mm <= 0.0:
        raise ValueError("Difference-operator spacings must be positive")
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    row = 0
    for z_index in range(z_count):
        for theta_index in range(theta_count):
            first = z_index * theta_count + theta_index
            second = z_index * theta_count + (theta_index + 1) % theta_count
            rows.extend((row, row))
            columns.extend((first, second))
            values.extend((-1.0 / theta_spacing_mm, 1.0 / theta_spacing_mm))
            row += 1
    for z_index in range(z_count - 1):
        for theta_index in range(theta_count):
            first = z_index * theta_count + theta_index
            second = (z_index + 1) * theta_count + theta_index
            rows.extend((row, row))
            columns.extend((first, second))
            values.extend((-1.0 / z_spacing_mm, 1.0 / z_spacing_mm))
            row += 1
    return sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(row, z_count * theta_count),
        dtype=np.float64,
    )


def expand_rotational_response(reference: np.ndarray) -> np.ndarray:
    """Expand TX=1 basis responses to all TX using ideal ring symmetry.

    ``reference`` has axes ``(z_basis, theta_relative, rx_relative, frequency)``.
    The returned array has axes ``(tx, rx, frequency, coefficient)`` with
    coefficient order ``z * theta_count + theta``.
    """

    array = np.asarray(reference)
    if array.ndim != 4:
        raise ValueError("reference must have shape (z, theta, rx, frequency)")
    z_count, theta_count, rx_count, frequency_count = array.shape
    if theta_count != rx_count:
        raise ValueError("Rotational expansion requires theta_count == rx_count")
    output = np.empty(
        (theta_count, rx_count, frequency_count, z_count * theta_count),
        dtype=array.dtype,
    )
    for tx_slot in range(theta_count):
        for rx_slot in range(rx_count):
            relative_rx = (rx_slot - tx_slot) % rx_count
            for z_index in range(z_count):
                for theta_slot in range(theta_count):
                    relative_theta = (theta_slot - tx_slot) % theta_count
                    column = z_index * theta_count + theta_slot
                    output[tx_slot, rx_slot, :, column] = array[
                        z_index, relative_theta, relative_rx, :
                    ]
    return output

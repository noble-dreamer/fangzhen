"""Evaluation-only metrics and previews for full-wave Rytov outputs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from skimage.metrics import structural_similarity


EPS = 1.0e-12


def load_target_mm(label_dir: Path, sample_id: str, shape: tuple[int, int]) -> np.ndarray:
    """Load a formal label only when the caller explicitly requested evaluation."""

    path = label_dir / f"{sample_id}_defect_depth_mm.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    target = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if target.ndim != 2 or not np.all(np.isfinite(target)):
        raise RuntimeError(f"Invalid formal label: {path}")
    if target.shape != shape:
        target_z, target_theta = shape
        source_z, source_theta = target.shape
        z_index = np.rint(np.linspace(0, source_z - 1, target_z)).astype(np.int64)
        theta_index = np.floor(
            np.arange(target_theta, dtype=np.float64) * source_theta / target_theta
        ).astype(np.int64)
        target = target[np.ix_(z_index, np.clip(theta_index, 0, source_theta - 1))]
    return np.asarray(target, dtype=np.float32)


def _pearson(prediction: np.ndarray, target: np.ndarray) -> float:
    x = np.asarray(prediction, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if np.std(x) <= EPS or np.std(y) <= EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _periodic_centroid(
    image: np.ndarray,
    theta_deg: np.ndarray,
    z_mm: np.ndarray,
) -> tuple[float, float]:
    weights = np.maximum(np.asarray(image, dtype=np.float64), 0.0)
    total = float(np.sum(weights))
    if total <= EPS:
        return float("nan"), float("nan")
    theta_rad = np.deg2rad(theta_deg)[None, :]
    theta = math.degrees(
        math.atan2(
            float(np.sum(weights * np.sin(theta_rad)) / total),
            float(np.sum(weights * np.cos(theta_rad)) / total),
        )
    ) % 360.0
    z = float(np.sum(weights * z_mm[:, None]) / total)
    return theta, z


def _surface_distance_mm(
    first: tuple[float, float],
    second: tuple[float, float],
    mid_radius_mm: float,
) -> float:
    if not all(np.isfinite((*first, *second))):
        return float("nan")
    delta_deg = (first[0] - second[0] + 180.0) % 360.0 - 180.0
    return float(math.hypot(mid_radius_mm * math.radians(delta_deg), first[1] - second[1]))


def compute_metrics(
    prediction_mm: np.ndarray,
    target_mm: np.ndarray,
    *,
    theta_deg: np.ndarray,
    z_mm: np.ndarray,
    mid_radius_mm: float,
    depth_limit_mm: float,
    threshold_mm: float,
) -> dict[str, float]:
    prediction = np.asarray(prediction_mm, dtype=np.float64)
    target = np.asarray(target_mm, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    if prediction.shape != (len(z_mm), len(theta_deg)):
        raise ValueError("Image and theta/z axes are inconsistent")
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(target)):
        raise ValueError("Prediction and target must be finite")

    error = prediction - target
    pred_mask = prediction >= threshold_mm
    target_mask = target >= threshold_mm
    intersection = int(np.sum(pred_mask & target_mask))
    union = int(np.sum(pred_mask | target_mask))
    mask_sum = int(np.sum(pred_mask) + np.sum(target_mask))
    prediction_mass = float(np.sum(prediction))
    target_mass = float(np.sum(target))
    top_count = max(1, int(round(0.05 * prediction.size)))
    top_indices = np.argpartition(prediction.reshape(-1), -top_count)[-top_count:]

    pred_centroid = _periodic_centroid(
        np.where(pred_mask, prediction, 0.0), theta_deg, z_mm
    )
    target_centroid = _periodic_centroid(
        np.where(target_mask, target, 0.0), theta_deg, z_mm
    )
    pred_peak = np.unravel_index(int(np.argmax(prediction)), prediction.shape)
    target_peak = np.unravel_index(int(np.argmax(target)), target.shape)
    pred_peak_position = (float(theta_deg[pred_peak[1]]), float(z_mm[pred_peak[0]]))
    target_peak_position = (float(theta_deg[target_peak[1]]), float(z_mm[target_peak[0]]))

    result = {
        "mae_mm": float(np.mean(np.abs(error))),
        "rmse_mm": float(np.sqrt(np.mean(error * error))),
        "ssim": float(
            structural_similarity(
                target,
                prediction,
                data_range=max(float(depth_limit_mm), EPS),
            )
        ),
        "pearson": _pearson(prediction, target),
        "iou": float(intersection / union) if union else float("nan"),
        "dice": float(2 * intersection / mask_sum) if mask_sum else float("nan"),
        "volume_error": float(abs(prediction_mass - target_mass) / max(target_mass, EPS)),
        "top5_hit_rate": float(np.mean(target_mask.reshape(-1)[top_indices])),
        "prediction_mass_in_label": float(
            np.sum(prediction[target_mask]) / max(prediction_mass, EPS)
        ),
        "max_depth_error_mm": float(abs(np.max(prediction) - np.max(target))),
        "prediction_max_mm": float(np.max(prediction)),
        "target_max_mm": float(np.max(target)),
        "prediction_mean_mm": float(np.mean(prediction)),
        "target_mean_mm": float(np.mean(target)),
        "centroid_error_mm": _surface_distance_mm(
            pred_centroid, target_centroid, mid_radius_mm
        ),
        "peak_location_error_mm": _surface_distance_mm(
            pred_peak_position, target_peak_position, mid_radius_mm
        ),
        "physical_limit_exceedance_fraction": float(np.mean(prediction > depth_limit_mm)),
        "defect_region_mae_mm": (
            float(np.mean(np.abs(error[target_mask]))) if np.any(target_mask) else float("nan")
        ),
        "false_positive_mean_mm": (
            float(np.mean(prediction[~target_mask])) if np.any(~target_mask) else float("nan")
        ),
    }
    return result


def _draw_panel(
    figure: Any,
    axis: Any,
    image: np.ndarray,
    *,
    title: str,
    extent: list[float],
    depth_limit_mm: float,
    cmap: str,
) -> None:
    artist = axis.imshow(
        image,
        origin="upper",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        vmin=0.0,
        vmax=depth_limit_mm,
        interpolation="nearest",
    )
    axis.set_title(title)
    axis.set_xlabel("theta (deg)")
    axis.set_ylabel("z (mm)")
    colorbar = figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.03)
    colorbar.set_label("Wall loss depth (mm)")


def save_preview(
    path: Path,
    *,
    prediction_mm: np.ndarray,
    theta_deg: np.ndarray,
    z_mm: np.ndarray,
    depth_limit_mm: float,
    title: str,
    target_mm: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extent = [float(theta_deg[0]), 360.0, float(z_mm[-1]), float(z_mm[0])]
    if target_mm is None:
        figure, axis = plt.subplots(1, 1, figsize=(5.3, 4.5), constrained_layout=True)
        _draw_panel(
            figure,
            axis,
            prediction_mm,
            title="Full-wave linearized Rytov",
            extent=extent,
            depth_limit_mm=depth_limit_mm,
            cmap="viridis",
        )
    else:
        target = np.asarray(target_mm)
        error = np.abs(np.asarray(prediction_mm) - target)
        figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.4), constrained_layout=True)
        _draw_panel(
            figure,
            axes[0],
            prediction_mm,
            title="Full-wave linearized Rytov",
            extent=extent,
            depth_limit_mm=depth_limit_mm,
            cmap="viridis",
        )
        _draw_panel(
            figure,
            axes[1],
            target,
            title="Ground truth",
            extent=extent,
            depth_limit_mm=depth_limit_mm,
            cmap="viridis",
        )
        _draw_panel(
            figure,
            axes[2],
            error,
            title="Absolute error",
            extent=extent,
            depth_limit_mm=depth_limit_mm,
            cmap="magma",
        )
    figure.suptitle(title)
    figure.savefig(path, dpi=170)
    plt.close(figure)

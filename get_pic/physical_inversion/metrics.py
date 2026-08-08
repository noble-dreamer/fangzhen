"""Physical-image metrics and rendering helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from skimage.metrics import structural_similarity


EPS = 1e-12


def _pearson(prediction: np.ndarray, target: np.ndarray) -> float:
    x = prediction.reshape(-1).astype(np.float64)
    y = target.reshape(-1).astype(np.float64)
    if np.std(x) <= EPS or np.std(y) <= EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _periodic_centroid(image: np.ndarray, theta_deg: np.ndarray, z_mm: np.ndarray) -> tuple[float, float]:
    weights = np.maximum(np.asarray(image, dtype=np.float64), 0.0)
    total = float(np.sum(weights))
    if total <= EPS:
        return float("nan"), float("nan")
    theta_rad = np.deg2rad(theta_deg)[None, :]
    sin_mean = float(np.sum(weights * np.sin(theta_rad)) / total)
    cos_mean = float(np.sum(weights * np.cos(theta_rad)) / total)
    theta = math.degrees(math.atan2(sin_mean, cos_mean)) % 360.0
    z = float(np.sum(weights * z_mm[:, None]) / total)
    return theta, z


def _centroid_distance_mm(
    prediction: np.ndarray,
    target: np.ndarray,
    theta_deg: np.ndarray,
    z_mm: np.ndarray,
    mid_radius_mm: float,
    threshold_mm: float,
) -> float:
    pred_weight = np.where(prediction >= threshold_mm, prediction, 0.0)
    target_weight = np.where(target >= threshold_mm, target, 0.0)
    pred_theta, pred_z = _periodic_centroid(pred_weight, theta_deg, z_mm)
    target_theta, target_z = _periodic_centroid(target_weight, theta_deg, z_mm)
    if not all(np.isfinite([pred_theta, pred_z, target_theta, target_z])):
        return float("nan")
    delta_deg = (pred_theta - target_theta + 180.0) % 360.0 - 180.0
    arc_mm = mid_radius_mm * math.radians(delta_deg)
    return float(math.hypot(arc_mm, pred_z - target_z))


def _peak_distance_mm(
    prediction: np.ndarray,
    target: np.ndarray,
    theta_deg: np.ndarray,
    z_mm: np.ndarray,
    mid_radius_mm: float,
) -> float:
    pred_z_index, pred_theta_index = np.unravel_index(np.argmax(prediction), prediction.shape)
    target_z_index, target_theta_index = np.unravel_index(np.argmax(target), target.shape)
    delta_deg = (
        float(theta_deg[pred_theta_index]) - float(theta_deg[target_theta_index]) + 180.0
    ) % 360.0 - 180.0
    arc_mm = mid_radius_mm * math.radians(delta_deg)
    axial_mm = float(z_mm[pred_z_index]) - float(z_mm[target_z_index])
    return float(math.hypot(arc_mm, axial_mm))


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
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(target)):
        raise ValueError("Prediction and target must be finite")
    error = prediction - target
    pred_mask = prediction >= threshold_mm
    target_mask = target >= threshold_mm
    intersection = int(np.sum(pred_mask & target_mask))
    union = int(np.sum(pred_mask | target_mask))
    mask_sum = int(np.sum(pred_mask) + np.sum(target_mask))
    target_volume = float(np.sum(target))
    top_count = max(1, int(round(0.05 * prediction.size)))
    top_indices = np.argpartition(prediction.reshape(-1), -top_count)[-top_count:]
    top5_hit = float(np.mean(target_mask.reshape(-1)[top_indices]))
    prediction_mass = float(np.sum(prediction))
    metrics = {
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
        "iou": float(intersection / union) if union > 0 else float("nan"),
        "dice": float(2 * intersection / mask_sum) if mask_sum > 0 else float("nan"),
        "volume_error": float(abs(np.sum(prediction) - target_volume) / max(abs(target_volume), EPS)),
        "top5_hit_rate": top5_hit,
        "prediction_mass_in_label": (
            float(np.sum(prediction[target_mask]) / max(prediction_mass, EPS))
        ),
        "max_depth_error_mm": float(abs(np.max(prediction) - np.max(target))),
        "prediction_max_mm": float(np.max(prediction)),
        "target_max_mm": float(np.max(target)),
        "prediction_mean_mm": float(np.mean(prediction)),
        "target_mean_mm": float(np.mean(target)),
        "centroid_error_mm": _centroid_distance_mm(
            prediction,
            target,
            theta_deg,
            z_mm,
            mid_radius_mm,
            threshold_mm,
        ),
        "peak_location_error_mm": _peak_distance_mm(
            prediction,
            target,
            theta_deg,
            z_mm,
            mid_radius_mm,
        ),
        "physical_limit_exceedance_fraction": float(np.mean(prediction > depth_limit_mm)),
    }
    if np.any(target_mask):
        metrics["defect_region_mae_mm"] = float(np.mean(np.abs(error[target_mask])))
    else:
        metrics["defect_region_mae_mm"] = float("nan")
    metrics["false_positive_mean_mm"] = (
        float(np.mean(prediction[~target_mask])) if np.any(~target_mask) else float("nan")
    )
    return metrics


def save_preview(
    path: Path,
    *,
    prediction_mm: np.ndarray,
    target_mm: np.ndarray,
    theta_deg: np.ndarray,
    z_mm: np.ndarray,
    depth_limit_mm: float,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extent = [float(theta_deg[0]), 360.0, float(z_mm[-1]), float(z_mm[0])]
    error = np.abs(np.asarray(prediction_mm) - np.asarray(target_mm))
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.4), constrained_layout=True)
    panels = (
        (prediction_mm, "Physical inversion", "viridis", 0.0, depth_limit_mm),
        (target_mm, "Ground truth", "viridis", 0.0, depth_limit_mm),
        (error, "Absolute error", "magma", 0.0, depth_limit_mm),
    )
    for axis, (image, panel_title, cmap, vmin, vmax) in zip(axes, panels):
        artist = axis.imshow(
            image,
            origin="upper",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        axis.set_title(panel_title)
        axis.set_xlabel("theta (deg)")
        axis.set_ylabel("z (mm)")
        colorbar = figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.03)
        colorbar.set_label("Wall loss depth (mm)")
    figure.suptitle(title)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")

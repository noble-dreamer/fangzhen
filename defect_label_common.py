"""Unfolded theta-z defect label generation for simple shell datasets."""

from __future__ import annotations

import json
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_THETA_COUNT = 512
DEFAULT_Z_COUNT = 512
DEFAULT_H_MIN_MM = 1.0
DEFAULT_DEFECT_LOSS_MAX_MM = 5.0
DEFAULT_DEFECT_WINDOW_POWER = 2
DEFAULT_MASK_THRESHOLD_MM = 0.01
DEFAULT_COLORMAP = 'viridis'


@dataclass(frozen=True)
class PipeConfig:
    length_mm: float = 1000.0
    outer_radius_mm: float = 160.0
    inner_radius_mm: float = 150.0

    @property
    def mid_radius_mm(self) -> float:
        return 0.5 * (self.outer_radius_mm + self.inner_radius_mm)

    @property
    def wall_thickness_mm(self) -> float:
        return self.outer_radius_mm - self.inner_radius_mm


DEFAULT_PIPE = PipeConfig()


def pipe_from_metadata(metadata: dict[str, Any] | None = None) -> PipeConfig:
    metadata = metadata or {}
    pipe = metadata.get('model', {}).get('pipe', metadata.get('pipe', {}))
    return PipeConfig(
        length_mm=float(pipe.get('length_mm', DEFAULT_PIPE.length_mm)),
        outer_radius_mm=float(pipe.get('outer_radius_mm', DEFAULT_PIPE.outer_radius_mm)),
        inner_radius_mm=float(pipe.get('inner_radius_mm', DEFAULT_PIPE.inner_radius_mm)),
    )


def sample_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    sample = metadata.get('sample', metadata)
    if not isinstance(sample, dict):
        raise ValueError('metadata does not contain a valid sample object')
    return sample


def safe_stem(value: Any, fallback: str) -> str:
    text = str(value if value not in (None, '') else fallback).strip()
    invalid = '<>:"/\\|?*'
    for char in invalid:
        text = text.replace(char, '_')
    text = text.strip(' ._')
    return text or fallback


def label_stem_from_metadata(metadata: dict[str, Any], fallback: str) -> str:
    model_path = metadata.get('model', {}).get('model_path')
    if model_path not in (None, ''):
        return safe_stem(Path(str(model_path)).stem, fallback)
    for key in ('sample_id', 'id'):
        value = metadata.get(key)
        if value not in (None, ''):
            return safe_stem(value, fallback)
    sample = metadata.get('sample')
    if isinstance(sample, dict):
        for key in ('sample_id', 'id'):
            value = sample.get(key)
            if value not in (None, ''):
                return safe_stem(value, fallback)
    return safe_stem(fallback, fallback)


def sample_id_from_metadata(metadata: dict[str, Any], fallback: str) -> str:
    return label_stem_from_metadata(metadata, fallback)


def theta_z_grid(
    pipe: PipeConfig,
    theta_count: int = DEFAULT_THETA_COUNT,
    z_count: int = DEFAULT_Z_COUNT,
) -> tuple[np.ndarray, np.ndarray]:
    if theta_count < 2:
        raise ValueError('theta_count must be at least 2')
    if z_count < 2:
        raise ValueError('z_count must be at least 2')
    theta_deg = np.linspace(0.0, 360.0, theta_count, endpoint=False, dtype=np.float64)
    z_mm = np.linspace(0.0, pipe.length_mm, z_count, dtype=np.float64)
    return theta_deg, z_mm


def circular_delta_deg(theta_deg: np.ndarray, center_deg: float) -> np.ndarray:
    return (theta_deg - center_deg + 180.0) % 360.0 - 180.0


def _as_positive_float(item: dict[str, Any], key: str) -> float | None:
    value = item.get(key)
    if value is None:
        return None
    value = float(value)
    if value <= 0.0:
        raise ValueError(f'{key} must be positive, got {value}')
    return value


def _radius_theta_mm(item: dict[str, Any]) -> float:
    radius = _as_positive_float(item, 'radius_theta_mm')
    if radius is not None:
        return radius
    diameter = _as_positive_float(item, 'diameter_theta_mm')
    if diameter is not None:
        return 0.5 * diameter
    radius = _as_positive_float(item, 'radius_mm')
    if radius is not None:
        return radius
    diameter = _as_positive_float(item, 'diameter_mm')
    if diameter is not None:
        return 0.5 * diameter
    raise ValueError(f'cannot infer circumferential radius from defect item: {item}')


def _radius_z_mm(item: dict[str, Any]) -> float:
    radius = _as_positive_float(item, 'radius_z_mm')
    if radius is not None:
        return radius
    diameter = _as_positive_float(item, 'diameter_z_mm')
    if diameter is not None:
        return 0.5 * diameter
    radius = _as_positive_float(item, 'radius_mm')
    if radius is not None:
        return radius
    diameter = _as_positive_float(item, 'diameter_mm')
    if diameter is not None:
        return 0.5 * diameter
    raise ValueError(f'cannot infer axial radius from defect item: {item}')


def _component_loss_mm(
    item: dict[str, Any],
    theta_grid_deg: np.ndarray,
    z_grid_mm: np.ndarray,
    pipe: PipeConfig,
) -> np.ndarray:
    theta0 = float(item['theta_deg'])
    z0 = float(item['z_mm'])
    depth = float(item['depth_mm'])
    rt = _radius_theta_mm(item)
    rz = _radius_z_mm(item)
    ds = pipe.mid_radius_mm * np.deg2rad(circular_delta_deg(theta_grid_deg, theta0))
    dz = z_grid_mm - z0
    window = np.exp(-((ds / rt) ** 2 + (dz / rz) ** 2) ** DEFAULT_DEFECT_WINDOW_POWER)
    return depth * window


def build_depth_map(
    sample: dict[str, Any],
    pipe: PipeConfig = DEFAULT_PIPE,
    theta_count: int = DEFAULT_THETA_COUNT,
    z_count: int = DEFAULT_Z_COUNT,
    h_min_mm: float = DEFAULT_H_MIN_MM,
    defect_loss_max_mm: float = DEFAULT_DEFECT_LOSS_MAX_MM,
) -> dict[str, np.ndarray | float | list[dict[str, Any]]]:
    """Build the same thickness-loss field used by the shell model.

    The returned depth map has shape ``(z_count, theta_count)`` so it can be
    used directly as an image: x is theta, y is axial z.
    """

    theta_deg, z_mm = theta_z_grid(pipe, theta_count=theta_count, z_count=z_count)
    theta_grid = theta_deg[None, :]
    z_grid = z_mm[:, None]
    depth_mm = np.zeros((z_count, theta_count), dtype=np.float64)

    defects = list(sample.get('defects', []))
    lobes = list(sample.get('lobes', []))
    for item in [*defects, *lobes]:
        if float(item.get('depth_mm', 0.0)) <= 0.0:
            continue
        depth_mm += _component_loss_mm(item, theta_grid, z_grid, pipe)

    max_wall_loss_mm = max(pipe.wall_thickness_mm - h_min_mm, 0.0)
    if max_wall_loss_mm > 0.0:
        depth_limit_mm = min(max_wall_loss_mm, defect_loss_max_mm)
        depth_mm = np.minimum(depth_mm, depth_limit_mm)
    else:
        depth_limit_mm = 0.0

    return {
        'theta_deg': theta_deg,
        'z_mm': z_mm,
        'depth_mm': depth_mm.astype(np.float32),
        'max_wall_loss_mm': float(max_wall_loss_mm),
        'depth_limit_mm': float(depth_limit_mm),
        'defects': defects,
        'lobes': lobes,
    }


def parula_like_rgb(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    controls = np.asarray(
        [
            [0.00, 49, 34, 145],
            [0.18, 42, 75, 185],
            [0.38, 26, 152, 199],
            [0.58, 32, 190, 170],
            [0.78, 151, 213, 74],
            [1.00, 246, 230, 38],
        ],
        dtype=np.float64,
    )
    x = controls[:, 0]
    channels = [np.interp(values, x, controls[:, channel]) for channel in (1, 2, 3)]
    return np.stack(channels, axis=-1).round().astype(np.uint8)


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    return (
        struct.pack('>I', len(payload))
        + chunk_type
        + payload
        + struct.pack('>I', zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    )


def write_rgb_png(path: Path, rgb: np.ndarray) -> None:
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f'RGB PNG expects shape (height, width, 3), got {image.shape}')
    height, width, _channels = image.shape
    rows = b''.join(b'\x00' + image[row].tobytes() for row in range(height))
    payload = zlib.compress(rows, level=6)
    header = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b'\x89PNG\r\n\x1a\n'
        + _png_chunk(b'IHDR', header)
        + _png_chunk(b'IDAT', payload)
        + _png_chunk(b'IEND', b'')
    )


def save_depth_png_fallback(depth_mm: np.ndarray, path: Path, preview_max_mm: float | None = None) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    if preview_max_mm is None:
        preview_max_mm = float(np.nanmax(depth_mm)) if np.size(depth_mm) else 0.0
    preview_max_mm = max(float(preview_max_mm), 1e-12)
    normalized = np.clip(depth_mm / preview_max_mm, 0.0, 1.0)
    rgb = parula_like_rgb(normalized)
    write_rgb_png(path, rgb)
    return preview_max_mm


def save_depth_png(
    depth_mm: np.ndarray,
    path: Path,
    theta_deg: np.ndarray,
    z_mm: np.ndarray,
    title: str,
    preview_max_mm: float | None = None,
    colormap: str = DEFAULT_COLORMAP,
) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    if preview_max_mm is None:
        preview_max_mm = float(np.nanmax(depth_mm)) if np.size(depth_mm) else 0.0
    preview_max_mm = max(float(preview_max_mm), 1e-12)
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return save_depth_png_fallback(depth_mm, path, preview_max_mm=preview_max_mm)

    z_min = float(np.nanmin(z_mm)) if z_mm.size else 0.0
    z_max = float(np.nanmax(z_mm)) if z_mm.size else float(depth_mm.shape[0] - 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.9), dpi=180, constrained_layout=True)
    image = ax.imshow(
        depth_mm,
        origin='lower',
        extent=[0.0, 360.0, z_min, z_max],
        aspect='auto',
        interpolation='nearest',
        cmap=colormap,
        vmin=0.0,
        vmax=preview_max_mm,
    )
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel('Circumferential angle theta (deg)', fontsize=9)
    ax.set_ylabel('Axial position z (mm)', fontsize=9)
    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(z_min, z_max)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.tick_params(labelsize=8)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.018)
    colorbar.set_label('Wall loss depth (mm)', fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.035)
    plt.close(fig)
    return preview_max_mm


def write_label_package(
    output_dir: Path,
    sample_id: str,
    sample: dict[str, Any],
    pipe: PipeConfig = DEFAULT_PIPE,
    theta_count: int = DEFAULT_THETA_COUNT,
    z_count: int = DEFAULT_Z_COUNT,
    h_min_mm: float = DEFAULT_H_MIN_MM,
    defect_loss_max_mm: float = DEFAULT_DEFECT_LOSS_MAX_MM,
    mask_threshold_mm: float = DEFAULT_MASK_THRESHOLD_MM,
    preview_max_mm: float | None = None,
    write_preview_png: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    label = build_depth_map(
        sample,
        pipe=pipe,
        theta_count=theta_count,
        z_count=z_count,
        h_min_mm=h_min_mm,
        defect_loss_max_mm=defect_loss_max_mm,
    )
    depth_mm = np.asarray(label['depth_mm'], dtype=np.float32)
    max_wall_loss_mm = float(label['max_wall_loss_mm'])
    norm_denominator = max(max_wall_loss_mm, 1e-12)
    depth_norm = np.clip(depth_mm / norm_denominator, 0.0, 1.0).astype(np.float32)
    mask = (depth_mm >= mask_threshold_mm).astype(np.uint8)

    stem = safe_stem(sample_id, 'defect_sample')
    depth_path = output_dir / f'{stem}_defect_depth_mm.npy'
    norm_path = output_dir / f'{stem}_defect_depth_norm.npy'
    mask_path = output_dir / f'{stem}_defect_mask.npy'
    png_path = output_dir / f'{stem}_defect_label.png'
    meta_path = output_dir / f'{stem}_defect_label_metadata.json'

    np.save(depth_path, depth_mm)
    np.save(norm_path, depth_norm)
    np.save(mask_path, mask)
    theta_deg = np.asarray(label['theta_deg'], dtype=np.float32)
    z_mm = np.asarray(label['z_mm'], dtype=np.float32)
    if write_preview_png:
        used_preview_max_mm = save_depth_png(
            depth_mm,
            png_path,
            theta_deg=theta_deg,
            z_mm=z_mm,
            title=f'{stem} defect depth label',
            preview_max_mm=preview_max_mm,
        )
        preview_png = str(png_path)
    else:
        if preview_max_mm is None:
            used_preview_max_mm = max(max_wall_loss_mm, 1e-12)
        else:
            used_preview_max_mm = max(float(preview_max_mm), 1e-12)
        preview_png = None

    metadata = {
        'sample_id': stem,
        'coordinate_system': 'unfolded outer pipe surface, theta-z',
        'array_shape': ['z_index', 'theta_index'],
        'depth_units': 'mm',
        'theta_axis': {
            'units': 'deg',
            'range': [0.0, 360.0],
            'count': int(theta_deg.size),
            'step': float(360.0 / max(theta_deg.size, 1)),
            'endpoint_included': False,
            'image_axis': 'x/columns',
        },
        'z_axis': {
            'units': 'mm',
            'range': [0.0, pipe.length_mm],
            'count': int(z_mm.size),
            'step': float(pipe.length_mm / max(z_mm.size - 1, 1)),
            'image_axis': 'y/rows',
        },
        'pipe': {
            'length_mm': pipe.length_mm,
            'outer_radius_mm': pipe.outer_radius_mm,
            'inner_radius_mm': pipe.inner_radius_mm,
            'mid_radius_mm': pipe.mid_radius_mm,
            'wall_thickness_mm': pipe.wall_thickness_mm,
        },
        'h_min_mm': h_min_mm,
        'max_wall_loss_mm': max_wall_loss_mm,
        'defect_loss_max_mm': defect_loss_max_mm,
        'depth_limit_mm': float(label['depth_limit_mm']),
        'normalization_denominator_mm': norm_denominator,
        'mask_threshold_mm': mask_threshold_mm,
        'preview_max_mm': used_preview_max_mm,
        'preview_colormap': DEFAULT_COLORMAP,
        'formula': (
            f'sum(depth_mm * exp(-(((Rm*dtheta)/rt)^2 + ((z-z0)/rz)^2)^{DEFAULT_DEFECT_WINDOW_POWER})), '
            'clipped by min(h0-h_min, defect_loss_max_mm)'
        ),
        'files': {
            'depth_mm_npy': str(depth_path),
            'depth_norm_npy': str(norm_path),
            'mask_npy': str(mask_path),
            'preview_png': preview_png,
            'metadata_json': str(meta_path),
        },
        'defect_count': len(label['defects']),
        'lobe_count': len(label['lobes']),
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return metadata


def pearson_correlation(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    true = np.asarray(target, dtype=np.float64).reshape(-1)
    valid = np.isfinite(pred) & np.isfinite(true)
    if not np.any(valid):
        return float('nan')
    pred = pred[valid]
    true = true[valid]
    pred_std = float(np.std(pred))
    true_std = float(np.std(true))
    if pred_std <= 0.0 or true_std <= 0.0:
        return float('nan')
    return float(np.corrcoef(pred, true)[0, 1])


def normalized_rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=np.float64)
    true = np.asarray(target, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f'shape mismatch: prediction {pred.shape} vs target {true.shape}')
    valid = np.isfinite(pred) & np.isfinite(true)
    if not np.any(valid):
        return float('nan')
    scale = max(float(np.nanmax(true[valid]) - np.nanmin(true[valid])), 1e-12)
    return float(np.sqrt(np.nanmean((pred[valid] - true[valid]) ** 2)) / scale)


def mask_iou(
    prediction: np.ndarray,
    target: np.ndarray,
    threshold: float = DEFAULT_MASK_THRESHOLD_MM,
) -> float:
    pred_mask = np.asarray(prediction) >= threshold
    true_mask = np.asarray(target) >= threshold
    if pred_mask.shape != true_mask.shape:
        raise ValueError(f'shape mismatch: prediction {pred_mask.shape} vs target {true_mask.shape}')
    union = np.logical_or(pred_mask, true_mask)
    if not np.any(union):
        return float('nan')
    return float(np.logical_and(pred_mask, true_mask).sum() / union.sum())


def compare_prediction(prediction: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, float]:
    if np.asarray(prediction).shape != np.asarray(target).shape:
        raise ValueError(f'shape mismatch: prediction {np.asarray(prediction).shape} vs target {np.asarray(target).shape}')
    return {
        'pearson_correlation': pearson_correlation(prediction, target),
        'normalized_rmse': normalized_rmse(prediction, target),
        'mask_iou': mask_iou(prediction, target, threshold=threshold),
    }

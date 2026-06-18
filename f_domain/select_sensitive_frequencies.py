"""Rank frequency-domain samples by healthy-damaged sensitivity."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import glob
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np


EPS = 1e-30
DEFAULT_RESPONSE_DIR = Path(__file__).resolve().parent / 'output' / 'streaming_dataset_a_frequency_shell' / 'frequency_response'
DEFAULT_DATASET_ROOT = DEFAULT_RESPONSE_DIR.parent
DEFAULT_LABEL_DIR = DEFAULT_DATASET_ROOT / 'labels'
DEFAULT_METADATA_DIR = DEFAULT_DATASET_ROOT / 'metadata'
DEFAULT_HEALTHY_RESPONSE = DEFAULT_RESPONSE_DIR / 'dataset_a_frequency_healthy_H_complex.npz'
DEFAULT_SAMPLE_TEMPLATE = 'dataset_a_frequency_sample_{sample_id:04d}_H_complex.npz'
GET_PIC_ROOT = Path(__file__).resolve().parents[1] / 'get_pic'
if str(GET_PIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GET_PIC_ROOT))

try:
    import coarse_map_common as cm
except Exception:
    cm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Select sensitive frequency points from healthy and damaged H_complex.npz files.'
    )
    parser.add_argument(
        '--healthy',
        type=Path,
        default=DEFAULT_HEALTHY_RESPONSE,
        help='Healthy *_H_complex.npz baseline.',
    )
    parser.add_argument(
        '--damaged',
        type=Path,
        nargs='*',
        default=[],
        help='One or more damaged *_H_complex.npz files.',
    )
    parser.add_argument(
        '--response-dir',
        type=Path,
        default=DEFAULT_RESPONSE_DIR,
        help='Directory containing standard *_H_complex.npz frequency responses.',
    )
    parser.add_argument(
        '--sample-ids',
        nargs='*',
        default=[],
        help='Damaged sample ids/ranges using standard names, e.g. 1,2,5-10 or 1 2 5-10.',
    )
    parser.add_argument(
        '--sample-name-template',
        default=DEFAULT_SAMPLE_TEMPLATE,
        help='Filename template used with --sample-ids; must contain {sample_id}.',
    )
    parser.add_argument(
        '--damaged-glob',
        nargs='*',
        default=[],
        help='Glob pattern(s) for damaged NPZ files. Relative patterns are checked from cwd and --response-dir.',
    )
    parser.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parent / 'output' / 'frequency_selection')
    parser.add_argument('--top-n', type=int, default=15)
    parser.add_argument(
        '--metric',
        choices=(
            'relative_l2',
            'relative_l1',
            'absolute_l2',
            'phase_weighted',
            'physics_tomography',
            'v1_label_guided',
        ),
        default='relative_l2',
    )
    parser.add_argument(
        '--label-dir',
        type=Path,
        default=DEFAULT_LABEL_DIR,
        help='Directory containing *_defect_depth_norm.npy labels; required by --metric v1_label_guided.',
    )
    parser.add_argument(
        '--metadata-dir',
        type=Path,
        default=DEFAULT_METADATA_DIR,
        help='Directory containing sample metadata; used by --metric v1_label_guided for geometry.',
    )
    parser.add_argument(
        '--v1-grid-size',
        type=int,
        default=128,
        help='Low-resolution theta/z grid used by the fast V1 label-guided frequency-selection metric.',
    )
    parser.add_argument(
        '--v1-sigma-ray-mm',
        type=float,
        default=25.0,
        help='Ray-tube sigma used by --metric v1_label_guided.',
    )
    parser.add_argument(
        '--v1-label-top-quantile',
        type=float,
        default=0.80,
        help='Top path-label-overlap quantile used for the contrast term in --metric v1_label_guided.',
    )
    parser.add_argument(
        '--jobs',
        type=int,
        default=1,
        help='Parallel worker count for loading NPZ files. Use 1 for serial loading, 0 for os.cpu_count().',
    )
    parser.add_argument(
        '--born-grid-size',
        type=int,
        default=128,
        help='Theta/z grid size used to build the compressed ray-Born Jacobian for --metric physics_tomography.',
    )
    parser.add_argument(
        '--born-rank',
        type=int,
        default=32,
        help='Low-rank dimension used for the compressed Fisher information matrix.',
    )
    parser.add_argument(
        '--born-sigma-ray-mm',
        type=float,
        default=25.0,
        help='Ray-tube sigma used to build the ray-Born Jacobian.',
    )
    parser.add_argument(
        '--born-info-scale',
        type=float,
        default=1.0,
        help='Scale applied to the Fisher matrix before logdet(I + scale * F).',
    )
    parser.add_argument(
        '--born-phase-weight',
        type=float,
        default=0.50,
        help='Relative weight of wrapped phase perturbation in the complex Rytov observation energy.',
    )
    parser.add_argument(
        '--born-noise-quantile',
        type=float,
        default=0.10,
        help='Lower-tail quantile used as a robust empirical noise-energy floor for Fisher weights.',
    )
    parser.add_argument(
        '--born-weight-clip-quantile',
        type=float,
        default=0.99,
        help='Upper quantile used to clip Fisher path weights and reduce outlier dominance.',
    )
    parser.add_argument(
        '--selection-strategy',
        choices=('top_score', 'greedy_d_optimal'),
        default='greedy_d_optimal',
        help='Frequency subset strategy. greedy_d_optimal is used only when Fisher matrices are available.',
    )
    parser.add_argument(
        '--min-healthy-abs-percentile',
        type=float,
        default=5.0,
        help='Drop frequencies whose healthy mean abs response is below this percentile across frequencies.',
    )
    parser.add_argument(
        '--frequency-min-khz',
        type=float,
        default=None,
        help='Optional lower bound for ranked frequencies.',
    )
    parser.add_argument(
        '--frequency-max-khz',
        type=float,
        default=None,
        help='Optional upper bound for ranked frequencies.',
    )
    parser.add_argument('--prefix', default='frequency_sensitivity')
    return parser.parse_args()


def parse_sample_ids(values: list[str]) -> list[int]:
    ids: list[int] = []
    for value in values:
        for token in str(value).split(','):
            token = token.strip()
            if not token:
                continue
            if '-' in token:
                start_text, stop_text = token.split('-', 1)
                start = int(start_text)
                stop = int(stop_text)
                if stop < start:
                    raise ValueError(f'sample id range stop must be >= start: {token}')
                ids.extend(range(start, stop + 1))
            else:
                ids.append(int(token))
    return list(dict.fromkeys(ids))


def expand_glob(pattern: str, response_dir: Path) -> list[Path]:
    matches = [Path(item) for item in glob.glob(pattern)]
    if matches:
        return sorted(matches)
    return sorted(response_dir.glob(pattern))


def unique_paths(paths: list[Path]) -> list[Path]:
    unique: dict[str, Path] = {}
    for path in paths:
        key = str(path.resolve(strict=False))
        if key not in unique:
            unique[key] = path
    return list(unique.values())


def collect_damaged_paths(args: argparse.Namespace) -> list[Path]:
    paths = list(args.damaged)
    for sample_id in parse_sample_ids(args.sample_ids):
        paths.append(args.response_dir / args.sample_name_template.format(sample_id=sample_id))
    for pattern in args.damaged_glob:
        paths.extend(expand_glob(pattern, args.response_dir))
    paths = unique_paths(paths)
    if not paths:
        raise RuntimeError('No damaged responses selected. Use --damaged, --sample-ids, or --damaged-glob.')
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise RuntimeError('Missing damaged response file(s): ' + ', '.join(str(path) for path in missing))
    return paths


def load_response(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f'Response file does not exist: {path}')
    data = np.load(path, allow_pickle=False)
    required = {'H_real', 'H_imag', 'tx_indices', 'rx_indices', 'frequencies_hz'}
    missing = required.difference(data.files)
    if missing:
        raise RuntimeError(f'{path} is missing fields: {sorted(missing)}')
    h = np.asarray(data['H_real'], dtype=float) + 1j * np.asarray(data['H_imag'], dtype=float)
    completed = np.asarray(data['completed_mask'], dtype=bool) if 'completed_mask' in data.files else np.ones(
        (h.shape[0], h.shape[2]),
        dtype=bool,
    )
    return {
        'path': path,
        'H': h,
        'completed_mask': completed,
        'tx_indices': tuple(int(item) for item in np.asarray(data['tx_indices']).tolist()),
        'rx_indices': tuple(int(item) for item in np.asarray(data['rx_indices']).tolist()),
        'frequencies_hz': tuple(float(item) for item in np.asarray(data['frequencies_hz']).tolist()),
    }


def load_responses_parallel(paths: list[Path], jobs: int) -> list[dict[str, Any]]:
    """Load many NPZ files in parallel. NumPy releases the GIL during decompression."""
    if jobs == 0:
        jobs = os.cpu_count() or 1
    jobs = max(1, min(int(jobs), len(paths) if paths else 1))
    if jobs == 1 or len(paths) <= 1:
        return [load_response(path) for path in paths]
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        return list(executor.map(load_response, paths))


def sample_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith('_H_complex.npz'):
        return name[:-len('_H_complex.npz')]
    return path.stem


def label_path_for_response(path: Path, label_dir: Path) -> Path:
    return label_dir / f'{sample_id_from_path(path)}_defect_depth_norm.npy'


def metadata_path_for_response(path: Path, metadata_dir: Path) -> Path:
    return metadata_dir / f'{sample_id_from_path(path)}.json'


def resample_label_nearest(label: np.ndarray, z_count: int, theta_count: int) -> np.ndarray:
    source = np.asarray(label, dtype=np.float64)
    if source.ndim != 2:
        raise RuntimeError(f'label must be 2D, got {source.shape}')
    z_indices = np.linspace(0, source.shape[0] - 1, z_count).round().astype(int)
    theta_indices = np.linspace(0, source.shape[1] - 1, theta_count).round().astype(int)
    return source[np.ix_(z_indices, theta_indices)]


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return float('nan')
    x = x[valid]
    y = y[valid]
    if float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def v1_path_label_overlap(
    *,
    damaged_path: Path,
    label_dir: Path,
    metadata_dir: Path,
    tx_indices: tuple[int, ...],
    rx_indices: tuple[int, ...],
    grid_size: int,
    sigma_ray_mm: float,
) -> dict[str, Any]:
    if cm is None:
        raise RuntimeError('coarse_map_common.py is required for --metric v1_label_guided')
    label_path = label_path_for_response(damaged_path, label_dir)
    if not label_path.exists():
        raise RuntimeError(f'Missing label for {damaged_path}: {label_path}')
    metadata_path = metadata_path_for_response(damaged_path, metadata_dir)
    metadata = cm.read_json(metadata_path) if metadata_path.exists() else {}
    geometry = cm.geometry_from_metadata(metadata)
    config = cm.CoarseMapConfig(
        theta_count=grid_size,
        z_count=grid_size,
        sigma_ray_mm=sigma_ray_mm,
    )
    theta_deg, z_mm = cm.label_grid(
        geometry,
        theta_count=config.theta_count,
        z_count=config.z_count,
    )
    label = resample_label_nearest(np.load(label_path), config.z_count, config.theta_count)
    overlaps: list[float] = []
    rays: list[tuple[int, int, int]] = []
    for tx_index in tx_indices:
        tx = geometry.tx_positions[tx_index]
        for rx_index in rx_indices:
            rx = geometry.rx_positions[rx_index]
            for order in config.helical_orders:
                kernel, _tube, _length = cm.ray_kernel(theta_deg, z_mm, geometry, tx, rx, order, config)
                weight_sum = float(np.sum(kernel))
                if weight_sum <= 0.0:
                    overlap = 0.0
                else:
                    overlap = float(np.sum(kernel * label) / weight_sum)
                overlaps.append(overlap)
                rays.append((tx_index, rx_index, order))
    return {
        'label_path': label_path,
        'metadata_path': metadata_path,
        'overlap': np.asarray(overlaps, dtype=np.float64),
        'rays': rays,
    }


def v1_label_guided_frequency_metric(
    healthy: dict[str, Any],
    damaged: dict[str, Any],
    *,
    freq_index: int,
    path_context: dict[str, Any],
    top_quantile: float,
) -> dict[str, float]:
    h0 = healthy['H'][:, :, freq_index]
    hd = damaged['H'][:, :, freq_index]
    completed_tx = healthy['completed_mask'][:, freq_index] & damaged['completed_mask'][:, freq_index]
    tx_lookup = {value: index for index, value in enumerate(healthy['tx_indices'])}
    rx_lookup = {value: index for index, value in enumerate(healthy['rx_indices'])}
    ray_values = []
    ray_overlap = []
    for ray_index, (tx_id, rx_id, _order) in enumerate(path_context['rays']):
        tx_i = tx_lookup[tx_id]
        rx_i = rx_lookup[rx_id]
        if not completed_tx[tx_i]:
            continue
        a = h0[tx_i, rx_i]
        b = hd[tx_i, rx_i]
        if not (np.isfinite(np.real(a)) and np.isfinite(np.imag(a)) and np.isfinite(np.real(b)) and np.isfinite(np.imag(b))):
            continue
        rel = abs(b - a) / (abs(a) + EPS)
        phase = abs(np.angle(b * np.conj(a)))
        # log1p bounds very large relative changes while preserving ordering.
        value = math.log1p(float(rel)) * (1.0 + 0.25 * float(abs(math.sin(phase))))
        ray_values.append(value)
        ray_overlap.append(float(path_context['overlap'][ray_index]))
    if len(ray_values) < 4:
        return {
            'score': float('nan'),
            'path_corr': float('nan'),
            'path_contrast': float('nan'),
            'energy_on_label_paths': float('nan'),
            'valid_ray_count': float(len(ray_values)),
        }
    values = np.asarray(ray_values, dtype=np.float64)
    overlaps = np.asarray(ray_overlap, dtype=np.float64)
    corr = safe_pearson(values, overlaps)
    corr_positive = max(float(corr), 0.0) if np.isfinite(corr) else 0.0
    threshold = float(np.nanquantile(overlaps, top_quantile))
    inside = overlaps >= threshold
    if np.sum(inside) < 2 or np.sum(~inside) < 2:
        contrast = 0.0
    else:
        inside_mean = float(np.nanmean(values[inside]))
        outside_mean = float(np.nanmean(values[~inside]))
        contrast = max(0.0, (inside_mean - outside_mean) / (abs(inside_mean) + abs(outside_mean) + EPS))
    max_overlap = max(float(np.nanmax(overlaps)), EPS)
    energy_on_label = float(np.sum(values * overlaps) / (np.sum(values) * max_overlap + EPS))
    energy_on_label = float(np.clip(energy_on_label, 0.0, 1.0))
    score = 0.50 * corr_positive + 0.30 * contrast + 0.20 * energy_on_label
    return {
        'score': float(score),
        'path_corr': float(corr),
        'path_contrast': float(contrast),
        'energy_on_label_paths': float(energy_on_label),
        'valid_ray_count': float(len(ray_values)),
    }


def assert_compatible(healthy: dict[str, Any], damaged: dict[str, Any]) -> None:
    for key in ('tx_indices', 'rx_indices', 'frequencies_hz'):
        if healthy[key] != damaged[key]:
            raise RuntimeError(
                f'{damaged["path"]} is not compatible with healthy baseline for {key}: '
                f'{damaged[key]} != {healthy[key]}'
            )
    if healthy['H'].shape != damaged['H'].shape:
        raise RuntimeError(f'{damaged["path"]} shape {damaged["H"].shape} != healthy shape {healthy["H"].shape}')


def finite_case_mask(healthy: dict[str, Any], damaged: dict[str, Any], freq_index: int) -> np.ndarray:
    healthy_h = healthy['H'][:, :, freq_index]
    damaged_h = damaged['H'][:, :, freq_index]
    completed = healthy['completed_mask'][:, freq_index] & damaged['completed_mask'][:, freq_index]
    finite_tx = np.all(np.isfinite(np.real(healthy_h)) & np.isfinite(np.imag(healthy_h)), axis=1)
    finite_tx &= np.all(np.isfinite(np.real(damaged_h)) & np.isfinite(np.imag(damaged_h)), axis=1)
    return completed & finite_tx


def sample_frequency_metric(
    healthy_h: np.ndarray,
    damaged_h: np.ndarray,
    *,
    metric: str,
) -> float:
    delta = damaged_h - healthy_h
    healthy_abs = np.abs(healthy_h)
    if metric == 'relative_l2':
        return float(np.linalg.norm(delta.ravel()) / (np.linalg.norm(healthy_h.ravel()) + EPS))
    if metric == 'relative_l1':
        return float(np.sum(np.abs(delta)) / (np.sum(healthy_abs) + EPS))
    if metric == 'absolute_l2':
        return float(np.linalg.norm(delta.ravel()))
    if metric == 'phase_weighted':
        phase_diff = np.angle(damaged_h * np.conj(healthy_h))
        relative_amp = np.abs(delta) / (healthy_abs + EPS)
        return float(np.nanmean(relative_amp * (1.0 + np.abs(np.sin(phase_diff)))))
    if metric == 'physics_tomography':
        return legacy_physics_tomography_frequency_metric(healthy_h, damaged_h)['score']
    raise ValueError(metric)


def participation_ratio(values: np.ndarray) -> float:
    """Effective normalized participation count in [0, 1]."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = np.where(np.isfinite(array), np.maximum(array, 0.0), 0.0)
    total = float(np.sum(array))
    square = float(np.sum(array * array))
    n = int(array.size)
    if n <= 0 or total <= 0.0 or square <= 0.0:
        return 0.0
    return float(np.clip((total * total) / (n * square + EPS), 0.0, 1.0))


def robust_contrast(values: np.ndarray) -> float:
    """Structured path contrast: low for uniform changes, high for path-dependent changes."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0
    q50 = float(np.nanquantile(array, 0.50))
    q90 = float(np.nanquantile(array, 0.90))
    return float(np.clip((q90 - q50) / (q90 + q50 + EPS), 0.0, 1.0))


def build_ray_born_context(
    healthy: dict[str, Any],
    *,
    metadata_dir: Path,
    grid_size: int,
    sigma_ray_mm: float,
    rank: int,
) -> dict[str, Any]:
    """Build a compressed ray-Born Jacobian used by the Fisher score.

    The full image-space Jacobian has one row per tx-rx-order ray and one
    column per theta/z pixel. For selection we only need path-space Gram
    products, so this function computes an orthonormal low-rank embedding of
    those ray kernels once and reuses it for all frequencies and samples.
    """
    if cm is None:
        raise RuntimeError('coarse_map_common.py is required for --metric physics_tomography')
    metadata_path = metadata_dir / 'dataset_a_frequency_healthy.json'
    metadata = cm.read_json(metadata_path) if metadata_path.exists() else {}
    geometry = cm.geometry_from_metadata(metadata)
    config = cm.CoarseMapConfig(
        theta_count=grid_size,
        z_count=grid_size,
        sigma_ray_mm=sigma_ray_mm,
    )
    theta_deg, z_mm = cm.label_grid(geometry, theta_count=config.theta_count, z_count=config.z_count)
    tx_indices = healthy['tx_indices']
    rx_indices = healthy['rx_indices']
    ray_rows: list[np.ndarray] = []
    ray_to_case: list[tuple[int, int, int, int]] = []
    tx_lookup = {value: index for index, value in enumerate(tx_indices)}
    rx_lookup = {value: index for index, value in enumerate(rx_indices)}
    for tx_id in tx_indices:
        if tx_id not in geometry.tx_positions:
            raise RuntimeError(f'Missing tx geometry for PZT {tx_id}')
        tx = geometry.tx_positions[tx_id]
        tx_i = tx_lookup[tx_id]
        for rx_id in rx_indices:
            if rx_id not in geometry.rx_positions:
                raise RuntimeError(f'Missing rx geometry for PZT {rx_id}')
            rx = geometry.rx_positions[rx_id]
            rx_i = rx_lookup[rx_id]
            for order in config.helical_orders:
                kernel, _tube, ray_length = cm.ray_kernel(theta_deg, z_mm, geometry, tx, rx, order, config)
                flat = np.asarray(kernel, dtype=np.float64).reshape(-1)
                norm = float(np.linalg.norm(flat))
                if norm <= 0.0 or not np.isfinite(norm):
                    continue
                ray_rows.append((flat / norm).astype(np.float32))
                ray_to_case.append((tx_i, rx_i, int(order), float(ray_length)))
    if not ray_rows:
        raise RuntimeError('The ray-Born Jacobian has no valid ray rows.')
    jacobian = np.stack(ray_rows, axis=0).astype(np.float32)
    requested_rank = max(1, min(int(rank), jacobian.shape[0], jacobian.shape[1]))
    # The path-space Gram matrix avoids an expensive pixel-scale SVD. The
    # resulting embedding A satisfies A A^T ~= J J^T in the retained subspace,
    # so each frequency can build a small Fisher matrix A^T W A.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        gram = jacobian @ jacobian.T
        eig_values, eig_vectors = np.linalg.eigh(0.5 * (gram + gram.T))
    order = np.argsort(eig_values)[::-1]
    eig_values = np.clip(eig_values[order[:requested_rank]], 0.0, None)
    eig_vectors = eig_vectors[:, order[:requested_rank]]
    embedding = (eig_vectors * np.sqrt(eig_values)[None, :]).astype(np.float64)
    ray_to_case_array = np.asarray(ray_to_case, dtype=np.float64)
    return {
        'embedding': embedding,
        'tx_case_indices': ray_to_case_array[:, 0].astype(np.int64),
        'rx_case_indices': ray_to_case_array[:, 1].astype(np.int64),
        'orders': ray_to_case_array[:, 2].astype(np.int64),
        'ray_lengths_mm': ray_to_case_array[:, 3].astype(np.float64),
        'rank': requested_rank,
        'grid_size': int(grid_size),
        'sigma_ray_mm': float(sigma_ray_mm),
    }


def rytov_observation_energy(
    healthy_h: np.ndarray,
    damaged_stack: np.ndarray,
    valid_stack: np.ndarray,
    *,
    phase_weight: float,
) -> np.ndarray:
    """Estimate per-sample, per-path perturbation energy from complex FRF.

    Under a first-order Born/Rytov approximation, log(Hd/H0) is a linearized
    projection of the material perturbation plus noise. We use the squared
    real log-amplitude perturbation and a wrapped phase perturbation as the
    empirical observation energy.
    """
    h0 = healthy_h[None, :, :, :]
    ratio = damaged_stack / (h0 + EPS)
    log_amp = np.log(np.abs(ratio) + EPS)
    phase = np.angle(ratio)
    energy = log_amp * log_amp + float(phase_weight) * phase * phase
    energy = np.where(valid_stack, energy, np.nan)
    return energy.astype(np.float64, copy=False)


def robust_noise_floor(values: np.ndarray, quantile: float) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 1.0
    q = float(np.nanquantile(array, np.clip(quantile, 0.0, 1.0)))
    median = float(np.nanmedian(array))
    return max(q, 1e-6 * median, EPS)


def fisher_logdet(matrix: np.ndarray, scale: float) -> float:
    sym = 0.5 * (matrix + matrix.T)
    sign, logdet = np.linalg.slogdet(np.eye(sym.shape[0], dtype=np.float64) + float(scale) * sym)
    if sign <= 0 or not np.isfinite(logdet):
        return float('nan')
    return float(logdet)


def physics_tomography_rank_frequencies(
    healthy: dict[str, Any],
    damaged_items: list[dict[str, Any]],
    *,
    min_healthy_abs_percentile: float,
    frequency_min_khz: float | None,
    frequency_max_khz: float | None,
    metadata_dir: Path,
    born_grid_size: int,
    born_rank: int,
    born_sigma_ray_mm: float,
    born_info_scale: float,
    born_phase_weight: float,
    born_noise_quantile: float,
    born_weight_clip_quantile: float,
) -> tuple[list[dict[str, Any]], list[np.ndarray | None], dict[str, Any]]:
    h0 = np.asarray(healthy['H'], dtype=np.complex128)
    damaged_stack = np.stack([np.asarray(item['H'], dtype=np.complex128) for item in damaged_items], axis=0)
    healthy_completed = np.asarray(healthy['completed_mask'], dtype=bool)[None, :, None, :]
    damaged_completed = np.stack([np.asarray(item['completed_mask'], dtype=bool) for item in damaged_items], axis=0)
    damaged_completed = damaged_completed[:, :, None, :]
    finite = np.isfinite(np.real(damaged_stack)) & np.isfinite(np.imag(damaged_stack))
    finite &= np.isfinite(np.real(h0))[None, :, :, :] & np.isfinite(np.imag(h0))[None, :, :, :]
    valid_stack = finite & healthy_completed & damaged_completed
    observation_energy = rytov_observation_energy(
        h0,
        damaged_stack,
        valid_stack,
        phase_weight=born_phase_weight,
    )
    context = build_ray_born_context(
        healthy,
        metadata_dir=metadata_dir,
        grid_size=born_grid_size,
        sigma_ray_mm=born_sigma_ray_mm,
        rank=born_rank,
    )
    embedding = np.asarray(context['embedding'], dtype=np.float64)
    tx_indices = np.asarray(context['tx_case_indices'], dtype=np.int64)
    rx_indices = np.asarray(context['rx_case_indices'], dtype=np.int64)
    ray_count, compressed_rank = embedding.shape
    frequencies_hz = healthy['frequencies_hz']
    healthy_mean_abs = np.nanmean(np.abs(h0), axis=(0, 1))
    floor = float(np.nanpercentile(healthy_mean_abs, min_healthy_abs_percentile))
    noise_floor = robust_noise_floor(observation_energy, born_noise_quantile)
    rows: list[dict[str, Any]] = []
    items: list[tuple[dict[str, Any], np.ndarray | None]] = []
    weight_clip_quantile = float(np.clip(born_weight_clip_quantile, 0.50, 1.0))
    for freq_index, frequency_hz in enumerate(frequencies_hz):
        frequency_khz = frequency_hz / 1000.0
        if frequency_min_khz is not None and frequency_khz < frequency_min_khz:
            continue
        if frequency_max_khz is not None and frequency_khz > frequency_max_khz:
            continue
        common = {
            'frequency_hz': frequency_hz,
            'frequency_khz': frequency_khz,
            'sample_count': 0,
            'valid_tx_count_min': 0,
            'valid_ray_count_min': '',
            'path_corr_mean': '',
            'path_contrast_mean': '',
            'energy_on_label_paths_mean': '',
            'healthy_mean_abs': float(healthy_mean_abs[freq_index]),
        }
        if healthy_mean_abs[freq_index] < floor:
            row = {
                **common,
                'sensitivity_mean': float('nan'),
                'sensitivity_median': float('nan'),
                'sensitivity_min': float('nan'),
                'sensitivity_max': float('nan'),
                'physics_fisher_logdet': '',
                'physics_fisher_trace': '',
                'physics_fisher_min_eig': '',
                'physics_effective_rank': '',
                'physics_observation_energy_mean': '',
                'physics_observation_energy_median': '',
                'physics_path_participation_mean': '',
                'physics_path_contrast_mean': '',
                'physics_tx_balance_mean': '',
                'physics_rx_balance_mean': '',
                'excluded_reason': 'healthy_response_below_floor',
            }
            rows.append(row)
            items.append((row, None))
            continue
        energy_sf = observation_energy[:, :, :, freq_index]
        path_energy = np.nanmean(energy_sf[:, tx_indices, rx_indices], axis=0)
        finite_weights = np.isfinite(path_energy)
        if not np.any(finite_weights):
            row = {
                **common,
                'sensitivity_mean': float('nan'),
                'sensitivity_median': float('nan'),
                'sensitivity_min': float('nan'),
                'sensitivity_max': float('nan'),
                'physics_fisher_logdet': '',
                'physics_fisher_trace': '',
                'physics_fisher_min_eig': '',
                'physics_effective_rank': '',
                'physics_observation_energy_mean': '',
                'physics_observation_energy_median': '',
                'physics_path_participation_mean': '',
                'physics_path_contrast_mean': '',
                'physics_tx_balance_mean': '',
                'physics_rx_balance_mean': '',
                'excluded_reason': 'no_valid_cases',
            }
            rows.append(row)
            items.append((row, None))
            continue
        clipped = np.array(path_energy, dtype=np.float64, copy=True)
        upper = float(np.nanquantile(clipped[finite_weights], weight_clip_quantile))
        clipped = np.where(finite_weights, np.clip(clipped, 0.0, max(upper, EPS)), 0.0)
        path_weights = clipped / (noise_floor + EPS)
        weighted_embedding = embedding * np.sqrt(path_weights[:, None])
        fisher = weighted_embedding.T @ weighted_embedding
        logdet = fisher_logdet(fisher, born_info_scale)
        eig = np.linalg.eigvalsh(0.5 * (fisher + fisher.T))
        eig = np.clip(eig, 0.0, None)
        eig_sum = float(np.sum(eig))
        eig_square = float(np.sum(eig * eig))
        effective_rank = float((eig_sum * eig_sum) / (eig_square + EPS)) if eig_sum > 0.0 else 0.0
        tx_energy = np.zeros(len(healthy['tx_indices']), dtype=np.float64)
        rx_energy = np.zeros(len(healthy['rx_indices']), dtype=np.float64)
        np.add.at(tx_energy, tx_indices, path_weights)
        np.add.at(rx_energy, rx_indices, path_weights)
        valid_case_counts = np.sum(np.any(valid_stack[:, :, :, freq_index], axis=2), axis=1)
        valid_sample_count = int(np.sum(valid_case_counts > 0))
        row = {
            **common,
            'sensitivity_mean': logdet,
            'sensitivity_median': logdet,
            'sensitivity_min': float(np.nanmin(path_weights[finite_weights])),
            'sensitivity_max': float(np.nanmax(path_weights[finite_weights])),
            'sample_count': valid_sample_count,
            'valid_tx_count_min': int(np.min(valid_case_counts)) if valid_case_counts.size else 0,
            'valid_ray_count_min': int(np.sum(finite_weights)),
            'physics_fisher_logdet': logdet,
            'physics_fisher_trace': float(np.trace(fisher)),
            'physics_fisher_min_eig': float(np.min(eig)) if eig.size else 0.0,
            'physics_effective_rank': effective_rank,
            'physics_observation_energy_mean': float(np.nanmean(path_energy[finite_weights])),
            'physics_observation_energy_median': float(np.nanmedian(path_energy[finite_weights])),
            'physics_path_participation_mean': participation_ratio(path_weights),
            'physics_path_contrast_mean': robust_contrast(path_weights),
            'physics_tx_balance_mean': participation_ratio(tx_energy),
            'physics_rx_balance_mean': participation_ratio(rx_energy),
            'healthy_mean_abs': float(healthy_mean_abs[freq_index]),
            'excluded_reason': '',
        }
        rows.append(row)
        items.append((row, fisher))
    items.sort(key=lambda item: (-np.nan_to_num(item[0]['sensitivity_mean'], nan=-np.inf), item[0]['frequency_hz']))
    rows = [row for row, _fisher in items]
    sorted_fisher = [fisher for _row, fisher in items]
    metadata = {
        'score_model': 'ray_born_rytov_fisher_information',
        'born_grid_size': int(born_grid_size),
        'born_rank': int(context['rank']),
        'born_sigma_ray_mm': float(born_sigma_ray_mm),
        'born_info_scale': float(born_info_scale),
        'born_phase_weight': float(born_phase_weight),
        'born_noise_quantile': float(born_noise_quantile),
        'born_weight_clip_quantile': float(born_weight_clip_quantile),
        'born_noise_floor': float(noise_floor),
        'born_ray_count': int(ray_count),
    }
    return rows, sorted_fisher, metadata


def greedy_d_optimal_selection(
    ranked_rows: list[dict[str, Any]],
    fisher_matrices: list[np.ndarray | None],
    *,
    top_n: int,
    scale: float,
) -> list[dict[str, Any]]:
    candidates = [
        (index, row, fisher_matrices[index])
        for index, row in enumerate(ranked_rows)
        if not row.get('excluded_reason') and np.isfinite(row.get('sensitivity_mean', float('nan')))
        and fisher_matrices[index] is not None
    ]
    if not candidates:
        return []
    rank = candidates[0][2].shape[0]
    current = np.zeros((rank, rank), dtype=np.float64)
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    current_score = fisher_logdet(current, scale)
    for step in range(min(top_n, len(candidates))):
        best_item = None
        best_gain = -float('inf')
        best_score = -float('inf')
        for index, row, fisher in candidates:
            if index in used:
                continue
            score = fisher_logdet(current + fisher, scale)
            gain = score - current_score
            if gain > best_gain or (math.isclose(gain, best_gain) and score > best_score):
                best_item = (index, row, fisher)
                best_gain = gain
                best_score = score
        if best_item is None:
            break
        index, row, fisher = best_item
        used.add(index)
        current += fisher
        current_score = best_score
        selected_row = dict(row)
        selected_row['greedy_step'] = step + 1
        selected_row['greedy_logdet_gain'] = float(best_gain)
        selected_row['greedy_cumulative_logdet'] = float(current_score)
        selected.append(selected_row)
    return selected


def legacy_physics_tomography_frequency_metric(healthy_h: np.ndarray, damaged_h: np.ndarray) -> dict[str, float]:
    """Legacy label-free guided-wave tomography frequency score.

    The main CLI path for --metric physics_tomography now uses the Born/Rytov
    Fisher information implementation in physics_tomography_rank_frequencies.
    This helper is kept for old callers that request a single-frequency score.

    The legacy score rewards frequencies that have:
    - robust healthy-damaged relative complex perturbation;
    - phase perturbation, not only amplitude change;
    - enough participating tx-rx paths;
    - non-uniform path contrast for localization;
    - balanced transmitter and receiver participation.
    """
    h0 = np.asarray(healthy_h, dtype=complex)
    hd = np.asarray(damaged_h, dtype=complex)
    valid = np.isfinite(np.real(h0)) & np.isfinite(np.imag(h0))
    valid &= np.isfinite(np.real(hd)) & np.isfinite(np.imag(hd))
    if not np.any(valid):
        return {
            'score': float('nan'),
            'robust_relative_change': float('nan'),
            'phase_activity': float('nan'),
            'path_participation': float('nan'),
            'path_contrast': float('nan'),
            'tx_balance': float('nan'),
            'rx_balance': float('nan'),
        }

    h0_abs = np.abs(h0)
    delta = hd - h0
    rel = np.zeros_like(h0_abs, dtype=np.float64)
    rel[valid] = np.abs(delta[valid]) / (h0_abs[valid] + EPS)
    rel_log = np.log1p(rel)
    phase = np.zeros_like(h0_abs, dtype=np.float64)
    phase[valid] = np.abs(np.sin(np.angle(hd[valid] * np.conj(h0[valid]))))

    # Phase is useful but less reliable than amplitude without dispersion calibration.
    path_signal = rel_log * (1.0 + 0.25 * phase)
    path_signal = np.where(valid, path_signal, 0.0)
    valid_values = path_signal[valid]
    robust_relative_change = float(
        0.70 * np.nanmedian(valid_values) + 0.30 * np.nanquantile(valid_values, 0.75)
    )
    phase_activity = float(np.nanmean(phase[valid]))
    path_participation = participation_ratio(path_signal[valid])
    path_contrast = robust_contrast(path_signal[valid])
    tx_energy = np.sum(path_signal, axis=1)
    rx_energy = np.sum(path_signal, axis=0)
    tx_balance = participation_ratio(tx_energy)
    rx_balance = participation_ratio(rx_energy)
    # Keep the metric physically interpretable:
    # 1) robust_relative_change is the base defect visibility term;
    # 2) phase_activity and path_contrast provide localization value;
    # 3) participation and tx/rx balance penalize frequencies dominated by very few paths.
    score = (
        robust_relative_change
        * (0.55 + 0.25 * np.clip(phase_activity, 0.0, 1.0))
        * (0.55 + 0.25 * np.clip(path_contrast, 0.0, 1.0))
        * (0.60 + 0.20 * np.clip(path_participation, 0.0, 1.0))
        * (0.75 + 0.15 * np.clip(tx_balance, 0.0, 1.0))
        * (0.75 + 0.15 * np.clip(rx_balance, 0.0, 1.0))
    )
    return {
        'score': float(score),
        'robust_relative_change': robust_relative_change,
        'phase_activity': phase_activity,
        'path_participation': path_participation,
        'path_contrast': path_contrast,
        'tx_balance': tx_balance,
        'rx_balance': rx_balance,
    }


def rank_frequencies(
    healthy: dict[str, Any],
    damaged_items: list[dict[str, Any]],
    *,
    metric: str,
    min_healthy_abs_percentile: float,
    frequency_min_khz: float | None,
    frequency_max_khz: float | None,
    label_contexts: dict[str, dict[str, Any]] | None = None,
    v1_label_top_quantile: float = 0.80,
) -> list[dict[str, Any]]:
    h0 = healthy['H']
    frequencies_hz = healthy['frequencies_hz']
    healthy_mean_abs = np.nanmean(np.abs(h0), axis=(0, 1))
    floor = float(np.nanpercentile(healthy_mean_abs, min_healthy_abs_percentile))
    rows: list[dict[str, Any]] = []
    for freq_index, frequency_hz in enumerate(frequencies_hz):
        frequency_khz = frequency_hz / 1000.0
        if frequency_min_khz is not None and frequency_khz < frequency_min_khz:
            continue
        if frequency_max_khz is not None and frequency_khz > frequency_max_khz:
            continue
        if healthy_mean_abs[freq_index] < floor:
            rows.append({
                'frequency_hz': frequency_hz,
                'frequency_khz': frequency_khz,
                'sensitivity_mean': float('nan'),
                'sensitivity_median': float('nan'),
                'sensitivity_min': float('nan'),
                'sensitivity_max': float('nan'),
                'sample_count': 0,
                'valid_tx_count_min': 0,
                'healthy_mean_abs': float(healthy_mean_abs[freq_index]),
                'excluded_reason': 'healthy_response_below_floor',
            })
            continue
        values = []
        path_corr_values = []
        path_contrast_values = []
        energy_on_label_values = []
        physics_components: dict[str, list[float]] = {
            'robust_relative_change': [],
            'phase_activity': [],
            'path_participation': [],
            'path_contrast': [],
            'tx_balance': [],
            'rx_balance': [],
        }
        valid_tx_counts = []
        valid_ray_counts = []
        for damaged in damaged_items:
            tx_mask = finite_case_mask(healthy, damaged, freq_index)
            valid_tx_counts.append(int(np.sum(tx_mask)))
            if not np.any(tx_mask):
                continue
            if metric == 'v1_label_guided':
                key = str(damaged['path'].resolve(strict=False))
                if not label_contexts or key not in label_contexts:
                    continue
                result = v1_label_guided_frequency_metric(
                    healthy,
                    damaged,
                    freq_index=freq_index,
                    path_context=label_contexts[key],
                    top_quantile=v1_label_top_quantile,
                )
                if np.isfinite(result['score']):
                    values.append(result['score'])
                    path_corr_values.append(result['path_corr'])
                    path_contrast_values.append(result['path_contrast'])
                    energy_on_label_values.append(result['energy_on_label_paths'])
                    valid_ray_counts.append(int(result['valid_ray_count']))
            else:
                healthy_slice = healthy['H'][tx_mask, :, freq_index]
                damaged_slice = damaged['H'][tx_mask, :, freq_index]
                if metric == 'physics_tomography':
                    result = legacy_physics_tomography_frequency_metric(healthy_slice, damaged_slice)
                    if np.isfinite(result['score']):
                        values.append(result['score'])
                        for key in physics_components:
                            physics_components[key].append(result[key])
                else:
                    values.append(sample_frequency_metric(healthy_slice, damaged_slice, metric=metric))
        if values:
            rows.append({
                'frequency_hz': frequency_hz,
                'frequency_khz': frequency_khz,
                'sensitivity_mean': float(np.mean(values)),
                'sensitivity_median': float(np.median(values)),
                'sensitivity_min': float(np.min(values)),
                'sensitivity_max': float(np.max(values)),
                'sample_count': len(values),
                'valid_tx_count_min': int(min(valid_tx_counts)) if valid_tx_counts else 0,
                'valid_ray_count_min': int(min(valid_ray_counts)) if valid_ray_counts else '',
                'path_corr_mean': float(np.nanmean(path_corr_values)) if path_corr_values else '',
                'path_contrast_mean': float(np.nanmean(path_contrast_values)) if path_contrast_values else '',
                'energy_on_label_paths_mean': float(np.nanmean(energy_on_label_values)) if energy_on_label_values else '',
                'physics_robust_relative_change_mean': (
                    float(np.nanmean(physics_components['robust_relative_change']))
                    if physics_components['robust_relative_change'] else ''
                ),
                'physics_phase_activity_mean': (
                    float(np.nanmean(physics_components['phase_activity']))
                    if physics_components['phase_activity'] else ''
                ),
                'physics_path_participation_mean': (
                    float(np.nanmean(physics_components['path_participation']))
                    if physics_components['path_participation'] else ''
                ),
                'physics_path_contrast_mean': (
                    float(np.nanmean(physics_components['path_contrast']))
                    if physics_components['path_contrast'] else ''
                ),
                'physics_tx_balance_mean': (
                    float(np.nanmean(physics_components['tx_balance']))
                    if physics_components['tx_balance'] else ''
                ),
                'physics_rx_balance_mean': (
                    float(np.nanmean(physics_components['rx_balance']))
                    if physics_components['rx_balance'] else ''
                ),
                'healthy_mean_abs': float(healthy_mean_abs[freq_index]),
                'excluded_reason': '',
            })
        else:
            rows.append({
                'frequency_hz': frequency_hz,
                'frequency_khz': frequency_khz,
                'sensitivity_mean': float('nan'),
                'sensitivity_median': float('nan'),
                'sensitivity_min': float('nan'),
                'sensitivity_max': float('nan'),
                'sample_count': 0,
                'valid_tx_count_min': int(min(valid_tx_counts)) if valid_tx_counts else 0,
                'valid_ray_count_min': int(min(valid_ray_counts)) if valid_ray_counts else '',
                'path_corr_mean': '',
                'path_contrast_mean': '',
                'energy_on_label_paths_mean': '',
                'physics_robust_relative_change_mean': '',
                'physics_phase_activity_mean': '',
                'physics_path_participation_mean': '',
                'physics_path_contrast_mean': '',
                'physics_tx_balance_mean': '',
                'physics_rx_balance_mean': '',
                'healthy_mean_abs': float(healthy_mean_abs[freq_index]),
                'excluded_reason': 'no_valid_cases',
            })
    rows.sort(key=lambda row: (-np.nan_to_num(row['sensitivity_mean'], nan=-np.inf), row['frequency_hz']))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    healthy = load_response(args.healthy)
    damaged_paths = collect_damaged_paths(args)
    damaged_items = load_responses_parallel(damaged_paths, args.jobs)
    for damaged in damaged_items:
        assert_compatible(healthy, damaged)
    label_contexts = None
    if args.metric == 'v1_label_guided':
        label_contexts = {}
        for path in damaged_paths:
            print(f'[v1_label_guided] building path-label overlap for {path.name}', flush=True)
            label_contexts[str(path.resolve(strict=False))] = v1_path_label_overlap(
                damaged_path=path,
                label_dir=args.label_dir,
                metadata_dir=args.metadata_dir,
                tx_indices=healthy['tx_indices'],
                rx_indices=healthy['rx_indices'],
                grid_size=args.v1_grid_size,
                sigma_ray_mm=args.v1_sigma_ray_mm,
            )
    physics_metadata: dict[str, Any] = {}
    selection_strategy_used = args.selection_strategy
    if args.metric == 'physics_tomography':
        ranked, fisher_matrices, physics_metadata = physics_tomography_rank_frequencies(
            healthy,
            damaged_items,
            min_healthy_abs_percentile=args.min_healthy_abs_percentile,
            frequency_min_khz=args.frequency_min_khz,
            frequency_max_khz=args.frequency_max_khz,
            metadata_dir=args.metadata_dir,
            born_grid_size=args.born_grid_size,
            born_rank=args.born_rank,
            born_sigma_ray_mm=args.born_sigma_ray_mm,
            born_info_scale=args.born_info_scale,
            born_phase_weight=args.born_phase_weight,
            born_noise_quantile=args.born_noise_quantile,
            born_weight_clip_quantile=args.born_weight_clip_quantile,
        )
        if args.selection_strategy == 'greedy_d_optimal':
            selected = greedy_d_optimal_selection(
                ranked,
                fisher_matrices,
                top_n=args.top_n,
                scale=args.born_info_scale,
            )
        else:
            selected = [
                row for row in ranked
                if not row['excluded_reason'] and np.isfinite(row['sensitivity_mean'])
            ][:args.top_n]
    else:
        ranked = rank_frequencies(
            healthy,
            damaged_items,
            metric=args.metric,
            min_healthy_abs_percentile=args.min_healthy_abs_percentile,
            frequency_min_khz=args.frequency_min_khz,
            frequency_max_khz=args.frequency_max_khz,
            label_contexts=label_contexts,
            v1_label_top_quantile=args.v1_label_top_quantile,
        )
        selected = [
            row for row in ranked
            if not row['excluded_reason'] and np.isfinite(row['sensitivity_mean'])
        ][:args.top_n]
        if args.selection_strategy == 'greedy_d_optimal':
            selection_strategy_used = 'top_score'
    args.output_root.mkdir(parents=True, exist_ok=True)
    ranked_csv = args.output_root / f'{args.prefix}_ranked.csv'
    selected_csv = args.output_root / f'{args.prefix}_top{args.top_n}.csv'
    selected_txt = args.output_root / f'{args.prefix}_top{args.top_n}_frequencies.txt'
    summary_json = args.output_root / f'{args.prefix}_summary.json'
    write_csv(ranked_csv, ranked)
    write_csv(selected_csv, selected)
    selected_text = ','.join(f'{row["frequency_hz"]:.12g}' for row in selected)
    selected_txt.write_text(selected_text + '\n', encoding='utf-8')
    summary = {
        'healthy': str(args.healthy),
        'damaged': [str(path) for path in damaged_paths],
        'response_dir': str(args.response_dir),
        'sample_ids': parse_sample_ids(args.sample_ids),
        'damaged_glob': args.damaged_glob,
        'metric': args.metric,
        'top_n': args.top_n,
        'min_healthy_abs_percentile': args.min_healthy_abs_percentile,
        'frequency_min_khz': args.frequency_min_khz,
        'frequency_max_khz': args.frequency_max_khz,
        'label_dir': str(args.label_dir),
        'metadata_dir': str(args.metadata_dir),
        'v1_grid_size': args.v1_grid_size,
        'v1_sigma_ray_mm': args.v1_sigma_ray_mm,
        'v1_label_top_quantile': args.v1_label_top_quantile,
        'jobs': args.jobs,
        'selection_strategy': selection_strategy_used,
        'physics_metadata': physics_metadata,
        'selected_frequencies_hz': [row['frequency_hz'] for row in selected],
        'selected_frequencies_khz': [row['frequency_khz'] for row in selected],
        'ranked_csv': str(ranked_csv),
        'selected_csv': str(selected_csv),
        'selected_txt': str(selected_txt),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Ranked CSV: {ranked_csv}')
    print(f'Selected CSV: {selected_csv}')
    print(f'Damaged samples: {len(damaged_paths)}')
    print(f'Selected frequencies: {selected_text}')


if __name__ == '__main__':
    main()

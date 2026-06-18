"""Solve one COMSOL shell case, export CSV, then discard the model.

This avoids saving transient solution fields into large MPH files.  The
preferred mode builds one model per sample, solves one transmitter/frequency
case at a time, exports the current receiver traces, clears the current
solution data, and reuses the same geometry/mesh/solver tree for the next
case.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import defect_label_common as defect_labels
import simple_shell_common as shell


HELICAL_ORDERS = (-1, 0, 1)
DEFAULT_HEARTBEAT_S = 30.0
# tqdm is intentionally disabled for COMSOL batch runs.  On some Windows
# conda/terminal combinations tqdm can be captured or delayed, while plain
# flushed stdout remains visible and works in scheduler logs.
TQDM_AVAILABLE = False


def console_log(message: str) -> None:
    print(message, flush=True)


@dataclass(frozen=True)
class Case:
    tx: int
    frequency_hz: float


@dataclass(frozen=True)
class SampleExportResult:
    sample_id: str
    dataset: str
    defect_state: str
    metadata_path: Path
    waveform_files: list[Path]
    feature_files: list[Path]
    case_count: int


def parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(',') if item.strip())
    if not values:
        raise ValueError('At least one transmitter index is required.')
    return values


def parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(',') if item.strip())
    if not values:
        raise ValueError('At least one frequency is required.')
    return values


def make_cases(tx_indices: tuple[int, ...], frequencies_hz: tuple[float, ...]) -> list[Case]:
    return [Case(tx=tx, frequency_hz=frequency) for tx in tx_indices for frequency in frequencies_hz]


def add_solver_arguments(parser) -> None:
    parser.add_argument('--dt-out-us', type=float, default=shell.SolverConfig.dt_out_us)
    parser.add_argument('--relative-tolerance', type=float, default=shell.SolverConfig.relative_tolerance)
    parser.add_argument(
        '--linear-solver',
        choices=('cudss', 'pardiso', 'mumps'),
        default=shell.SolverConfig.direct_linear_solver,
    )
    parser.add_argument(
        '--cudss-precision',
        choices=('single', 'double'),
        default=shell.SolverConfig.cudss_precision,
    )
    parser.add_argument('--pardiso-use-cluster', action='store_true')
    parser.add_argument(
        '--rebuild-each-case',
        action='store_true',
        help='Rebuild geometry/mesh/solver tree for every tx/frequency case. Slower, but useful as a fallback.',
    )
    parser.add_argument(
        '--include-comsol-marker-datasets',
        action='store_true',
        help='Create receiver/transmitter CutPoint datasets in COMSOL for manual visual inspection. Disabled by default for streaming exports.',
    )


def apply_solver_arguments(args) -> None:
    shell.SOLVER = shell.SolverConfig(
        solve=False,
        dt_out_us=args.dt_out_us,
        relative_tolerance=args.relative_tolerance,
        direct_linear_solver=args.linear_solver,
        cudss_precision=args.cudss_precision,
        pardiso_use_cluster=args.pardiso_use_cluster,
    )
    console_log(
        '[solver] '
        f'linear_solver={args.linear_solver}, '
        f'cudss_precision={args.cudss_precision}, '
        f'dt_out_us={args.dt_out_us}, '
        f'relative_tolerance={args.relative_tolerance}'
    )
    if args.linear_solver == 'cudss' and args.cudss_precision == 'single':
        console_log(
            '[solver] warning: cuDSS single precision is a speed-test mode for this shell model. '
            'If COMSOL reports an internal cuDSS error, cannot find consistent initial values, '
            'or last time step not converged, rerun with --linear-solver pardiso or '
            '--linear-solver cudss --cudss-precision double.'
        )
    shell.CREATE_RECEIVER_DATASETS = bool(args.include_comsol_marker_datasets)
    shell.CREATE_VISUAL_MARKER_DATASETS = bool(args.include_comsol_marker_datasets)


def radial_expression(position: dict[str, float]) -> str:
    return shell.radial_displacement_expr(position)


def project_position_to_shell_midsurface(position: dict[str, Any]) -> dict[str, Any]:
    """Return the nearest receiver point on the cylindrical shell midsurface."""
    x_mm = float(position.get('x_mm', float('nan')))
    y_mm = float(position.get('y_mm', float('nan')))
    if math.isfinite(x_mm) and math.isfinite(y_mm) and math.hypot(x_mm, y_mm) > 0.0:
        theta = math.atan2(y_mm, x_mm)
    else:
        theta = math.radians(float(position['theta_deg']))

    z_mm = min(max(float(position['z_mm']), 0.0), shell.PIPE.length_mm)
    radius_mm = shell.PIPE.mid_radius_mm
    projected = dict(position)
    projected.update({
        'theta_deg': math.degrees(theta) % 360.0,
        'x_mm': radius_mm * math.cos(theta),
        'y_mm': radius_mm * math.sin(theta),
        'z_mm': z_mm,
        'radius_mm': radius_mm,
    })
    return projected


def receiver_shell_positions() -> list[dict[str, Any]]:
    return [project_position_to_shell_midsurface(position) for position in shell.receiver_positions()]


def solution_dataset_node(model):
    solution_datasets = [dataset for dataset in model / 'datasets' if dataset.type() == 'Solution']
    if not solution_datasets:
        available = [f'{dataset.name()}:{dataset.type()}' for dataset in model / 'datasets']
        raise RuntimeError('No COMSOL Solution dataset found. Available datasets: ' + '; '.join(available))

    def score(dataset) -> tuple[int, str]:
        props = dataset.properties()
        name = dataset.name().lower()
        value = 0
        if props.get('solution'):
            value += 4
        if 'simple shell displacement' in name:
            value += 2
        if 'parametric' not in name and '参数化' not in name:
            value += 1
        return value, dataset.name()

    selected = max(solution_datasets, key=score)
    return selected


def solution_dataset_tag(model) -> str:
    return solution_dataset_node(model).tag()


def shell_boundary_selection_node(model):
    for selection in model / 'selections':
        if selection.name() == 'open pipe cylindrical shell boundary':
            return selection
    for selection in model / 'selections':
        if 'cylindrical shell boundary' in selection.name().lower():
            return selection
    available = [f'{selection.name()}:{selection.type()}' for selection in model / 'selections']
    raise RuntimeError('No cylindrical shell boundary selection found. Available selections: ' + '; '.join(available))


def receiver_points_for_current_solution(model):
    solution_dataset = solution_dataset_node(model)
    positions = receiver_shell_positions()
    expressions = [radial_expression(position) for position in positions]
    return solution_dataset, positions, expressions


def shell_coordinate_mm(position: dict[str, Any]) -> np.ndarray:
    projected = project_position_to_shell_midsurface(position)
    return np.asarray(
        [[projected['x_mm']], [projected['y_mm']], [projected['z_mm']]],
        dtype=float,
    )


def time_trace_from_comsol_data(values: np.ndarray, time_count: int) -> np.ndarray | None:
    array = np.asarray(values)
    if array.size == 0:
        return None
    array = np.real_if_close(array)
    if np.iscomplexobj(array):
        array = np.real(array)

    squeezed = np.asarray(array).squeeze()
    if squeezed.size == time_count:
        return np.asarray(squeezed, dtype=float).reshape(time_count)

    for axis, size in enumerate(array.shape):
        if size == time_count:
            if array.size // time_count != 1:
                return None
            moved = np.moveaxis(array, axis, 0).reshape(time_count, -1)
            return np.asarray(moved[:, 0], dtype=float)
    return None


def finite_time_trace(values: np.ndarray, time_count: int) -> np.ndarray | None:
    trace = time_trace_from_comsol_data(values, time_count)
    if trace is not None and trace.size == time_count and np.all(np.isfinite(trace)):
        return trace
    return None


def evaluate_point_cutpoint(model, solution_dataset, position: dict[str, Any], expression: str, time_count: int) -> np.ndarray:
    """Evaluate a receiver trace on a temporary CutPoint3D dataset."""
    projected = project_position_to_shell_midsurface(position)
    point_dataset = None
    evaluation = None
    try:
        point_dataset = (model / 'datasets').create(
            'CutPoint3D',
            name=f'stream export receiver {projected["index"]:02d} shell point',
        )
        point_dataset.property('data', solution_dataset)
        point_dataset.property('pointx', f'{projected["x_mm"]:.12g}[mm]')
        point_dataset.property('pointy', f'{projected["y_mm"]:.12g}[mm]')
        point_dataset.property('pointz', f'{projected["z_mm"]:.12g}[mm]')

        evaluation = (model / 'evaluations').create('Eval')
        evaluation.property('data', point_dataset)
        evaluation.property('expr', expression)
        evaluation.property('unit', 'm')
        java = evaluation.java
        values = np.asarray(java.getData())
        if java.isComplex():
            values = values.astype(complex) + 1j * np.asarray(java.getImagData())
        trace = finite_time_trace(values, time_count)
        if trace is not None:
            return trace
        raise RuntimeError(f'raw CutPoint data shape={values.shape}, finite={bool(np.all(np.isfinite(values))) if values.size else False}')
    finally:
        if evaluation is not None:
            evaluation.remove()
        if point_dataset is not None:
            point_dataset.remove()


def evaluate_point_interpolation(model, solution_dataset, position: dict[str, Any], expression: str, time_count: int) -> np.ndarray:
    """Evaluate one expression at one receiver coordinate using COMSOL Interp."""
    base_coordinate_mm = shell_coordinate_mm(position)
    shell_selection = shell_boundary_selection_node(model)
    last_error: Exception | None = None
    attempts: list[str] = []
    for coordinate_label, coordinate in (('model-mm', base_coordinate_mm), ('si-m', base_coordinate_mm * 1e-3)):
        for edim in ('2', 'auto', '3'):
            interpolation = (model / 'evaluations').create('Interp')
            try:
                interpolation.property('data', solution_dataset)
                interpolation.property('expr', expression)
                interpolation.property('unit', 'm')
                interpolation.property('edim', edim)
                interpolation.select(shell_selection)
                shell.set_if_possible(interpolation, 'coorderr', False)
                shell.set_if_possible(interpolation, 'ext', 1.0)
                interpolation.property('coord', coordinate)
                java = interpolation.java
                values = np.asarray(java.getData())
                if java.isComplex():
                    values = values.astype(complex) + 1j * np.asarray(java.getImagData())
                trace = finite_time_trace(values, time_count)
                if trace is not None:
                    return trace
                attempts.append(f'{coordinate_label}/edim={edim}:shape={values.shape}')
            except Exception as error:
                last_error = error
                attempts.append(f'{coordinate_label}/edim={edim}:error={type(error).__name__}:{error}')
            finally:
                interpolation.remove()
    if last_error is not None:
        raise RuntimeError(
            f'COMSOL point interpolation failed at receiver {position["index"]}: {last_error}. '
            f'Attempts: {"; ".join(attempts)}'
        ) from last_error
    raise RuntimeError(
        f'COMSOL point interpolation returned no finite data at receiver {position["index"]}. '
        f'Attempts: {"; ".join(attempts)}'
    )


def field_time_space_from_comsol_data(values: np.ndarray, time_count: int) -> np.ndarray:
    array = np.asarray(values)
    array = np.real_if_close(array)
    if np.iscomplexobj(array):
        array = np.real(array)
    array = np.asarray(array, dtype=float).squeeze()
    if array.ndim == 1:
        raise RuntimeError(f'COMSOL field data has no spatial axis: shape={array.shape}')
    for axis, size in enumerate(array.shape):
        if size == time_count:
            return np.moveaxis(array, axis, 0).reshape(time_count, -1)
    raise RuntimeError(f'COMSOL field data has no time axis of length {time_count}: shape={array.shape}')


def evaluate_shell_displacement_field(model, solution_dataset, time_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return actual shell sample coordinates plus u/v over all stored times."""
    evaluation = (model / 'evaluations').create('Eval')
    try:
        evaluation.property('data', solution_dataset)
        evaluation.property('expr', ['u', 'v'])
        evaluation.property('unit', ['m', 'm'])
        evaluation.property('edim', '2')
        evaluation.select(shell_boundary_selection_node(model))
        java = evaluation.java
        values = np.asarray(java.getData())
        if java.isComplex():
            values = values.astype(complex) + 1j * np.asarray(java.getImagData())
        coordinates_mm = np.asarray(java.getCoordinates(), dtype=float)
    finally:
        evaluation.remove()

    if coordinates_mm.ndim != 2 or coordinates_mm.shape[0] != 3 or coordinates_mm.shape[1] == 0:
        raise RuntimeError(f'COMSOL shell evaluation returned invalid coordinates: shape={coordinates_mm.shape}')
    if values.shape[0] < 2:
        raise RuntimeError(f'COMSOL shell evaluation returned invalid u/v data shape: {values.shape}')

    u_field = field_time_space_from_comsol_data(values[0], time_count)
    v_field = field_time_space_from_comsol_data(values[1], time_count)
    if u_field.shape != v_field.shape or u_field.shape[1] != coordinates_mm.shape[1]:
        raise RuntimeError(
            'COMSOL shell field shape mismatch: '
            f'u={u_field.shape}, v={v_field.shape}, coordinates={coordinates_mm.shape}'
        )
    if not (np.all(np.isfinite(coordinates_mm)) and np.all(np.isfinite(u_field)) and np.all(np.isfinite(v_field))):
        raise RuntimeError('COMSOL shell field export returned non-finite coordinates or displacement values.')
    return coordinates_mm, u_field, v_field


def nearest_shell_channels(
    coordinates_mm: np.ndarray,
    u_field: np.ndarray,
    v_field: np.ndarray,
    positions: list[dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    coordinate_columns = coordinates_mm.T
    channels = []
    nearest: list[dict[str, Any]] = []
    for position in positions:
        projected = project_position_to_shell_midsurface(position)
        target = np.asarray([projected['x_mm'], projected['y_mm'], projected['z_mm']], dtype=float)
        distances2 = np.sum((coordinate_columns - target) ** 2, axis=1)
        nearest_index = int(np.argmin(distances2))
        theta = math.radians(projected['theta_deg'])
        trace = math.cos(theta) * u_field[:, nearest_index] + math.sin(theta) * v_field[:, nearest_index]
        channels.append(trace)
        nearest.append({
            'receiver': int(projected['index']),
            'nearest_index': nearest_index,
            'distance_mm': float(math.sqrt(float(distances2[nearest_index]))),
            'target_x_mm': float(target[0]),
            'target_y_mm': float(target[1]),
            'target_z_mm': float(target[2]),
            'actual_x_mm': float(coordinates_mm[0, nearest_index]),
            'actual_y_mm': float(coordinates_mm[1, nearest_index]),
            'actual_z_mm': float(coordinates_mm[2, nearest_index]),
        })
    return np.vstack(channels).T, nearest


def evaluate_current_solution_from_nearest_shell_points(
    model,
    solution_dataset,
    positions: list[dict[str, Any]],
    time_s: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    coordinates_mm, u_field, v_field = evaluate_shell_displacement_field(model, solution_dataset, time_s.size)
    channels, nearest = nearest_shell_channels(coordinates_mm, u_field, v_field, positions)
    max_distance = max(item['distance_mm'] for item in nearest) if nearest else float('nan')
    console_log(
        '[export] sampled receiver traces from nearest actual shell points; '
        f'shell_points={coordinates_mm.shape[1]}, max_receiver_distance_mm={max_distance:.3f}'
    )
    return channels, nearest


def global_result_time_channels(values: np.ndarray, time_count: int, channel_count: int) -> np.ndarray:
    array = np.asarray(values)
    array = np.real_if_close(array)
    if np.iscomplexobj(array):
        array = np.real(array)
    array = np.asarray(array, dtype=float).squeeze()
    if array.ndim == 1:
        if channel_count == 1 and array.size == time_count:
            return array.reshape(time_count, 1)
        if array.size == time_count * channel_count:
            return array.reshape(channel_count, time_count).T
    if array.ndim == 2:
        if array.shape == (channel_count, time_count):
            return array.T
        if array.shape == (time_count, channel_count):
            return array
    for time_axis, size in enumerate(array.shape):
        if size != time_count:
            continue
        moved = np.moveaxis(array, time_axis, 0).reshape(time_count, -1)
        if moved.shape[1] == channel_count:
            return moved
    raise RuntimeError(
        f'COMSOL weighted receiver data shape {array.shape} does not match '
        f'time_count={time_count}, channel_count={channel_count}'
    )


def evaluate_receiver_weighted_averages(model, solution_dataset, time_count: int) -> np.ndarray:
    expressions = shell.receiver_weighted_average_expressions()
    evaluation = (model / 'evaluations').create('EvalGlobal')
    try:
        evaluation.property('data', solution_dataset)
        evaluation.property('expr', expressions)
        evaluation.property('unit', ['m'] * len(expressions))
        evaluation.property('probetag', 'none')
        java = evaluation.java
        raw = np.asarray(java.computeResult())
        if java.isComplex():
            values = raw[0].astype(complex) + 1j * raw[1]
        else:
            values = raw[0]
        channels = global_result_time_channels(values, time_count, len(expressions))
    finally:
        evaluation.remove()
    if not np.all(np.isfinite(channels)):
        raise RuntimeError('COMSOL weighted receiver average export returned non-finite data.')
    console_log(
        '[export] sampled receiver traces from patch-weighted shell averages '
        f'using {shell.RECEIVER_INTEGRATION_OPERATOR}.'
    )
    return channels


def evaluate_current_solution(model, solution_dataset, positions: list[dict[str, Any]], expressions: list[str]) -> tuple[np.ndarray, np.ndarray]:
    time_s = np.asarray(model.evaluate('t', dataset=solution_dataset), dtype=float).reshape(-1)
    try:
        data = evaluate_receiver_weighted_averages(model, solution_dataset, time_s.size)
        if data.shape[0] != time_s.shape[0]:
            raise RuntimeError(f'time/channel length mismatch: {time_s.shape[0]} != {data.shape[0]}')
        return time_s, data
    except Exception as weighted_error:
        console_log(
            '[export] patch-weighted receiver export failed; falling back to point/nearest shell export. '
            f'{type(weighted_error).__name__}: {weighted_error}'
        )

    channels = []
    point_errors = []
    for position, expression in zip(positions, expressions, strict=True):
        try:
            values = evaluate_point_interpolation(model, solution_dataset, position, expression, time_s.size)
        except Exception as error:
            point_errors.append(f'rx{position["index"]:02d}: {type(error).__name__}: {error}')
            break
        channels.append(np.asarray(values, dtype=float).reshape(-1))
    if point_errors:
        console_log(
            '[export] direct point interpolation failed; falling back to nearest actual shell points. '
            + point_errors[0]
        )
        data, _nearest = evaluate_current_solution_from_nearest_shell_points(model, solution_dataset, positions, time_s)
    else:
        data = np.vstack(channels).T
    if data.shape[0] != time_s.shape[0]:
        raise RuntimeError(f'time/channel length mismatch: {time_s.shape[0]} != {data.shape[0]}')
    return time_s, data


def analytic_envelope(signal: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fft(signal)
    multiplier = np.zeros(signal.size)
    if signal.size % 2 == 0:
        multiplier[0] = 1.0
        multiplier[signal.size // 2] = 1.0
        multiplier[1:signal.size // 2] = 2.0
    else:
        multiplier[0] = 1.0
        multiplier[1:(signal.size + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(spectrum * multiplier))


def first_arrival(time_s: np.ndarray, envelope: np.ndarray, threshold_ratio: float) -> float:
    if envelope.size == 0:
        return float('nan')
    threshold = float(np.nanmax(envelope)) * threshold_ratio
    indices = np.flatnonzero(envelope >= threshold)
    return float(time_s[int(indices[0])]) if indices.size else float('nan')


def fft_at_frequency(time_s: np.ndarray, signal: np.ndarray, frequency_hz: float) -> tuple[float, float, float]:
    dt = float(np.median(np.diff(time_s)))
    window = np.hanning(signal.size)
    spectrum = np.fft.rfft(signal * window)
    freqs = np.fft.rfftfreq(signal.size, dt)
    index = int(np.argmin(np.abs(freqs - frequency_hz)))
    value = spectrum[index]
    coherent_gain = max(float(np.sum(window)) / signal.size, 1e-12)
    amplitude = 2.0 * abs(value) / signal.size / coherent_gain
    return float(freqs[index]), float(amplitude), float(np.angle(value))


def circular_delta(theta_to_deg: float, theta_from_deg: float, order: int) -> float:
    base = (theta_to_deg - theta_from_deg + 180.0) % 360.0 - 180.0
    if order > 0 and base < 0:
        base += 360.0
    elif order < 0 and base > 0:
        base -= 360.0
    return base + 360.0 * order


def arrival_time_for_order(tx_pos: dict[str, float], rx_pos: dict[str, float], order: int, group_velocity: float) -> float:
    radius_m = shell.PIPE.mid_radius_mm * 1e-3
    dz_m = (rx_pos['z_mm'] - tx_pos['z_mm']) * 1e-3
    arc_m = math.radians(circular_delta(rx_pos['theta_deg'], tx_pos['theta_deg'], order)) * radius_m
    return math.hypot(dz_m, arc_m) / group_velocity


def window_peak(time_s: np.ndarray, envelope: np.ndarray, center_s: float, half_width_s: float) -> tuple[float, float]:
    mask = (time_s >= center_s - half_width_s) & (time_s <= center_s + half_width_s)
    if not np.any(mask):
        return float('nan'), float('nan')
    local_indices = np.flatnonzero(mask)
    peak_local = int(np.argmax(envelope[mask]))
    index = int(local_indices[peak_local])
    return float(envelope[index]), float(time_s[index])


def write_waveform(path: Path, time_s: np.ndarray, channels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ['time_s'] + [f'rx{i:02d}_ur_m' for i in range(1, channels.shape[1] + 1)]
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for index, time_value in enumerate(time_s):
            writer.writerow([f'{time_value:.12g}', *[f'{value:.12e}' for value in channels[index]]])


def write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_feature_csvs(
    feature_files: list[Path],
    feature_rows: list[dict[str, Any]],
    helical_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    """Checkpoint accumulated per-case feature rows to disk."""
    write_dict_csv(feature_files[0], feature_rows)
    write_dict_csv(feature_files[1], helical_rows)
    write_dict_csv(feature_files[2], summary_rows)


def read_waveform(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=',', names=True)
    if data.dtype.names is None or len(data.dtype.names) < 2:
        raise RuntimeError(f'No waveform channels found in {path}')
    return np.vstack([data[name] for name in data.dtype.names[1:]]).T


def load_healthy_waveform(root: Path | None, case: Case, healthy_sample_id: str | None = None) -> np.ndarray | None:
    if root is None:
        return None
    if healthy_sample_id:
        path = root / f'{healthy_sample_id}_tx{case.tx:02d}_f{int(case.frequency_hz)}Hz_waveforms.csv'
        if path.exists():
            return read_waveform(path)
    pattern = f'*_tx{case.tx:02d}_f{int(case.frequency_hz)}Hz_waveforms.csv'
    matches = sorted(root.glob(pattern))
    healthy_matches = [item for item in matches if 'healthy' in item.name.lower()]
    selected = healthy_matches[0] if healthy_matches else (matches[0] if matches else None)
    return read_waveform(selected) if selected is not None else None


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return 'unknown'
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, sec = divmod(remainder, 60)
    if hours:
        return f'{hours:d}h {minutes:02d}m {sec:02d}s'
    if minutes:
        return f'{minutes:d}m {sec:02d}s'
    return f'{sec:d}s'


def append_progress(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as file:
        file.write(json.dumps(event, ensure_ascii=False, default=str) + '\n')


def progress_bar(completed: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return '[' + ('-' * width) + ']'
    ratio = min(max(completed / total, 0.0), 1.0)
    filled = int(round(ratio * width))
    return '[' + ('#' * filled) + ('-' * (width - filled)) + ']'


def progress_event(
    *,
    sample_id: str,
    dataset: str,
    defect_state: str,
    case: Case | None,
    status: str,
    case_index: int,
    case_count: int,
    started_s: float,
    message: str,
) -> dict[str, Any]:
    elapsed_s = time.monotonic() - started_s
    if status in {'case_done', 'sample_done'}:
        completed = max(case_index, 0)
    else:
        completed = max(case_index - 1, 0) if case_index > 0 else 0
    average_s = elapsed_s / completed if completed > 0 else None
    remaining = max(case_count - completed, 0)
    eta_s = average_s * remaining if average_s is not None else None
    return {
        'wall_time_s': time.time(),
        'dataset': dataset,
        'sample_id': sample_id,
        'defect_state': defect_state,
        'status': status,
        'case_index': case_index,
        'case_count': case_count,
        'completed_cases': completed,
        'tx': None if case is None else case.tx,
        'frequency_hz': None if case is None else case.frequency_hz,
        'elapsed_s': elapsed_s,
        'average_case_s': average_s,
        'eta_s': eta_s,
        'message': message,
    }


def progress_line(event: dict[str, Any]) -> str:
    case_text = (
        'sample'
        if event['tx'] is None
        else f'tx={event["tx"]} f={event["frequency_hz"]}Hz'
    )
    percent = 100.0 * event['completed_cases'] / event['case_count'] if event['case_count'] else 100.0
    return (
        '[progress] '
        f'{progress_bar(int(event["completed_cases"]), int(event["case_count"]))} '
        f'{percent:5.1f}% '
        f'{event["sample_id"]} {event["status"]} '
        f'{event["case_index"]}/{event["case_count"]} '
        f'{case_text} '
        f'elapsed={format_duration(float(event["elapsed_s"]))} '
        f'eta={format_duration(float(event["eta_s"])) if event["eta_s"] is not None else "unknown"} '
        f'- {event["message"]}'
    )


def print_progress(event: dict[str, Any]) -> None:
    console_log(progress_line(event))


class BlockingHeartbeat:
    def __init__(
        self,
        *,
        progress_path: Path,
        sample_id: str,
        dataset: str,
        defect_state: str,
        case: Case | None,
        case_index: int,
        case_count: int,
        sample_started_s: float,
        operation_started_s: float,
        status: str,
        message: str,
        interval_s: float = DEFAULT_HEARTBEAT_S,
        pbar=None,
        postfix_prefix: str = 'running',
    ):
        self.progress_path = progress_path
        self.sample_id = sample_id
        self.dataset = dataset
        self.defect_state = defect_state
        self.case = case
        self.case_index = case_index
        self.case_count = case_count
        self.sample_started_s = sample_started_s
        self.operation_started_s = operation_started_s
        self.status = status
        self.message = message
        self.interval_s = max(interval_s, 5.0)
        self.pbar = pbar
        self.postfix_prefix = postfix_prefix
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        self._thread.join(timeout=1.0)
        if sys.stdout.isatty():
            print('', flush=True)

    def _event(self) -> dict[str, Any]:
        elapsed_s = time.monotonic() - self.sample_started_s
        operation_elapsed_s = time.monotonic() - self.operation_started_s
        completed = max(self.case_index - 1, 0) if self.case_index > 0 else 0
        average_s = elapsed_s / completed if completed > 0 else None
        remaining = max(self.case_count - completed, 0)
        eta_s = average_s * remaining if average_s is not None else None
        return {
            'wall_time_s': time.time(),
            'dataset': self.dataset,
            'sample_id': self.sample_id,
            'defect_state': self.defect_state,
            'status': self.status,
            'case_index': self.case_index,
            'case_count': self.case_count,
            'completed_cases': completed,
            'tx': None if self.case is None else self.case.tx,
            'frequency_hz': None if self.case is None else self.case.frequency_hz,
            'elapsed_s': elapsed_s,
            'operation_elapsed_s': operation_elapsed_s,
            'average_case_s': average_s,
            'eta_s': eta_s,
            'message': self.message,
        }

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            event = self._event()
            append_progress(self.progress_path, event)
            elapsed = format_duration(float(event['operation_elapsed_s']))
            line = progress_line(event) + f' operation_elapsed={elapsed}'
            if self.pbar is not None:
                self.pbar.set_postfix_str(f'{self.postfix_prefix} {elapsed}', refresh=True)
                console_log(line)
            else:
                console_log(line)


class SolveHeartbeat:
    def __init__(
        self,
        *,
        progress_path: Path,
        sample_id: str,
        dataset: str,
        defect_state: str,
        case: Case,
        case_index: int,
        case_count: int,
        sample_started_s: float,
        case_started_s: float,
        interval_s: float = DEFAULT_HEARTBEAT_S,
        pbar=None,
    ):
        self.progress_path = progress_path
        self.sample_id = sample_id
        self.dataset = dataset
        self.defect_state = defect_state
        self.case = case
        self.case_index = case_index
        self.case_count = case_count
        self.sample_started_s = sample_started_s
        self.case_started_s = case_started_s
        self.interval_s = max(interval_s, 5.0)
        self.pbar = pbar
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        self._thread.join(timeout=1.0)
        if sys.stdout.isatty():
            print('', flush=True)

    def _event(self) -> dict[str, Any]:
        elapsed_s = time.monotonic() - self.sample_started_s
        case_elapsed_s = time.monotonic() - self.case_started_s
        completed = max(self.case_index - 1, 0)
        average_s = elapsed_s / completed if completed > 0 else None
        remaining = max(self.case_count - completed, 0)
        eta_s = average_s * remaining if average_s is not None else None
        return {
            'wall_time_s': time.time(),
            'dataset': self.dataset,
            'sample_id': self.sample_id,
            'defect_state': self.defect_state,
            'status': 'comsol_solve_running',
            'case_index': self.case_index,
            'case_count': self.case_count,
            'completed_cases': completed,
            'tx': self.case.tx,
            'frequency_hz': self.case.frequency_hz,
            'elapsed_s': elapsed_s,
            'case_elapsed_s': case_elapsed_s,
            'average_case_s': average_s,
            'eta_s': eta_s,
            'message': (
                'COMSOL model.solve() is still running; internal time-step '
                'percentage is not available through mph during the blocking call'
            ),
        }

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            event = self._event()
            append_progress(self.progress_path, event)
            line = progress_line(event) + f' case_elapsed={format_duration(float(event["case_elapsed_s"]))}'
            if self.pbar is not None:
                self.pbar.set_postfix_str(
                    f'case {self.case_index}/{self.case_count} solve {format_duration(float(event["case_elapsed_s"]))}',
                    refresh=True,
                )
                console_log(line)
            elif sys.stdout.isatty():
                print('\r' + line, end='', flush=True)
            else:
                console_log(line)


def run_model_solve_with_heartbeat(
    model,
    *,
    progress_path: Path,
    sample_id: str,
    dataset: str,
    defect_state: str,
    case: Case,
    case_index: int,
    case_count: int,
    sample_started_s: float,
    heartbeat_s: float,
    pbar=None,
) -> dict[str, Any]:
    solve_started_s = time.monotonic()
    with SolveHeartbeat(
        progress_path=progress_path,
        sample_id=sample_id,
        dataset=dataset,
        defect_state=defect_state,
        case=case,
        case_index=case_index,
        case_count=case_count,
        sample_started_s=sample_started_s,
        case_started_s=solve_started_s,
        interval_s=heartbeat_s,
        pbar=pbar,
    ):
        model.solve()
    return {
        'solve_elapsed_s': time.monotonic() - solve_started_s,
        'post_solve_problems': model.problems(),
    }


def clear_solution_data(model) -> list[str]:
    """Drop transient field data while keeping geometry, mesh, and solver tree."""
    cleared: list[str] = []
    failures: list[str] = []
    for solution in model / 'solutions':
        try:
            solution.java.clearSolutionData()
            cleared.append(f'{solution.name()}:clearSolutionData')
            continue
        except Exception:
            pass
        try:
            solution.java.clearSolution()
            cleared.append(f'{solution.name()}:clearSolution')
        except Exception as error:
            failures.append(f'{solution.name()}:clear_failed:{type(error).__name__}:{error}')
    if failures:
        raise RuntimeError('Failed to clear COMSOL solution data: ' + '; '.join(failures))
    return cleared


def validate_waveforms(time_s: np.ndarray, channels: np.ndarray) -> dict[str, Any]:
    finite = bool(np.all(np.isfinite(time_s)) and np.all(np.isfinite(channels)))
    max_abs = np.nanmax(np.abs(channels), axis=0) if channels.size else np.asarray([])
    nonzero_channels = int(np.sum(max_abs > 0.0))
    dead_channels = [
        int(index + 1)
        for index, value in enumerate(max_abs)
        if not np.isfinite(value) or value <= 0.0
    ]
    return {
        'time_count': int(time_s.size),
        'channel_count': int(channels.shape[1]) if channels.ndim == 2 else 0,
        'finite': finite,
        'nonzero_channels': nonzero_channels,
        'dead_channels': dead_channels,
        'max_abs_per_channel_m': [float(value) for value in max_abs],
        'global_max_abs_m': float(np.nanmax(np.abs(channels))) if channels.size else float('nan'),
    }


def append_case_features(
    *,
    feature_rows: list[dict[str, Any]],
    helical_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    sample_id: str,
    case: Case,
    time_s: np.ndarray,
    channels: np.ndarray,
    waveform_path: Path,
    healthy: np.ndarray | None,
    threshold_ratio: float,
    window_us: float,
    group_velocity: float,
) -> None:
    positions = shell.receiver_positions()
    tx_by_index = {
        item['index']: item
        for item in shell.transducer_positions()
        if item['ring'] == 'tx'
    }
    tx_pos = tx_by_index[case.tx]
    for rx_channel, rx_pos in enumerate(positions, start=1):
        signal = channels[:, rx_channel - 1]
        envelope = analytic_envelope(signal)
        peak_index = int(np.nanargmax(envelope))
        fft_freq, fft_amp, fft_phase = fft_at_frequency(time_s, signal, case.frequency_hz)
        tof = first_arrival(time_s, envelope, threshold_ratio)
        healthy_minus_current = float('nan')
        if healthy is not None and healthy.shape == channels.shape:
            healthy_envelope = analytic_envelope(healthy[:, rx_channel - 1])
            healthy_minus_current = float(np.nanmax(healthy_envelope) - np.nanmax(envelope))

        feature_rows.append({
            'sample_id': sample_id,
            'tx': case.tx,
            'rx_channel': rx_channel,
            'rx_pzt': rx_pos['index'],
            'frequency_hz': case.frequency_hz,
            'tof_first_s': tof,
            'hilbert_peak_amplitude_m': float(envelope[peak_index]),
            'hilbert_peak_time_s': float(time_s[peak_index]),
            'fft_bin_hz': fft_freq,
            'fft_amplitude_m': fft_amp,
            'fft_phase_rad': fft_phase,
            'max_abs_displacement_m': float(np.nanmax(np.abs(signal))),
            'healthy_minus_current_peak_m': healthy_minus_current,
            'waveform_csv': waveform_path.name,
        })
        summary_rows.append({
            'sample_id': sample_id,
            'tx': case.tx,
            'frequency_hz': case.frequency_hz,
            'rx_channel': rx_channel,
            'rx_pzt': rx_pos['index'],
            'theta_deg': rx_pos['theta_deg'],
            'x_mm': rx_pos['x_mm'],
            'y_mm': rx_pos['y_mm'],
            'z_mm': rx_pos['z_mm'],
            'max_abs_displacement_m': float(np.nanmax(np.abs(signal))),
            'hilbert_peak_amplitude_m': float(envelope[peak_index]),
            'hilbert_peak_time_s': float(time_s[peak_index]),
            'tof_first_s': tof,
        })
        for order in HELICAL_ORDERS:
            predicted = arrival_time_for_order(tx_pos, rx_pos, order, group_velocity)
            amp, peak_time = window_peak(time_s, envelope, predicted, window_us * 1e-6)
            helical_rows.append({
                'sample_id': sample_id,
                'tx': case.tx,
                'rx_channel': rx_channel,
                'rx_pzt': rx_pos['index'],
                'frequency_hz': case.frequency_hz,
                'helical_order': order,
                'predicted_arrival_s': predicted,
                'window_half_width_s': window_us * 1e-6,
                'order_peak_amplitude_m': amp,
                'order_peak_time_s': peak_time,
            })


def solve_export_sample(
    *,
    client,
    dataset: str,
    sample_id: str,
    defect_state: str,
    output_root: Path,
    cases: list[Case],
    defects,
    lobes,
    sample_metadata: dict[str, Any],
    healthy_waveform_root: Path | None = None,
    healthy_sample_id: str | None = None,
    threshold_ratio: float = 0.15,
    window_us: float = 35.0,
    group_velocity: float = 2522.0,
    clear_each_case: bool = True,
    heartbeat_s: float = DEFAULT_HEARTBEAT_S,
    reuse_sample_model: bool = True,
) -> SampleExportResult:
    waveform_dir = output_root / 'csv' / 'waveforms'
    feature_dir = output_root / 'csv' / 'tomography_features'
    metadata_dir = output_root / 'metadata'
    progress_path = output_root / 'progress' / f'{sample_id}_progress.jsonl'
    feature_files = [
        feature_dir / f'{sample_id}_tomography_features.csv',
        feature_dir / f'{sample_id}_helical_order_projections.csv',
        feature_dir / f'{sample_id}_receiver_summary.csv',
    ]
    waveform_files: list[Path] = []
    case_problems: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    helical_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    sample_started_s = time.monotonic()
    pbar = None
    if TQDM_AVAILABLE:
        pbar = tqdm(
            total=len(cases),
            desc=sample_id,
            unit='case',
            dynamic_ncols=True,
            leave=True,
        )

    def emit(case: Case | None, status: str, case_index: int, message: str) -> None:
        event = progress_event(
            sample_id=sample_id,
            dataset=dataset,
            defect_state=defect_state,
            case=case,
            status=status,
            case_index=case_index,
            case_count=len(cases),
            started_s=sample_started_s,
            message=message,
        )
        print_progress(event)
        append_progress(progress_path, event)

    def solve_export_case(
        *,
        model,
        case: Case,
        case_index: int,
        build_problems,
        reused_model: bool,
        start_message: str,
        clear_solution_after_export: bool,
    ) -> None:
        try:
            if pbar is not None:
                pbar.set_description(f'{sample_id} tx{case.tx:02d} f{int(case.frequency_hz)}')
                pbar.set_postfix_str('setting parameters', refresh=True)
            emit(case, 'case_start', case_index, start_message)
            model.parameter('tx', str(case.tx))
            model.parameter('pzt_fc', f'{case.frequency_hz:.12g}[Hz]')
            if pbar is not None:
                pbar.set_postfix_str('COMSOL solve running', refresh=True)
            solve_info = run_model_solve_with_heartbeat(
                model,
                progress_path=progress_path,
                sample_id=sample_id,
                dataset=dataset,
                defect_state=defect_state,
                case=case,
                case_index=case_index,
                case_count=len(cases),
                sample_started_s=sample_started_s,
                heartbeat_s=heartbeat_s,
                pbar=pbar,
            )

            if pbar is not None:
                pbar.set_postfix_str('exporting waveforms/features', refresh=True)
            solution_dataset, positions, expressions = receiver_points_for_current_solution(model)
            time_s, channels = evaluate_current_solution(model, solution_dataset, positions, expressions)
            waveform_check = validate_waveforms(time_s, channels)

            waveform_path = waveform_dir / f'{sample_id}_tx{case.tx:02d}_f{int(case.frequency_hz)}Hz_waveforms.csv'
            write_waveform(waveform_path, time_s, channels)
            waveform_files.append(waveform_path)

            healthy = load_healthy_waveform(healthy_waveform_root, case, healthy_sample_id)
            append_case_features(
                feature_rows=feature_rows,
                helical_rows=helical_rows,
                summary_rows=summary_rows,
                sample_id=sample_id,
                case=case,
                time_s=time_s,
                channels=channels,
                waveform_path=waveform_path,
                healthy=healthy,
                threshold_ratio=threshold_ratio,
                window_us=window_us,
                group_velocity=group_velocity,
            )
            write_feature_csvs(feature_files, feature_rows, helical_rows, summary_rows)
            cleared_solution_data = clear_solution_data(model) if clear_solution_after_export else []
            case_problems.append({
                'tx': case.tx,
                'frequency_hz': case.frequency_hz,
                'build_problems': build_problems,
                'post_solve_problems': solve_info['post_solve_problems'],
                'solve_elapsed_s': solve_info['solve_elapsed_s'],
                'waveform_check': waveform_check,
                'waveform_csv': str(waveform_path),
                'solution_dataset': solution_dataset.tag(),
                'saved_mph': False,
                'reused_sample_model': reused_model,
                'cleared_solution_data': cleared_solution_data,
            })
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix_str(
                    f'done; nonzero {waveform_check["nonzero_channels"]}/{waveform_check["channel_count"]}',
                    refresh=True,
                )
            emit(
                case,
                'case_done',
                case_index,
                (
                    f'wrote {waveform_path}; solve_elapsed={format_duration(solve_info["solve_elapsed_s"])}; '
                    f'nonzero_channels={waveform_check["nonzero_channels"]}/{waveform_check["channel_count"]}'
                ),
            )
        except Exception as error:
            emit(case, 'case_failed', case_index, f'{type(error).__name__}: {error}')
            raise

    try:
        emit(None, 'sample_start', 0, 'starting sample solve/export')

        if reuse_sample_model:
            model = None
            try:
                tx_indices = tuple(dict.fromkeys(case.tx for case in cases))
                frequencies = tuple(dict.fromkeys(case.frequency_hz for case in cases))
                shell.SWEEP = shell.SweepConfig(
                    transmitter_indices=tx_indices,
                    frequencies_hz=frequencies,
                    use_parametric_sweep=False,
                )
                if pbar is not None:
                    pbar.set_postfix_str('building reusable sample model', refresh=True)
                emit(None, 'sample_model_build_start', 0, 'building one reusable COMSOL model/mesh for this sample')
                build_started_s = time.monotonic()
                with BlockingHeartbeat(
                    progress_path=progress_path,
                    sample_id=sample_id,
                    dataset=dataset,
                    defect_state=defect_state,
                    case=None,
                    case_index=0,
                    case_count=len(cases),
                    sample_started_s=sample_started_s,
                    operation_started_s=build_started_s,
                    status='sample_model_build_running',
                    message='COMSOL reusable sample model build/mesh is still running',
                    interval_s=heartbeat_s,
                    pbar=pbar,
                    postfix_prefix='building model',
                ):
                    model, build_problems = shell.build_model_object(client, sample_id, defects=defects, lobes=lobes)
                emit(
                    None,
                    'sample_model_build_done',
                    0,
                    f'built reusable sample model in {format_duration(time.monotonic() - build_started_s)}',
                )
                for case_index, case in enumerate(cases, start=1):
                    solve_export_case(
                        model=model,
                        case=case,
                        case_index=case_index,
                        build_problems=build_problems if case_index == 1 else [],
                        reused_model=True,
                        start_message='setting tx/frequency and solving reusable COMSOL model',
                        clear_solution_after_export=True,
                    )
            finally:
                if model is not None:
                    client.remove(model)
                if clear_each_case:
                    client.clear()
        else:
            for case_index, case in enumerate(cases, start=1):
                shell.SWEEP = shell.SweepConfig(
                    transmitter_indices=(case.tx,),
                    frequencies_hz=(case.frequency_hz,),
                    use_parametric_sweep=False,
                )
                model_name = f'{sample_id}_tx{case.tx:02d}_f{int(case.frequency_hz)}Hz'
                model = None
                try:
                    if pbar is not None:
                        pbar.set_description(f'{sample_id} tx{case.tx:02d} f{int(case.frequency_hz)}')
                        pbar.set_postfix_str('building', refresh=True)
                    build_started_s = time.monotonic()
                    with BlockingHeartbeat(
                        progress_path=progress_path,
                        sample_id=sample_id,
                        dataset=dataset,
                        defect_state=defect_state,
                        case=case,
                        case_index=case_index,
                        case_count=len(cases),
                        sample_started_s=sample_started_s,
                        operation_started_s=build_started_s,
                        status='case_model_build_running',
                        message='COMSOL per-case model build/mesh is still running',
                        interval_s=heartbeat_s,
                        pbar=pbar,
                        postfix_prefix='building model',
                    ):
                        model, build_problems = shell.build_model_object(client, model_name, defects=defects, lobes=lobes)
                    solve_export_case(
                        model=model,
                        case=case,
                        case_index=case_index,
                        build_problems=build_problems,
                        reused_model=False,
                        start_message='solving per-case COMSOL model',
                        clear_solution_after_export=False,
                    )
                finally:
                    if model is not None:
                        client.remove(model)
                    if clear_each_case:
                        client.clear()
    finally:
        if pbar is not None:
            pbar.close()

    write_feature_csvs(feature_files, feature_rows, helical_rows, summary_rows)

    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / f'{sample_id}.json'
    label_metadata = defect_labels.write_label_package(
        output_root / 'labels',
        sample_id,
        sample_metadata,
        pipe=shell.PIPE,
    )
    metadata = {
        'dataset': dataset,
        'sample_id': sample_id,
        'defect_state': defect_state,
        'sample': sample_metadata,
        'model': shell.model_metadata(dataset, defect_state, None, {'case_problems': case_problems}),
        'defect_label': label_metadata,
        'streaming_export': {
            'saved_mph': False,
            'case_count': len(cases),
            'tx': [case.tx for case in cases],
            'frequencies_hz': [case.frequency_hz for case in cases],
            'clear_each_case': clear_each_case,
            'reuse_sample_model': reuse_sample_model,
            'clear_solution_after_each_export': reuse_sample_model,
            'heartbeat_s': heartbeat_s,
            'progress_file': str(progress_path),
            'waveform_files': [str(item) for item in waveform_files],
            'feature_files': [str(item) for item in feature_files],
            'note': 'COMSOL solution fields were evaluated in-session, exported, and then discarded.',
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

    done_event = progress_event(
        sample_id=sample_id,
        dataset=dataset,
        defect_state=defect_state,
        case=None,
        status='sample_done',
        case_index=len(cases),
        case_count=len(cases),
        started_s=sample_started_s,
        message=f'wrote metadata {metadata_path}',
    )
    print_progress(done_event)
    append_progress(progress_path, done_event)

    return SampleExportResult(
        sample_id=sample_id,
        dataset=dataset,
        defect_state=defect_state,
        metadata_path=metadata_path,
        waveform_files=waveform_files,
        feature_files=feature_files,
        case_count=len(cases),
    )


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

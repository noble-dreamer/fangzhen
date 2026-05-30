"""Solve one COMSOL shell case, export CSV, then discard the model.

This avoids saving transient solution fields into large MPH files.  Each
case uses one transmitter and one center frequency, so COMSOL never stores
the full 16 x 3 parametric solution set in one model.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import simple_shell_common as shell


HELICAL_ORDERS = (-1, 0, 1)


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


def radial_expression(position: dict[str, float]) -> str:
    theta = math.radians(position['theta_deg'])
    return f'({math.cos(theta):.12g})*u+({math.sin(theta):.12g})*v'


def ensure_cutpoint(model, solution_dataset: str, position: dict[str, Any]):
    name = f'receiver PZT {position["index"]:02d} point'
    for dataset in model / 'datasets':
        if dataset.name() == name:
            dataset.property('data', solution_dataset)
            return dataset
    dataset = (model / 'datasets').create('CutPoint3D', name=name)
    dataset.property('data', solution_dataset)
    dataset.property('pointx', f'{position["x_mm"]:.12g}[mm]')
    dataset.property('pointy', f'{position["y_mm"]:.12g}[mm]')
    dataset.property('pointz', f'{position["z_mm"]:.12g}[mm]')
    return dataset


def evaluate_current_solution(model, datasets, expressions: list[str]) -> tuple[np.ndarray, np.ndarray]:
    first_dataset = datasets[0].name()
    time_s = np.asarray(model.evaluate('t', dataset=first_dataset), dtype=float).reshape(-1)
    channels = []
    for dataset, expression in zip(datasets, expressions, strict=True):
        values = np.asarray(
            model.evaluate(expression, unit='m', dataset=dataset.name()),
            dtype=float,
        ).reshape(-1)
        channels.append(values)
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


def print_progress(event: dict[str, Any]) -> None:
    case_text = (
        'sample'
        if event['tx'] is None
        else f'tx={event["tx"]} f={event["frequency_hz"]}Hz'
    )
    print(
        '[progress] '
        f'{event["sample_id"]} {event["status"]} '
        f'{event["case_index"]}/{event["case_count"]} '
        f'{case_text} '
        f'elapsed={format_duration(float(event["elapsed_s"]))} '
        f'eta={format_duration(float(event["eta_s"])) if event["eta_s"] is not None else "unknown"} '
        f'- {event["message"]}',
        flush=True,
    )


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
) -> SampleExportResult:
    waveform_dir = output_root / 'csv' / 'waveforms'
    feature_dir = output_root / 'csv' / 'tomography_features'
    metadata_dir = output_root / 'metadata'
    progress_path = output_root / 'progress' / f'{sample_id}_progress.jsonl'
    waveform_files: list[Path] = []
    case_problems: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    helical_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    sample_started_s = time.monotonic()

    start_event = progress_event(
        sample_id=sample_id,
        dataset=dataset,
        defect_state=defect_state,
        case=None,
        status='sample_start',
        case_index=0,
        case_count=len(cases),
        started_s=sample_started_s,
        message='starting sample solve/export',
    )
    print_progress(start_event)
    append_progress(progress_path, start_event)

    for case_index, case in enumerate(cases, start=1):
        shell.SWEEP = shell.SweepConfig(
            transmitter_indices=(case.tx,),
            frequencies_hz=(case.frequency_hz,),
            use_parametric_sweep=False,
        )
        model_name = f'{sample_id}_tx{case.tx:02d}_f{int(case.frequency_hz)}Hz'
        model = None
        try:
            event = progress_event(
                sample_id=sample_id,
                dataset=dataset,
                defect_state=defect_state,
                case=case,
                status='case_start',
                case_index=case_index,
                case_count=len(cases),
                started_s=sample_started_s,
                message='building and solving COMSOL case without saving MPH',
            )
            print_progress(event)
            append_progress(progress_path, event)
            model, problems = shell.build_model_object(client, model_name, defects=defects, lobes=lobes)
            model.parameter('tx', str(case.tx))
            model.parameter('pzt_fc', f'{case.frequency_hz:.12g}[Hz]')
            model.solve()

            positions = shell.receiver_positions()
            datasets = [ensure_cutpoint(model, 'dset1', position) for position in positions]
            expressions = [radial_expression(position) for position in positions]
            time_s, channels = evaluate_current_solution(model, datasets, expressions)

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
            case_problems.append({
                'tx': case.tx,
                'frequency_hz': case.frequency_hz,
                'build_problems': problems,
                'post_solve_problems': model.problems(),
                'waveform_csv': str(waveform_path),
                'saved_mph': False,
            })
            event = progress_event(
                sample_id=sample_id,
                dataset=dataset,
                defect_state=defect_state,
                case=case,
                status='case_done',
                case_index=case_index,
                case_count=len(cases),
                started_s=sample_started_s,
                message=f'wrote {waveform_path}',
            )
            print_progress(event)
            append_progress(progress_path, event)
        except Exception as error:
            event = progress_event(
                sample_id=sample_id,
                dataset=dataset,
                defect_state=defect_state,
                case=case,
                status='case_failed',
                case_index=case_index,
                case_count=len(cases),
                started_s=sample_started_s,
                message=f'{type(error).__name__}: {error}',
            )
            print_progress(event)
            append_progress(progress_path, event)
            raise
        finally:
            if model is not None:
                client.remove(model)
            if clear_each_case:
                client.clear()

    feature_files = [
        feature_dir / f'{sample_id}_tomography_features.csv',
        feature_dir / f'{sample_id}_helical_order_projections.csv',
        feature_dir / f'{sample_id}_receiver_summary.csv',
    ]
    write_dict_csv(feature_files[0], feature_rows)
    write_dict_csv(feature_files[1], helical_rows)
    write_dict_csv(feature_files[2], summary_rows)

    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / f'{sample_id}.json'
    metadata = {
        'dataset': dataset,
        'sample_id': sample_id,
        'defect_state': defect_state,
        'sample': sample_metadata,
        'model': shell.model_metadata(dataset, defect_state, None, {'case_problems': case_problems}),
        'streaming_export': {
            'saved_mph': False,
            'case_count': len(cases),
            'tx': [case.tx for case in cases],
            'frequencies_hz': [case.frequency_hz for case in cases],
            'clear_each_case': clear_each_case,
            'progress_file': str(progress_path),
            'waveform_files': [str(item) for item in waveform_files],
            'feature_files': [str(item) for item in feature_files],
            'note': 'COMSOL solution fields were evaluated in-session and then discarded.',
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

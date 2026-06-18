"""Frequency-domain COMSOL helpers for the simple Dataset A shell model."""

from __future__ import annotations

import csv
import inspect
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
SIMPLE_ROOT = ROOT.parent
if str(SIMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMPLE_ROOT))

import defect_label_common as defect_labels
import simple_shell_common as shell
import streaming_export_common as streaming


OUTPUT_ROOT = ROOT / 'output'
DEFAULT_FREQUENCIES = '30000,35000,40000,45000,50000,55000,60000,65000,70000'
DEFAULT_SWEEP_START_KHZ = 20.0
DEFAULT_SWEEP_STOP_KHZ = 100.0
DEFAULT_SWEEP_STEP_KHZ = 5.0
DEFAULT_TX = ','.join(str(index) for index in range(1, 17))

_PATCHED = False
_ORIGINAL_LOAD_VECTOR_EXPRESSION = None
_ORIGINAL_CREATE_FUNCTIONS = None
_ORIGINAL_CREATE_STUDY = None


@dataclass(frozen=True)
class FrequencyCase:
    tx: int
    frequency_hz: float


@dataclass(frozen=True)
class FrequencySampleExportResult:
    sample_id: str
    dataset: str
    defect_state: str
    metadata_path: Path
    response_files: list[Path]
    feature_files: list[Path]
    case_count: int


def parse_int_list(text: str) -> tuple[int, ...]:
    return streaming.parse_int_list(text)


def parse_float_list(text: str) -> tuple[float, ...]:
    return streaming.parse_float_list(text)


def frequency_range_hz(start_khz: float, stop_khz: float, step_khz: float) -> tuple[float, ...]:
    if step_khz <= 0.0:
        raise ValueError(f'frequency step must be positive, got {step_khz}')
    if stop_khz < start_khz:
        raise ValueError(f'frequency stop must be >= start, got {start_khz}..{stop_khz}')
    values = []
    current = float(start_khz)
    # Include the endpoint despite small floating-point roundoff.
    while current <= stop_khz + 1e-9:
        values.append(round(current * 1000.0, 9))
        current += step_khz
    if not values:
        raise ValueError('frequency range produced no frequencies')
    return tuple(values)


def format_frequency_list(frequencies_hz: tuple[float, ...]) -> str:
    return ','.join(f'{frequency:.12g}' for frequency in frequencies_hz)


def make_cases(tx_indices: tuple[int, ...], frequencies_hz: tuple[float, ...]) -> list[FrequencyCase]:
    return [FrequencyCase(tx=tx, frequency_hz=frequency) for tx in tx_indices for frequency in frequencies_hz]


def configure_dataset_a_frequency(*, use_parametric_sweep: bool = False) -> None:
    """Configure the shared shell module for Dataset A frequency-domain runs."""
    shell.MODEL_FAMILY = 'Dataset A simple shell frequency-domain solve/export'
    shell.POSITION_PERTURBATIONS = {}
    shell.AMPLITUDE_SCALE = {}
    shell.MATERIAL = shell.MaterialConfig()
    shell.ABSORBING_LAYER = shell.AbsorbingLayerConfig(enabled=True)
    shell.DEFECT_MODEL = shell.DefectModelConfig(corrosion_surface='outer')
    shell.SOLVER = shell.SolverConfig(solve=False, direct_linear_solver='pardiso')
    shell.SWEEP = shell.SweepConfig(
        transmitter_indices=tuple(range(1, 17)),
        frequencies_hz=tuple(parse_float_list(DEFAULT_FREQUENCIES)),
        use_parametric_sweep=use_parametric_sweep,
    )
    shell.RECEIVER_INDICES = tuple(range(17, 33))
    shell.DATASET_NOTES = [
        'Dataset A frequency-domain shell model.',
        'Shell end absorbing layers are enabled to suppress end reflections.',
        'No transducer position, amplitude, or material perturbation.',
        'PZT solids are replaced by equivalent shell face-load windows.',
        'Frequency-domain excitation uses harmonic load amplitude, not pztpulse(t).',
        'Receivers are patch-weighted radial displacement averages using intop_shell.',
    ]
    enable_frequency_domain_mode()


def apply_solver_arguments(args) -> None:
    shell.SOLVER = shell.SolverConfig(
        solve=False,
        relative_tolerance=args.relative_tolerance,
        direct_linear_solver=args.linear_solver,
        cudss_precision=args.cudss_precision,
        pardiso_use_cluster=args.pardiso_use_cluster,
    )
    streaming.console_log(
        '[solver] '
        f'analysis=frequency_domain, '
        f'linear_solver={args.linear_solver}, '
        f'cudss_precision={args.cudss_precision}, '
        f'relative_tolerance={args.relative_tolerance}'
    )
    shell.CREATE_RECEIVER_DATASETS = bool(args.include_comsol_marker_datasets)
    shell.CREATE_VISUAL_MARKER_DATASETS = bool(args.include_comsol_marker_datasets)


def add_frequency_solver_arguments(parser) -> None:
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
        help='Rebuild geometry/mesh/solver tree for every tx/frequency case.',
    )
    parser.add_argument(
        '--include-comsol-marker-datasets',
        action='store_true',
        help='Create receiver/transmitter CutPoint datasets for manual visual inspection.',
    )


def harmonic_load_vector_expression() -> list[str]:
    """Equivalent shell face-load amplitude for frequency-domain harmonic solves."""
    x_terms: list[str] = []
    y_terms: list[str] = []
    for item in shell.transmitter_positions():
        theta = math.radians(item['theta_deg'])
        gate = f'if(tx=={item["index"]},1,0)'
        window = shell.patch_window_expr(item['theta_deg'], item['z_mm'])
        scale = f'{item["amplitude_scale"]:.9g}*{gate}*F0/pzt_A*({window})'
        x_terms.append(f'({math.cos(theta):.12g})*({scale})')
        y_terms.append(f'({math.sin(theta):.12g})*({scale})')
    return ['+'.join(x_terms) or '0', '+'.join(y_terms) or '0', '0']


def create_frequency_functions(model) -> None:
    """Frequency-domain load is harmonic, so no time pulse function is needed."""
    return None


def create_frequency_study(model):
    study = (model / 'studies').create(name='simple shell displacement frequency domain')
    study.java.setGenPlots(False)
    study.java.setGenConv(False)
    step = study.create('Frequency', name='frequency domain')
    step.property('plist', 'pzt_fc')
    if shell.SWEEP.use_parametric_sweep:
        parametric = study.create('Parametric', name='tx and frequency sweep')
        parametric.property('pname', ['tx', 'pzt_fc'])
        parametric.property('sweeptype', 'filled')
        parametric.property('plistarr', [
            ' '.join(str(index) for index in shell.SWEEP.transmitter_indices),
            ' '.join(f'{freq:.9g}[Hz]' for freq in shell.SWEEP.frequencies_hz),
        ])
        parametric.property('punit', ['', 'Hz'])
    study.java.createAutoSequences('sol')
    shell.tune_solver(model)
    return study


def enable_frequency_domain_mode() -> None:
    """Patch the shared shell builder so new models are frequency-domain models."""
    global _PATCHED
    global _ORIGINAL_LOAD_VECTOR_EXPRESSION
    global _ORIGINAL_CREATE_FUNCTIONS
    global _ORIGINAL_CREATE_STUDY

    if not _PATCHED:
        _ORIGINAL_LOAD_VECTOR_EXPRESSION = shell.load_vector_expression
        _ORIGINAL_CREATE_FUNCTIONS = shell.create_functions
        _ORIGINAL_CREATE_STUDY = shell.create_study
        _PATCHED = True
    shell.load_vector_expression = harmonic_load_vector_expression
    shell.create_functions = create_frequency_functions
    shell.create_study = create_frequency_study


def solution_dataset_node(model):
    return streaming.solution_dataset_node(model)


def complex_receiver_channels(model, solution_dataset) -> np.ndarray:
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
            values = np.asarray(raw[0]) + 1j * np.asarray(raw[1])
        else:
            values = np.asarray(raw[0], dtype=float).astype(complex)
    finally:
        evaluation.remove()
    channels = complex_result_channels(values, len(expressions))
    if not np.all(np.isfinite(np.real(channels))) or not np.all(np.isfinite(np.imag(channels))):
        raise RuntimeError('COMSOL frequency receiver export returned non-finite complex data.')
    return channels


def complex_result_channels(values: np.ndarray, channel_count: int) -> np.ndarray:
    array = np.asarray(values)
    squeezed = np.asarray(array).squeeze()
    if squeezed.ndim == 0 and channel_count == 1:
        return squeezed.reshape(1)
    if squeezed.size == channel_count:
        return squeezed.reshape(channel_count)
    for axis, size in enumerate(array.shape):
        if size != channel_count:
            continue
        moved = np.moveaxis(array, axis, 0).reshape(channel_count, -1)
        if moved.shape[1] == 1:
            return moved[:, 0]
    raise RuntimeError(
        f'COMSOL frequency receiver data shape {array.shape} does not match channel_count={channel_count}'
    )


def response_rows(
    *,
    sample_id: str,
    dataset: str,
    defect_state: str,
    case: FrequencyCase,
    channels: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    positions = shell.receiver_positions()
    for rx_channel, (position, value) in enumerate(zip(positions, channels, strict=True), start=1):
        rows.append({
            'sample_id': sample_id,
            'dataset': dataset,
            'defect_state': defect_state,
            'tx': case.tx,
            'frequency_hz': case.frequency_hz,
            'rx_channel': rx_channel,
            'rx_pzt': position['index'],
            'theta_deg': position['theta_deg'],
            'x_mm': position['x_mm'],
            'y_mm': position['y_mm'],
            'z_mm': position['z_mm'],
            'real_ur_m': float(np.real(value)),
            'imag_ur_m': float(np.imag(value)),
            'abs_ur_m': float(abs(value)),
            'phase_rad': float(np.angle(value)),
        })
    return rows


def write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_frequency_npz(
    path: Path,
    *,
    h_matrix: np.ndarray,
    tx_indices: tuple[int, ...],
    frequencies_hz: tuple[float, ...],
    rx_indices: tuple[int, ...],
    completed_mask: np.ndarray,
    sample_id: str,
    dataset: str,
    defect_state: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        H_real=np.real(h_matrix),
        H_imag=np.imag(h_matrix),
        completed_mask=completed_mask,
        tx_indices=np.asarray(tx_indices, dtype=np.int32),
        rx_indices=np.asarray(rx_indices, dtype=np.int32),
        frequencies_hz=np.asarray(frequencies_hz, dtype=float),
        sample_id=np.asarray(sample_id),
        dataset=np.asarray(dataset),
        defect_state=np.asarray(defect_state),
    )


def write_label_package_compatible(
    output_dir: Path,
    sample_id: str,
    sample_metadata: dict[str, Any],
    *,
    write_label_preview: bool,
) -> dict[str, Any]:
    kwargs = {
        'pipe': shell.PIPE,
    }
    if 'write_preview_png' in inspect.signature(defect_labels.write_label_package).parameters:
        kwargs['write_preview_png'] = write_label_preview
    elif not write_label_preview:
        streaming.console_log(
            '[label] defect_label_common.write_label_package does not support --skip-label-preview; '
            'continuing with preview PNG enabled. Update simple/defect_label_common.py to enable skipping it.'
        )
    return defect_labels.write_label_package(
        output_dir,
        sample_id,
        sample_metadata,
        **kwargs,
    )


def write_frequency_build_log(path: Path, saved: list[Path], problems: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    notes = '\n'.join(f'- {note}' for note in shell.DATASET_NOTES) or '- None'
    path.write_text(
        f"""# Frequency-Domain Simple Shell Build Log

## Generated files

{chr(10).join(f'- `{item}`' for item in saved)}

## Model family

{shell.MODEL_FAMILY}

## Notes

{notes}

## Frequency-domain changes

- Study type: `Frequency`, with frequency list expression `pzt_fc`.
- Time pulse `pztpulse(t)` is not used in the face load.
- The equivalent load is a harmonic shell face-load amplitude `F0/pzt_A * window_tx`.
- Dataset A absorbing layers are enabled through the same axial Rayleigh damping ramp as the time-domain Dataset A model.
- Receivers are the same patch-weighted averages: `intop_shell(w_rx*u_r)/intop_shell(w_rx)`.

## COMSOL Model Builder checks

- Equivalent excitation: `Component 1 > Shell Mechanics > equivalent transducer face load`.
- Active transmitter/frequency: `Global Definitions > Parameters`, then `tx` and `pzt_fc`.
- Frequency study: `Study > simple shell displacement frequency domain`.
- Receiver weighted averages: `Results > Derived Values > receiver patch weighted average radial displacement`.
- Optional marker datasets: `Results > Datasets > transmitter PZT marker points` and `receiver PZT marker points`.

## COMSOL self-check

```json
{json.dumps(problems, ensure_ascii=False, indent=2, default=str)}
```
""",
        encoding='utf-8',
    )


def model_metadata(
    dataset: str,
    defect_state: str,
    model_path: Path | str | None,
    problems: Any,
    *,
    analysis_type: str = 'frequency_domain',
) -> dict[str, Any]:
    metadata = shell.model_metadata(dataset, defect_state, model_path, problems)
    metadata['analysis_type'] = analysis_type
    metadata['frequency_domain'] = {
        'study': 'Frequency',
        'frequency_expression': 'pzt_fc',
        'load_expression': 'F0/pzt_A * window_tx, harmonic amplitude; pztpulse(t) is not used',
        'receiver_export': 'complex patch-weighted radial displacement',
        'output_units': 'm complex amplitude',
    }
    return metadata


def solve_export_frequency_sample(
    *,
    client,
    dataset: str,
    sample_id: str,
    defect_state: str,
    output_root: Path,
    cases: list[FrequencyCase],
    defects,
    lobes,
    sample_metadata: dict[str, Any],
    clear_each_case: bool = True,
    heartbeat_s: float = streaming.DEFAULT_HEARTBEAT_S,
    reuse_sample_model: bool = True,
    write_label_preview: bool = True,
    keep_case_csv: bool = False,
) -> FrequencySampleExportResult:
    response_dir = output_root / 'csv' / 'frequency_response'
    npz_dir = output_root / 'frequency_response'
    metadata_dir = output_root / 'metadata'
    progress_path = output_root / 'progress' / f'{sample_id}_progress.jsonl'
    cumulative_csv = response_dir / f'{sample_id}_frequency_response.csv'
    npz_path = npz_dir / f'{sample_id}_H_complex.npz'
    tx_indices = tuple(dict.fromkeys(case.tx for case in cases))
    frequencies_hz = tuple(dict.fromkeys(case.frequency_hz for case in cases))
    rx_indices = tuple(position['index'] for position in shell.receiver_positions())
    h_matrix = np.full(
        (len(tx_indices), len(rx_indices), len(frequencies_hz)),
        np.nan + 1j * np.nan,
        dtype=complex,
    )
    completed_mask = np.zeros((len(tx_indices), len(frequencies_hz)), dtype=bool)
    response_files: list[Path] = []
    response_table_rows: list[dict[str, Any]] = []
    case_problems: list[dict[str, Any]] = []
    sample_started_s = time.monotonic()

    def emit(case: FrequencyCase | None, status: str, case_index: int, message: str) -> None:
        event = streaming.progress_event(
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
        streaming.print_progress(event)
        streaming.append_progress(progress_path, event)

    def checkpoint_outputs() -> None:
        write_dict_csv(cumulative_csv, response_table_rows)
        write_frequency_npz(
            npz_path,
            h_matrix=h_matrix,
            tx_indices=tx_indices,
            frequencies_hz=frequencies_hz,
            rx_indices=rx_indices,
            completed_mask=completed_mask,
            sample_id=sample_id,
            dataset=dataset,
            defect_state=defect_state,
        )

    def solve_export_case(
        *,
        model,
        case: FrequencyCase,
        case_index: int,
        build_problems,
        reused_model: bool,
        start_message: str,
        clear_solution_after_export: bool,
    ) -> None:
        try:
            emit(case, 'case_start', case_index, start_message)
            model.parameter('tx', str(case.tx))
            model.parameter('pzt_fc', f'{case.frequency_hz:.12g}[Hz]')
            solve_info = streaming.run_model_solve_with_heartbeat(
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
            )
            solution_dataset = solution_dataset_node(model)
            channels = complex_receiver_channels(model, solution_dataset)
            rows = response_rows(
                sample_id=sample_id,
                dataset=dataset,
                defect_state=defect_state,
                case=case,
                channels=channels,
            )
            case_csv = response_dir / f'{sample_id}_tx{case.tx:02d}_f{int(case.frequency_hz)}Hz_frequency_response.csv'
            if keep_case_csv:
                write_dict_csv(case_csv, rows)
                response_files.append(case_csv)
            response_table_rows.extend(rows)
            tx_index = tx_indices.index(case.tx)
            frequency_index = frequencies_hz.index(case.frequency_hz)
            h_matrix[tx_index, :, frequency_index] = channels
            completed_mask[tx_index, frequency_index] = True
            checkpoint_outputs()
            cleared_solution_data = streaming.clear_solution_data(model) if clear_solution_after_export else []
            max_abs = float(np.nanmax(np.abs(channels))) if channels.size else float('nan')
            nonzero_channels = int(np.sum(np.abs(channels) > 0.0))
            case_problems.append({
                'tx': case.tx,
                'frequency_hz': case.frequency_hz,
                'build_problems': build_problems,
                'post_solve_problems': solve_info['post_solve_problems'],
                'solve_elapsed_s': solve_info['solve_elapsed_s'],
                'response_csv': str(case_csv) if keep_case_csv else None,
                'solution_dataset': solution_dataset.tag(),
                'nonzero_channels': nonzero_channels,
                'channel_count': int(channels.size),
                'max_abs_ur_m': max_abs,
                'saved_mph': False,
                'reused_sample_model': reused_model,
                'cleared_solution_data': cleared_solution_data,
            })
            emit(
                case,
                'case_done',
                case_index,
                (
                    f'wrote {cumulative_csv.name}'
                    + (f' and {case_csv.name}' if keep_case_csv else '')
                    + f'; solve_elapsed={streaming.format_duration(solve_info["solve_elapsed_s"])}; '
                    f'nonzero_channels={nonzero_channels}/{channels.size}; max_abs_ur_m={max_abs:.6e}'
                ),
            )
        except Exception as error:
            emit(case, 'case_failed', case_index, f'{type(error).__name__}: {error}')
            raise

    emit(None, 'sample_start', 0, 'starting frequency-domain sample solve/export')
    emit(None, 'label_start', 0, 'writing defect label arrays before COMSOL model build')
    try:
        label_metadata = write_label_package_compatible(
            output_root / 'labels',
            sample_id,
            sample_metadata,
            write_label_preview=write_label_preview,
        )
    except Exception as error:
        emit(None, 'label_failed', 0, f'{type(error).__name__}: {error}')
        raise
    emit(None, 'label_done', 0, 'defect label arrays written')
    try:
        if reuse_sample_model:
            model = None
            try:
                shell.SWEEP = shell.SweepConfig(
                    transmitter_indices=tx_indices,
                    frequencies_hz=frequencies_hz,
                    use_parametric_sweep=False,
                )
                emit(None, 'sample_model_build_start', 0, 'building one reusable frequency-domain COMSOL model/mesh')
                build_started_s = time.monotonic()
                with streaming.BlockingHeartbeat(
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
                    message='COMSOL reusable frequency-domain model build/mesh is still running',
                    interval_s=heartbeat_s,
                    postfix_prefix='building frequency model',
                ):
                    model, build_problems = shell.build_model_object(client, sample_id, defects=defects, lobes=lobes)
                emit(
                    None,
                    'sample_model_build_done',
                    0,
                    f'built reusable frequency-domain model in {streaming.format_duration(time.monotonic() - build_started_s)}',
                )
                for case_index, case in enumerate(cases, start=1):
                    solve_export_case(
                        model=model,
                        case=case,
                        case_index=case_index,
                        build_problems=build_problems if case_index == 1 else [],
                        reused_model=True,
                        start_message='setting tx/frequency and solving reusable frequency-domain COMSOL model',
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
                    build_started_s = time.monotonic()
                    with streaming.BlockingHeartbeat(
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
                        message='COMSOL per-case frequency-domain model build/mesh is still running',
                        interval_s=heartbeat_s,
                        postfix_prefix='building frequency model',
                    ):
                        model, build_problems = shell.build_model_object(client, model_name, defects=defects, lobes=lobes)
                    solve_export_case(
                        model=model,
                        case=case,
                        case_index=case_index,
                        build_problems=build_problems,
                        reused_model=False,
                        start_message='solving per-case frequency-domain COMSOL model',
                        clear_solution_after_export=False,
                    )
                finally:
                    if model is not None:
                        client.remove(model)
                    if clear_each_case:
                        client.clear()
    finally:
        checkpoint_outputs()

    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / f'{sample_id}.json'
    metadata = {
        'dataset': dataset,
        'sample_id': sample_id,
        'defect_state': defect_state,
        'sample': sample_metadata,
        'model': model_metadata(dataset, defect_state, None, {'case_problems': case_problems}),
        'defect_label': label_metadata,
        'frequency_export': {
            'saved_mph': False,
            'case_count': len(cases),
            'tx': [case.tx for case in cases],
            'frequencies_hz': [case.frequency_hz for case in cases],
            'clear_each_case': clear_each_case,
            'reuse_sample_model': reuse_sample_model,
            'clear_solution_after_each_export': reuse_sample_model,
            'heartbeat_s': heartbeat_s,
            'keep_case_csv': keep_case_csv,
            'progress_file': str(progress_path),
            'response_files': [str(item) for item in response_files],
            'cumulative_response_csv': str(cumulative_csv),
            'complex_response_npz': str(npz_path),
            'note': 'Frequency-domain COMSOL solutions were evaluated in-session, exported, and then discarded.',
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    done_event = streaming.progress_event(
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
    streaming.print_progress(done_event)
    streaming.append_progress(progress_path, done_event)
    return FrequencySampleExportResult(
        sample_id=sample_id,
        dataset=dataset,
        defect_state=defect_state,
        metadata_path=metadata_path,
        response_files=response_files,
        feature_files=[cumulative_csv, npz_path],
        case_count=len(cases),
    )


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[str, dict[str, Any]] = {}
    if path.exists():
        with path.open('r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                sample_id = row.get('sample_id')
                if sample_id:
                    merged[sample_id] = dict(row)
    for row in rows:
        sample_id = str(row.get('sample_id', ''))
        if sample_id:
            if row.get('status') == 'skipped_existing' and sample_id in merged:
                continue
            merged[sample_id] = row
    output_rows = list(merged.values())
    fieldnames: list[str] = []
    for row in output_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

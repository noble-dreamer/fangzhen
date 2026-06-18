"""Stream-solve Dataset A shell cases in the frequency domain."""

from __future__ import annotations

import argparse
import faulthandler
import re
from pathlib import Path

faulthandler.enable(all_threads=True)

print('[startup] importing COMSOL/mph modules...', flush=True)
import frequency_domain_common as fcommon
import simple_defect_common as defects
print('[startup] imports loaded.', flush=True)


OUTPUT_ROOT = fcommon.OUTPUT_ROOT / 'streaming_dataset_a_frequency_shell'
SAMPLE_ID_PATTERN = re.compile(r'dataset_a_frequency_sample_(\d{4,})')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Solve Dataset A simple shell frequency-domain cases and export complex receiver responses.'
    )
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument(
        '--start-id',
        type=int,
        default=None,
        help='First damaged sample numeric id. Default: auto-select the first non-overlapping id segment.',
    )
    parser.add_argument('--seed0', type=int, default=710000)
    parser.add_argument('--tx', nargs='+', default=[fcommon.DEFAULT_TX])
    parser.add_argument('--frequencies', nargs='+', default=[fcommon.DEFAULT_FREQUENCIES])
    parser.add_argument(
        '--frequency-start-khz',
        type=float,
        default=None,
        help='Start of automatic frequency sweep in kHz. Use with --frequency-stop-khz and --frequency-step-khz.',
    )
    parser.add_argument(
        '--frequency-stop-khz',
        type=float,
        default=None,
        help='End of automatic frequency sweep in kHz, inclusive.',
    )
    parser.add_argument(
        '--frequency-step-khz',
        type=float,
        default=None,
        help='Automatic frequency sweep spacing in kHz, e.g. 2.5 or 5.',
    )
    parser.add_argument('--include-healthy', action='store_true')
    parser.add_argument('--only-healthy', action='store_true')
    parser.add_argument('--healthy-sample-id', default='dataset_a_frequency_healthy')
    parser.add_argument(
        '--force-healthy',
        action='store_true',
        help='Recompute the healthy baseline even when a complete matching H_complex.npz already exists.',
    )
    parser.add_argument(
        '--overwrite-existing',
        action='store_true',
        help='Allow damaged sample ids to overwrite existing output files. Use only for intentional reruns.',
    )
    parser.add_argument('--heartbeat-s', type=float, default=fcommon.streaming.DEFAULT_HEARTBEAT_S)
    parser.add_argument(
        '--skip-label-preview',
        action='store_true',
        help='Write label arrays/metadata but skip preview PNG generation.',
    )
    parser.add_argument(
        '--keep-case-csv',
        action='store_true',
        help='Also keep per-case tx/frequency CSV files. Default is to keep only the cumulative sample CSV.',
    )
    fcommon.add_frequency_solver_arguments(parser)
    parser.add_argument('--cores', type=int, default=None)
    parser.add_argument('--keep-client-cache', action='store_true')
    return parser.parse_args()


def result_row(
    result: fcommon.FrequencySampleExportResult,
    seed: int | None,
    defect_count: int,
    lobe_count: int,
    status: str = 'solved',
    note: str = '',
) -> dict:
    return {
        'sample_id': result.sample_id,
        'dataset': result.dataset,
        'defect_state': result.defect_state,
        'seed': seed,
        'case_count': result.case_count,
        'response_file_count': len(result.response_files),
        'feature_file_count': len(result.feature_files),
        'defect_count': defect_count,
        'lobe_count': lobe_count,
        'metadata': str(result.metadata_path),
        'saved_mph': False,
        'analysis_type': 'frequency_domain',
        'status': status,
        'note': note,
    }


def skipped_row(
    *,
    sample_id: str,
    defect_state: str,
    output_root: Path,
    case_count: int,
    note: str,
) -> dict:
    return {
        'sample_id': sample_id,
        'dataset': 'A_frequency',
        'defect_state': defect_state,
        'seed': None,
        'case_count': case_count,
        'response_file_count': 0,
        'feature_file_count': 0,
        'defect_count': 0,
        'lobe_count': 0,
        'metadata': str(output_root / 'metadata' / f'{sample_id}.json'),
        'saved_mph': False,
        'analysis_type': 'frequency_domain',
        'status': 'skipped_existing',
        'note': note,
    }


def sample_name(sample_id: int) -> str:
    return f'dataset_a_frequency_sample_{sample_id:04d}'


def cli_list_text(values: list[str] | str) -> str:
    if isinstance(values, str):
        return values
    return ','.join(str(value) for value in values)


def existing_damaged_sample_ids(output_root: Path) -> set[int]:
    ids: set[int] = set()
    roots = [
        output_root / 'frequency_response',
        output_root / 'metadata',
        output_root / 'labels',
        output_root / 'progress',
        output_root / 'csv' / 'frequency_response',
    ]
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob('dataset_a_frequency_sample_*'):
            match = SAMPLE_ID_PATTERN.search(path.name)
            if match:
                ids.add(int(match.group(1)))
    return ids


def first_free_sample_segment(output_root: Path, sample_count: int) -> int:
    if sample_count <= 0:
        return 1
    existing = existing_damaged_sample_ids(output_root)
    candidate = 1
    while True:
        segment = range(candidate, candidate + sample_count)
        collisions = [item for item in segment if item in existing]
        if not collisions:
            return candidate
        candidate = max(collisions) + 1


def plan_damaged_sample_ids(args: argparse.Namespace) -> tuple[int, ...]:
    if args.samples < 0:
        raise ValueError(f'--samples must be >= 0, got {args.samples}')
    if args.only_healthy or args.samples == 0:
        return ()
    if args.start_id is None:
        start_id = first_free_sample_segment(args.output_root, args.samples)
        fcommon.streaming.console_log(
            f'[samples] auto-selected start-id={start_id}; '
            f'planned ids {start_id}-{start_id + args.samples - 1}'
        )
    else:
        start_id = args.start_id
        fcommon.streaming.console_log(
            f'[samples] using requested start-id={start_id}; '
            f'planned ids {start_id}-{start_id + args.samples - 1}'
        )
    if start_id <= 0:
        raise ValueError(f'--start-id must be positive, got {start_id}')
    planned = tuple(range(start_id, start_id + args.samples))
    if not args.overwrite_existing:
        existing = existing_damaged_sample_ids(args.output_root)
        collisions = sorted(item for item in planned if item in existing)
        if collisions:
            examples = ', '.join(sample_name(item) for item in collisions[:8])
            raise RuntimeError(
                'Refusing to overwrite existing damaged sample outputs: '
                f'{examples}. Use a different --start-id, omit --start-id for auto allocation, '
                'or pass --overwrite-existing for an intentional rerun.'
            )
    return planned


def requested_tx_and_frequencies(cases: list[fcommon.FrequencyCase]) -> tuple[tuple[int, ...], tuple[float, ...]]:
    tx_indices = tuple(dict.fromkeys(case.tx for case in cases))
    frequencies_hz = tuple(dict.fromkeys(case.frequency_hz for case in cases))
    return tx_indices, frequencies_hz


def healthy_response_status(
    *,
    path: Path,
    tx_indices: tuple[int, ...],
    frequencies_hz: tuple[float, ...],
) -> tuple[bool, str]:
    if not path.exists():
        return False, 'missing healthy response npz'
    try:
        with fcommon.np.load(path, allow_pickle=False) as data:
            required = {'H_real', 'H_imag', 'completed_mask', 'tx_indices', 'frequencies_hz'}
            missing = required.difference(data.files)
            if missing:
                return False, f'missing fields {sorted(missing)}'
            existing_tx = tuple(int(item) for item in fcommon.np.asarray(data['tx_indices']).tolist())
            existing_freq = tuple(float(item) for item in fcommon.np.asarray(data['frequencies_hz']).tolist())
            if existing_tx != tx_indices:
                return False, f'tx mismatch existing={existing_tx} requested={tx_indices}'
            if len(existing_freq) != len(frequencies_hz) or not fcommon.np.allclose(
                existing_freq,
                frequencies_hz,
                rtol=0.0,
                atol=1e-6,
            ):
                return False, f'frequency mismatch existing={existing_freq} requested={frequencies_hz}'
            completed = fcommon.np.asarray(data['completed_mask'], dtype=bool)
            expected_shape = (len(tx_indices), len(frequencies_hz))
            if completed.shape != expected_shape:
                return False, f'completed_mask shape {completed.shape} != {expected_shape}'
            if not fcommon.np.all(completed):
                done = int(fcommon.np.sum(completed))
                return False, f'incomplete healthy response {done}/{completed.size} cases'
            h_real = fcommon.np.asarray(data['H_real'], dtype=float)
            h_imag = fcommon.np.asarray(data['H_imag'], dtype=float)
            expected_h_shape = (len(tx_indices), 16, len(frequencies_hz))
            if h_real.shape != expected_h_shape or h_imag.shape != expected_h_shape:
                return False, f'H shape real={h_real.shape} imag={h_imag.shape} expected={expected_h_shape}'
            if not fcommon.np.all(fcommon.np.isfinite(h_real)) or not fcommon.np.all(fcommon.np.isfinite(h_imag)):
                return False, 'healthy response contains non-finite values'
    except Exception as error:
        return False, f'failed to read existing healthy response: {type(error).__name__}: {error}'
    return True, 'complete matching healthy response exists'


def should_solve_healthy(
    args: argparse.Namespace,
    cases: list[fcommon.FrequencyCase],
) -> tuple[bool, str]:
    if not (args.include_healthy or args.only_healthy):
        return False, 'healthy baseline was not requested'
    healthy_npz = args.output_root / 'frequency_response' / f'{args.healthy_sample_id}_H_complex.npz'
    tx_indices, frequencies_hz = requested_tx_and_frequencies(cases)
    complete, reason = healthy_response_status(
        path=healthy_npz,
        tx_indices=tx_indices,
        frequencies_hz=frequencies_hz,
    )
    if args.force_healthy:
        return True, f'--force-healthy set; recomputing healthy baseline ({reason})'
    if complete:
        return False, reason
    if healthy_npz.exists() and ('mismatch' in reason or 'shape' in reason):
        raise RuntimeError(
            f'Existing healthy baseline {healthy_npz} is not compatible with the requested cases: {reason}. '
            'Use a different --healthy-sample-id to keep both baselines, or pass --force-healthy to overwrite it.'
        )
    return True, reason


def main() -> None:
    args = parse_args()
    fcommon.configure_dataset_a_frequency(use_parametric_sweep=False)
    fcommon.apply_solver_arguments(args)
    if any(value is not None for value in (args.frequency_start_khz, args.frequency_stop_khz, args.frequency_step_khz)):
        start_khz = (
            fcommon.DEFAULT_SWEEP_START_KHZ
            if args.frequency_start_khz is None
            else args.frequency_start_khz
        )
        stop_khz = (
            fcommon.DEFAULT_SWEEP_STOP_KHZ
            if args.frequency_stop_khz is None
            else args.frequency_stop_khz
        )
        step_khz = (
            fcommon.DEFAULT_SWEEP_STEP_KHZ
            if args.frequency_step_khz is None
            else args.frequency_step_khz
        )
        frequencies_hz = fcommon.frequency_range_hz(start_khz, stop_khz, step_khz)
        fcommon.streaming.console_log(
            '[frequency] '
            f'using range {start_khz:g}-{stop_khz:g} kHz step {step_khz:g} kHz; '
            f'count={len(frequencies_hz)}'
        )
    else:
        frequencies_hz = fcommon.parse_float_list(cli_list_text(args.frequencies))
    cases = fcommon.make_cases(
        fcommon.parse_int_list(cli_list_text(args.tx)),
        frequencies_hz,
    )
    damaged_sample_ids = plan_damaged_sample_ids(args)
    solve_healthy, healthy_reason = should_solve_healthy(args, cases)
    if args.include_healthy or args.only_healthy:
        if solve_healthy:
            fcommon.streaming.console_log(f'[healthy] solving healthy baseline: {healthy_reason}')
        else:
            fcommon.streaming.console_log(f'[healthy] skipping healthy baseline: {healthy_reason}')
    clear_each_case = not args.keep_client_cache
    rows: list[dict] = []

    if not solve_healthy and not damaged_sample_ids:
        rows.append(skipped_row(
            sample_id=args.healthy_sample_id,
            defect_state='healthy_no_defect',
            output_root=args.output_root,
            case_count=len(cases),
            note=healthy_reason,
        ))
        fcommon.write_manifest(args.output_root / 'manifest.csv', rows)
        print(f'Manifest: {args.output_root / "manifest.csv"}')
        print('No COMSOL solve was needed.')
        return

    fcommon.streaming.console_log('[startup] starting COMSOL client...')
    client = fcommon.shell.start_client(cores=args.cores)
    fcommon.streaming.console_log('[startup] COMSOL client started.')
    try:
        if args.include_healthy or args.only_healthy:
            if not solve_healthy:
                rows.append(skipped_row(
                    sample_id=args.healthy_sample_id,
                    defect_state='healthy_no_defect',
                    output_root=args.output_root,
                    case_count=len(cases),
                    note=healthy_reason,
                ))
            else:
                result = fcommon.solve_export_frequency_sample(
                    client=client,
                    dataset='A_frequency',
                    sample_id=args.healthy_sample_id,
                    defect_state='healthy_no_defect',
                    output_root=args.output_root,
                    cases=cases,
                    defects=[],
                    lobes=[],
                    sample_metadata={'sample_id': 0, 'seed': None, 'defects': [], 'lobes': []},
                clear_each_case=clear_each_case,
                heartbeat_s=args.heartbeat_s,
                reuse_sample_model=not args.rebuild_each_case,
                write_label_preview=not args.skip_label_preview,
                keep_case_csv=args.keep_case_csv,
            )
                rows.append(result_row(result, None, 0, 0))

        if not args.only_healthy:
            sampling = defects.DefectSamplingConfig(
                min_defects=1,
                max_defects=3,
                aspect_ratio_range=(0.75, 1.45),
                irregular_lobes=True,
            )
            for sample_id in damaged_sample_ids:
                seed = args.seed0 + sample_id
                sample = defects.generate_sample(sample_id, seed, sampling)
                model_defects, model_lobes = defects.to_shell_defects(sample)
                name = sample_name(sample_id)
                result = fcommon.solve_export_frequency_sample(
                    client=client,
                    dataset='A_frequency',
                    sample_id=name,
                    defect_state='damaged_random_outer_corrosion',
                    output_root=args.output_root,
                    cases=cases,
                    defects=model_defects,
                    lobes=model_lobes,
                    sample_metadata=defects.sample_to_dict(sample),
                    clear_each_case=clear_each_case,
                    heartbeat_s=args.heartbeat_s,
                    reuse_sample_model=not args.rebuild_each_case,
                    write_label_preview=not args.skip_label_preview,
                    keep_case_csv=args.keep_case_csv,
                )
                rows.append(result_row(result, seed, len(sample.defects), len(sample.lobes)))
    finally:
        client.clear()

    fcommon.write_manifest(args.output_root / 'manifest.csv', rows)
    print(f'Manifest: {args.output_root / "manifest.csv"}')
    print('Frequency-domain responses were exported directly from COMSOL memory and no solved MPH files were saved.')


if __name__ == '__main__':
    main()

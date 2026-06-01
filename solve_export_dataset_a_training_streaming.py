"""Stream-solve Dataset A training cases with random outer corrosion defects."""

from __future__ import annotations

import argparse
from pathlib import Path

print('[startup] importing COMSOL/mph modules...', flush=True)
import simple_defect_common as defects
import simple_shell_common as shell
import streaming_export_common as streaming
print('[startup] imports loaded.', flush=True)


OUTPUT_ROOT = shell.OUTPUT_ROOT / 'streaming_dataset_a_training_shell'

DEFAULT_TX = ','.join(str(index) for index in range(1, 17))
DEFAULT_FREQUENCIES = '40000,50000,60000'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Solve Dataset A training shell cases with random outer-surface corrosion defects.'
    )
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--start-id', type=int, default=1)
    parser.add_argument('--seed0', type=int, default=510000)
    parser.add_argument('--tx', default=DEFAULT_TX)
    parser.add_argument('--frequencies', default=DEFAULT_FREQUENCIES)
    parser.add_argument('--include-healthy', action='store_true')
    parser.add_argument('--only-healthy', action='store_true')
    parser.add_argument('--healthy-sample-id', default='dataset_a_training_healthy')
    parser.add_argument('--healthy-waveform-root', type=Path, default=None)
    parser.add_argument('--healthy-waveform-sample-id', default=None)
    parser.add_argument('--threshold-ratio', type=float, default=0.15)
    parser.add_argument('--window-us', type=float, default=35.0)
    parser.add_argument('--group-velocity', type=float, default=2522.0)
    parser.add_argument('--heartbeat-s', type=float, default=streaming.DEFAULT_HEARTBEAT_S)
    streaming.add_solver_arguments(parser)
    parser.add_argument('--cores', type=int, default=None)
    parser.add_argument('--keep-client-cache', action='store_true')
    return parser.parse_args()


def configure() -> None:
    shell.MODEL_FAMILY = 'Dataset A training shell streaming solve/export'
    shell.POSITION_PERTURBATIONS = {}
    shell.AMPLITUDE_SCALE = {}
    shell.MATERIAL = shell.MaterialConfig()
    shell.ABSORBING_LAYER = shell.AbsorbingLayerConfig(enabled=True)
    shell.DEFECT_MODEL = shell.DefectModelConfig(corrosion_surface='outer')
    shell.SOLVER = shell.SolverConfig(solve=False)
    shell.RECEIVER_INDICES = tuple(range(17, 33))
    shell.DATASET_NOTES = [
        'Dataset A training uses random outer corrosion defects.',
        'Shell end absorbing layers are enabled to suppress end reflections.',
        'No transducer position, amplitude, or material perturbation.',
        'Each tx/frequency case is solved and exported in-session, then discarded.',
    ]


def result_row(result: streaming.SampleExportResult, seed: int | None, defect_count: int, lobe_count: int) -> dict:
    return {
        'sample_id': result.sample_id,
        'dataset': result.dataset,
        'defect_state': result.defect_state,
        'seed': seed,
        'case_count': result.case_count,
        'waveform_count': len(result.waveform_files),
        'feature_file_count': len(result.feature_files),
        'defect_count': defect_count,
        'lobe_count': lobe_count,
        'metadata': str(result.metadata_path),
        'saved_mph': False,
    }


def main() -> None:
    args = parse_args()
    configure()
    streaming.apply_solver_arguments(args)
    cases = streaming.make_cases(
        streaming.parse_int_list(args.tx),
        streaming.parse_float_list(args.frequencies),
    )
    clear_each_case = not args.keep_client_cache
    rows: list[dict] = []
    healthy_root = args.healthy_waveform_root
    healthy_sample_id = args.healthy_waveform_sample_id

    streaming.console_log('[startup] starting COMSOL client...')
    client = shell.start_client(cores=args.cores)
    streaming.console_log('[startup] COMSOL client started.')
    try:
        if args.include_healthy or args.only_healthy:
            result = streaming.solve_export_sample(
                client=client,
                dataset='A_training',
                sample_id=args.healthy_sample_id,
                defect_state='healthy_no_defect',
                output_root=args.output_root,
                cases=cases,
                defects=[],
                lobes=[],
                sample_metadata={'sample_id': 0, 'seed': None, 'defects': [], 'lobes': []},
                threshold_ratio=args.threshold_ratio,
                window_us=args.window_us,
                group_velocity=args.group_velocity,
                clear_each_case=clear_each_case,
                heartbeat_s=args.heartbeat_s,
                reuse_sample_model=not args.rebuild_each_case,
            )
            rows.append(result_row(result, None, 0, 0))
            healthy_root = args.output_root / 'csv' / 'waveforms'
            healthy_sample_id = args.healthy_sample_id

        if not args.only_healthy:
            sampling = defects.DefectSamplingConfig(
                min_defects=1,
                max_defects=4,
                aspect_ratio_range=(0.6, 1.8),
                irregular_lobes=True,
            )
            for sample_id in range(args.start_id, args.start_id + args.samples):
                seed = args.seed0 + sample_id
                sample = defects.generate_sample(sample_id, seed, sampling)
                model_defects, model_lobes = defects.to_shell_defects(sample)
                name = f'dataset_a_training_sample_{sample_id:04d}'
                result = streaming.solve_export_sample(
                    client=client,
                    dataset='A_training',
                    sample_id=name,
                    defect_state='damaged_random_outer_corrosion',
                    output_root=args.output_root,
                    cases=cases,
                    defects=model_defects,
                    lobes=model_lobes,
                    sample_metadata=defects.sample_to_dict(sample),
                    healthy_waveform_root=healthy_root,
                    healthy_sample_id=healthy_sample_id,
                    threshold_ratio=args.threshold_ratio,
                    window_us=args.window_us,
                    group_velocity=args.group_velocity,
                    clear_each_case=clear_each_case,
                    heartbeat_s=args.heartbeat_s,
                    reuse_sample_model=not args.rebuild_each_case,
                )
                rows.append(result_row(result, seed, len(sample.defects), len(sample.lobes)))
    finally:
        client.clear()

    streaming.write_manifest(args.output_root / 'manifest.csv', rows)
    print(f'Manifest: {args.output_root / "manifest.csv"}')
    print('Solved fields were exported directly from COMSOL memory and no solved MPH files were saved.')


if __name__ == '__main__':
    main()

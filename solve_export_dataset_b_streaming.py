"""Stream-solve Dataset B shell cases and export CSV without saved MPH files."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

print('[startup] importing COMSOL/mph modules...', flush=True)
import simple_defect_common as defects
import simple_shell_common as shell
import streaming_export_common as streaming
print('[startup] imports loaded.', flush=True)


OUTPUT_ROOT = shell.OUTPUT_ROOT / 'streaming_dataset_b_shell'

DEFAULT_TX = ','.join(str(index) for index in range(1, 17))
DEFAULT_FREQUENCIES = '40000,50000,60000'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Solve Dataset B simple shell cases, export CSV immediately, and do not save solved MPH files.'
    )
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--samples', type=int, default=1, help='Number of damaged samples to generate.')
    parser.add_argument('--start-id', type=int, default=1)
    parser.add_argument('--seed0', type=int, default=610000)
    parser.add_argument('--tx', default=DEFAULT_TX)
    parser.add_argument('--frequencies', default=DEFAULT_FREQUENCIES)
    parser.add_argument('--include-healthy', action='store_true')
    parser.add_argument('--only-healthy', action='store_true')
    parser.add_argument('--healthy-sample-id', default='dataset_b_shell_healthy')
    parser.add_argument('--healthy-waveform-root', type=Path, default=None)
    parser.add_argument('--healthy-waveform-sample-id', default=None)
    parser.add_argument('--threshold-ratio', type=float, default=0.15)
    parser.add_argument('--window-us', type=float, default=35.0)
    parser.add_argument('--group-velocity', type=float, default=2522.0)
    parser.add_argument('--heartbeat-s', type=float, default=streaming.DEFAULT_HEARTBEAT_S)
    streaming.add_solver_arguments(parser)
    parser.add_argument(
        '--cores',
        type=int,
        default=None,
        help='COMSOL core count. Leave unset so COMSOL allocates cores by itself.',
    )
    parser.add_argument(
        '--keep-client-cache',
        action='store_true',
        help='Do not call client.clear() after each case. Faster, but uses more memory/temp space.',
    )
    return parser.parse_args()


def uniform(rng: random.Random, low: float, high: float) -> float:
    return low + (high - low) * rng.random()


def configure_b(seed: int) -> None:
    rng = random.Random(seed + 991)
    total = shell.TRANSDUCER.count_per_ring * 2
    shell.POSITION_PERTURBATIONS = {
        index: {
            'dz_mm': uniform(rng, -1.0, 1.0),
            'dtheta_deg': uniform(rng, -1.0, 1.0),
        }
        for index in range(1, total + 1)
    }
    shell.AMPLITUDE_SCALE = {
        index: uniform(rng, 0.85, 1.15)
        for index in range(1, total + 1)
    }
    shell.MATERIAL = shell.MaterialConfig(
        density_kg_m3=uniform(rng, 2680.0, 2720.0),
        young_gpa=uniform(rng, 68.0, 72.0),
        poisson=uniform(rng, 0.32, 0.34),
        rayleigh_alpha_1_s=0.0,
        rayleigh_beta_s=uniform(rng, 3e-8, 9e-8),
    )
    shell.ABSORBING_LAYER = shell.AbsorbingLayerConfig(enabled=False)
    shell.DEFECT_MODEL = shell.DefectModelConfig(corrosion_surface='outer')
    shell.SOLVER = shell.SolverConfig(solve=False)
    shell.RECEIVER_INDICES = tuple(range(17, 33))
    shell.MODEL_FAMILY = 'Dataset B simple shell streaming solve/export'
    shell.DATASET_NOTES = [
        f'Random seed: {seed}.',
        'Dataset B uses material, transmitter, receiver, and amplitude perturbations.',
        'Defects are outer-surface corrosion represented by thickness loss and shell offset.',
        'Pipe ends are left free for experimental matching.',
        'Noise and trigger jitter should be added after waveform CSV export.',
        'Each tx/frequency case is solved and exported in-session, then discarded.',
        'Solved MPH files are intentionally not saved.',
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
            configure_b(args.seed0)
            streaming.apply_solver_arguments(args)
            result = streaming.solve_export_sample(
                client=client,
                dataset='B',
                sample_id=args.healthy_sample_id,
                defect_state='healthy_no_defect',
                output_root=args.output_root,
                cases=cases,
                defects=[],
                lobes=[],
                sample_metadata={'sample_id': 0, 'seed': args.seed0, 'defects': [], 'lobes': []},
                threshold_ratio=args.threshold_ratio,
                window_us=args.window_us,
                group_velocity=args.group_velocity,
                clear_each_case=clear_each_case,
                heartbeat_s=args.heartbeat_s,
                reuse_sample_model=not args.rebuild_each_case,
            )
            rows.append(result_row(result, args.seed0, 0, 0))
            healthy_root = args.output_root / 'csv' / 'waveforms'
            healthy_sample_id = args.healthy_sample_id

        if not args.only_healthy:
            sampling = defects.DefectSamplingConfig()
            for sample_id in range(args.start_id, args.start_id + args.samples):
                seed = args.seed0 + sample_id
                configure_b(seed)
                streaming.apply_solver_arguments(args)
                sample = defects.generate_sample(sample_id, seed, sampling)
                model_defects, model_lobes = defects.to_shell_defects(sample)
                name = f'dataset_b_shell_sample_{sample_id:04d}'
                result = streaming.solve_export_sample(
                    client=client,
                    dataset='B',
                    sample_id=name,
                    defect_state='damaged_thickness_loss',
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

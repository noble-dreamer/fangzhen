"""Stream-solve Dataset A validation cases with one regular outer corrosion defect."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import simple_shell_common as shell
import streaming_export_common as streaming


OUTPUT_ROOT = Path(r'D:\lab_ultr\fz\simple\output\streaming_dataset_a_validation_shell')

DEFAULT_TX = ','.join(str(index) for index in range(1, 17))
DEFAULT_FREQUENCIES = '40000,50000,60000'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Solve Dataset A validation shell cases with one regular outer-surface corrosion defect.'
    )
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--tx', default=DEFAULT_TX)
    parser.add_argument('--frequencies', default=DEFAULT_FREQUENCIES)
    parser.add_argument('--include-healthy', action='store_true')
    parser.add_argument('--only-healthy', action='store_true')
    parser.add_argument('--healthy-sample-id', default='dataset_a_validation_healthy')
    parser.add_argument('--sample-id', default='dataset_a_validation_single_defect')
    parser.add_argument('--theta-deg', type=float, default=0.0)
    parser.add_argument('--z-mm', type=float, default=500.0)
    parser.add_argument('--diameter-theta-mm', type=float, default=120.0)
    parser.add_argument('--diameter-z-mm', type=float, default=120.0)
    parser.add_argument('--depth-mm', type=float, default=3.0)
    parser.add_argument('--threshold-ratio', type=float, default=0.15)
    parser.add_argument('--window-us', type=float, default=35.0)
    parser.add_argument('--group-velocity', type=float, default=2522.0)
    parser.add_argument('--cores', type=int, default=None)
    parser.add_argument('--keep-client-cache', action='store_true')
    return parser.parse_args()


def configure() -> None:
    shell.MODEL_FAMILY = 'Dataset A validation shell streaming solve/export'
    shell.POSITION_PERTURBATIONS = {}
    shell.AMPLITUDE_SCALE = {}
    shell.MATERIAL = shell.MaterialConfig()
    shell.ABSORBING_LAYER = shell.AbsorbingLayerConfig(enabled=True)
    shell.DEFECT_MODEL = shell.DefectModelConfig(corrosion_surface='outer')
    shell.SOLVER = shell.SolverConfig(solve=False)
    shell.RECEIVER_INDICES = tuple(range(17, 33))
    shell.DATASET_NOTES = [
        'Dataset A validation uses one regular circular/elliptical outer corrosion defect.',
        'Shell end absorbing layers are enabled to suppress end reflections.',
        'No transducer position, amplitude, or material perturbation.',
        'Each tx/frequency case is solved and exported in-session, then discarded.',
    ]


def result_row(result: streaming.SampleExportResult, defect_count: int) -> dict:
    return {
        'sample_id': result.sample_id,
        'dataset': result.dataset,
        'defect_state': result.defect_state,
        'case_count': result.case_count,
        'waveform_count': len(result.waveform_files),
        'feature_file_count': len(result.feature_files),
        'defect_count': defect_count,
        'lobe_count': 0,
        'metadata': str(result.metadata_path),
        'saved_mph': False,
    }


def main() -> None:
    args = parse_args()
    configure()
    cases = streaming.make_cases(
        streaming.parse_int_list(args.tx),
        streaming.parse_float_list(args.frequencies),
    )
    clear_each_case = not args.keep_client_cache
    rows: list[dict] = []
    healthy_root = None
    healthy_sample_id = None

    client = shell.start_client(cores=args.cores)
    try:
        if args.include_healthy or args.only_healthy:
            healthy = streaming.solve_export_sample(
                client=client,
                dataset='A_validation',
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
            )
            rows.append(result_row(healthy, 0))
            healthy_root = args.output_root / 'csv' / 'waveforms'
            healthy_sample_id = args.healthy_sample_id

        if not args.only_healthy:
            defect = shell.DefectConfig(
                theta_deg=args.theta_deg,
                z_mm=args.z_mm,
                radius_mm=0.5 * max(args.diameter_theta_mm, args.diameter_z_mm),
                depth_mm=args.depth_mm,
                radius_theta_mm=0.5 * args.diameter_theta_mm,
                radius_z_mm=0.5 * args.diameter_z_mm,
            )
            sample_metadata = {
                'sample_id': 1,
                'seed': None,
                'defects': [
                    {
                        'theta_deg': args.theta_deg,
                        'z_mm': args.z_mm,
                        'diameter_theta_mm': args.diameter_theta_mm,
                        'diameter_z_mm': args.diameter_z_mm,
                        'depth_mm': args.depth_mm,
                        'shape': 'regular_superellipse_outer_corrosion',
                    }
                ],
                'lobes': [],
                'absorbing_layer': asdict(shell.ABSORBING_LAYER),
                'defect_model': asdict(shell.DEFECT_MODEL),
            }
            damaged = streaming.solve_export_sample(
                client=client,
                dataset='A_validation',
                sample_id=args.sample_id,
                defect_state='damaged_single_regular_outer_corrosion',
                output_root=args.output_root,
                cases=cases,
                defects=[defect],
                lobes=[],
                sample_metadata=sample_metadata,
                healthy_waveform_root=healthy_root,
                healthy_sample_id=healthy_sample_id,
                threshold_ratio=args.threshold_ratio,
                window_us=args.window_us,
                group_velocity=args.group_velocity,
                clear_each_case=clear_each_case,
            )
            rows.append(result_row(damaged, 1))
    finally:
        client.clear()

    streaming.write_manifest(args.output_root / 'manifest.csv', rows)
    print(f'Manifest: {args.output_root / "manifest.csv"}')
    print('Solved fields were exported directly from COMSOL memory and no solved MPH files were saved.')


if __name__ == '__main__':
    main()

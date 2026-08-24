"""Stream frequency responses for textured irregular single defects."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
SIMPLE_ROOT = HERE.parents[1]
F_DOMAIN = HERE.parent
for path in (SIMPLE_ROOT, F_DOMAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import frequency_domain_common as fcommon
from irregular_defect_common import (
    GENERATOR_NAME,
    generate_irregular_defect,
    irregular_shell_field,
    write_comsol_table,
    write_label_package,
)


DEFAULT_OUTPUT = HERE / 'output_dataset' / 'streaming_dataset_a_frequency_shell'
SELECTED_FREQUENCIES = '40000,32500,20000,42500,50000,47500,52500,70000,72500,67500,75000,80000,82500,77500,95000'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--start-id', type=int, default=1)
    parser.add_argument('--seed0', type=int, default=20260820)
    parser.add_argument('--grid-count', type=int, default=129)
    parser.add_argument('--tx', nargs='+', default=[fcommon.DEFAULT_TX])
    parser.add_argument('--frequencies', nargs='+', default=[SELECTED_FREQUENCIES])
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--include-healthy', action='store_true')
    parser.add_argument('--only-healthy', action='store_true')
    parser.add_argument('--overwrite-existing', action='store_true')
    parser.add_argument('--skip-label-preview', action='store_true')
    parser.add_argument('--keep-case-csv', action='store_true')
    parser.add_argument('--checkpoint-every-cases', type=int, default=1)
    parser.add_argument('--heartbeat-s', type=float, default=fcommon.streaming.DEFAULT_HEARTBEAT_S)
    parser.add_argument('--cores', type=int, default=None)
    fcommon.add_frequency_solver_arguments(parser)
    return parser.parse_args()


def list_text(values: list[str]) -> str:
    return ','.join(str(value) for value in values)


def ensure_available(root: Path, sample_id: str, overwrite: bool) -> None:
    expected = (
        root / 'frequency_response' / f'{sample_id}_H_complex.npz',
        root / 'metadata' / f'{sample_id}.json',
    )
    collisions = [path for path in expected if path.exists()]
    if collisions and not overwrite:
        raise RuntimeError(f'Refusing to overwrite existing output: {collisions[0]}')


def result_row(result, seed: int | None, generator: str, size_class: str | None = None) -> dict[str, object]:
    return {
        'sample_id': result.sample_id,
        'dataset': result.dataset,
        'defect_state': result.defect_state,
        'seed': seed,
        'case_count': result.case_count,
        'defect_count': 0 if seed is None else 1,
        'generator': generator,
        'size_class': size_class,
        'metadata': str(result.metadata_path),
        'saved_mph': False,
        'status': 'solved',
    }


def solve_irregular(client, args, cases, sample_index: int):
    sample_id = f'dataset_a_frequency_irregular_sample_{sample_index:04d}'
    ensure_available(args.output_root, sample_id, args.overwrite_existing)
    seed = args.seed0 + sample_index
    field = generate_irregular_defect(seed, grid_count=args.grid_count)
    table_path = write_comsol_table(field, args.output_root / 'defect_tables' / f'{sample_id}.txt')
    sample_metadata = {
        'sample_id': sample_index,
        'seed': seed,
        'defects': [],
        'lobes': [],
        'irregular_defect': {**field.metadata(), 'comsol_table': str(table_path)},
    }
    original_label_writer = fcommon.write_label_package_compatible

    def label_writer(output_dir, current_id, _sample_metadata, *, write_label_preview):
        return write_label_package(
            fcommon.defect_labels,
            output_dir,
            current_id,
            field,
            write_preview_png=write_label_preview,
        )

    fcommon.write_label_package_compatible = label_writer
    try:
        with irregular_shell_field(fcommon.shell, field, table_path):
            result = fcommon.solve_export_frequency_sample(
                client=client,
                dataset='A_frequency_irregular',
                sample_id=sample_id,
                defect_state='damaged_irregular_outer_corrosion',
                output_root=args.output_root,
                cases=cases,
                defects=[field],
                lobes=[],
                sample_metadata=sample_metadata,
                clear_each_case=True,
                heartbeat_s=args.heartbeat_s,
                reuse_sample_model=not args.rebuild_each_case,
                write_label_preview=not args.skip_label_preview,
                keep_case_csv=args.keep_case_csv,
                checkpoint_every_cases=args.checkpoint_every_cases,
            )
    finally:
        fcommon.write_label_package_compatible = original_label_writer
    return result, seed, field.size_class


def main() -> None:
    args = parse_args()
    if args.samples < 0 or args.start_id <= 0 or args.checkpoint_every_cases < 0:
        raise ValueError('samples/checkpoint must be nonnegative and start-id must be positive')
    fcommon.configure_dataset_a_frequency(use_parametric_sweep=False)
    fcommon.apply_solver_arguments(args)
    tx_indices = fcommon.parse_int_list(list_text(args.tx))
    frequencies = fcommon.parse_float_list(list_text(args.frequencies))
    cases = fcommon.make_cases(tx_indices, frequencies)
    fcommon.shell.MESH = replace(fcommon.shell.MESH, max_frequency_hz=max(frequencies))
    rows = []
    client = fcommon.shell.start_client(cores=args.cores)
    try:
        if args.include_healthy or args.only_healthy:
            healthy_id = 'dataset_a_frequency_irregular_healthy'
            ensure_available(args.output_root, healthy_id, args.overwrite_existing)
            result = fcommon.solve_export_frequency_sample(
                client=client,
                dataset='A_frequency_irregular',
                sample_id=healthy_id,
                defect_state='healthy_no_defect',
                output_root=args.output_root,
                cases=cases,
                defects=[],
                lobes=[],
                sample_metadata={'sample_id': 0, 'seed': None, 'defects': [], 'lobes': []},
                clear_each_case=True,
                heartbeat_s=args.heartbeat_s,
                reuse_sample_model=not args.rebuild_each_case,
                write_label_preview=not args.skip_label_preview,
                keep_case_csv=args.keep_case_csv,
                checkpoint_every_cases=args.checkpoint_every_cases,
            )
            rows.append(result_row(result, None, 'healthy'))
        if not args.only_healthy:
            for sample_index in range(args.start_id, args.start_id + args.samples):
                result, seed, size_class = solve_irregular(client, args, cases, sample_index)
                rows.append(result_row(result, seed, GENERATOR_NAME, size_class))
    finally:
        client.clear()
    fcommon.write_manifest(args.output_root / 'manifest.csv', rows)
    print(f'Manifest: {args.output_root / "manifest.csv"}')


if __name__ == '__main__':
    main()

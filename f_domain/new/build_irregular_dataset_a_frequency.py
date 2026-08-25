"""Build one unsolved irregular-defect frequency Shell model for inspection."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SIMPLE_ROOT = HERE.parents[1]
F_DOMAIN = HERE.parent
for path in (SIMPLE_ROOT, F_DOMAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import frequency_domain_common as fcommon
from irregular_defect_common import generate_irregular_defect, irregular_shell_field, write_comsol_table


DEFAULT_OUTPUT = HERE / 'output_dataset' / 'dataset_a_frequency_shell'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20260821)
    parser.add_argument('--grid-count', type=int, default=129)
    parser.add_argument('--max-frequency-hz', type=float, default=95_000.0)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--cores', type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_frequency_hz <= 0.0:
        raise ValueError('--max-frequency-hz must be positive')
    stem = f'irregular_sample_{args.sample_id:04d}'
    model_name = f'pipe_shell_frequency_{stem}'
    field = generate_irregular_defect(args.seed, grid_count=args.grid_count)
    table_path = write_comsol_table(field, args.output_dir / 'defect_tables' / f'{stem}.txt')
    fcommon.configure_dataset_a_frequency(use_parametric_sweep=True)
    fcommon.shell.MESH = replace(fcommon.shell.MESH, max_frequency_hz=args.max_frequency_hz)
    fcommon.shell.CREATE_RECEIVER_DATASETS = True
    fcommon.shell.CREATE_VISUAL_MARKER_DATASETS = True
    client = fcommon.shell.start_client(cores=args.cores)
    try:
        with irregular_shell_field(fcommon.shell, field, table_path):
            model_path, problems = fcommon.shell.build_model(
                client,
                model_name,
                args.output_dir,
                defects=[field],
            )
        metadata = fcommon.model_metadata('A_frequency_irregular', 'damaged_irregular_outer_corrosion', model_path, problems)
        metadata['irregular_defect'] = {**field.metadata(), 'comsol_table': str(table_path)}
        metadata_dir = args.output_dir / 'metadata'
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = metadata_dir / f'{model_name}.json'
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
        fcommon.write_frequency_build_log(
            args.output_dir / 'dataset_a_frequency_shell_build_log.md',
            [model_path, table_path, metadata_path],
            {model_name: problems},
        )
    finally:
        client.clear()
    print(f'Saved {model_path}')
    print(f'Defect table: {table_path}')
    print('No frequency-domain study was solved.')


if __name__ == '__main__':
    main()

"""Generate Dataset A shell models with random thickness-loss defects."""

from __future__ import annotations

import csv
from pathlib import Path

import simple_defect_common as defects
import simple_shell_common as shell


OUTPUT_DIR = shell.OUTPUT_ROOT / 'generated_dataset_a_shell'
SAMPLE_COUNT = 1
SEED0 = 510000
SOLVE = False


def configure() -> None:
    shell.MODEL_FAMILY = 'Generated Dataset A simple shell defects'
    shell.POSITION_PERTURBATIONS = {}
    shell.AMPLITUDE_SCALE = {}
    shell.MATERIAL = shell.MaterialConfig()
    shell.ABSORBING_LAYER = shell.AbsorbingLayerConfig(enabled=True)
    shell.DEFECT_MODEL = shell.DefectModelConfig(corrosion_surface='outer')
    shell.SOLVER = shell.SolverConfig(solve=SOLVE)
    shell.SWEEP = shell.SweepConfig(
        transmitter_indices=tuple(range(1, 17)),
        frequencies_hz=(40_000.0, 50_000.0, 60_000.0),
        use_parametric_sweep=True,
    )
    shell.RECEIVER_INDICES = tuple(range(17, 33))
    shell.DATASET_NOTES = [
        'Random defect Dataset A shell model.',
        'Defects are outer-surface shell-thickness reductions.',
        'Dataset A uses shell end absorbing layers to suppress end reflections.',
        'No transducer or material perturbations.',
        f'SOLVE={SOLVE}.',
    ]


def main() -> None:
    configure()
    sampling = defects.DefectSamplingConfig()
    client = shell.start_client()
    rows = []
    try:
        for sample_id in range(1, SAMPLE_COUNT + 1):
            seed = SEED0 + sample_id
            sample = defects.generate_sample(sample_id, seed, sampling)
            model_defects, model_lobes = defects.to_shell_defects(sample)
            model_name = f'dataset_a_shell_sample_{sample_id:04d}'
            path, problems = shell.build_model(
                client,
                model_name,
                OUTPUT_DIR / 'models',
                defects=model_defects,
                lobes=model_lobes,
            )
            metadata_path = OUTPUT_DIR / 'metadata' / f'{model_name}.json'
            defects.write_metadata(
                metadata_path,
                sample,
                'A',
                shell.model_metadata('A', 'damaged_thickness_loss', path, problems),
                {'sampling': sampling.__dict__},
            )
            rows.append({
                'sample_id': sample_id,
                'seed': seed,
                'model': str(path),
                'metadata': str(metadata_path),
                'defect_count': len(sample.defects),
                'lobe_count': len(sample.lobes),
                'problems': problems,
            })
            print(f'Saved {path}')
    finally:
        client.clear()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = OUTPUT_DIR / 'manifest.csv'
    with manifest.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Manifest: {manifest}')
    print('No transient study was solved.' if not SOLVE else 'Transient solve was enabled.')


if __name__ == '__main__':
    main()

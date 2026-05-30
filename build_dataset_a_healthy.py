"""Build the ideal healthy no-defect shell model for Dataset A."""

from __future__ import annotations

import json
from pathlib import Path

import simple_shell_common as shell


OUTPUT_DIR = shell.OUTPUT_ROOT / 'dataset_a_shell'
MODEL_NAME = 'pipe_shell_healthy'


def configure() -> None:
    shell.MODEL_FAMILY = 'Dataset A simple shell: ideal healthy baseline'
    shell.POSITION_PERTURBATIONS = {}
    shell.AMPLITUDE_SCALE = {}
    shell.MATERIAL = shell.MaterialConfig()
    shell.ABSORBING_LAYER = shell.AbsorbingLayerConfig(enabled=True)
    shell.DEFECT_MODEL = shell.DefectModelConfig(corrosion_surface='outer')
    shell.SOLVER = shell.SolverConfig(solve=False)
    shell.SWEEP = shell.SweepConfig(
        transmitter_indices=tuple(range(1, 17)),
        frequencies_hz=(40_000.0, 50_000.0, 60_000.0),
        use_parametric_sweep=True,
    )
    shell.RECEIVER_INDICES = tuple(range(17, 33))
    shell.DATASET_NOTES = [
        'Ideal no-defect shell model.',
        'Dataset A uses shell end absorbing layers to suppress end reflections.',
        'No transducer position or amplitude perturbation.',
        'PZT solids are replaced by equivalent shell face-load windows.',
    ]


def main() -> None:
    configure()
    client = shell.start_client()
    saved = []
    problems = {}
    try:
        path, model_problems = shell.build_model(client, MODEL_NAME, OUTPUT_DIR)
        saved.append(path)
        problems[MODEL_NAME] = model_problems
        metadata = shell.model_metadata('A', 'healthy_no_defect', path, model_problems)
        (OUTPUT_DIR / 'metadata').mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'metadata' / f'{MODEL_NAME}.json').write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
            encoding='utf-8',
        )
        shell.write_build_log(OUTPUT_DIR / 'dataset_a_shell_build_log.md', saved, problems)
    finally:
        client.clear()
    print(f'Saved {saved[0]}')
    print('No transient study was solved.')


if __name__ == '__main__':
    main()

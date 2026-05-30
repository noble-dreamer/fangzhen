"""Build the realistic healthy no-defect shell model for Dataset B."""

from __future__ import annotations

import json
import random

import simple_shell_common as shell


OUTPUT_DIR = shell.OUTPUT_ROOT / 'dataset_b_shell'
MODEL_NAME = 'pipe_shell_healthy'
SEED = 20260528


def uniform(rng: random.Random, low: float, high: float) -> float:
    return low + (high - low) * rng.random()


def configure(seed: int = SEED) -> None:
    rng = random.Random(seed)
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
    shell.SWEEP = shell.SweepConfig(
        transmitter_indices=tuple(range(1, 17)),
        frequencies_hz=(40_000.0, 50_000.0, 60_000.0),
        use_parametric_sweep=True,
    )
    shell.RECEIVER_INDICES = tuple(range(17, 33))
    shell.MODEL_FAMILY = 'Dataset B simple shell: realistic healthy baseline'
    shell.DATASET_NOTES = [
        f'Random seed: {seed}.',
        'No-defect shell model with material, transmitter, and receiver perturbations.',
        'PZT-domain uncertainty is represented as equivalent tx/rx position and amplitude error.',
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
        metadata = shell.model_metadata('B', 'healthy_no_defect', path, model_problems)
        metadata['seed'] = SEED
        metadata['postprocessing_noise_plan'] = {
            'snr_db_range': [20.0, 40.0],
            'trigger_jitter_us_range': [-0.2, 0.2],
            'baseline_drift': True,
        }
        (OUTPUT_DIR / 'metadata').mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'metadata' / f'{MODEL_NAME}.json').write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
            encoding='utf-8',
        )
        shell.write_build_log(OUTPUT_DIR / 'dataset_b_shell_build_log.md', saved, problems)
    finally:
        client.clear()
    print(f'Saved {saved[0]}')
    print('No transient study was solved.')


if __name__ == '__main__':
    main()

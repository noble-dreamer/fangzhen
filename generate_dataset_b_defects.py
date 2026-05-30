"""Generate Dataset B shell models with defects plus tx/rx uncertainty."""

from __future__ import annotations

import csv
import random

import simple_defect_common as defects
import simple_shell_common as shell


OUTPUT_DIR = shell.OUTPUT_ROOT / 'generated_dataset_b_shell'
SAMPLE_COUNT = 1
SEED0 = 610000
SOLVE = False


def uniform(rng: random.Random, low: float, high: float) -> float:
    return low + (high - low) * rng.random()


def configure(seed: int) -> None:
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
    shell.SOLVER = shell.SolverConfig(solve=SOLVE)
    shell.SWEEP = shell.SweepConfig(
        transmitter_indices=tuple(range(1, 17)),
        frequencies_hz=(40_000.0, 50_000.0, 60_000.0),
        use_parametric_sweep=True,
    )
    shell.RECEIVER_INDICES = tuple(range(17, 33))
    shell.MODEL_FAMILY = 'Generated Dataset B simple shell defects'
    shell.DATASET_NOTES = [
        f'Random seed: {seed}.',
        'Defects are outer-surface shell-thickness reductions.',
        'PZT uncertainty is represented by transmitter/receiver position and amplitude errors.',
        'Noise and trigger jitter are intended for postprocessing after waveform export.',
        f'SOLVE={SOLVE}.',
    ]


def main() -> None:
    sampling = defects.DefectSamplingConfig()
    client = shell.start_client()
    rows = []
    try:
        for sample_id in range(1, SAMPLE_COUNT + 1):
            seed = SEED0 + sample_id
            configure(seed)
            sample = defects.generate_sample(sample_id, seed, sampling)
            model_defects, model_lobes = defects.to_shell_defects(sample)
            model_name = f'dataset_b_shell_sample_{sample_id:04d}'
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
                'B',
                shell.model_metadata('B', 'damaged_thickness_loss', path, problems),
                {
                    'sampling': sampling.__dict__,
                    'postprocessing_noise_plan': {
                        'snr_db_range': [20.0, 40.0],
                        'trigger_jitter_us_range': [-0.2, 0.2],
                        'baseline_drift': True,
                    },
                },
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

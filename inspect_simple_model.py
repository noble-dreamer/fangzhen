"""Inspect simple shell MPH files for receiver count and sweep setup."""

from __future__ import annotations

import argparse
from pathlib import Path

import mph


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='*', type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = args.models or [
        ROOT / 'output' / 'dataset_a_shell' / 'pipe_shell_healthy.mph',
        ROOT / 'output' / 'dataset_b_shell' / 'pipe_shell_healthy.mph',
        ROOT / 'output' / 'generated_dataset_a_shell' / 'models' / 'dataset_a_shell_sample_0001.mph',
        ROOT / 'output' / 'generated_dataset_b_shell' / 'models' / 'dataset_b_shell_sample_0001.mph',
    ]
    client = mph.start(cores=1)
    try:
        for path in paths:
            if not path.exists():
                continue
            model = client.load(path.resolve())
            print(f'\nMODEL {path}')
            receivers = [item for item in model / 'datasets' if item.type() == 'CutPoint3D']
            print(f'receiver point datasets: {len(receivers)}')
            for study in model / 'studies':
                print('study', study.name(), study.tag())
                for feature in study.children():
                    props = feature.properties()
                    interesting = {
                        key: props[key]
                        for key in ('pname', 'plistarr', 'sweeptype', 'tlist')
                        if key in props
                    }
                    print(' ', feature.name(), feature.tag(), feature.type(), interesting)
            for mesh in model / 'meshes':
                print('mesh', mesh.name(), mesh.tag())
                for feature in mesh.children():
                    props = feature.properties()
                    interesting = {key: props[key] for key in ('hmax', 'hmin') if key in props}
                    print(' ', feature.name(), feature.tag(), feature.type(), interesting)
            client.remove(model)
    finally:
        client.clear()


if __name__ == '__main__':
    main()

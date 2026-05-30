"""Probe Shell FaceLoad active fields for simple models."""

from __future__ import annotations

import argparse
from pathlib import Path

import mph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = mph.start(cores=1)
    model = client.load(args.model.resolve())
    try:
        for physics in model / 'physics':
            for child in physics.children():
                if child.type() != 'FaceLoad':
                    continue
                print('FaceLoad:', child.name(), child.tag())
                props = child.properties()
                for key in [
                    'LoadTypeForce',
                    'F',
                    'Ff',
                    'FfTot',
                    'M',
                    'Mf',
                    'loadLocation',
                    'coordinateSystem',
                    'tractionType',
                ]:
                    if key in props:
                        print(key, '=', props[key])
    finally:
        client.remove(model)


if __name__ == '__main__':
    main()

"""Print all Shell thickness/offset fields for COMSOL version checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import mph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('--cores', type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = mph.start(cores=args.cores)
    model = client.load(args.model.resolve())
    try:
        for physics in model / 'physics':
            for child in physics.children():
                if child.type() == 'ThicknessOffset':
                    print('ThicknessOffset:', child.name(), child.tag())
                    for key, value in sorted(child.properties().items()):
                        print(f'{key} = {value}')
                    for key in sorted(child.properties()):
                        try:
                            allowed = child.java.getAllowedPropertyValues(key)
                            if allowed is not None:
                                print(f'ALLOWED {key} = {list(allowed)}')
                        except Exception:
                            pass
                    for candidate, value in [
                        ('OffsetDefinition', 'MidSurface'),
                        ('OffsetDefinition', 'TopSurface'),
                        ('OffsetDefinition', 'BottomSurface'),
                        ('OffsetDefinition', 'UserDefined'),
                        ('OffsetDefinition', 'UserDefinedOffset'),
                        ('OffsetDefinition', 'NoOffset'),
                        ('zrel_offset', '0'),
                        ('z_rel_offset', '0'),
                        ('zoffset', '0'),
                        ('z_offset', '0'),
                        ('Position', 'UserDefined'),
                        ('position', 'UserDefined'),
                    ]:
                        try:
                            child.property(candidate, value)
                            print('SET OK', candidate, value)
                        except Exception as error:
                            message = str(error).splitlines()[0] if str(error) else repr(error)
                            print('SET NO', candidate, value, type(error).__name__, message[:160])
    finally:
        client.remove(model)


if __name__ == '__main__':
    main()

"""Print important simple-shell model-tree properties."""

from __future__ import annotations

import argparse
from pathlib import Path

import mph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    return parser.parse_args()


def selected_properties(props: dict, keys: list[str]) -> dict:
    return {key: props.get(key) for key in keys if key in props}


def main() -> None:
    args = parse_args()
    client = mph.start(cores=1)
    model = client.load(args.model.resolve())
    try:
        print('PARAMETERS')
        for key in ['rho_al', 'E_al', 'nu_al', 'h0', 'h_min', 'F0', 'pzt_A']:
            try:
                print(key, '=', model.parameter(key))
            except Exception as error:
                print(key, 'ERROR', type(error).__name__, error)

        print('\nMATERIALS')
        for material in model / 'materials':
            print(material.name(), material.tag(), material.type())
            try:
                print(' selection:', material.selection())
            except Exception as error:
                print(' selection error:', type(error).__name__, error)
            props = material.java.propertyGroup('def')
            for key in ['density', 'youngsmodulus', 'poissonsratio']:
                try:
                    print(f' {key}:', props.getString(key))
                except Exception as error:
                    print(f' {key}: ERROR {type(error).__name__}: {error}')

        print('\nPHYSICS')
        for physics in model / 'physics':
            print(physics.name(), physics.tag(), physics.type())
            try:
                print(' physics selection:', physics.selection())
            except Exception as error:
                print(' physics selection error:', type(error).__name__, error)
            for child in physics.children():
                props = child.properties()
                print(' child:', child.name(), child.tag(), child.type())
                print('  ', selected_properties(
                    props,
                    [
                        'E_mat', 'E', 'nu_mat', 'nu', 'rho_mat', 'rho',
                        'd', 'OffsetDefinition', 'F', 'Ff', 'LoadTypeForce',
                    ],
                ))

        print('\nRECEIVER DATASETS')
        for dataset in model / 'datasets':
            if dataset.type() == 'CutPoint3D':
                props = dataset.properties()
                print(dataset.name(), dataset.tag(), selected_properties(
                    props, ['data', 'pointx', 'pointy', 'pointz', 'method', 'locdef']
                ))
    finally:
        client.remove(model)


if __name__ == '__main__':
    main()

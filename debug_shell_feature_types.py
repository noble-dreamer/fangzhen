"""Probe COMSOL Shell feature type names used by version-specific nodes."""

from __future__ import annotations

import argparse

import simple_shell_common as shell


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cores', type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = shell.start_client(cores=args.cores)
    model = None
    try:
        shell.SWEEP = shell.SweepConfig(
            transmitter_indices=(1,),
            frequencies_hz=(50_000.0,),
            use_parametric_sweep=False,
        )
        shell.MESH = shell.MeshConfig(build_mesh=False)
        model, _problems = shell.build_model_object(client, 'probe_shell_feature_types')
        physics = next(iter(model / 'physics'))
        candidates = [
            'LowReflectingBoundary',
            'LowReflecting',
            'LowReflectingBoundaryCondition',
            'LRB',
            'LowReflectingBnd',
            'AbsorbingBoundary',
            'SpringFoundation',
            'EdgeSpringFoundation',
            'Dashpot',
            'Damper',
            'Fixed',
            'Free',
            'EdgeLoad',
        ]
        for feature_type in candidates:
            for dim in (1, 2):
                try:
                    feature = physics.create(
                        feature_type,
                        dim,
                        name=f'probe {feature_type} dim{dim}',
                    )
                    print('OK', feature_type, dim, feature.tag(), feature.type(), feature.properties())
                except Exception as error:
                    message = str(error).splitlines()[0] if str(error) else repr(error)
                    print('NO', feature_type, dim, type(error).__name__, message[:180])
    finally:
        if model is not None:
            client.remove(model)
        client.clear()


if __name__ == '__main__':
    main()

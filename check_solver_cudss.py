"""Build an in-memory simple shell model and verify the direct solver setting."""

from __future__ import annotations

import argparse

import simple_shell_common as shell


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cores', type=int, default=1)
    parser.add_argument('--linear-solver', choices=('cudss', 'pardiso', 'mumps'), default='cudss')
    parser.add_argument('--cudss-precision', choices=('single', 'double'), default='single')
    parser.add_argument('--dt-out-us', type=float, default=1.0)
    parser.add_argument('--relative-tolerance', type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shell.SWEEP = shell.SweepConfig(
        transmitter_indices=(1,),
        frequencies_hz=(50_000.0,),
        use_parametric_sweep=False,
    )
    shell.MESH = shell.MeshConfig(build_mesh=False)
    shell.SOLVER = shell.SolverConfig(
        solve=False,
        dt_out_us=args.dt_out_us,
        relative_tolerance=args.relative_tolerance,
        direct_linear_solver=args.linear_solver,
        cudss_precision=args.cudss_precision,
    )

    client = shell.start_client(cores=args.cores)
    model = None
    try:
        model, problems = shell.build_model_object(client, 'check_solver_cudss')
        print('problems =', problems)
        found = False
        for solution in model / 'solutions':
            for feature in shell.walk_solver_features(solution):
                if feature.type() == 'Direct':
                    props = feature.properties()
                    print('direct_solver_node =', feature.name(), feature.tag())
                    print('linsolver =', props.get('linsolver'))
                    print('cudssprecision =', props.get('cudssprecision'))
                    print('cudsshybridmemory =', props.get('cudsshybridmemory'))
                    print('cudssusehybridcompute =', props.get('cudssusehybridcompute'))
                    print('cudssmultigpusinglenode =', props.get('cudssmultigpusinglenode'))
                    print('clusterpardiso =', props.get('clusterpardiso'))
                    found = True
        if not found:
            raise RuntimeError('No Direct solver node found.')
    finally:
        if model is not None:
            client.remove(model)
        client.clear()


if __name__ == '__main__':
    main()

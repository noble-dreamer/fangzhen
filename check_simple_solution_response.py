"""Check whether a solved simple-shell model has nonzero displacement response."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import mph
import numpy as np

import simple_shell_common as shell


PARAM_DATASET = 'simple shell displacement transient//参数化解 1'


@dataclass(frozen=True)
class Case:
    outer: int
    tx: int
    frequency_hz: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--tx', default='1')
    parser.add_argument('--frequencies', default='50000')
    parser.add_argument('--dataset', default=PARAM_DATASET)
    parser.add_argument('--cores', type=int, default=4)
    return parser.parse_args()


def parse_int_list(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(',') if item.strip())


def parse_float_list(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in text.split(',') if item.strip())


def cases(tx_indices: tuple[int, ...], frequencies_hz: tuple[float, ...]) -> list[Case]:
    result = []
    outer = 1
    for tx in tx_indices:
        for frequency_hz in frequencies_hz:
            result.append(Case(outer, tx, frequency_hz))
            outer += 1
    return result


def radial_expression(position: dict) -> str:
    theta = math.radians(position['theta_deg'])
    return f'({math.cos(theta):.12g})*u+({math.sin(theta):.12g})*v'


def ensure_cutpoint(model, solution_dataset: str, position: dict):
    name = f'receiver PZT {position["index"]:02d} point'
    for dataset in model / 'datasets':
        if dataset.name() == name:
            dataset.property('data', solution_dataset)
            return dataset
    dataset = (model / 'datasets').create('CutPoint3D', name=name)
    dataset.property('data', solution_dataset)
    dataset.property('pointx', f'{position["x_mm"]:.12g}[mm]')
    dataset.property('pointy', f'{position["y_mm"]:.12g}[mm]')
    dataset.property('pointz', f'{position["z_mm"]:.12g}[mm]')
    return dataset


def main() -> None:
    args = parse_args()
    client = mph.start(cores=args.cores)
    model = client.load(args.model.resolve())
    try:
        positions = shell.receiver_positions()
        datasets = [ensure_cutpoint(model, args.dataset, position) for position in positions]
        expressions = [radial_expression(position) for position in positions]
        for case in cases(parse_int_list(args.tx), parse_float_list(args.frequencies)):
            print(f'CASE tx={case.tx:02d}, f={case.frequency_hz:.0f} Hz, outer={case.outer}')
            time_s = np.asarray(model.evaluate('t', dataset=datasets[0].name(), outer=case.outer), dtype=float).reshape(-1)
            print(f'  time points: {time_s.size}, t0={time_s[0]:.6g}, t1={time_s[-1]:.6g}')
            maxima = []
            for index, (dataset, expression) in enumerate(zip(datasets, expressions, strict=True), start=1):
                values = np.asarray(
                    model.evaluate(expression, unit='m', dataset=dataset.name(), outer=case.outer),
                    dtype=float,
                ).reshape(-1)
                maxima.append(float(np.nanmax(np.abs(values))))
                print(f'  rx{index:02d}: max_abs_ur_m={maxima[-1]:.6e}')
            print(f'  all_rx_max_abs_ur_m={max(maxima):.6e}')
    finally:
        client.remove(model)


if __name__ == '__main__':
    main()

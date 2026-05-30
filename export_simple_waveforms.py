"""Export 16 receiver time traces from solved simple shell MPH files."""

from __future__ import annotations

import argparse
import csv
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
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--sample-id', default=None)
    parser.add_argument('--tx', default='1')
    parser.add_argument('--frequencies', default='50000')
    parser.add_argument('--dataset', default=PARAM_DATASET)
    parser.add_argument('--cores', type=int, default=1)
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


def evaluate_case(model, case: Case, datasets, expressions):
    first_dataset = datasets[0].name()
    time_s = np.asarray(model.evaluate('t', dataset=first_dataset, outer=case.outer), dtype=float).reshape(-1)
    channels = []
    for dataset, expression in zip(datasets, expressions, strict=True):
        values = np.asarray(
            model.evaluate(expression, unit='m', dataset=dataset.name(), outer=case.outer),
            dtype=float,
        ).reshape(-1)
        channels.append(values)
    return time_s, np.vstack(channels).T


def write_waveform(path: Path, time_s: np.ndarray, channels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ['time_s'] + [f'rx{i:02d}_ur_m' for i in range(1, channels.shape[1] + 1)]
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row, time_value in zip(channels, time_s, strict=True):
            writer.writerow([f'{time_value:.12g}', *[f'{value:.12e}' for value in row]])


def main() -> None:
    args = parse_args()
    tx_indices = parse_int_list(args.tx)
    frequencies_hz = parse_float_list(args.frequencies)
    sample_id = args.sample_id or args.model.stem
    waveform_dir = args.output_root / 'csv' / 'waveforms'

    client = mph.start(cores=args.cores)
    model = client.load(args.model.resolve())
    try:
        positions = shell.receiver_positions()
        datasets = [ensure_cutpoint(model, args.dataset, position) for position in positions]
        expressions = [radial_expression(position) for position in positions]
        for case in cases(tx_indices, frequencies_hz):
            print(f'Exporting tx={case.tx:02d}, f={case.frequency_hz:.0f} Hz, outer={case.outer}')
            time_s, channels = evaluate_case(model, case, datasets, expressions)
            write_waveform(
                waveform_dir / f'{sample_id}_tx{case.tx:02d}_f{int(case.frequency_hz)}Hz_waveforms.csv',
                time_s,
                channels,
            )
    finally:
        client.remove(model)


if __name__ == '__main__':
    main()

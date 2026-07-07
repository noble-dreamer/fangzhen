"""Create a frequency-subset H_complex.npz from an existing response file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subset a frequency-domain H_complex.npz by frequency list.")
    parser.add_argument("--input", type=Path, required=True, help="Source *_H_complex.npz with full frequency axis.")
    parser.add_argument("--output", type=Path, required=True, help="Destination subset *_H_complex.npz.")
    parser.add_argument(
        "--frequencies",
        nargs="+",
        default=[],
        help="Requested frequencies in Hz. Values may be separated by spaces or commas.",
    )
    parser.add_argument(
        "--frequency-file",
        type=Path,
        default=None,
        help="Text file containing comma/newline separated frequencies in Hz.",
    )
    parser.add_argument("--sample-id", default=None, help="Override sample_id stored in output.")
    parser.add_argument("--dataset", default=None, help="Override dataset stored in output.")
    parser.add_argument("--defect-state", default=None, help="Override defect_state stored in output.")
    parser.add_argument(
        "--tolerance-hz",
        type=float,
        default=1e-6,
        help="Frequency matching tolerance in Hz.",
    )
    return parser.parse_args()


def parse_frequency_values(tokens: list[str]) -> list[float]:
    values: list[float] = []
    for token in tokens:
        for item in str(token).replace("\n", ",").replace(";", ",").split(","):
            item = item.strip()
            if item:
                values.append(float(item))
    return values


def scalar_string(data: np.lib.npyio.NpzFile, key: str, fallback: str) -> str:
    if key not in data.files:
        return fallback
    value = np.asarray(data[key])
    if value.shape == ():
        return str(value.item())
    return str(value.tolist())


def main() -> None:
    args = parse_args()
    requested = parse_frequency_values(args.frequencies)
    if args.frequency_file is not None:
        requested.extend(parse_frequency_values([args.frequency_file.read_text(encoding="utf-8")]))
    if not requested:
        raise RuntimeError("No requested frequencies. Use --frequencies or --frequency-file.")

    with np.load(args.input, allow_pickle=False) as data:
        required = {"H_real", "H_imag", "completed_mask", "tx_indices", "rx_indices", "frequencies_hz"}
        missing = required.difference(data.files)
        if missing:
            raise RuntimeError(f"{args.input} missing fields: {sorted(missing)}")
        frequencies = np.asarray(data["frequencies_hz"], dtype=float)
        indices: list[int] = []
        missing_freqs: list[float] = []
        for value in requested:
            matches = np.where(np.abs(frequencies - value) <= args.tolerance_hz)[0]
            if matches.size == 0:
                missing_freqs.append(value)
            else:
                indices.append(int(matches[0]))
        if missing_freqs:
            raise RuntimeError(
                "Requested frequencies are not present in source file: "
                + ",".join(f"{value:g}" for value in missing_freqs)
            )
        if len(set(indices)) != len(indices):
            raise RuntimeError("Requested frequency list contains duplicates after matching source frequencies.")

        h_real = np.asarray(data["H_real"])[:, :, indices]
        h_imag = np.asarray(data["H_imag"])[:, :, indices]
        completed = np.asarray(data["completed_mask"], dtype=bool)[:, indices]
        sample_id = args.sample_id or scalar_string(data, "sample_id", args.output.stem.replace("_H_complex", ""))
        dataset = args.dataset or scalar_string(data, "dataset", "")
        defect_state = args.defect_state or scalar_string(data, "defect_state", "")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            H_real=h_real,
            H_imag=h_imag,
            completed_mask=completed,
            tx_indices=np.asarray(data["tx_indices"], dtype=np.int32),
            rx_indices=np.asarray(data["rx_indices"], dtype=np.int32),
            frequencies_hz=frequencies[indices],
            sample_id=np.asarray(sample_id),
            dataset=np.asarray(dataset),
            defect_state=np.asarray(defect_state),
        )
    print(args.output)
    print(",".join(f"{value:.12g}" for value in requested))


if __name__ == "__main__":
    main()

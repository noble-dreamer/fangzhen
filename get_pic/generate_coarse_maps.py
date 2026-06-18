"""Generate V1 frequency-domain coarse maps.

Run with:
    conda run -n get_pic python simple/get_pic/generate_coarse_maps.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import coarse_map_common as cm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate V1 ray-tube coarse maps from frequency responses.")
    parser.add_argument("--healthy", type=Path, default=cm.DEFAULT_HEALTHY_RESPONSE)
    parser.add_argument("--healthy-metadata", type=Path, default=cm.DEFAULT_HEALTHY_METADATA)
    parser.add_argument(
        "--damaged",
        type=Path,
        nargs="*",
        default=None,
        help="Damaged *_H_complex.npz files. If omitted, standard dataset_a_frequency_sample_* files are used.",
    )
    parser.add_argument("--response-dir", type=Path, default=cm.DEFAULT_RESPONSE_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=cm.DEFAULT_METADATA_DIR)
    parser.add_argument("--output-root", type=Path, default=cm.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=cm.ROOT / "configs" / "dataset_a_v1.json")
    parser.add_argument(
        "--selected-frequencies",
        type=Path,
        default=None,
        help="Optional txt/csv frequency selection. Defaults to f_domain frequency_selection output if it exists.",
    )
    parser.add_argument(
        "--no-default-frequency-selection",
        action="store_true",
        help="Use all valid completed frequencies instead of the default frequency_selection txt when present.",
    )
    parser.add_argument("--sample-ids", nargs="*", default=[], help="Standard sample ids/ranges, e.g. 1,3,5-8.")
    parser.add_argument("--preview", action="store_true", help="Also write preview PNGs.")
    return parser.parse_args()


def parse_sample_ids(values: list[str]) -> list[int]:
    ids: list[int] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                start_text, stop_text = token.split("-", 1)
                start = int(start_text)
                stop = int(stop_text)
                if stop < start:
                    raise ValueError(f"Invalid sample id range: {token}")
                ids.extend(range(start, stop + 1))
            else:
                ids.append(int(token))
    return list(dict.fromkeys(ids))


def collect_damaged(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.damaged:
        paths.extend(args.damaged)
    for sample_id in parse_sample_ids(args.sample_ids):
        paths.append(args.response_dir / f"dataset_a_frequency_sample_{sample_id:04d}_H_complex.npz")
    if not paths:
        paths.extend(sorted(args.response_dir.glob("dataset_a_frequency_sample_*_H_complex.npz")))
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise RuntimeError(
            f"No damaged frequency responses found in {args.response_dir}. "
            "Upload/generate damaged *_H_complex.npz files or pass --damaged."
        )
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing damaged response files: " + ", ".join(str(path) for path in missing))
    return paths


def metadata_for_response(path: Path, metadata_dir: Path) -> Path:
    return metadata_dir / f"{cm.sample_id_from_response_path(path)}.json"


def main() -> None:
    args = parse_args()
    config = cm.CoarseMapConfig.from_json(args.config if args.config.exists() else None)
    selected_frequency_file = args.selected_frequencies
    if selected_frequency_file is None and not args.no_default_frequency_selection and cm.DEFAULT_SELECTION_TXT.exists():
        selected_frequency_file = cm.DEFAULT_SELECTION_TXT

    healthy = cm.load_frequency_response(args.healthy)
    healthy_metadata = cm.read_json(args.healthy_metadata) if args.healthy_metadata.exists() else {}
    damaged_paths = collect_damaged(args)
    rows = []
    for damaged_path in damaged_paths:
        damaged = cm.load_frequency_response(damaged_path)
        damaged_metadata_path = metadata_for_response(damaged_path, args.metadata_dir)
        damaged_metadata = cm.read_json(damaged_metadata_path) if damaged_metadata_path.exists() else {}
        products = cm.make_v1_coarse_map(
            healthy,
            damaged,
            healthy_metadata=healthy_metadata,
            damaged_metadata=damaged_metadata,
            selected_frequency_file=selected_frequency_file,
            config=config,
        )
        coarse_path, x_path = cm.write_projection_outputs(
            products,
            output_root=args.output_root,
            healthy=healthy,
            damaged=damaged,
            healthy_metadata_path=args.healthy_metadata if args.healthy_metadata.exists() else None,
            damaged_metadata_path=damaged_metadata_path if damaged_metadata_path.exists() else None,
        )
        if args.preview:
            import preview_coarse_maps

            preview_path = args.output_root / "previews" / f"{damaged.sample_id}_coarse_preview.png"
            preview_coarse_maps.write_preview(coarse_path, preview_path)
        rows.append({
            "sample_id": damaged.sample_id,
            "damaged_npz": str(damaged_path),
            "coarse_map_npz": str(coarse_path),
            "x_matrix_npz": str(x_path),
            "selected_frequency_file": "" if selected_frequency_file is None else str(selected_frequency_file),
            "valid_tx_rx_frequency_count": products.metadata["valid_tx_rx_frequency_count"],
        })
        print(json.dumps(rows[-1], ensure_ascii=False))
    cm.write_manifest(args.output_root / "manifest.csv", rows)


if __name__ == "__main__":
    main()

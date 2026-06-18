"""Compare V1 images from selected frequencies against all completed frequencies.

Example:
    conda run -n get_pic python simple/get_pic/compare_frequency_selection.py --sample-id 14
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import coarse_map_common as cm
import evaluate_coarse_maps
import preview_coarse_maps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare selected-frequency V1 imaging against all-frequency V1 imaging.")
    parser.add_argument("--sample-id", type=int, default=14)
    parser.add_argument("--healthy", type=Path, default=cm.DEFAULT_HEALTHY_RESPONSE)
    parser.add_argument("--healthy-metadata", type=Path, default=cm.DEFAULT_HEALTHY_METADATA)
    parser.add_argument("--response-dir", type=Path, default=cm.DEFAULT_RESPONSE_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=cm.DEFAULT_METADATA_DIR)
    parser.add_argument("--label-dir", type=Path, default=cm.DEFAULT_LABEL_DIR)
    parser.add_argument("--selected-frequencies", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=cm.DEFAULT_OUTPUT_ROOT / "frequency_selection_compare")
    parser.add_argument("--config", type=Path, default=cm.ROOT / "configs" / "dataset_a_v1.json")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--preview", action="store_true")
    return parser.parse_args()


def sample_name(sample_id: int) -> str:
    return f"dataset_a_frequency_sample_{sample_id:04d}"


def write_single_frequency_txt(path: Path, frequency_hz: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{frequency_hz:.12g}\n", encoding="utf-8")
    return path


def build_one(
    *,
    healthy: cm.FrequencyResponse,
    damaged: cm.FrequencyResponse,
    healthy_metadata: dict,
    damaged_metadata: dict,
    healthy_metadata_path: Path | None,
    damaged_metadata_path: Path | None,
    selected_file: Path | None,
    output_root: Path,
    config: cm.CoarseMapConfig,
) -> tuple[Path, Path]:
    products = cm.make_v1_coarse_map(
        healthy,
        damaged,
        healthy_metadata=healthy_metadata,
        damaged_metadata=damaged_metadata,
        selected_frequency_file=selected_file,
        config=config,
    )
    return cm.write_projection_outputs(
        products,
        output_root=output_root,
        healthy=healthy,
        damaged=damaged,
        healthy_metadata_path=healthy_metadata_path,
        damaged_metadata_path=damaged_metadata_path,
    )


def best_metric(report: dict, channel: str = "ray_relative_delta") -> dict:
    return report["metrics"].get(channel, {})


def main() -> None:
    args = parse_args()
    config = cm.CoarseMapConfig.from_json(args.config if args.config.exists() else None)
    name = sample_name(args.sample_id)
    damaged_path = args.response_dir / f"{name}_H_complex.npz"
    damaged_metadata_path = args.metadata_dir / f"{name}.json"
    label_path = args.label_dir / f"{name}_defect_depth_norm.npy"
    if not label_path.exists():
        raise FileNotFoundError(label_path)
    healthy = cm.load_frequency_response(args.healthy)
    damaged = cm.load_frequency_response(damaged_path)
    healthy_metadata = cm.read_json(args.healthy_metadata) if args.healthy_metadata.exists() else {}
    damaged_metadata = cm.read_json(damaged_metadata_path) if damaged_metadata_path.exists() else {}

    cases = [
        ("selected", args.selected_frequencies),
        ("all_completed", None),
    ]
    rows = []
    for tag, selected_file in cases:
        case_root = args.output_root / tag
        coarse_path, x_path = build_one(
            healthy=healthy,
            damaged=damaged,
            healthy_metadata=healthy_metadata,
            damaged_metadata=damaged_metadata,
            healthy_metadata_path=args.healthy_metadata if args.healthy_metadata.exists() else None,
            damaged_metadata_path=damaged_metadata_path if damaged_metadata_path.exists() else None,
            selected_file=selected_file,
            output_root=case_root,
            config=config,
        )
        report = evaluate_coarse_maps.evaluate_one(coarse_path, label_path, args.threshold, raw=False)
        report_path = case_root / "reports" / f"{coarse_path.stem}_metrics.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        preview_path = ""
        if args.preview:
            preview_file = case_root / "previews" / f"{coarse_path.stem}_preview.png"
            preview_coarse_maps.write_preview(coarse_path, preview_file)
            preview_path = str(preview_file)
        m = best_metric(report)
        rows.append({
            "case": tag,
            "sample_id": name,
            "selected_frequency_file": "" if selected_file is None else str(selected_file),
            "coarse_map": str(coarse_path),
            "x_matrix": str(x_path),
            "report": str(report_path),
            "preview": preview_path,
            "ray_relative_delta_pearson": m.get("pearson", ""),
            "ray_relative_delta_nrmse": m.get("nrmse", ""),
            "ray_relative_delta_mask_iou": m.get("mask_iou", ""),
            "ray_relative_delta_centroid_error_mm": m.get("centroid_error_mm", ""),
        })

    summary_csv = args.output_root / f"{name}_selection_vs_all_summary.csv"
    cm.write_manifest(summary_csv, rows)
    summary_json = args.output_root / f"{name}_selection_vs_all_summary.json"
    summary_json.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_csv)
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()

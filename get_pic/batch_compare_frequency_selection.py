"""Batch-compare frequency-selection methods against all completed frequencies."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev

import compare_frequency_selection as compare_one
import coarse_map_common as cm
import evaluate_coarse_maps
import preview_coarse_maps


METRIC_SPECS = [
    ("pearson", "ray_relative_delta_pearson", True),
    ("nrmse", "ray_relative_delta_nrmse", False),
    ("mask_iou", "ray_relative_delta_mask_iou", True),
    ("top5_hit_rate", "ray_relative_delta_top5_hit_rate", True),
    ("prediction_mass_in_label", "ray_relative_delta_prediction_mass_in_label", True),
    ("centroid_error_mm", "ray_relative_delta_centroid_error_mm", False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch compare selected-frequency V1 images against all-frequency images.")
    parser.add_argument("--sample-ids", nargs="*", default=["1-12", "14-23"])
    parser.add_argument("--healthy", type=Path, default=cm.DEFAULT_HEALTHY_RESPONSE)
    parser.add_argument("--healthy-metadata", type=Path, default=cm.DEFAULT_HEALTHY_METADATA)
    parser.add_argument("--response-dir", type=Path, default=cm.DEFAULT_RESPONSE_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=cm.DEFAULT_METADATA_DIR)
    parser.add_argument("--label-dir", type=Path, default=cm.DEFAULT_LABEL_DIR)
    parser.add_argument("--config", type=Path, default=cm.ROOT / "configs" / "dataset_a_v1.json")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--preview-first", type=int, default=0, help="Only write preview PNG for the first N samples.")
    parser.add_argument("--output-root", type=Path, default=cm.DEFAULT_OUTPUT_ROOT / "batch_frequency_selection_compare")
    parser.add_argument("--physics-selected", type=Path, required=True)
    parser.add_argument("--relative-selected", type=Path, required=True)
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
                ids.extend(range(start, stop + 1))
            else:
                ids.append(int(token))
    return list(dict.fromkeys(ids))


def read_selected_frequencies(path: Path) -> list[float]:
    text = path.read_text(encoding="utf-8").replace("\n", ",")
    values: list[float] = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    return values


def numeric_value(row: dict, key: str) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return float("nan")
    return float(value)


def write_selection_snapshot(args: argparse.Namespace, sample_ids: list[int]) -> dict:
    physics_hz = read_selected_frequencies(args.physics_selected)
    relative_hz = read_selected_frequencies(args.relative_selected)
    snapshot = {
        "sample_ids": sample_ids,
        "sample_count": len(sample_ids),
        "preview_first": args.preview_first,
        "physics_selected_file": str(args.physics_selected),
        "physics_selected_frequencies_hz": physics_hz,
        "physics_selected_frequencies_khz": [value / 1000.0 for value in physics_hz],
        "relative_selected_file": str(args.relative_selected),
        "relative_selected_frequencies_hz": relative_hz,
        "relative_selected_frequencies_khz": [value / 1000.0 for value in relative_hz],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "physics_tomography_tuned_top15_frequencies.txt").write_text(
        args.physics_selected.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (args.output_root / "relative_l2_top15_frequencies.txt").write_text(
        args.relative_selected.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    snapshot_path = args.output_root / "batch_compare_selected_frequencies.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return snapshot


def evaluate_method(
    *,
    sample_id: int,
    selected_file: Path | None,
    method_name: str,
    args: argparse.Namespace,
    preview: bool,
    include_all_completed: bool,
) -> list[dict]:
    name = compare_one.sample_name(sample_id)
    healthy = cm.load_frequency_response(args.healthy)
    damaged_path = args.response_dir / f"{name}_H_complex.npz"
    damaged = cm.load_frequency_response(damaged_path)
    healthy_metadata = cm.read_json(args.healthy_metadata) if args.healthy_metadata.exists() else {}
    damaged_metadata_path = args.metadata_dir / f"{name}.json"
    damaged_metadata = cm.read_json(damaged_metadata_path) if damaged_metadata_path.exists() else {}
    config = cm.CoarseMapConfig.from_json(args.config if args.config.exists() else None)
    label_path = args.label_dir / f"{name}_defect_depth_norm.npy"

    def build_or_reuse(
        *,
        case_root: Path,
        chosen: Path | None,
        preview_enabled: bool,
    ) -> tuple[Path, Path, Path, str]:
        coarse_path = case_root / "coarse_maps" / f"{name}_coarse_maps.npz"
        x_path = case_root / "x_matrix" / f"{name}_x_matrix.npz"
        report_path = case_root / "reports" / f"{name}_coarse_maps_metrics.json"
        preview_path = case_root / "previews" / f"{name}_coarse_maps_preview.png"
        if not (coarse_path.exists() and x_path.exists()):
            coarse_path, x_path = compare_one.build_one(
                healthy=healthy,
                damaged=damaged,
                healthy_metadata=healthy_metadata,
                damaged_metadata=damaged_metadata,
                healthy_metadata_path=args.healthy_metadata if args.healthy_metadata.exists() else None,
                damaged_metadata_path=damaged_metadata_path if damaged_metadata_path.exists() else None,
                selected_file=chosen,
                output_root=case_root,
                config=config,
            )
        if not report_path.exists():
            report = evaluate_coarse_maps.evaluate_one(coarse_path, label_path, args.threshold, raw=False)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if preview_enabled and not preview_path.exists():
            preview_coarse_maps.write_preview(coarse_path, preview_path)
        return coarse_path, x_path, report_path, (str(preview_path) if preview_enabled and preview_path.exists() else "")

    rows = []
    selected_root = args.output_root / method_name / f"sample_{sample_id:04d}" / method_name
    coarse_path, x_path, report_path, preview_path = build_or_reuse(
        case_root=selected_root,
        chosen=selected_file,
        preview_enabled=preview,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    m = compare_one.best_metric(report)
    rows.append({
        "method": method_name,
        "baseline": "all_completed",
        "sample_id": name,
        "selected_frequency_file": "" if selected_file is None else str(selected_file),
        "coarse_map": str(coarse_path),
        "x_matrix": str(x_path),
        "report": str(report_path),
        "preview": preview_path,
        "ray_relative_delta_pearson": m.get("pearson", ""),
        "ray_relative_delta_nrmse": m.get("nrmse", ""),
        "ray_relative_delta_mask_iou": m.get("mask_iou", ""),
        "ray_relative_delta_top5_hit_rate": m.get("top5_hit_rate", ""),
        "ray_relative_delta_prediction_mass_in_label": m.get("prediction_mass_in_label", ""),
        "ray_relative_delta_centroid_error_mm": m.get("centroid_error_mm", ""),
    })

    if include_all_completed:
        shared_all_root = args.output_root / "_shared_all_completed" / f"sample_{sample_id:04d}" / "all_completed"
        coarse_path, x_path, report_path, preview_path = build_or_reuse(
            case_root=shared_all_root,
            chosen=None,
            preview_enabled=preview,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        m = compare_one.best_metric(report)
        rows.append({
            "method": "all_completed",
            "baseline": "all_completed",
            "sample_id": name,
            "selected_frequency_file": "",
            "coarse_map": str(coarse_path),
            "x_matrix": str(x_path),
            "report": str(report_path),
            "preview": preview_path,
            "ray_relative_delta_pearson": m.get("pearson", ""),
            "ray_relative_delta_nrmse": m.get("nrmse", ""),
            "ray_relative_delta_mask_iou": m.get("mask_iou", ""),
            "ray_relative_delta_top5_hit_rate": m.get("top5_hit_rate", ""),
            "ray_relative_delta_prediction_mass_in_label": m.get("prediction_mass_in_label", ""),
            "ray_relative_delta_centroid_error_mm": m.get("centroid_error_mm", ""),
        })
    return rows


def summarize_pairs(rows: list[dict], method_name: str) -> dict:
    selected = [row for row in rows if row["method"] == method_name]
    all_rows = [row for row in rows if row["method"] == "all_completed"]
    by_sample_all = {row["sample_id"]: row for row in all_rows}
    deltas = {
        "pearson_delta": [],
        "nrmse_delta": [],
        "top5_hit_rate_delta": [],
        "prediction_mass_in_label_delta": [],
        "centroid_error_mm_delta": [],
    }
    for row in selected:
        base = by_sample_all[row["sample_id"]]
        deltas["pearson_delta"].append(row["ray_relative_delta_pearson"] - base["ray_relative_delta_pearson"])
        deltas["nrmse_delta"].append(row["ray_relative_delta_nrmse"] - base["ray_relative_delta_nrmse"])
        deltas["top5_hit_rate_delta"].append(
            row["ray_relative_delta_top5_hit_rate"] - base["ray_relative_delta_top5_hit_rate"]
        )
        deltas["prediction_mass_in_label_delta"].append(
            row["ray_relative_delta_prediction_mass_in_label"] - base["ray_relative_delta_prediction_mass_in_label"]
        )
        deltas["centroid_error_mm_delta"].append(
            row["ray_relative_delta_centroid_error_mm"] - base["ray_relative_delta_centroid_error_mm"]
        )
    summary = {"method": method_name, "sample_count": len(selected)}
    for key, values in deltas.items():
        summary[f"{key}_mean"] = mean(values) if values else float("nan")
        summary[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return summary


def summarize_pairwise(rows: list[dict], method_a: str, method_b: str) -> tuple[list[dict], dict]:
    rows_a = [row for row in rows if row["method"] == method_a]
    rows_b = {row["sample_id"]: row for row in rows if row["method"] == method_b}
    sample_rows: list[dict] = []
    comparison = f"{method_a}_vs_{method_b}"
    for row_a in rows_a:
        sample_id = row_a["sample_id"]
        row_b = rows_b.get(sample_id)
        if row_b is None:
            continue
        pair_row = {
            "comparison": comparison,
            "method_a": method_a,
            "method_b": method_b,
            "sample_id": sample_id,
        }
        for short_name, column, higher_is_better in METRIC_SPECS:
            value_a = numeric_value(row_a, column)
            value_b = numeric_value(row_b, column)
            delta = value_a - value_b
            improvement = delta if higher_is_better else -delta
            pair_row[f"{short_name}_a"] = value_a
            pair_row[f"{short_name}_b"] = value_b
            pair_row[f"{short_name}_delta"] = delta
            pair_row[f"{short_name}_improvement"] = improvement
        sample_rows.append(pair_row)

    summary = {
        "comparison": comparison,
        "method_a": method_a,
        "method_b": method_b,
        "sample_count": len(sample_rows),
    }
    for short_name, _column, _higher_is_better in METRIC_SPECS:
        deltas = [
            row[f"{short_name}_delta"]
            for row in sample_rows
            if math.isfinite(row[f"{short_name}_delta"])
        ]
        improvements = [
            row[f"{short_name}_improvement"]
            for row in sample_rows
            if math.isfinite(row[f"{short_name}_improvement"])
        ]
        summary[f"{short_name}_delta_mean"] = mean(deltas) if deltas else float("nan")
        summary[f"{short_name}_delta_std"] = pstdev(deltas) if len(deltas) > 1 else 0.0
        summary[f"{short_name}_improvement_mean"] = mean(improvements) if improvements else float("nan")
        summary[f"{short_name}_improvement_std"] = pstdev(improvements) if len(improvements) > 1 else 0.0
        summary[f"{short_name}_better_count"] = sum(1 for value in improvements if value > 0.0)
        summary[f"{short_name}_worse_count"] = sum(1 for value in improvements if value < 0.0)
    return sample_rows, summary


def main() -> None:
    args = parse_args()
    sample_ids = parse_sample_ids(args.sample_ids)
    selection_snapshot = write_selection_snapshot(args, sample_ids)
    rows: list[dict] = []
    for index, sample_id in enumerate(sample_ids):
        preview = index < args.preview_first
        rows.extend(
            evaluate_method(
                sample_id=sample_id,
                selected_file=args.physics_selected,
                method_name="physics_tomography_tuned",
                args=args,
                preview=preview,
                include_all_completed=True,
            )
        )
        rows.extend(
            evaluate_method(
                sample_id=sample_id,
                selected_file=args.relative_selected,
                method_name="relative_l2",
                args=args,
                preview=preview,
                include_all_completed=False,
            )
        )
    manifest = args.output_root / "batch_compare_manifest.csv"
    cm.write_manifest(manifest, rows)
    legacy_summaries = [
        summarize_pairs(rows, "physics_tomography_tuned"),
        summarize_pairs(rows, "relative_l2"),
    ]
    pairwise_rows: list[dict] = []
    pairwise_summaries: list[dict] = []
    for method_a, method_b in (
        ("physics_tomography_tuned", "all_completed"),
        ("relative_l2", "all_completed"),
        ("physics_tomography_tuned", "relative_l2"),
    ):
        sample_rows, summary = summarize_pairwise(rows, method_a, method_b)
        pairwise_rows.extend(sample_rows)
        pairwise_summaries.append(summary)
    pairwise_csv = args.output_root / "batch_compare_pairwise.csv"
    cm.write_manifest(pairwise_csv, pairwise_rows)
    summary_csv = args.output_root / "batch_compare_summary.csv"
    cm.write_manifest(summary_csv, pairwise_summaries)
    summary_json = args.output_root / "batch_compare_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "selection_snapshot": selection_snapshot,
                "legacy_vs_all_summaries": legacy_summaries,
                "pairwise_summaries": pairwise_summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(manifest)
    print(pairwise_csv)
    print(summary_csv)
    print(summary_json)


if __name__ == "__main__":
    main()

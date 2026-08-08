"""Compare physical inversion and EDM predictions on the same millimetre scale."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simple.get_pic.physical_inversion.inversion import (  # noqa: E402
    InversionConfig,
    PhysicalInverter,
    parse_sample_ids,
    sample_name,
)
from simple.get_pic.physical_inversion.metrics import compute_metrics, write_json  # noqa: E402


DEFAULT_CONFIG = HERE / "configs" / "dataset_a_rytov_tv.json"
LOWER_IS_BETTER = {
    "mae_mm",
    "rmse_mm",
    "volume_error",
    "max_depth_error_mm",
    "centroid_error_mm",
    "peak_location_error_mm",
    "defect_region_mae_mm",
    "false_positive_mean_mm",
    "physical_limit_exceedance_fraction",
}
HIGHER_IS_BETTER = {
    "ssim",
    "pearson",
    "iou",
    "dice",
    "top5_hit_rate",
    "prediction_mass_in_label",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare v2 simulation-derived physical inversion with EDM outputs."
    )
    parser.add_argument("--edm-pred-dir", type=Path, required=True)
    parser.add_argument("--physical-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sample-ids", nargs="+", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--allow-smoke", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required manifest not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = row.get("sample_id", "")
        if not sample_id:
            raise RuntimeError(f"Manifest row without sample_id: {path}")
        if sample_id in output:
            raise RuntimeError(f"Duplicate sample_id {sample_id} in {path}")
        output[sample_id] = row
    return output


def resolve_recorded_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    candidates = [path] if path.is_absolute() else [PROJECT_ROOT / path, manifest_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Manifest path does not exist: {value}")


def load_image(path: Path) -> np.ndarray:
    array = np.asarray(np.load(path), dtype=np.float32).squeeze()
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{path} must contain one finite 2D image")
    return array


def save_comparison_preview(
    path: Path,
    *,
    label: np.ndarray,
    physical: np.ndarray,
    edm: np.ndarray,
    theta_deg: np.ndarray,
    z_mm: np.ndarray,
    depth_limit_mm: float,
    title: str,
) -> None:
    extent = [float(theta_deg[0]), 360.0, float(z_mm[-1]), float(z_mm[0])]
    panels = (
        (label, "Ground truth", "viridis"),
        (physical, "Physical inversion", "viridis"),
        (edm, "EDM posterior mean", "viridis"),
        (np.abs(physical - label), "Physical abs. error", "magma"),
        (np.abs(edm - label), "EDM abs. error", "magma"),
    )
    figure, axes = plt.subplots(1, 5, figsize=(21.0, 4.2), constrained_layout=True)
    for axis, (image, panel_title, cmap) in zip(axes, panels):
        artist = axis.imshow(
            image,
            origin="upper",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=0.0,
            vmax=depth_limit_mm,
            interpolation="nearest",
        )
        axis.set_title(panel_title)
        axis.set_xlabel("theta (deg)")
        axis.set_ylabel("z (mm)")
        colorbar = figure.colorbar(artist, ax=axis, fraction=0.042, pad=0.025)
        colorbar.set_label("Wall loss depth (mm)")
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def finite_mean(rows: list[dict[str, Any]], prefix: str, key: str) -> float:
    values = np.asarray([row.get(f"{prefix}_{key}", np.nan) for row in rows], dtype=np.float64)
    return float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")


def main() -> None:
    args = parse_args()
    config = InversionConfig.from_json(args.config.resolve())
    inverter = PhysicalInverter(config)
    physical_root = args.physical_root.resolve() if args.physical_root else config.output_path
    edm_root = args.edm_pred_dir.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else physical_root / "comparison_with_edm"
    )
    sample_ids = (
        parse_sample_ids(args.sample_ids)
        if args.sample_ids is not None
        else list(config.validation_sample_ids)
    )
    physical_manifest_path = physical_root / "manifest.csv"
    edm_manifest_path = edm_root / "manifest.csv"
    physical_manifest = load_manifest(physical_manifest_path)
    edm_manifest = load_manifest(edm_manifest_path)
    if not args.allow_smoke:
        checkpoint_values = [row.get("checkpoint", "") for row in edm_manifest.values()]
        if "smoke" in str(edm_root).lower() or any("smoke" in value.lower() for value in checkpoint_values):
            raise RuntimeError("Refusing a model-quality comparison against smoke outputs; use --allow-smoke for code testing only")
    theta_deg, z_mm = inverter.grid_axes()
    rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        name = sample_name(sample_id)
        if name not in physical_manifest or name not in edm_manifest:
            raise KeyError(f"{name} is missing from one of the two manifests")
        physical_row = physical_manifest[name]
        edm_row = edm_manifest[name]
        physical_path = resolve_recorded_path(physical_row["prediction_mm_npy"], physical_manifest_path)
        edm_path = resolve_recorded_path(edm_row["prediction_mm_npy"], edm_manifest_path)
        physical = load_image(physical_path)
        edm = load_image(edm_path)
        if physical.shape != (config.image_size, config.image_size):
            raise ValueError(f"Unexpected physical image shape for {name}: {physical.shape}")
        if edm.shape != physical.shape:
            raise ValueError(f"EDM/physical image shape mismatch for {name}: {edm.shape} vs {physical.shape}")
        label = inverter.load_label_mm(sample_id)
        metadata = json.loads(
            (inverter.label_dir / f"{name}_defect_label_metadata.json").read_text(encoding="utf-8")
        )
        depth_limit_mm = float(metadata["depth_limit_mm"])
        denominator_mm = float(metadata["normalization_denominator_mm"])
        if not np.isclose(float(physical_row["depth_limit_mm"]), depth_limit_mm):
            raise RuntimeError(f"Physical manifest depth limit mismatch for {name}")
        if not np.isclose(float(physical_row["normalization_denominator_mm"]), denominator_mm):
            raise RuntimeError(f"Physical manifest normalization mismatch for {name}")
        if not np.isclose(float(edm_row["physical_depth_limit_mm"]), depth_limit_mm):
            raise RuntimeError(f"EDM manifest depth limit mismatch for {name}")
        if not np.isclose(float(edm_row["normalization_denominator_mm"]), denominator_mm):
            raise RuntimeError(f"EDM manifest normalization mismatch for {name}")
        physical_metrics = compute_metrics(
            physical,
            label,
            theta_deg=theta_deg,
            z_mm=z_mm,
            mid_radius_mm=inverter.geometry.mid_radius_mm,
            depth_limit_mm=depth_limit_mm,
            threshold_mm=config.defect_threshold_mm,
        )
        edm_metrics = compute_metrics(
            edm,
            label,
            theta_deg=theta_deg,
            z_mm=z_mm,
            mid_radius_mm=inverter.geometry.mid_radius_mm,
            depth_limit_mm=depth_limit_mm,
            threshold_mm=config.defect_threshold_mm,
        )
        row: dict[str, Any] = {
            "sample_id": name,
            "physical_prediction": str(physical_path),
            "edm_prediction": str(edm_path),
            "edm_checkpoint": edm_row.get("checkpoint", ""),
            "edm_state_key": edm_row.get("state_key", ""),
            "edm_steps": edm_row.get("steps", ""),
            "edm_num_posterior_samples": edm_row.get("num_posterior_samples", ""),
        }
        row.update({f"physical_{key}": value for key, value in physical_metrics.items()})
        row.update({f"edm_{key}": value for key, value in edm_metrics.items()})
        rows.append(row)
        if not args.no_preview:
            save_comparison_preview(
                output_root / name / f"{name}_physical_vs_edm.png",
                label=label,
                physical=physical,
                edm=edm,
                theta_deg=theta_deg,
                z_mm=z_mm,
                depth_limit_mm=depth_limit_mm,
                title=name,
            )
        print(f"Compared {name}")

    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "metrics.csv"
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    metric_keys = sorted(LOWER_IS_BETTER | HIGHER_IS_BETTER)
    summary_metrics: dict[str, Any] = {}
    for key in metric_keys:
        physical_mean = finite_mean(rows, "physical", key)
        edm_mean = finite_mean(rows, "edm", key)
        if key in LOWER_IS_BETTER:
            edm_better = sum(row[f"edm_{key}"] < row[f"physical_{key}"] for row in rows)
        elif key in HIGHER_IS_BETTER:
            edm_better = sum(row[f"edm_{key}"] > row[f"physical_{key}"] for row in rows)
        else:
            continue
        summary_metrics[key] = {
            "physical_mean": physical_mean,
            "edm_mean": edm_mean,
            "edm_minus_physical": edm_mean - physical_mean,
            "edm_better_count": edm_better,
            "sample_count": len(rows),
        }
    write_json(
        output_root / "summary.json",
        {
            "sample_ids": [sample_name(value) for value in sample_ids],
            "comparison_role": "fixed_validation_not_independent_test",
            "shared_depth_limit_mm": float(metadata["depth_limit_mm"]),
            "normalization_denominator_mm": float(metadata["normalization_denominator_mm"]),
            "metrics": summary_metrics,
            "metrics_csv": str(metrics_path),
            "edm_checkpoints": sorted({row["edm_checkpoint"] for row in rows}),
            "edm_state_keys": sorted({row["edm_state_key"] for row in rows}),
            "edm_steps": sorted({row["edm_steps"] for row in rows}),
            "edm_num_posterior_samples": sorted({row["edm_num_posterior_samples"] for row in rows}),
            "warning": (
                "This split selected EDM best.pt and is validation, not an unbiased final test. "
                "Do not interpret smoke checkpoints as model quality."
            ),
        },
    )
    print(f"Comparison metrics: {metrics_path}")


if __name__ == "__main__":
    main()

"""Invert formal responses with the distribution-matched COMSOL channel prior."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simple.get_pic import coarse_map_common as cm  # noqa: E402
from simple.get_pic.physical_inversion.inversion import (  # noqa: E402
    InversionConfig,
    PhysicalInverter,
    load_label_metadata,
    parse_sample_ids,
    physical_limits,
    sample_name,
)
from simple.get_pic.physical_inversion.metrics import compute_metrics, save_preview, write_json  # noqa: E402
from simple.get_pic.physical_inversion.simulation_prior.channel_prior import (  # noqa: E402
    SimulationChannelPrior,
)


DEFAULT_CONFIG = HERE / "configs" / "dataset_a_rytov_tv.json"
DEFAULT_PRIOR = HERE / "simulation_prior" / "output_matched_corpus" / "channel_prior.json"
DEFAULT_OUTPUT_ROOT = HERE / "simulation_prior" / "output_dataset_matched_corpus"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Physical inversion using a distribution-matched COMSOL complex-Rytov prior."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--sample-ids", nargs="+", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument(
        "--evaluate-labels",
        dest="evaluate_labels",
        action="store_true",
        default=True,
        help="Read formal labels only after inversion for metrics and preview rendering.",
    )
    parser.add_argument(
        "--no-evaluate-labels",
        dest="evaluate_labels",
        action="store_false",
        help="Do not read formal labels at all.",
    )
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


def finite_mean(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float64)
    return float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key != "sample_id" and isinstance(value, (float, int, np.floating, np.integer))
        }
    )
    return {"sample_count": len(rows), "mean": {key: finite_mean(rows, key) for key in keys}}


def default_sample_ids(config: InversionConfig) -> list[int]:
    ids = list(config.formal_sample_ids)
    if not ids:
        raise ValueError("No formal sample IDs are configured")
    return ids


def main() -> None:
    args = parse_args()
    config = InversionConfig.from_json(args.config.resolve())
    prior = SimulationChannelPrior.load(args.prior.resolve())
    inverter = PhysicalInverter(config, helical_order_weights=prior.order_weights)
    prior.assert_compatible(inverter)
    output_root = args.output_root.resolve() if args.output_root else DEFAULT_OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    sample_ids = parse_sample_ids(args.sample_ids) if args.sample_ids is not None else default_sample_ids(config)
    if not sample_ids:
        raise ValueError("No formal sample IDs requested")
    iterations = config.sirt_iterations if args.iterations is None else int(args.iterations)
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    contract_path = output_root / "run_contract.json"
    run_contract = {
        "schema_version": 1,
        "algorithm": "distribution_matched_comsol_complex_rytov_ray_tomography",
        "prior_metadata": str(prior.metadata_path),
        "prior_model_sha256": prior.model_sha256,
        "real_formal_sample_labels_used_for_inversion": False,
        "evaluation_labels_requested": bool(args.evaluate_labels),
        "sample_ids": [sample_name(value) for value in sample_ids],
        "sirt_iterations": iterations,
        "helical_orders": prior.helical_orders.tolist(),
        "order_weights": prior.order_weights.tolist(),
        "normalization_denominator_mm": prior.normalization_denominator_mm,
        "depth_limit_mm": prior.depth_limit_mm,
        "config": config.to_dict(),
    }
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != run_contract:
            raise RuntimeError(f"Existing output root has a different run contract: {contract_path}")
    else:
        write_json(contract_path, run_contract)

    theta_deg, z_mm = inverter.grid_axes()
    manifest_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for index, sample_id in enumerate(sample_ids, start=1):
        name = sample_name(sample_id)
        damaged = cm.load_frequency_response(inverter.damaged_path(sample_id))
        observation, response_diagnostics = prior.estimate_path_depth(inverter, damaged)
        raw_grid, solver_diagnostics = inverter.invert_path_depth(
            observation,
            depth_limit_mm=prior.depth_limit_mm,
            iterations=iterations,
        )
        prediction = np.clip(inverter.output_grid(raw_grid), 0.0, prior.depth_limit_mm).astype(np.float32)
        prediction_norm = (prediction / prior.normalization_denominator_mm).astype(np.float32)
        sample_dir = output_root / "samples" / name
        sample_dir.mkdir(parents=True, exist_ok=True)
        raw_path = sample_dir / f"{name}_simulation_channel_prior_raw_grid.npy"
        mm_path = sample_dir / f"{name}_simulation_channel_prior_mm.npy"
        norm_path = sample_dir / f"{name}_simulation_channel_prior_norm.npy"
        observation_path = sample_dir / f"{name}_path_observation.npz"
        np.save(raw_path, raw_grid.astype(np.float32))
        np.save(mm_path, prediction)
        np.save(norm_path, prediction_norm)
        np.savez_compressed(
            observation_path,
            path_depth_mm=observation.values_mm,
            weights=observation.weights,
            frequency_hz=observation.frequencies_hz,
            tx_indices=inverter.operator.tx_indices,
            rx_indices=inverter.operator.rx_indices,
            order_weights=prior.order_weights,
            helical_orders=prior.helical_orders,
            theta_deg=inverter.operator.theta_deg,
            z_mm=inverter.operator.z_mm,
            coverage=inverter.operator.coverage,
        )

        metrics: dict[str, float] = {}
        preview_path: Path | None = None
        label_path = inverter.label_path(sample_id)
        if args.evaluate_labels:
            label = inverter.load_label_mm(sample_id)
            metadata = load_label_metadata(inverter.label_dir, sample_id)
            denominator, depth_limit = physical_limits(metadata)
            if not np.isclose(denominator, prior.normalization_denominator_mm) or not np.isclose(
                depth_limit, prior.depth_limit_mm
            ):
                raise RuntimeError(f"Physical scale mismatch for {name}")
            metrics = compute_metrics(
                prediction,
                label,
                theta_deg=theta_deg,
                z_mm=z_mm,
                mid_radius_mm=inverter.geometry.mid_radius_mm,
                depth_limit_mm=prior.depth_limit_mm,
                threshold_mm=config.defect_threshold_mm,
            )
            metric_rows.append({"sample_id": name, **metrics})
            if not args.no_preview:
                preview_path = sample_dir / f"{name}_simulation_channel_prior_preview.png"
                save_preview(
                    preview_path,
                    prediction_mm=prediction,
                    target_mm=label,
                    theta_deg=theta_deg,
                    z_mm=z_mm,
                    depth_limit_mm=prior.depth_limit_mm,
                    title=f"{name} distribution-matched simulation prior",
                )
        diagnostics = {
            "sample_id": name,
            "algorithm": "distribution_matched_comsol_complex_rytov_ray_tomography",
            "source_healthy_npz": str(inverter.healthy_path),
            "source_damaged_npz": str(damaged.path),
            "source_label_used_after_inversion_for_evaluation_only": str(label_path) if args.evaluate_labels else None,
            "real_formal_sample_labels_used_for_inversion": False,
            "prior_metadata": str(prior.metadata_path),
            "prior_model_npz": str(prior.model_path),
            "prior_model_sha256": prior.model_sha256,
            "helical_orders": prior.helical_orders.tolist(),
            "order_weights": prior.order_weights.tolist(),
            "normalization_denominator_mm": prior.normalization_denominator_mm,
            "depth_limit_mm": prior.depth_limit_mm,
            "response_to_path_mm": response_diagnostics,
            "solver": solver_diagnostics,
            "metrics": metrics,
        }
        diagnostics_path = sample_dir / f"{name}_diagnostics.json"
        write_json(diagnostics_path, diagnostics)
        manifest_rows.append(
            {
                "sample_id": name,
                "split": "validation" if sample_id in config.validation_sample_ids else "formal_reference",
                "source_damaged_npz": str(damaged.path),
                "prediction_mm_npy": str(mm_path),
                "prediction_norm_npy": str(norm_path),
                "raw_grid_npy": str(raw_path),
                "path_observation_npz": str(observation_path),
                "preview_png": "" if preview_path is None else str(preview_path),
                "diagnostics_json": str(diagnostics_path),
                "normalization_denominator_mm": prior.normalization_denominator_mm,
                "depth_limit_mm": prior.depth_limit_mm,
                "labels_read_for_evaluation": bool(args.evaluate_labels),
                **metrics,
            }
        )
        print(f"[{index}/{len(sample_ids)}] {name} -> {sample_dir}")
    manifest_path = output_root / "manifest.csv"
    fields = list(dict.fromkeys(key for row in manifest_rows for key in row))
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    validation_names = {sample_name(value) for value in config.validation_sample_ids}
    formal_reference_names = {
        sample_name(value)
        for value in config.formal_sample_ids
        if value not in config.validation_sample_ids
    }
    summary = {
        "algorithm": "distribution_matched_comsol_complex_rytov_ray_tomography",
        "all_requested_descriptive_only": metric_summary(metric_rows),
        "formal_reference_descriptive_only": metric_summary(
            [row for row in metric_rows if row["sample_id"] in formal_reference_names]
        ),
        "configured_validation_descriptive_only": metric_summary(
            [row for row in metric_rows if row["sample_id"] in validation_names]
        ),
        "real_formal_sample_labels_used_for_inversion": False,
        "labels_read_for_post_inversion_evaluation": bool(args.evaluate_labels),
        "prior_model_sha256": prior.model_sha256,
        "held_out_synthetic_corpus_validation": prior.held_out_validation,
        "validation_sample_ids": sorted(validation_names),
        "formal_reference_sample_ids": sorted(formal_reference_names),
        "interpretation_warning": (
            "The prior was fitted only on independent COMSOL corpus labels. Formal labels are read after "
            "inversion solely for descriptive metrics and previews."
        ),
        "manifest_csv": str(manifest_path),
    }
    write_json(output_root / "summary.json", summary)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

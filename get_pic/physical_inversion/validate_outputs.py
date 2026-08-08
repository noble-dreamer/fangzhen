"""Validate v2 distribution-matched physical-inversion artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simple.get_pic.physical_inversion.inversion import InversionConfig, PhysicalInverter  # noqa: E402
from simple.get_pic.physical_inversion.simulation_prior.channel_prior import (  # noqa: E402
    SimulationChannelPrior,
)


DEFAULT_CONFIG = HERE / "configs" / "dataset_a_rytov_tv.json"
DEFAULT_PRIOR = HERE / "simulation_prior" / "output_matched_corpus" / "channel_prior.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate v2 matched-corpus physical inversion outputs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def resolve_recorded_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [manifest_path.parent / path, PROJECT_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def main() -> None:
    args = parse_args()
    config = InversionConfig.from_json(args.config.resolve())
    prior = SimulationChannelPrior.load(args.prior.resolve())
    inverter = PhysicalInverter(config, helical_order_weights=prior.order_weights)
    prior.assert_compatible(inverter)
    output_root = args.output_root.resolve() if args.output_root else config.output_path
    contract_path = output_root / "run_contract.json"
    manifest_path = output_root / "manifest.csv"
    if not contract_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Missing v2 output contract or manifest under {output_root}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("algorithm") != "distribution_matched_comsol_complex_rytov_ray_tomography":
        raise RuntimeError("Output root is not a v2 matched-corpus inversion")
    if contract.get("real_formal_sample_labels_used_for_inversion") is not False:
        raise RuntimeError("v2 run contract incorrectly claims formal labels were used for inversion")
    if contract.get("prior_model_sha256") != prior.model_sha256:
        raise RuntimeError("Output run contract references a different v2 prior")
    if contract.get("config") != config.to_dict():
        raise RuntimeError("Output run contract does not match the current v2 inversion config")
    expected_contract_ids = [f"dataset_a_frequency_sample_{sample_id:04d}" for sample_id in config.formal_sample_ids]
    if contract.get("sample_ids") != expected_contract_ids:
        raise RuntimeError("Output run contract contains a stale formal-sample selection")
    with manifest_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    expected_names = {f"dataset_a_frequency_sample_{sample_id:04d}" for sample_id in config.formal_sample_ids}
    actual_names = {row.get("sample_id", "") for row in rows}
    if actual_names != expected_names:
        raise RuntimeError(
            f"v2 manifest sample IDs differ from config; missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )
    validation_names = {f"dataset_a_frequency_sample_{sample_id:04d}" for sample_id in config.validation_sample_ids}
    for row in rows:
        expected_split = "validation" if row["sample_id"] in validation_names else "formal_reference"
        if row.get("split") != expected_split:
            raise RuntimeError(
                f"v2 manifest split is stale for {row['sample_id']}: {row.get('split')!r}"
            )
    for row in rows:
        sample_id = row["sample_id"]
        mm_path = resolve_recorded_path(row["prediction_mm_npy"], manifest_path)
        norm_path = resolve_recorded_path(row["prediction_norm_npy"], manifest_path)
        raw_path = resolve_recorded_path(row["raw_grid_npy"], manifest_path)
        observation_path = resolve_recorded_path(row["path_observation_npz"], manifest_path)
        diagnostics_path = resolve_recorded_path(row["diagnostics_json"], manifest_path)
        preview_path = resolve_recorded_path(row["preview_png"], manifest_path) if row.get("preview_png") else None
        required = [mm_path, norm_path, raw_path, observation_path, diagnostics_path]
        if preview_path is not None:
            required.append(preview_path)
        for path in required:
            if not path.exists():
                raise FileNotFoundError(path)
        prediction_mm = np.asarray(np.load(mm_path), dtype=np.float32)
        prediction_norm = np.asarray(np.load(norm_path), dtype=np.float32)
        raw = np.asarray(np.load(raw_path), dtype=np.float32)
        if prediction_mm.shape != (config.image_size, config.image_size):
            raise RuntimeError(f"Unexpected v2 prediction shape for {sample_id}: {prediction_mm.shape}")
        if raw.shape != (config.inversion_grid_size, config.inversion_grid_size):
            raise RuntimeError(f"Unexpected v2 raw-grid shape for {sample_id}: {raw.shape}")
        if not all(np.all(np.isfinite(item)) for item in (prediction_mm, prediction_norm, raw)):
            raise RuntimeError(f"Non-finite v2 output for {sample_id}")
        if float(np.min(prediction_mm)) < 0.0 or float(np.max(prediction_mm)) > prior.depth_limit_mm + 1e-6:
            raise RuntimeError(f"Physical bounds violated for {sample_id}")
        if not np.allclose(
            prediction_norm,
            prediction_mm / prior.normalization_denominator_mm,
            atol=1e-6,
            rtol=1e-6,
        ):
            raise RuntimeError(f"Normalized/mm relationship is wrong for {sample_id}")
        with np.load(observation_path, allow_pickle=False) as data:
            if data["path_depth_mm"].shape != (256,) or not np.all(np.isfinite(data["path_depth_mm"])):
                raise RuntimeError(f"Invalid v2 path observation for {sample_id}")
            if not np.all(np.diff(data["frequency_hz"]) > 0.0):
                raise RuntimeError(f"Frequency axis is not ascending for {sample_id}")
            if not np.array_equal(data["tx_indices"], inverter.operator.tx_indices):
                raise RuntimeError(f"TX path order mismatch for {sample_id}")
            if not np.array_equal(data["rx_indices"], inverter.operator.rx_indices):
                raise RuntimeError(f"RX path order mismatch for {sample_id}")
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        if diagnostics.get("sample_id") != sample_id:
            raise RuntimeError(f"Diagnostics sample mismatch for {sample_id}")
        if diagnostics.get("real_formal_sample_labels_used_for_inversion") is not False:
            raise RuntimeError(f"Diagnostics show formal label leakage for {sample_id}")
        if diagnostics.get("prior_model_sha256") != prior.model_sha256:
            raise RuntimeError(f"Diagnostics use a different prior for {sample_id}")
    summary_path = output_root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("algorithm") != contract["algorithm"]:
        raise RuntimeError("Summary algorithm does not match the v2 run contract")
    if summary.get("prior_model_sha256") != prior.model_sha256:
        raise RuntimeError("Summary references a different v2 prior")
    if summary.get("real_formal_sample_labels_used_for_inversion") is not False:
        raise RuntimeError("Summary incorrectly claims formal labels were used for inversion")
    required_summary_sections = {
        "all_requested_descriptive_only",
        "formal_reference_descriptive_only",
        "configured_validation_descriptive_only",
    }
    missing_sections = sorted(required_summary_sections.difference(summary))
    if missing_sections:
        raise RuntimeError(f"Summary is missing v2 sections: {missing_sections}")
    print(
        f"validated {len(rows)} v2 samples; shape={config.image_size}x{config.image_size}; "
        f"range=[0,{prior.depth_limit_mm:g}] mm; denominator={prior.normalization_denominator_mm:g} mm"
    )


if __name__ == "__main__":
    main()

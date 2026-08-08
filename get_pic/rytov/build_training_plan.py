"""Build the deterministic COMSOL weak-basis training plan."""

from __future__ import annotations

import argparse
import json
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simple.get_pic.rytov.common import (  # noqa: E402
    assert_axes,
    healthy_rotational_relative_l2,
    load_response,
    sha256_file,
    sha256_json,
    write_json,
)
from simple.get_pic.rytov.config import RytovConfig  # noqa: E402


DEFAULT_CONFIG = HERE / "configs" / "dataset_a_fullwave_rytov.json"
ARTIFACT_KIND = "fullwave_rytov_training_plan"
FORMAL_TX_INDICES = tuple(range(1, 17))
FORMAL_RX_INDICES = tuple(range(17, 33))
MODEL_CONTRACT_KEYS = (
    "model_family",
    "pipe",
    "transducer",
    "material",
    "absorbing_layer",
    "defect_model",
    "mesh",
    "receiver_indices",
    "receiver_model",
    "actuation_model",
    "position_perturbations",
    "amplitude_scale",
    "analysis_type",
    "frequency_domain",
)
PHYSICAL_SOURCE_PATHS = (
    "simple/f_domain/frequency_domain_common.py",
    "simple/simple_shell_common.py",
    "simple/streaming_export_common.py",
    "simple/defect_label_common.py",
    "simple/get_pic/rytov/basis.py",
)
WINDOW_POWER = 3
SAFE_SAMPLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the deterministic full-wave Rytov weak-basis COMSOL plan."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def guarded_rytov_path(path: Path, *, description: str) -> Path:
    """Resolve a project output and reject every destination outside this module."""

    resolved = path.resolve()
    root = HERE.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{description} must remain inside {root}, got {resolved}") from error
    if not relative.parts:
        raise ValueError(f"{description} may not be the Rytov source root itself: {resolved}")
    return resolved


def validate_sample_id(value: str, *, description: str) -> str:
    sample_id = str(value)
    if not SAFE_SAMPLE_ID.fullmatch(sample_id) or sample_id in {".", ".."}:
        raise ValueError(f"{description} is not a safe filename token: {sample_id!r}")
    return sample_id


def config_sha256(config: RytovConfig) -> str:
    return sha256_json(config.to_dict())


def plan_sha256(plan: dict[str, Any]) -> str:
    payload = dict(plan)
    payload.pop("plan_sha256", None)
    return sha256_json(payload)


def validate_plan_sha256(plan: dict[str, Any]) -> None:
    stored = str(plan.get("plan_sha256", ""))
    actual = plan_sha256(plan)
    if not stored or stored != actual:
        raise RuntimeError(f"Training-plan SHA mismatch: stored={stored!r}, actual={actual}")


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def model_contract(metadata: dict[str, Any]) -> dict[str, Any]:
    model = metadata.get("model")
    if not isinstance(model, dict):
        raise RuntimeError("Healthy metadata has no model object")
    missing = [key for key in MODEL_CONTRACT_KEYS if key not in model]
    if missing:
        raise RuntimeError(f"Healthy model metadata is missing contract fields: {missing}")
    return {key: model[key] for key in MODEL_CONTRACT_KEYS}


def physical_source_sha256() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in PHYSICAL_SOURCE_PATHS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = sha256_file(path)
    return result


def depth_token(depth_mm: float) -> str:
    micrometers = Decimal(str(depth_mm)) * Decimal("1000")
    normalized = micrometers.normalize()
    if normalized == normalized.to_integral_value():
        return f"d{int(normalized):04d}um"
    text = format(normalized, "f").rstrip("0").rstrip(".")
    whole, fraction = text.split(".")
    return f"d{int(whole):04d}p{fraction}um"


def sample_id_for(z_index: int, theta_index: int, depth_mm: float) -> str:
    return f"rytov_basis_z{z_index:02d}_t{theta_index:02d}_{depth_token(depth_mm)}"


def build_probe_entries(config: RytovConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    theta_centers_deg = np.linspace(
        0.0, 360.0, config.theta_basis_count, endpoint=False, dtype=np.float64
    )
    z_centers_mm = np.linspace(
        config.z_min_mm, config.z_max_mm, config.z_basis_count, dtype=np.float64
    )
    coefficient_count = int(theta_centers_deg.size * z_centers_mm.size)
    basis = {
        "theta_centers_deg": [float(value) for value in theta_centers_deg],
        "z_centers_mm": [float(value) for value in z_centers_mm],
        "radius_theta_mm": float(config.basis_radius_theta_mm),
        "radius_z_mm": float(config.basis_radius_z_mm),
        "window_power": int(WINDOW_POWER),
        "coefficient_count": coefficient_count,
        "coefficient_order": "z_major_theta_minor",
        "coefficient_index_formula": "z_index * theta_basis_count + theta_index",
    }
    samples: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    radius_theta = float(config.basis_radius_theta_mm)
    radius_z = float(config.basis_radius_z_mm)
    for z_index, z_mm in enumerate(z_centers_mm):
        for theta_index, theta_deg in enumerate(theta_centers_deg):
            basis_index = z_index * theta_centers_deg.size + theta_index
            for depth_index, depth_mm in enumerate(config.perturbation_depths_mm):
                sample_id = sample_id_for(z_index, theta_index, float(depth_mm))
                if sample_id in identifiers:
                    raise RuntimeError(
                        "Perturbation depths produce duplicate sample IDs after unit encoding: "
                        f"{sample_id}"
                    )
                identifiers.add(sample_id)
                defect = {
                    "theta_deg": float(theta_deg),
                    "z_mm": float(z_mm),
                    "radius_mm": radius_theta,
                    "radius_theta_mm": radius_theta,
                    "radius_z_mm": radius_z,
                    "diameter_mm": 2.0 * radius_theta,
                    "diameter_theta_mm": 2.0 * radius_theta,
                    "diameter_z_mm": 2.0 * radius_z,
                    "depth_mm": float(depth_mm),
                    "lobe_count": 0,
                }
                samples.append(
                    {
                        "sample_id": sample_id,
                        "type": "weak_basis_probe",
                        "split": "training",
                        "basis_index": int(basis_index),
                        "z_index": int(z_index),
                        "theta_index": int(theta_index),
                        "theta_deg": float(theta_deg),
                        "z_mm": float(z_mm),
                        "depth_index": int(depth_index),
                        "depth_mm": float(depth_mm),
                        "basis_peak_mm": float(depth_mm),
                        "defect": defect,
                        "response_npz": (
                            f"training_corpus/frequency_response/{sample_id}_H_complex.npz"
                        ),
                    }
                )
    expected = coefficient_count * len(config.perturbation_depths_mm)
    if len(samples) != expected:
        raise RuntimeError(f"Built {len(samples)} probes, expected {expected}")
    return basis, samples


def build_plan(config: RytovConfig) -> dict[str, Any]:
    guarded_rytov_path(config.output_path, description="config output_root")
    guarded_rytov_path(config.training_path, description="training corpus root")
    validate_sample_id(
        config.training_healthy_sample_id,
        description="training_healthy_sample_id",
    )
    formal = load_response(config.healthy_path)
    assert_axes(
        formal,
        tx_indices=np.asarray(FORMAL_TX_INDICES, dtype=np.int32),
        rx_indices=np.asarray(FORMAL_RX_INDICES, dtype=np.int32),
        frequencies_hz=np.asarray(config.frequencies_hz, dtype=np.float64),
    )
    if formal.sample_id != config.healthy_sample_id:
        raise RuntimeError(
            f"Formal healthy sample ID mismatch: {formal.sample_id!r} != {config.healthy_sample_id!r}"
        )
    if not config.healthy_metadata_path.is_file():
        raise FileNotFoundError(config.healthy_metadata_path)
    metadata = load_json_object(config.healthy_metadata_path)
    if str(metadata.get("sample_id", "")) != config.healthy_sample_id:
        raise RuntimeError("Formal healthy metadata sample_id does not match the config")
    contract = model_contract(metadata)
    symmetry_relative_l2 = healthy_rotational_relative_l2(formal.h)
    if (
        config.jacobian_assembly_mode == "rotational_tx1"
        and symmetry_relative_l2 > config.maximum_healthy_symmetry_relative_l2
    ):
        raise RuntimeError(
            f"Formal healthy rotational relative L2 {symmetry_relative_l2:.6g} exceeds "
            f"{config.maximum_healthy_symmetry_relative_l2:.6g}; use the all_tx config"
        )
    basis, samples = build_probe_entries(config)
    plan: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "purpose": (
            "COMSOL numerical weak-defect Jacobian for full-wave linearized complex Rytov inversion"
        ),
        "config": config.to_dict(),
        "config_sha256": config_sha256(config),
        "jacobian_assembly_mode": config.jacobian_assembly_mode,
        "healthy_npz": str(config.healthy_path.resolve()),
        "healthy_npz_sha256": sha256_file(config.healthy_path),
        "healthy_metadata": str(config.healthy_metadata_path.resolve()),
        "healthy_metadata_sha256": sha256_file(config.healthy_metadata_path),
        "healthy_model_contract": contract,
        "healthy_model_contract_sha256": sha256_json(contract),
        "healthy_rotational_relative_l2": float(symmetry_relative_l2),
        "formal_tx_indices": list(FORMAL_TX_INDICES),
        "tx_indices": [int(value) for value in config.training_tx_indices],
        "rx_indices": list(FORMAL_RX_INDICES),
        "frequency_hz_ordered": [float(value) for value in config.frequencies_hz],
        "physical_source_sha256": physical_source_sha256(),
        "training_corpus_root": "training_corpus",
        "training_healthy_sample_id": config.training_healthy_sample_id,
        "training_healthy_npz": (
            "training_corpus/frequency_response/"
            f"{config.training_healthy_sample_id}_H_complex.npz"
        ),
        "basis": basis,
        "sample_count": len(samples),
        "samples": samples,
    }
    plan["plan_sha256"] = plan_sha256(plan)
    return plan


def validate_plan_for_config(plan: dict[str, Any], config: RytovConfig) -> None:
    if plan.get("artifact_kind") != ARTIFACT_KIND:
        raise RuntimeError(f"Unexpected training-plan artifact: {plan.get('artifact_kind')!r}")
    validate_plan_sha256(plan)
    if plan.get("config_sha256") != config_sha256(config):
        raise RuntimeError("Training plan does not match the selected Rytov config")
    expected = build_plan(config)
    if plan != expected:
        differing = sorted(
            key for key in set(plan).union(expected) if plan.get(key) != expected.get(key)
        )
        raise RuntimeError(
            "Training plan no longer matches its formal healthy/source contract; "
            f"differing fields: {differing}"
        )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = RytovConfig.from_json(config_path)
    output = guarded_rytov_path(
        args.output.resolve() if args.output else config.plan_path,
        description="training plan output",
    )
    plan = build_plan(config)
    print(f"Config: {config_path}")
    print(f"Formal healthy: {config.healthy_path}")
    print(f"Plan: {output}")
    print(
        f"mode={config.jacobian_assembly_mode}, probes={len(plan['samples'])}, "
        f"cases/probe={len(config.training_tx_indices) * len(config.frequencies_hz)}, "
        f"total probe cases={len(plan['samples']) * len(config.training_tx_indices) * len(config.frequencies_hz)}"
    )
    print(f"plan_sha256={plan['plan_sha256']}")
    if args.dry_run:
        print("Dry run complete; no files were written.")
        return
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --force")
    write_json(output, plan)
    print(f"Wrote deterministic training plan: {output}")


if __name__ == "__main__":
    main()

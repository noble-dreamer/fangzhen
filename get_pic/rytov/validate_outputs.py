"""Validate saved formal full-wave Rytov inversion outputs without reading labels."""

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

from simple.get_pic.rytov.common import sha256_file, sha256_json  # noqa: E402
from simple.get_pic.rytov.config import RytovConfig, project_path  # noqa: E402
from simple.get_pic.rytov.formal_selection import (  # noqa: E402
    FormalSelection,
    batch_output_root,
    sample_name,
)
from simple.get_pic.rytov.rytov_operator import FullWaveRytovOperator  # noqa: E402


DEFAULT_CONFIG = HERE / "configs" / "dataset_a_fullwave_rytov.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate saved full-wave Rytov outputs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--operator", type=Path, default=None)
    parser.add_argument(
        "--start-id",
        type=int,
        default=None,
        help=(
            "Inclusive formal-response ID start used to locate an automatic batch root; "
            "must be paired with --end-id."
        ),
    )
    parser.add_argument(
        "--end-id",
        type=int,
        default=None,
        help=(
            "Inclusive formal-response ID end used to locate an automatic batch root; "
            "must be paired with --start-id."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Accept a manifest containing a strict subset of the recorded formal selection.",
    )
    return parser.parse_args()


def _requested_range(start_id: int | None, end_id: int | None) -> tuple[int, int] | None:
    if (start_id is None) != (end_id is None):
        raise ValueError("--start-id and --end-id must be supplied together")
    if start_id is None or end_id is None:
        return None
    if start_id <= 0 or end_id < start_id:
        raise ValueError("--start-id/--end-id must be a positive inclusive range")
    return start_id, end_id


def _safe_output_root(
    value: Path | None,
    config: RytovConfig,
    requested_range: tuple[int, int] | None,
) -> Path:
    if value is None:
        path = (
            batch_output_root(config.formal_output_path, *requested_range)
            if requested_range is not None
            else config.formal_output_path
        )
    else:
        path = project_path(value)
    resolved = path.resolve()
    try:
        resolved.relative_to(HERE.resolve())
    except ValueError as error:
        raise ValueError(f"Output root must stay under {HERE}: {resolved}") from error
    if requested_range is not None and resolved == config.formal_output_path.resolve():
        raise ValueError(
            "A formal source-ID range must not use the shared output_dataset root. "
            "Omit --output-root or pass the dedicated batch root."
        )
    return resolved


def _resolve_artifact(value: str, output_root: Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(output_root.resolve())
    except ValueError as error:
        raise RuntimeError(f"Manifest artifact escapes output root: {resolved}") from error
    return resolved


def _load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise RuntimeError(f"Empty manifest: {path}")
    return rows


def _selection_from_contract(contract: dict[str, Any], config: RytovConfig) -> FormalSelection:
    """Read v2 batch provenance, with a narrow compatibility path for v1 runs."""

    raw = contract.get("formal_selection")
    if raw is None:
        if contract.get("schema_version") != 1:
            raise RuntimeError("Run contract is missing formal selection metadata")
        return FormalSelection(sample_ids=config.formal_sample_ids, kind="configured")
    if not isinstance(raw, dict):
        raise RuntimeError("Run contract formal_selection must be an object")
    raw_ids = raw.get("sample_ids")
    if not isinstance(raw_ids, list):
        raise RuntimeError("Run contract formal_selection.sample_ids must be a list")
    try:
        sample_ids = tuple(int(value) for value in raw_ids)
        kind = str(raw["kind"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Invalid formal selection in run contract") from error
    try:
        if kind == "source_range":
            if raw.get("range_is_inclusive") is not True:
                raise RuntimeError("Formal source-ID range contract is not marked inclusive")
            if raw.get("source") != "dataset_frequency_response_filename_ids":
                raise RuntimeError("Unexpected formal source-ID range provenance")
            return FormalSelection(
                sample_ids=sample_ids,
                kind=kind,
                start_id=int(raw["start_id"]),
                end_id=int(raw["end_id"]),
            )
        selection = FormalSelection(sample_ids=sample_ids, kind=kind)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Invalid formal selection in run contract") from error
    if selection.kind == "configured" and selection.sample_ids != config.formal_sample_ids:
        raise RuntimeError("Configured formal selection differs from the signed config")
    return selection


def _expected_sample_names(selection: FormalSelection) -> set[str]:
    return {sample_name(sample_id) for sample_id in selection.sample_ids}


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = RytovConfig.from_json(config_path)
    operator_path = args.operator.resolve() if args.operator else config.operator_metadata_path.resolve()
    operator = FullWaveRytovOperator.load(operator_path, config=config)
    operator.assert_compatible(config)
    requested_range = _requested_range(args.start_id, args.end_id)
    output_root = _safe_output_root(args.output_root, config, requested_range)
    contract_path = output_root / "run_contract.json"
    summary_path = output_root / "summary.json"
    manifest_path = output_root / "manifest.csv"
    for path in (contract_path, summary_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("artifact_kind") != "fullwave_linearized_rytov_formal_inversion":
        raise RuntimeError(f"Unexpected run contract: {contract.get('artifact_kind')}")
    if contract.get("schema_version") not in {1, 2}:
        raise RuntimeError(f"Unsupported run contract schema: {contract.get('schema_version')}")
    if contract.get("config_sha256") != sha256_json(config.to_dict()):
        raise RuntimeError("Run contract was generated with another config")
    if contract.get("operator_model_sha256") != operator.model_sha256:
        raise RuntimeError("Run contract was generated with another operator")
    if contract.get("operator_metadata_sha256") != sha256_file(operator.metadata_path):
        raise RuntimeError("Operator metadata changed after the formal inversion")
    if contract.get("formal_labels_used_for_inversion") is not False:
        raise RuntimeError("Run contract does not guarantee label-free inversion")
    if contract.get("image_shape") != [config.image_size, config.image_size]:
        raise RuntimeError("Run contract image shape differs from config")
    selection = _selection_from_contract(contract, config)
    if requested_range is not None:
        start_id, end_id = requested_range
        if (
            not selection.is_source_range
            or selection.start_id != start_id
            or selection.end_id != end_id
        ):
            raise RuntimeError(
                "Requested --start-id/--end-id does not match the batch selection recorded "
                "in run_contract.json"
            )

    rows = _load_manifest(manifest_path)
    names = [row.get("sample_id", "") for row in rows]
    if len(names) != len(set(names)):
        raise RuntimeError("Manifest contains duplicate sample IDs")
    expected_names = _expected_sample_names(selection)
    actual_names = set(names)
    if args.allow_subset:
        if not actual_names or not actual_names.issubset(expected_names):
            raise RuntimeError("Manifest is not a non-empty subset of the recorded formal selection")
    elif actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise RuntimeError(
            f"Manifest sample set mismatch; missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    expected_image_shape = (config.image_size, config.image_size)
    expected_coefficients = config.theta_basis_count * config.z_basis_count
    expected_data_shape = operator.jacobian.shape[:3]
    relative_residuals: list[float] = []
    evaluated_count = 0
    for row in rows:
        name = row["sample_id"]
        if row.get("status") != "complete":
            raise RuntimeError(f"Incomplete manifest row: {name}")
        if row.get("operator_model_sha256") != operator.model_sha256:
            raise RuntimeError(f"Operator mismatch in manifest row: {name}")
        if row.get("formal_labels_used_for_inversion", "").lower() != "false":
            raise RuntimeError(f"Manifest does not guarantee label-free inversion: {name}")

        prediction_path = _resolve_artifact(row["prediction_mm"], output_root)
        norm_path = _resolve_artifact(row["prediction_norm"], output_root)
        coefficients_path = _resolve_artifact(row["coefficients_mm"], output_root)
        data_fit_path = _resolve_artifact(row["data_fit_npz"], output_root)
        diagnostics_path = _resolve_artifact(row["diagnostics_json"], output_root)
        required = (prediction_path, norm_path, coefficients_path, data_fit_path, diagnostics_path)
        missing_files = [str(path) for path in required if not path.exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing artifacts for {name}: {missing_files}")
        preview_value = row.get("preview_png", "")
        if preview_value and not _resolve_artifact(preview_value, output_root).exists():
            raise FileNotFoundError(f"Missing preview for {name}")

        prediction = np.asarray(np.load(prediction_path, allow_pickle=False), dtype=np.float64)
        prediction_norm = np.asarray(np.load(norm_path, allow_pickle=False), dtype=np.float64)
        coefficients = np.asarray(np.load(coefficients_path, allow_pickle=False), dtype=np.float64)
        if prediction.shape != expected_image_shape or prediction_norm.shape != expected_image_shape:
            raise RuntimeError(f"Invalid image shape for {name}")
        if coefficients.shape != (expected_coefficients,):
            raise RuntimeError(f"Invalid coefficient shape for {name}: {coefficients.shape}")
        if not all(
            np.all(np.isfinite(value)) for value in (prediction, prediction_norm, coefficients)
        ):
            raise RuntimeError(f"Non-finite prediction artifact for {name}")
        tolerance = 2.0e-5
        if np.min(prediction) < -tolerance or np.max(prediction) > config.depth_limit_mm + tolerance:
            raise RuntimeError(f"Prediction outside physical range for {name}")
        if np.min(coefficients) < -tolerance or np.max(coefficients) > config.depth_limit_mm + tolerance:
            raise RuntimeError(f"Coefficients outside configured bounds for {name}")
        if not np.allclose(
            prediction_norm,
            prediction / config.normalization_denominator_mm,
            rtol=1.0e-6,
            atol=1.0e-7,
        ):
            raise RuntimeError(f"Millimeter normalization mismatch for {name}")

        with np.load(data_fit_path, allow_pickle=False) as data:
            required_keys = {
                "observed_real",
                "observed_imag",
                "predicted_real",
                "predicted_imag",
                "residual_real",
                "residual_imag",
                "data_weights",
                "tx_indices",
                "rx_indices",
                "frequencies_hz",
            }
            missing_keys = sorted(required_keys.difference(data.files))
            if missing_keys:
                raise RuntimeError(f"Data-fit artifact for {name} is missing {missing_keys}")
            observed = np.asarray(data["observed_real"], dtype=np.float64) + 1j * np.asarray(
                data["observed_imag"], dtype=np.float64
            )
            predicted = np.asarray(data["predicted_real"], dtype=np.float64) + 1j * np.asarray(
                data["predicted_imag"], dtype=np.float64
            )
            residual = np.asarray(data["residual_real"], dtype=np.float64) + 1j * np.asarray(
                data["residual_imag"], dtype=np.float64
            )
            weights = np.asarray(data["data_weights"], dtype=np.float64)
            tx_indices = np.asarray(data["tx_indices"], dtype=np.int32)
            rx_indices = np.asarray(data["rx_indices"], dtype=np.int32)
            frequencies_hz = np.asarray(data["frequencies_hz"], dtype=np.float64)
        if any(value.shape != expected_data_shape for value in (observed, predicted, residual, weights)):
            raise RuntimeError(f"Data-fit measurement shape mismatch for {name}")
        if not all(
            np.all(np.isfinite(value))
            for value in (observed.real, observed.imag, predicted.real, predicted.imag, residual.real, residual.imag, weights)
        ):
            raise RuntimeError(f"Non-finite data-fit artifact for {name}")
        if np.any(weights < 0.0) or not np.any(weights > 0.0):
            raise RuntimeError(f"Invalid data weights for {name}")
        if not np.allclose(residual, observed - predicted, rtol=2.0e-5, atol=1.0e-7):
            raise RuntimeError(f"Stored complex residual is inconsistent for {name}")
        if not np.array_equal(tx_indices, operator.tx_indices):
            raise RuntimeError(f"TX axis mismatch in data-fit artifact for {name}")
        if not np.array_equal(rx_indices, operator.rx_indices):
            raise RuntimeError(f"RX axis mismatch in data-fit artifact for {name}")
        if not np.allclose(frequencies_hz, operator.frequencies_hz, rtol=0.0, atol=1.0e-6):
            raise RuntimeError(f"Frequency axis mismatch in data-fit artifact for {name}")

        weighted_residual = np.sqrt(weights) * residual
        weighted_observed = np.sqrt(weights) * observed
        relative_residual = float(
            np.linalg.norm(weighted_residual)
            / max(float(np.linalg.norm(weighted_observed)), 1.0e-12)
        )
        recorded_residual = float(row["weighted_complex_relative_residual"])
        if not np.isclose(relative_residual, recorded_residual, rtol=2.0e-5, atol=1.0e-7):
            raise RuntimeError(f"Recorded data residual mismatch for {name}")
        relative_residuals.append(relative_residual)

        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        if diagnostics.get("sample_id") != name:
            raise RuntimeError(f"Diagnostics sample ID mismatch for {name}")
        if diagnostics.get("operator_model_sha256") != operator.model_sha256:
            raise RuntimeError(f"Diagnostics operator mismatch for {name}")
        if diagnostics.get("formal_labels_used_for_inversion") is not False:
            raise RuntimeError(f"Diagnostics do not guarantee label-free inversion for {name}")
        evaluated = diagnostics.get("formal_labels_used_for_evaluation") is True
        if evaluated:
            evaluated_count += 1
            if not diagnostics.get("metrics"):
                raise RuntimeError(f"Evaluated output has no metrics for {name}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("artifact_kind") != "fullwave_linearized_rytov_output_summary":
        raise RuntimeError("Unexpected summary artifact")
    if int(summary.get("sample_count", -1)) != len(rows):
        raise RuntimeError("Summary sample count differs from manifest")
    if summary.get("operator_model_sha256") != operator.model_sha256:
        raise RuntimeError("Summary operator mismatch")
    if summary.get("formal_labels_used_for_inversion") is not False:
        raise RuntimeError("Summary does not guarantee label-free inversion")
    if contract.get("schema_version") == 2 and summary.get("formal_selection") != selection.to_contract():
        raise RuntimeError("Summary formal selection differs from run contract")
    print(
        f"Outputs valid: samples={len(rows)}, evaluated={evaluated_count}, "
        f"mean_data_residual={np.mean(relative_residuals):.6f}, "
        f"shape={expected_image_shape}, range=0..{config.depth_limit_mm:g} mm"
    )
    print("Validation did not read formal labels.")


if __name__ == "__main__":
    main()

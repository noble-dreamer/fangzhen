"""Run the frozen full-wave linearized Rytov operator on formal responses."""

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

from simple.get_pic.rytov.common import load_response, sha256_file, sha256_json, write_json  # noqa: E402
from simple.get_pic.rytov.basis import output_axes  # noqa: E402
from simple.get_pic.rytov.config import RytovConfig, project_path  # noqa: E402
from simple.get_pic.rytov.formal_selection import (  # noqa: E402
    FormalSelection,
    batch_output_root,
    sample_name,
    select_formal_sample_ids,
)
from simple.get_pic.rytov.inversion import invert_response  # noqa: E402
from simple.get_pic.rytov.metrics import compute_metrics, load_target_mm, save_preview  # noqa: E402
from simple.get_pic.rytov.rytov_operator import FullWaveRytovOperator  # noqa: E402


DEFAULT_CONFIG = HERE / "configs" / "dataset_a_fullwave_rytov.json"
MID_RADIUS_MM = 155.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Invert formal complex responses with a frozen COMSOL-derived Rytov operator."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--operator", type=Path, default=None, help="Operator metadata JSON.")
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=None,
        help="Legacy explicit IDs or inclusive tokens such as 1-4,5.",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=None,
        help=(
            "Inclusive numeric formal-response ID start. Must be paired with --end-id; "
            "this is not a training-basis probe ID."
        ),
    )
    parser.add_argument(
        "--end-id",
        type=int,
        default=None,
        help=(
            "Inclusive numeric formal-response ID end. Must be paired with --start-id; "
            "this is not a training-basis probe ID."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Output root. Source-ID ranges otherwise use a distinct automatic "
            "output_dataset/batches/ids_<start>_<end> directory."
        ),
    )
    parser.add_argument(
        "--evaluate-labels",
        action="store_true",
        help="Read formal labels only after prediction for metrics and three-panel previews.",
    )
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Recompute compatible sample outputs that already exist.",
    )
    return parser.parse_args()


def _safe_output_root(
    value: Path | None,
    config: RytovConfig,
    selection: FormalSelection,
) -> Path:
    if value is None:
        path = (
            batch_output_root(
                config.formal_output_path,
                selection.start_id,
                selection.end_id,
            )
            if selection.is_source_range
            else config.formal_output_path
        )
    else:
        path = project_path(value)
    resolved = path.resolve()
    try:
        resolved.relative_to(HERE.resolve())
    except ValueError as error:
        raise ValueError(f"Output root must remain below {HERE}: {resolved}") from error
    if selection.is_source_range and resolved == config.formal_output_path.resolve():
        raise ValueError(
            "A formal source-ID range must not write to the shared output_dataset root. "
            "Omit --output-root for its automatic batch directory, or choose a dedicated root."
        )
    return resolved


def _contract_is_compatible(
    existing: dict[str, Any],
    expected: dict[str, Any],
    selection: FormalSelection,
) -> bool:
    if existing == expected:
        return True
    # A pre-batch run at the unchanged default root did not record selection
    # metadata. Keep it reusable only for the complete configured sample set.
    if existing.get("schema_version") == 1 and selection.kind == "configured":
        legacy_expected = dict(expected)
        legacy_expected["schema_version"] = 1
        legacy_expected.pop("formal_selection", None)
        return existing == legacy_expected
    return False


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _record_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _finite_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = np.asarray(
        [float(row[key]) for row in rows if key in row and row[key] not in (None, "")],
        dtype=np.float64,
    )
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else None


def _validate_prediction(
    prediction_mm: np.ndarray,
    prediction_norm: np.ndarray,
    coefficients_mm: np.ndarray,
    config: RytovConfig,
) -> None:
    expected_shape = (config.image_size, config.image_size)
    if prediction_mm.shape != expected_shape or prediction_norm.shape != expected_shape:
        raise RuntimeError(
            f"Inversion returned {prediction_mm.shape}/{prediction_norm.shape}, expected {expected_shape}"
        )
    if coefficients_mm.shape != (config.theta_basis_count * config.z_basis_count,):
        raise RuntimeError(f"Unexpected coefficient shape: {coefficients_mm.shape}")
    for name, value in (
        ("prediction_mm", prediction_mm),
        ("prediction_norm", prediction_norm),
        ("coefficients_mm", coefficients_mm),
    ):
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"{name} contains non-finite values")
    tolerance = 2.0e-5
    if float(np.min(prediction_mm)) < -tolerance or float(np.max(prediction_mm)) > config.depth_limit_mm + tolerance:
        raise RuntimeError("Prediction violates the configured physical wall-loss range")
    expected_norm = prediction_mm / config.normalization_denominator_mm
    if not np.allclose(prediction_norm, expected_norm, rtol=1.0e-6, atol=1.0e-7):
        raise RuntimeError("prediction_norm is not prediction_mm / normalization_denominator_mm")


def _load_existing(
    *,
    prediction_path: Path,
    norm_path: Path,
    coefficients_path: Path,
    data_fit_path: Path,
    diagnostics_path: Path,
    operator: FullWaveRytovOperator,
    config: RytovConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]] | None:
    required = (prediction_path, norm_path, coefficients_path, data_fit_path, diagnostics_path)
    if not all(path.exists() for path in required):
        if any(path.exists() for path in required):
            raise RuntimeError(
                f"Incomplete existing output under {prediction_path.parent}; pass --overwrite-existing"
            )
        return None
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    if diagnostics.get("operator_model_sha256") != operator.model_sha256:
        raise RuntimeError(
            f"Existing output uses another operator under {prediction_path.parent}; "
            "pass --overwrite-existing"
        )
    prediction = np.asarray(np.load(prediction_path, allow_pickle=False), dtype=np.float32)
    prediction_norm = np.asarray(np.load(norm_path, allow_pickle=False), dtype=np.float32)
    coefficients = np.asarray(np.load(coefficients_path, allow_pickle=False), dtype=np.float64)
    with np.load(data_fit_path, allow_pickle=False) as data:
        fit = {key: np.asarray(data[key]) for key in data.files}
    _validate_prediction(prediction, prediction_norm, coefficients, config)
    return prediction, prediction_norm, coefficients, fit, diagnostics


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = RytovConfig.from_json(config_path)
    operator_path = args.operator.resolve() if args.operator else config.operator_metadata_path.resolve()
    operator = FullWaveRytovOperator.load(operator_path, config=config)
    operator.assert_compatible(config)
    selection = select_formal_sample_ids(
        dataset_path=config.dataset_path,
        configured_ids=config.formal_sample_ids,
        explicit_ids=args.sample_ids,
        start_id=args.start_id,
        end_id=args.end_id,
    )
    output_root = _safe_output_root(args.output_root, config, selection)
    output_root.mkdir(parents=True, exist_ok=True)
    selected_ids = selection.sample_ids
    if selection.is_source_range:
        print(
            f"Formal source-ID range: {selection.start_id}..{selection.end_id} "
            f"(inclusive, samples={len(selected_ids)})"
        )
    else:
        print(f"Formal sample selection: kind={selection.kind}, samples={len(selected_ids)}")
    print(f"Formal inversion output root: {output_root}")
    output_theta_deg, output_z_mm = output_axes(config.image_size)

    contract = {
        "schema_version": 2,
        "artifact_kind": "fullwave_linearized_rytov_formal_inversion",
        "config_path": _record_path(config_path),
        "config_sha256": sha256_json(config.to_dict()),
        "operator_metadata": _record_path(operator.metadata_path),
        "operator_metadata_sha256": sha256_file(operator.metadata_path),
        "operator_model": _record_path(operator.model_path),
        "operator_model_sha256": operator.model_sha256,
        "jacobian_assembly_mode": operator.assembly_mode,
        "formal_labels_used_for_inversion": False,
        "normalization_denominator_mm": config.normalization_denominator_mm,
        "depth_limit_mm": config.depth_limit_mm,
        "image_shape": [config.image_size, config.image_size],
        "formal_selection": selection.to_contract(),
    }
    contract_path = output_root / "run_contract.json"
    if contract_path.exists():
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if not _contract_is_compatible(existing_contract, contract, selection):
            raise RuntimeError(
                f"Existing run contract is incompatible: {contract_path}. Use another output root."
            )
    else:
        write_json(contract_path, contract)

    rows: list[dict[str, Any]] = []
    label_dir = config.dataset_path / "labels"
    for index, sample_id in enumerate(selected_ids, start=1):
        name = sample_name(sample_id)
        response_path = config.dataset_path / "frequency_response" / f"{name}_H_complex.npz"
        sample_dir = output_root / "samples" / name
        sample_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = sample_dir / f"{name}_rytov_prediction_mm.npy"
        norm_path = sample_dir / f"{name}_rytov_prediction_norm.npy"
        coefficients_path = sample_dir / f"{name}_rytov_coefficients_mm.npy"
        data_fit_path = sample_dir / f"{name}_rytov_data_fit.npz"
        diagnostics_path = sample_dir / f"{name}_rytov_diagnostics.json"
        preview_path = sample_dir / f"{name}_rytov_preview.png"

        existing = None
        if not args.overwrite_existing:
            existing = _load_existing(
                prediction_path=prediction_path,
                norm_path=norm_path,
                coefficients_path=coefficients_path,
                data_fit_path=data_fit_path,
                diagnostics_path=diagnostics_path,
                operator=operator,
                config=config,
            )
        if existing is None:
            damaged = load_response(response_path)
            result = invert_response(
                config=config,
                operator=operator,
                damaged_response=damaged,
            )
            prediction = np.asarray(result.image_mm, dtype=np.float32)
            prediction_norm = np.asarray(result.image_norm, dtype=np.float32)
            coefficients = np.asarray(result.coefficients_mm, dtype=np.float64)
            observation = np.asarray(result.observation, dtype=np.complex128)
            predicted = np.asarray(result.predicted_rytov, dtype=np.complex128)
            weights = np.asarray(result.data_weights, dtype=np.float64)
            _validate_prediction(prediction, prediction_norm, coefficients, config)
            if observation.shape != operator.jacobian.shape[:3] or predicted.shape != observation.shape:
                raise RuntimeError("Inversion data-fit arrays do not match the operator measurement axes")
            if weights.shape != observation.shape or np.any(weights < 0.0):
                raise RuntimeError("Inversion returned invalid data weights")
            residual = observation - predicted
            np.save(prediction_path, prediction)
            np.save(norm_path, prediction_norm)
            np.save(coefficients_path, coefficients.astype(np.float32))
            np.savez_compressed(
                data_fit_path,
                observed_real=np.real(observation).astype(np.float32),
                observed_imag=np.imag(observation).astype(np.float32),
                predicted_real=np.real(predicted).astype(np.float32),
                predicted_imag=np.imag(predicted).astype(np.float32),
                residual_real=np.real(residual).astype(np.float32),
                residual_imag=np.imag(residual).astype(np.float32),
                data_weights=weights.astype(np.float32),
                tx_indices=operator.tx_indices.astype(np.int32),
                rx_indices=operator.rx_indices.astype(np.int32),
                frequencies_hz=operator.frequencies_hz.astype(np.float64),
            )
            fit = {
                "observed_real": np.real(observation),
                "observed_imag": np.imag(observation),
                "predicted_real": np.real(predicted),
                "predicted_imag": np.imag(predicted),
                "data_weights": weights,
            }
            diagnostics = {
                "schema_version": 1,
                "artifact_kind": "fullwave_linearized_rytov_sample_diagnostics",
                "sample_id": name,
                "operator_model_sha256": operator.model_sha256,
                "jacobian_assembly_mode": operator.assembly_mode,
                "formal_labels_used_for_inversion": False,
                "formal_labels_used_for_evaluation": False,
                "inversion": _json_safe(result.diagnostics),
            }
        else:
            prediction, prediction_norm, coefficients, fit, diagnostics = existing
            print(f"[{index}/{len(selected_ids)}] reusing compatible prediction: {name}")

        observed = np.asarray(fit["observed_real"], dtype=np.float64) + 1j * np.asarray(
            fit["observed_imag"], dtype=np.float64
        )
        predicted = np.asarray(fit["predicted_real"], dtype=np.float64) + 1j * np.asarray(
            fit["predicted_imag"], dtype=np.float64
        )
        weights = np.asarray(fit["data_weights"], dtype=np.float64)
        weighted_residual = np.sqrt(np.maximum(weights, 0.0)) * (observed - predicted)
        weighted_observation = np.sqrt(np.maximum(weights, 0.0)) * observed
        relative_residual = float(
            np.linalg.norm(weighted_residual)
            / max(float(np.linalg.norm(weighted_observation)), 1.0e-12)
        )

        target: np.ndarray | None = None
        metrics: dict[str, float] = {}
        if args.evaluate_labels:
            target = load_target_mm(label_dir, name, prediction.shape)
            metrics = compute_metrics(
                prediction,
                target,
                theta_deg=np.asarray(output_theta_deg, dtype=np.float64),
                z_mm=np.asarray(output_z_mm, dtype=np.float64),
                mid_radius_mm=MID_RADIUS_MM,
                depth_limit_mm=config.depth_limit_mm,
                threshold_mm=config.defect_threshold_mm,
            )
        if not args.no_preview:
            save_preview(
                preview_path,
                prediction_mm=prediction,
                target_mm=target,
                theta_deg=output_theta_deg,
                z_mm=output_z_mm,
                depth_limit_mm=config.depth_limit_mm,
                title=f"{name} full-wave linearized Rytov",
            )

        diagnostics.update(
            {
                "formal_labels_used_for_inversion": False,
                "formal_labels_used_for_evaluation": bool(args.evaluate_labels),
                "weighted_complex_relative_residual": relative_residual,
                "metrics": _json_safe(metrics),
                "paths": {
                    "prediction_mm": _record_path(prediction_path),
                    "prediction_norm": _record_path(norm_path),
                    "coefficients_mm": _record_path(coefficients_path),
                    "data_fit": _record_path(data_fit_path),
                    "preview": None if args.no_preview else _record_path(preview_path),
                },
            }
        )
        write_json(diagnostics_path, _json_safe(diagnostics))

        row: dict[str, Any] = {
            "sample_id": name,
            "status": "complete",
            "operator_model_sha256": operator.model_sha256,
            "jacobian_assembly_mode": operator.assembly_mode,
            "prediction_mm": _record_path(prediction_path),
            "prediction_norm": _record_path(norm_path),
            "coefficients_mm": _record_path(coefficients_path),
            "data_fit_npz": _record_path(data_fit_path),
            "diagnostics_json": _record_path(diagnostics_path),
            "preview_png": "" if args.no_preview else _record_path(preview_path),
            "formal_labels_used_for_inversion": False,
            "formal_labels_used_for_evaluation": bool(args.evaluate_labels),
            "weighted_complex_relative_residual": relative_residual,
            "prediction_max_mm": float(np.max(prediction)),
        }
        row.update(metrics)
        rows.append(row)
        print(
            f"[{index}/{len(selected_ids)}] {name}: "
            f"data_residual={relative_residual:.4f}, max={np.max(prediction):.3f} mm"
        )

    manifest_path = output_root / "manifest.csv"
    _write_manifest(manifest_path, rows)
    metric_keys = (
        "pearson",
        "rmse_mm",
        "mae_mm",
        "ssim",
        "dice",
        "iou",
        "peak_location_error_mm",
        "false_positive_mean_mm",
    )
    summary = {
        "schema_version": 1,
        "artifact_kind": "fullwave_linearized_rytov_output_summary",
        "sample_count": len(rows),
        "sample_ids": [row["sample_id"] for row in rows],
        "formal_selection": selection.to_contract(),
        "operator_model_sha256": operator.model_sha256,
        "jacobian_assembly_mode": operator.assembly_mode,
        "formal_labels_used_for_inversion": False,
        "formal_labels_used_for_evaluation": bool(args.evaluate_labels),
        "mean_weighted_complex_relative_residual": _finite_mean(
            rows, "weighted_complex_relative_residual"
        ),
        "mean_metrics": {key: _finite_mean(rows, key) for key in metric_keys},
        "label_policy": (
            "Formal labels were loaded only after each frozen prediction for descriptive metrics."
            if args.evaluate_labels
            else "Formal labels were not read."
        ),
    }
    write_json(output_root / "summary.json", _json_safe(summary))
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()

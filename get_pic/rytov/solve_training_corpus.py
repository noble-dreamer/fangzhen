"""Stream-solve the signed full-wave Rytov weak-basis training corpus."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
SIMPLE_ROOT = PROJECT_ROOT / "simple"
F_DOMAIN_ROOT = SIMPLE_ROOT / "f_domain"
for import_root in (PROJECT_ROOT, SIMPLE_ROOT, F_DOMAIN_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple.get_pic.rytov.build_training_plan import (  # noqa: E402
    DEFAULT_CONFIG,
    guarded_rytov_path,
    load_json_object,
    model_contract,
    validate_plan_for_config,
)
from simple.get_pic.rytov.common import (  # noqa: E402
    FrequencyResponse,
    assert_axes,
    load_response,
    sha256_json,
)
from simple.get_pic.rytov.config import RytovConfig  # noqa: E402

import frequency_domain_common as fcommon  # noqa: E402
import simple_shell_common as shell  # noqa: E402


DATASET_NAME = "A_frequency_fullwave_rytov_training"
HEALTHY_DEFECT_STATE = "healthy_no_defect"
PROBE_DEFECT_STATE = "single_weak_outer_corrosion_basis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve the signed COMSOL corpus for the full-wave linearized Rytov operator."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Must equal the training_corpus path fixed by the signed config.",
    )
    parser.add_argument(
        "--probe-sample-id",
        "--sample-id",
        dest="probe_sample_id",
        nargs="+",
        default=None,
        help=(
            "Solve only the listed Rytov training-probe artifact IDs. Comma-separated "
            "tokens are accepted; these are not formal f_domain sample IDs."
        ),
    )
    parser.add_argument(
        "--start-basis-index",
        type=int,
        default=None,
        help=(
            "Inclusive zero-based basis_index from the signed training plan. All weak "
            "perturbation depths for each selected basis are included."
        ),
    )
    parser.add_argument(
        "--end-basis-index",
        type=int,
        default=None,
        help=(
            "Inclusive zero-based basis_index; defaults to the final planned basis. "
            "Cannot be combined with --probe-sample-id."
        ),
    )
    # Keep the first batching interface usable for existing server jobs, but do not
    # expose its ambiguous names in the normal help.  The values are converted to
    # the corresponding one-based basis ordinal below.
    parser.add_argument(
        "--start-id",
        dest="legacy_start_id",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--end-id",
        dest="legacy_end_id",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--healthy-baseline",
        choices=("auto", "require-existing"),
        default="auto",
        help=(
            "Shared COMSOL healthy baseline scheduling: auto (default) validates/reuses "
            "it and solves it if absent; require-existing refuses a batch before COMSOL "
            "if the validated baseline is absent."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    collision = parser.add_mutually_exclusive_group()
    collision.add_argument(
        "--resume-incomplete",
        action="store_true",
        help="Rerun whole samples whose artifacts exist but fail the signed completion contract.",
    )
    collision.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Rerun matching complete samples as well as incomplete samples.",
    )
    parser.add_argument("--skip-label-preview", action="store_true")
    parser.add_argument("--checkpoint-final-only", action="store_true")
    parser.add_argument(
        "--keep-solution-data-between-cases",
        action="store_true",
        help="Do not clear COMSOL solution fields between frequency cases; may use much more memory.",
    )
    parser.add_argument(
        "--heartbeat-s",
        type=float,
        default=fcommon.streaming.DEFAULT_HEARTBEAT_S,
    )
    parser.add_argument("--cores", type=int, default=None)
    fcommon.add_frequency_solver_arguments(parser)
    return parser.parse_args()


def selected_samples(
    plan: dict[str, Any],
    values: list[str] | None,
    training_healthy_sample_id: str,
    *,
    start_basis_index: int | None = None,
    end_basis_index: int | None = None,
    legacy_start_id: int | None = None,
    legacy_end_id: int | None = None,
) -> tuple[list[dict[str, Any]], tuple[int, int] | None, bool]:
    samples = list(plan.get("samples", []))
    samples.sort(
        key=lambda item: (
            int(item["basis_index"]),
            int(item.get("depth_index", 0)),
            str(item["sample_id"]),
        )
    )
    if not samples:
        raise ValueError("The deterministic training plan contains no weak-basis probes")
    basis_indices = sorted({int(item["basis_index"]) for item in samples})
    expected_basis_indices = list(range(basis_indices[0], basis_indices[-1] + 1))
    if basis_indices != expected_basis_indices:
        raise ValueError(
            "Training plan basis_index values must form one contiguous range; "
            f"got {basis_indices}"
        )
    has_basis_range = start_basis_index is not None or end_basis_index is not None
    has_legacy_range = legacy_start_id is not None or legacy_end_id is not None
    if has_basis_range and has_legacy_range:
        raise ValueError(
            "Use --start-basis-index/--end-basis-index instead of the legacy "
            "--start-id/--end-id aliases; do not combine both forms."
        )
    if values is not None and (has_basis_range or has_legacy_range):
        raise ValueError(
            "--probe-sample-id/--sample-id cannot be combined with a basis-index range"
        )
    if has_legacy_range:
        # The old interface counted one-based probe ordinals.  Convert those
        # ordinals to actual plan basis_index values so multi-depth plans do not
        # accidentally split a basis function in the middle of its depth trials.
        ordinal_first = 1 if legacy_start_id is None else int(legacy_start_id)
        ordinal_last = len(basis_indices) if legacy_end_id is None else int(legacy_end_id)
        if ordinal_first < 1 or ordinal_last < ordinal_first or ordinal_last > len(basis_indices):
            raise ValueError(
                "Invalid legacy weak-basis ordinal range "
                f"{ordinal_first}..{ordinal_last}; valid inclusive range is "
                f"1..{len(basis_indices)}"
            )
        first = basis_indices[ordinal_first - 1]
        last = basis_indices[ordinal_last - 1]
        return (
            [item for item in samples if first <= int(item["basis_index"]) <= last],
            (first, last),
            True,
        )
    if has_basis_range:
        first = basis_indices[0] if start_basis_index is None else int(start_basis_index)
        last = basis_indices[-1] if end_basis_index is None else int(end_basis_index)
        if first < basis_indices[0] or last < first or last > basis_indices[-1]:
            raise ValueError(
                "Invalid inclusive basis_index range "
                f"{first}..{last}; valid range is {basis_indices[0]}..{basis_indices[-1]}"
            )
        return (
            [item for item in samples if first <= int(item["basis_index"]) <= last],
            (first, last),
            False,
        )
    if values is None:
        return samples, None, False
    requested = {
        token.strip()
        for value in values
        for token in str(value).split(",")
        if token.strip()
    }
    requested.discard(training_healthy_sample_id)
    by_id = {str(item.get("sample_id", "")): item for item in samples}
    missing = sorted(requested.difference(by_id))
    if missing:
        raise ValueError(f"Unknown training sample IDs: {missing}")
    return [item for item in samples if item["sample_id"] in requested], None, False


def healthy_sample_metadata(plan: dict[str, Any]) -> dict[str, Any]:
    sample_id = str(plan["training_healthy_sample_id"])
    payload: dict[str, Any] = {
        "sample_id": sample_id,
        "artifact_kind": "fullwave_rytov_training_healthy",
        "training_plan_sha256": plan["plan_sha256"],
        "config_sha256": plan["config_sha256"],
        "formal_healthy_npz_sha256": plan["healthy_npz_sha256"],
        "jacobian_assembly_mode": plan["jacobian_assembly_mode"],
        "tx_indices": list(plan["tx_indices"]),
        "frequencies_hz": list(plan["frequency_hz_ordered"]),
        "defects": [],
        "lobes": [],
    }
    payload["sample_contract_sha256"] = sha256_json(payload)
    return payload


def probe_sample_metadata(entry: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    defect = deepcopy(entry["defect"])
    payload: dict[str, Any] = {
        "sample_id": str(entry["sample_id"]),
        "artifact_kind": "fullwave_rytov_weak_basis_sample",
        "training_plan_sha256": plan["plan_sha256"],
        "config_sha256": plan["config_sha256"],
        "formal_healthy_npz_sha256": plan["healthy_npz_sha256"],
        "jacobian_assembly_mode": plan["jacobian_assembly_mode"],
        "basis_index": int(entry["basis_index"]),
        "z_index": int(entry["z_index"]),
        "theta_index": int(entry["theta_index"]),
        "depth_index": int(entry["depth_index"]),
        "theta_deg": float(entry["theta_deg"]),
        "z_mm": float(entry["z_mm"]),
        "basis_peak_mm": float(entry["basis_peak_mm"]),
        "basis_radius_theta_mm": float(plan["basis"]["radius_theta_mm"]),
        "basis_radius_z_mm": float(plan["basis"]["radius_z_mm"]),
        "basis_window_power": int(plan["basis"]["window_power"]),
        "defects": [defect],
        "lobes": [],
    }
    payload["sample_contract_sha256"] = sha256_json(payload)
    return payload


def artifact_paths(output_root: Path, sample_id: str) -> tuple[Path, ...]:
    return (
        output_root / "frequency_response" / f"{sample_id}_H_complex.npz",
        output_root / "metadata" / f"{sample_id}.json",
        output_root / "progress" / f"{sample_id}_progress.jsonl",
        output_root / "csv" / "frequency_response" / f"{sample_id}_frequency_response.csv",
        output_root / "labels" / f"{sample_id}_defect_depth_mm.npy",
        output_root / "labels" / f"{sample_id}_defect_depth_norm.npy",
        output_root / "labels" / f"{sample_id}_defect_mask.npy",
        output_root / "labels" / f"{sample_id}_defect_label_metadata.json",
        output_root / "labels" / f"{sample_id}_defect_label.png",
    )


def artifacts_exist(output_root: Path, sample_id: str) -> bool:
    return any(path.exists() for path in artifact_paths(output_root, sample_id))


def expected_axes(
    config: RytovConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(config.training_tx_indices, dtype=np.int32),
        np.arange(17, 33, dtype=np.int32),
        np.asarray(config.frequencies_hz, dtype=np.float64),
    )


def metadata_case_axes(metadata: dict[str, Any]) -> tuple[list[int], np.ndarray]:
    export = metadata.get("frequency_export")
    if not isinstance(export, dict):
        raise RuntimeError("metadata has no frequency_export object")
    return (
        [int(value) for value in export.get("tx", [])],
        np.asarray(export.get("frequencies_hz", []), dtype=np.float64),
    )


def completion_status(
    *,
    output_root: Path,
    sample_id: str,
    defect_state: str,
    expected_sample: dict[str, Any],
    config: RytovConfig,
    plan: dict[str, Any],
) -> tuple[bool, str, FrequencyResponse | None]:
    response_path = output_root / "frequency_response" / f"{sample_id}_H_complex.npz"
    metadata_path = output_root / "metadata" / f"{sample_id}.json"
    if not response_path.is_file() or not metadata_path.is_file():
        return False, "response NPZ or metadata JSON is missing", None
    try:
        response = load_response(response_path)
        tx_indices, rx_indices, frequencies_hz = expected_axes(config)
        assert_axes(
            response,
            tx_indices=tx_indices,
            rx_indices=rx_indices,
            frequencies_hz=frequencies_hz,
        )
        if response.sample_id != sample_id:
            raise RuntimeError(
                f"response sample_id {response.sample_id!r} does not equal {sample_id!r}"
            )
        metadata = load_json_object(metadata_path)
        if metadata.get("dataset") != DATASET_NAME:
            raise RuntimeError("metadata dataset contract mismatch")
        if metadata.get("sample_id") != sample_id:
            raise RuntimeError("metadata sample_id contract mismatch")
        if metadata.get("defect_state") != defect_state:
            raise RuntimeError("metadata defect_state contract mismatch")
        if metadata.get("sample") != expected_sample:
            raise RuntimeError("metadata sample contract mismatch")
        contract = model_contract(metadata)
        if sha256_json(contract) != plan["healthy_model_contract_sha256"]:
            raise RuntimeError("COMSOL model contract differs from the formal healthy model")
        export_tx, export_frequency = metadata_case_axes(metadata)
        expected_tx = [
            int(tx)
            for tx in config.training_tx_indices
            for _frequency in config.frequencies_hz
        ]
        expected_frequency = np.asarray(
            [
                float(frequency)
                for _tx in config.training_tx_indices
                for frequency in config.frequencies_hz
            ],
            dtype=np.float64,
        )
        if export_tx != expected_tx:
            raise RuntimeError("metadata case TX order mismatch")
        if export_frequency.shape != expected_frequency.shape or not np.allclose(
            export_frequency, expected_frequency, rtol=0.0, atol=1.0e-6
        ):
            raise RuntimeError("metadata case frequency order mismatch")
        export = metadata["frequency_export"]
        if int(export.get("case_count", -1)) != expected_frequency.size:
            raise RuntimeError("metadata case_count mismatch")
        label = metadata.get("defect_label")
        if not isinstance(label, dict):
            raise RuntimeError("metadata has no defect_label contract")
        expected_defects = len(expected_sample["defects"])
        if int(label.get("defect_count", -1)) != expected_defects:
            raise RuntimeError("label defect count mismatch")
        if int(label.get("lobe_count", -1)) != 0:
            raise RuntimeError("training label unexpectedly contains lobes")
        if str(label.get("sample_id", "")) != sample_id:
            raise RuntimeError("label sample_id contract mismatch")
        if not np.isclose(
            float(label.get("normalization_denominator_mm", float("nan"))),
            config.normalization_denominator_mm,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise RuntimeError("label millimeter normalization contract mismatch")
        required_local_files = (
            output_root / "csv" / "frequency_response" / f"{sample_id}_frequency_response.csv",
            output_root / "labels" / f"{sample_id}_defect_depth_mm.npy",
            output_root / "labels" / f"{sample_id}_defect_depth_norm.npy",
            output_root / "labels" / f"{sample_id}_defect_mask.npy",
            output_root / "labels" / f"{sample_id}_defect_label_metadata.json",
        )
        missing_local = [str(path) for path in required_local_files if not path.is_file()]
        if missing_local:
            raise RuntimeError(f"signed training artifacts are missing: {missing_local}")
    except Exception as error:
        return False, f"{type(error).__name__}: {error}", None
    return True, "complete signed response and metadata", response


def formal_training_response(
    formal: FrequencyResponse,
    config: RytovConfig,
) -> np.ndarray:
    positions = []
    for tx_id in config.training_tx_indices:
        matches = np.flatnonzero(formal.tx_indices == int(tx_id))
        if matches.size != 1:
            raise RuntimeError(f"Formal healthy response does not contain TX {tx_id} exactly once")
        positions.append(int(matches[0]))
    return formal.h[np.asarray(positions, dtype=np.int64)]


def local_healthy_relative_l2(
    local: FrequencyResponse,
    formal: FrequencyResponse,
    config: RytovConfig,
) -> float:
    reference = formal_training_response(formal, config)
    denominator = float(np.linalg.norm(reference))
    if denominator <= 0.0:
        raise RuntimeError("Formal healthy response has zero norm")
    return float(np.linalg.norm(local.h - reference) / denominator)


def classify_sample(
    *,
    output_root: Path,
    sample_id: str,
    defect_state: str,
    expected_sample: dict[str, Any],
    config: RytovConfig,
    plan: dict[str, Any],
    overwrite_existing: bool,
    resume_incomplete: bool,
) -> tuple[str, str, FrequencyResponse | None]:
    complete, reason, response = completion_status(
        output_root=output_root,
        sample_id=sample_id,
        defect_state=defect_state,
        expected_sample=expected_sample,
        config=config,
        plan=plan,
    )
    if complete:
        return ("pending", "--overwrite-existing requested", response) if overwrite_existing else (
            "complete",
            reason,
            response,
        )
    if artifacts_exist(output_root, sample_id):
        if overwrite_existing or resume_incomplete:
            return "pending", f"rerunning incompatible/incomplete artifacts ({reason})", None
        return "collision", reason, None
    return "pending", "no existing artifacts", None


def defect_from_entry(entry: dict[str, Any]) -> shell.DefectConfig:
    item = entry["defect"]
    return shell.DefectConfig(
        theta_deg=float(item["theta_deg"]),
        z_mm=float(item["z_mm"]),
        radius_mm=float(item["radius_mm"]),
        depth_mm=float(item["depth_mm"]),
        radius_theta_mm=float(item["radius_theta_mm"]),
        radius_z_mm=float(item["radius_z_mm"]),
    )


def manifest_row(
    *,
    output_root: Path,
    sample_id: str,
    defect_state: str,
    plan: dict[str, Any],
    entry: dict[str, Any] | None,
    status: str,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "dataset": DATASET_NAME,
        "defect_state": defect_state,
        "artifact_kind": "training_healthy" if entry is None else "weak_basis_probe",
        "basis_index": "" if entry is None else int(entry["basis_index"]),
        "depth_index": "" if entry is None else int(entry["depth_index"]),
        "depth_mm": "" if entry is None else float(entry["depth_mm"]),
        "case_count": len(plan["tx_indices"]) * len(plan["frequency_hz_ordered"]),
        "metadata": str(output_root / "metadata" / f"{sample_id}.json"),
        "response_npz": str(
            output_root / "frequency_response" / f"{sample_id}_H_complex.npz"
        ),
        "plan_sha256": plan["plan_sha256"],
        "status": status,
    }


def current_manifest_rows(
    *,
    output_root: Path,
    config: RytovConfig,
    plan: dict[str, Any],
    solved_ids: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    healthy_id = config.training_healthy_sample_id
    healthy_expected = healthy_sample_metadata(plan)
    healthy_complete, _reason, _response = completion_status(
        output_root=output_root,
        sample_id=healthy_id,
        defect_state=HEALTHY_DEFECT_STATE,
        expected_sample=healthy_expected,
        config=config,
        plan=plan,
    )
    if healthy_complete:
        rows.append(
            manifest_row(
                output_root=output_root,
                sample_id=healthy_id,
                defect_state=HEALTHY_DEFECT_STATE,
                plan=plan,
                entry=None,
                status="solved" if healthy_id in solved_ids else "reused_complete",
            )
        )
    for entry in plan["samples"]:
        sample_id = str(entry["sample_id"])
        expected_sample = probe_sample_metadata(entry, plan)
        complete, _reason, _response = completion_status(
            output_root=output_root,
            sample_id=sample_id,
            defect_state=PROBE_DEFECT_STATE,
            expected_sample=expected_sample,
            config=config,
            plan=plan,
        )
        if not complete:
            continue
        rows.append(
            manifest_row(
                output_root=output_root,
                sample_id=sample_id,
                defect_state=PROBE_DEFECT_STATE,
                plan=plan,
                entry=entry,
                status="solved" if sample_id in solved_ids else "reused_complete",
            )
        )
    return rows


def write_current_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "dataset",
        "defect_state",
        "artifact_kind",
        "basis_index",
        "depth_index",
        "depth_mm",
        "case_count",
        "metadata",
        "response_npz",
        "plan_sha256",
        "status",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def solve_one(
    *,
    client,
    output_root: Path,
    cases: list[fcommon.FrequencyCase],
    sample_id: str,
    defect_state: str,
    defects: list[shell.DefectConfig],
    sample_metadata: dict[str, Any],
    args: argparse.Namespace,
    config: RytovConfig,
) -> fcommon.FrequencySampleExportResult:
    return fcommon.solve_export_frequency_sample(
        client=client,
        dataset=DATASET_NAME,
        sample_id=sample_id,
        defect_state=defect_state,
        output_root=output_root,
        cases=cases,
        defects=defects,
        lobes=[],
        sample_metadata=sample_metadata,
        clear_each_case=True,
        heartbeat_s=args.heartbeat_s,
        reuse_sample_model=not args.rebuild_each_case,
        write_label_preview=not args.skip_label_preview,
        keep_case_csv=False,
        checkpoint_every_cases=(
            0 if args.checkpoint_final_only else len(config.frequencies_hz)
        ),
        clear_solution_after_each_export=not args.keep_solution_data_between_cases,
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = RytovConfig.from_json(config_path)
    guarded_rytov_path(config.output_path, description="config output_root")
    planned_output_root = guarded_rytov_path(
        config.training_path, description="training corpus root"
    )
    if args.output_root is not None:
        requested_output_root = guarded_rytov_path(
            args.output_root.resolve(), description="requested training corpus root"
        )
        if requested_output_root != planned_output_root:
            raise ValueError(
                "--output-root must equal the path fixed by the signed config: "
                f"{planned_output_root}"
            )
    output_root = planned_output_root
    plan_path = guarded_rytov_path(
        args.plan.resolve() if args.plan else config.plan_path,
        description="training plan",
    )
    if not plan_path.is_file():
        raise FileNotFoundError(
            f"Training plan not found: {plan_path}. Run build_training_plan.py first."
        )
    plan = load_json_object(plan_path)
    validate_plan_for_config(plan, config)
    formal = load_response(config.healthy_path)
    training_healthy_sample_id = config.training_healthy_sample_id
    samples, basis_range, used_legacy_range = selected_samples(
        plan,
        args.probe_sample_id,
        training_healthy_sample_id,
        start_basis_index=args.start_basis_index,
        end_basis_index=args.end_basis_index,
        legacy_start_id=args.legacy_start_id,
        legacy_end_id=args.legacy_end_id,
    )
    if args.probe_sample_id is not None and not samples and training_healthy_sample_id not in {
        token.strip()
        for value in args.probe_sample_id
        for token in str(value).split(",")
    }:
        raise ValueError("The sample filter selected no weak-basis probes")
    if used_legacy_range:
        print(
            "WARNING: --start-id/--end-id are deprecated aliases for a one-based "
            "Rytov basis ordinal. Use --start-basis-index/--end-basis-index "
            "(zero-based plan basis_index) to avoid confusing them with formal sample IDs.",
            file=sys.stderr,
        )

    healthy_metadata = healthy_sample_metadata(plan)
    healthy_state, healthy_reason, local_healthy = classify_sample(
        output_root=output_root,
        sample_id=training_healthy_sample_id,
        defect_state=HEALTHY_DEFECT_STATE,
        expected_sample=healthy_metadata,
        config=config,
        plan=plan,
        overwrite_existing=args.overwrite_existing,
        resume_incomplete=args.resume_incomplete,
    )
    if healthy_state == "complete" and local_healthy is not None:
        mismatch = local_healthy_relative_l2(local_healthy, formal, config)
        if mismatch > config.maximum_healthy_closure_relative_l2:
            healthy_state = (
                "pending"
                if args.overwrite_existing or args.resume_incomplete
                else "collision"
            )
            healthy_reason = (
                f"local/formal healthy relative L2 {mismatch:.6g} exceeds "
                f"{config.maximum_healthy_closure_relative_l2:.6g}"
            )
            local_healthy = None
        else:
            healthy_reason += f"; local/formal relative L2={mismatch:.6g}"
    if args.healthy_baseline == "require-existing" and healthy_state != "complete":
        raise RuntimeError(
            "The selected serial batch requires an existing validated training healthy baseline, "
            f"but its state is {healthy_state}: {healthy_reason}. Run the first batch once with "
            "--healthy-baseline auto, then rerun this batch."
        )
    healthy_scheduled = args.healthy_baseline == "auto" and healthy_state == "pending"

    pending: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    collisions: list[str] = []
    complete_count = 0
    for entry in samples:
        sample_id = str(entry["sample_id"])
        expected_sample = probe_sample_metadata(entry, plan)
        state, reason, _response = classify_sample(
            output_root=output_root,
            sample_id=sample_id,
            defect_state=PROBE_DEFECT_STATE,
            expected_sample=expected_sample,
            config=config,
            plan=plan,
            overwrite_existing=args.overwrite_existing,
            resume_incomplete=args.resume_incomplete,
        )
        if state == "pending":
            pending.append((entry, expected_sample, reason))
        elif state == "complete":
            complete_count += 1
        else:
            collisions.append(f"{sample_id}: {reason}")

    print(f"Config: {config_path}")
    print(f"Plan: {plan_path} ({plan['plan_sha256']})")
    print(f"Output root: {output_root}")
    print(
        f"mode={config.jacobian_assembly_mode}, tx={list(config.training_tx_indices)}, "
        f"frequencies={len(config.frequencies_hz)}"
    )
    print(
        f"Training healthy: {healthy_state} ({healthy_reason}); "
        f"schedule={args.healthy_baseline}, scheduled={healthy_scheduled}"
    )
    print(
        f"Selected probes={len(samples)}, complete={complete_count}, "
        f"pending={len(pending)}, incompatible={len(collisions)}"
    )
    if basis_range is not None:
        first, last = basis_range
        print(
            "Selected inclusive plan basis_index interval="
            f"{first}..{last} ({last - first + 1} basis functions; "
            f"{len(samples)} weak-probe samples including all configured depths)"
        )
    else:
        if samples:
            basis_indices = sorted({int(item["basis_index"]) for item in samples})
            print(
                "Selected all requested weak probes; plan basis_index values="
                f"{basis_indices[0]}..{basis_indices[-1]} "
                f"({len(basis_indices)} basis functions)"
            )
        else:
            print("Selected no weak probes; this invocation only handles the training healthy baseline.")
    print(
        "Pending COMSOL cases: "
        f"{(int(healthy_scheduled) + len(pending)) * len(config.training_tx_indices) * len(config.frequencies_hz)}"
    )
    if collisions:
        raise RuntimeError(
            "Incompatible training artifacts exist. Pass --resume-incomplete to rerun them or "
            "--overwrite-existing for an intentional full rerun: "
            + "; ".join(collisions[:8])
        )
    if healthy_state == "collision":
        raise RuntimeError(
            "The local training healthy response is incompatible. Pass --resume-incomplete "
            f"or --overwrite-existing: {healthy_reason}"
        )
    if args.dry_run:
        print("Dry run complete; COMSOL was not started and no files were written.")
        return
    manifest_path = guarded_rytov_path(
        output_root / "manifest.csv", description="training manifest"
    )
    if not healthy_scheduled and not pending:
        rows = current_manifest_rows(
            output_root=output_root,
            config=config,
            plan=plan,
            solved_ids=set(),
        )
        write_current_manifest(manifest_path, rows)
        print(f"Training manifest: {manifest_path} ({len(rows)} complete records)")
        print("All selected signed training artifacts are complete; COMSOL was not started.")
        return

    fcommon.configure_dataset_a_frequency(use_parametric_sweep=False)
    fcommon.apply_solver_arguments(args)
    cases = fcommon.make_cases(config.training_tx_indices, config.frequencies_hz)
    client = shell.start_client(cores=args.cores)
    solved_ids: set[str] = set()
    try:
        if healthy_scheduled:
            print(f"[healthy] solving {training_healthy_sample_id}")
            solve_one(
                client=client,
                output_root=output_root,
                cases=cases,
                sample_id=training_healthy_sample_id,
                defect_state=HEALTHY_DEFECT_STATE,
                defects=[],
                sample_metadata=healthy_metadata,
                args=args,
                config=config,
            )
            complete, reason, local_healthy = completion_status(
                output_root=output_root,
                sample_id=training_healthy_sample_id,
                defect_state=HEALTHY_DEFECT_STATE,
                expected_sample=healthy_metadata,
                config=config,
                plan=plan,
            )
            if not complete or local_healthy is None:
                raise RuntimeError(f"New local healthy response failed validation: {reason}")
            mismatch = local_healthy_relative_l2(local_healthy, formal, config)
            if mismatch > config.maximum_healthy_closure_relative_l2:
                raise RuntimeError(
                    f"New local/formal healthy relative L2 {mismatch:.6g} exceeds "
                    f"{config.maximum_healthy_closure_relative_l2:.6g}"
                )
            solved_ids.add(training_healthy_sample_id)
            print(f"[healthy] local/formal relative L2={mismatch:.6g}")

        for index, (entry, expected_sample, reason) in enumerate(pending, start=1):
            sample_id = str(entry["sample_id"])
            print(f"[{index}/{len(pending)}] solving {sample_id}: {reason}")
            solve_one(
                client=client,
                output_root=output_root,
                cases=cases,
                sample_id=sample_id,
                defect_state=PROBE_DEFECT_STATE,
                defects=[defect_from_entry(entry)],
                sample_metadata=expected_sample,
                args=args,
                config=config,
            )
            complete, validation_reason, _response = completion_status(
                output_root=output_root,
                sample_id=sample_id,
                defect_state=PROBE_DEFECT_STATE,
                expected_sample=expected_sample,
                config=config,
                plan=plan,
            )
            if not complete:
                raise RuntimeError(
                    f"New probe response failed validation for {sample_id}: {validation_reason}"
                )
            solved_ids.add(sample_id)
    finally:
        client.clear()
    rows = current_manifest_rows(
        output_root=output_root,
        config=config,
        plan=plan,
        solved_ids=solved_ids,
    )
    write_current_manifest(manifest_path, rows)
    print(f"Training manifest: {manifest_path} ({len(rows)} complete records)")


if __name__ == "__main__":
    main()

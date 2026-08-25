"""Publish a solid-corrected schema-v3 library for Shell-proxy EDM simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dispersion_tracking as tracking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shell-library", type=Path, required=True)
    parser.add_argument("--mesh-report", type=Path, required=True)
    parser.add_argument("--solid-calibration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--readiness-scope", choices=("shell_proxy_simulation_only",), required=True)
    parser.add_argument("--frequency-range-khz", type=float, nargs=2, default=(15.0, 110.0))
    parser.add_argument("--max-mesh-complex-relative", type=float, default=0.05)
    parser.add_argument("--max-mesh-phase-rmse-rad", type=float, default=0.05)
    parser.add_argument("--min-solid-valid-fraction", type=float, default=0.5)
    parser.add_argument("--min-calibrated-point-fraction", type=float, default=0.1)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def validate_inputs(shell: dict[str, np.ndarray], mesh: dict, solid: dict[str, np.ndarray], args: argparse.Namespace, shell_sha: str) -> None:
    if int(shell.get("schema_version", 0)) != 3 or not np.asarray(shell["completed_mask"]).all():
        raise RuntimeError("Shell library must be complete schema v3")
    if str(shell.get("run_kind", "smoke")) != "formal" or bool(shell.get("scientific_ready", False)):
        raise RuntimeError("Input Shell library must be uncalibrated formal output")
    thresholds = mesh.get("thresholds", {})
    if not mesh.get("passed", False):
        raise RuntimeError("Finite-pipe mesh validation did not pass")
    if int(mesh.get("schema_version", 0)) != 1 or mesh.get("validation_scope") != "finite-pipe patch-averaged axial receiver mesh convergence":
        raise RuntimeError("Unexpected finite-pipe mesh report schema or scope")
    if float(thresholds.get("complex_relative_l2", np.inf)) > args.max_mesh_complex_relative:
        raise RuntimeError("Mesh report used a relaxed complex-response threshold")
    if float(thresholds.get("phase_rmse_rad", np.inf)) > args.max_mesh_phase_rmse_rad:
        raise RuntimeError("Mesh report used a relaxed phase threshold")
    metrics = mesh.get("metrics", [])
    if not metrics or any(not row.get("passed", False) for row in metrics):
        raise RuntimeError("Mesh report contains missing or failed convergence cases")
    if any(float(row.get("complex_relative_l2", np.inf)) > args.max_mesh_complex_relative or float(row.get("phase_rmse_rad", np.inf)) > args.max_mesh_phase_rmse_rad for row in metrics):
        raise RuntimeError("Mesh report metrics exceed the publication thresholds")
    if int(solid.get("schema_version", 0)) != 1 or bool(solid.get("scientific_ready", True)):
        raise RuntimeError("Solid calibration must be an unpromoted schema-v1 checkpoint")
    if not np.asarray(solid["completed_mask"]).all() or str(solid["shell_library_sha256"]) != shell_sha:
        raise RuntimeError("Solid calibration is incomplete or references another Shell library")
    valid_fraction = float(np.asarray(solid["calibration_valid_mask"], dtype=bool).mean())
    if valid_fraction < args.min_solid_valid_fraction:
        raise RuntimeError(f"Solid valid-mode fraction {valid_fraction:.3f} is below the gate")


def correction_fields(shell: dict[str, np.ndarray], solid: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    shell_frequency = np.asarray(shell["frequency_hz"], dtype=float)
    shell_branch = np.asarray(shell["branch_id"], dtype=int)
    ratio = np.ones_like(shell_frequency)
    confidence = np.zeros_like(shell_frequency)
    valid = np.asarray(solid["calibration_valid_mask"], dtype=bool)
    matched_branch = np.asarray(solid["matched_shell_branch_id"], dtype=int)
    observed_ratio = np.asarray(solid["frequency_ratio_solid_over_shell"], dtype=float)
    observed_confidence = np.asarray(solid["calibration_confidence"], dtype=float)
    indices = np.indices(valid.shape)
    h_axis = np.asarray(solid["thickness_mm"], dtype=float)
    k_axis = np.asarray(solid["kz_rad_m"], dtype=float)
    h_scale = max(float(np.ptp(h_axis)), 1.0)
    k_scale = max(float(np.ptp(k_axis)), 1.0)
    for branch in np.unique(matched_branch[valid]):
        observation = valid & (matched_branch == branch)
        obs_h = h_axis[indices[0][observation]]
        obs_k = k_axis[indices[1][observation]]
        obs_ratio = observed_ratio[observation]
        obs_conf = observed_confidence[observation]
        for hi, ki, bi in zip(*np.nonzero(shell_branch == branch), strict=True):
            h_value = float(shell["thickness_mm"][hi])
            k_value = float(shell["kz_rad_m"][ki])
            if not (obs_h.min() <= h_value <= obs_h.max() and obs_k.min() <= k_value <= obs_k.max()):
                continue
            distance2 = ((h_value - obs_h) / h_scale) ** 2 + ((k_value - obs_k) / k_scale) ** 2
            weights = obs_conf / (distance2 + 1e-6)
            if weights.sum() <= 0.0:
                continue
            ratio[hi, ki, bi] = float(np.sum(weights * obs_ratio) / weights.sum())
            local_conf = float(np.sum(weights * obs_conf) / weights.sum())
            confidence[hi, ki, bi] = local_conf * float(np.exp(-distance2.min()))
    return ratio, np.clip(confidence, 0.0, 1.0)


def corrected_derivatives(shell: dict[str, np.ndarray], frequency: np.ndarray, confidence: np.ndarray) -> dict[str, np.ndarray]:
    tracked = {
        "frequency_hz": frequency,
        "polarization": shell["polarization"],
        "circumferential_order": shell["circumferential_order"],
        "observability": shell["observability"],
        "frequency_imag_hz": shell["frequency_imag_hz"],
        "order_confidence": shell["order_confidence"],
        "tracking_residual": shell["tracking_residual"],
        "thickness_tracking_residual": shell["thickness_tracking_residual"],
        "tracking_confidence": shell["tracking_confidence"] * confidence,
        "branch_id": shell["branch_id"],
    }
    derived = tracking.derive_dispersion(tracked, shell["thickness_mm"], shell["kz_rad_m"])
    calibrated = confidence > 0.0
    for mask_name, value_name in (
        ("group_velocity_valid_mask", "group_velocity_m_s"),
        ("thickness_derivative_valid_mask", "df_dh_hz_per_mm"),
        ("derivative_valid_mask", "dk_dh_rad_m_per_mm"),
    ):
        derived[mask_name] &= calibrated
        derived[value_name][~derived[mask_name]] = np.nan
    return derived


def main() -> None:
    args = parse_args()
    shell_sha = sha256(args.shell_library)
    mesh_sha = sha256(args.mesh_report)
    shell = load_npz(args.shell_library)
    solid = load_npz(args.solid_calibration)
    mesh = load_json(args.mesh_report)
    validate_inputs(shell, mesh, solid, args, shell_sha)
    ratio, calibration_confidence = correction_fields(shell, solid)
    corrected_frequency = np.asarray(shell["frequency_hz"], dtype=float) * ratio
    derived = corrected_derivatives(shell, corrected_frequency, calibration_confidence)
    original_valid = np.asarray(shell["derivative_valid_mask"], dtype=bool)
    calibrated_fraction = float(((calibration_confidence > 0.0) & original_valid).sum() / max(original_valid.sum(), 1))
    if calibrated_fraction < args.min_calibrated_point_fraction:
        raise RuntimeError(f"Calibrated derivative-point fraction {calibrated_fraction:.3f} is below the gate")
    payload = dict(shell)
    payload["uncalibrated_shell_frequency_hz"] = shell["frequency_hz"]
    payload["solid_frequency_correction_ratio"] = ratio
    payload["solid_calibration_confidence"] = calibration_confidence
    for key in ("frequency_hz", "tracking_confidence", "phase_velocity_m_s", "group_velocity_m_s", "df_dh_hz_per_mm", "dk_dh_rad_m_per_mm", "group_velocity_valid_mask", "thickness_derivative_valid_mask", "derivative_valid_mask"):
        payload[key] = derived[key]
    low_hz, high_hz = 1000.0 * np.asarray(args.frequency_range_khz)
    payload["frequency_in_range_mask"] = (payload["frequency_hz"] >= low_hz) & (payload["frequency_hz"] <= high_hz)
    payload["scientific_ready"] = np.asarray(True)
    payload["readiness_scope"] = np.asarray(args.readiness_scope)
    payload["source_shell_sha256"] = np.asarray(shell_sha)
    payload["source_mesh_report_sha256"] = np.asarray(mesh_sha)
    payload["source_solid_calibration_sha256"] = np.asarray(sha256(args.solid_calibration))
    output_path = args.output_root / "dispersion" / "axisymmetric_shell_dispersion_library.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(output_path)
    metadata = {"schema_version": 3, "scientific_ready": True, "readiness_scope": args.readiness_scope, "calibrated_derivative_point_fraction": calibrated_fraction, "source_shell_library": str(args.shell_library), "source_shell_sha256": shell_sha, "mesh_report": str(args.mesh_report), "mesh_report_sha256": mesh_sha, "solid_calibration": str(args.solid_calibration), "output": str(output_path), "limitation": "Qualified only for the current Shell-proxy simulation pipeline; not real-pipe certification."}
    metadata_path = args.output_root / "metadata" / "axisymmetric_shell_dispersion.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Calibrated library: {output_path}; derivative coverage={calibrated_fraction:.3f}")


if __name__ == "__main__":
    main()

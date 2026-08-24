"""Solve at most 36 full-solid points and match them to a formal Shell library."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
HERE = Path(__file__).resolve().parent
SIMPLE_ROOT = HERE.parents[1]
for path in (HERE, SIMPLE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dispersion_tracking as tracking
import solid_dispersion_common as solid
import streaming_export_common as streaming

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shell-library", type=Path, required=True)
    parser.add_argument("--thickness-mm", type=float, nargs="+", default=(5.0, 7.5, 10.0))
    parser.add_argument("--k-indices", type=int, nargs="+", default=(0, 20, 40))
    parser.add_argument("--circumferential-orders", type=int, nargs="+", default=(8, 10, 12))
    parser.add_argument("--mode-count", type=int, default=4)
    parser.add_argument("--max-solves", type=int, default=36)
    parser.add_argument("--mesh-hmax-mm", type=float, default=1.25)
    parser.add_argument("--eigen-shift-khz", type=float, default=60.0)
    parser.add_argument("--frequency-range-khz", type=float, nargs=2, default=(15.0, 110.0))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--allow-smoke-library", action="store_true", help="Code-smoke only.")
    return parser.parse_args()

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def load_shell(path: Path, allow_smoke: bool) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {"schema_version", "completed_mask", "thickness_mm", "kz_rad_m", "frequency_hz", "polarization", "circumferential_order", "tracking_confidence", "branch_id"}
        missing = required.difference(data.files)
        if missing or int(data["schema_version"]) != 3 or not np.asarray(data["completed_mask"]).all():
            raise RuntimeError(f"Shell library is incomplete schema v3; missing={sorted(missing)}")
        run_kind = str(data["run_kind"]) if "run_kind" in data.files else "smoke"
        if run_kind != "formal" and not allow_smoke:
            raise RuntimeError("Solid calibration requires a formal Shell library")
        return {key: np.asarray(data[key]) for key in required if key not in {"schema_version", "completed_mask"}}

def axis_indices(axis: np.ndarray, requested: list[float]) -> np.ndarray:
    indices = []
    for value in requested:
        matches = np.flatnonzero(np.isclose(axis, value, rtol=0.0, atol=1e-9))
        if matches.size != 1:
            raise RuntimeError(f"Requested axis value {value} is absent or duplicated")
        indices.append(int(matches[0]))
    return np.asarray(indices, dtype=int)

def write_checkpoint(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)

def initial_payload(path: Path, shape: tuple[int, ...], axes: dict[str, np.ndarray], shell_sha: str) -> dict[str, np.ndarray]:
    payload = {
        "schema_version": np.asarray(1, dtype=np.int32),
        "scientific_ready": np.asarray(False),
        "shell_library_sha256": np.asarray(shell_sha),
        **axes,
        "completed_mask": np.zeros(shape[:3], dtype=bool),
        "solid_frequency_hz": np.full(shape, np.nan),
        "solid_frequency_imag_hz": np.full(shape, np.nan),
        "solid_polarization": np.full((*shape, 3), np.nan),
    }
    if not path.exists():
        return payload
    with np.load(path, allow_pickle=False) as data:
        for key in axes:
            if key not in data.files or not np.array_equal(data[key], axes[key]):
                raise RuntimeError(f"Existing solid checkpoint axis {key} is incompatible")
        if str(data["shell_library_sha256"]) != shell_sha:
            raise RuntimeError("Existing solid checkpoint uses a different Shell library")
        for key in ("completed_mask", "solid_frequency_hz", "solid_frequency_imag_hz", "solid_polarization"):
            payload[key] = np.asarray(data[key], dtype=payload[key].dtype)
    return payload

def match_shell(payload: dict[str, np.ndarray], shell_data: dict[str, np.ndarray], h_indices: np.ndarray, k_indices: np.ndarray, args: argparse.Namespace) -> dict[str, np.ndarray]:
    shape = payload["solid_frequency_hz"].shape
    outputs = {
        "matched_shell_frequency_hz": np.full(shape, np.nan),
        "matched_shell_branch_id": np.full(shape, -1, dtype=np.int32),
        "match_cost": np.full(shape, np.nan),
        "frequency_ratio_solid_over_shell": np.full(shape, np.nan),
        "calibration_confidence": np.zeros(shape),
        "calibration_valid_mask": np.zeros(shape, dtype=bool),
    }
    low_hz, high_hz = 1000.0 * np.asarray(args.frequency_range_khz)
    shift_hz = args.eigen_shift_khz * 1000.0
    for hi, shell_hi in enumerate(h_indices):
        for ki, shell_ki in enumerate(k_indices):
            for ni, order in enumerate(payload["circumferential_orders"]):
                mask = shell_data["circumferential_order"][shell_hi, shell_ki] == order
                candidates = np.flatnonzero(mask & np.isfinite(shell_data["frequency_hz"][shell_hi, shell_ki]))
                if candidates.size < args.mode_count:
                    raise RuntimeError(f"Too few Shell modes for h/k/n={hi}/{ki}/{order}")
                shell_f_all = shell_data["frequency_hz"][shell_hi, shell_ki]
                candidates = candidates[np.argsort(np.abs(shell_f_all[candidates] - shift_hz))[: args.mode_count]]
                shell_f = shell_f_all[candidates]
                shell_p = shell_data["polarization"][shell_hi, shell_ki, candidates]
                solid_f = payload["solid_frequency_hz"][hi, ki, ni]
                solid_p = payload["solid_polarization"][hi, ki, ni]
                cost = np.abs(np.log(np.maximum(solid_f[:, None], 1.0) / np.maximum(shell_f[None, :], 1.0)))
                cost += 0.25 * np.sum(np.abs(solid_p[:, None] - shell_p[None, :]), axis=2)
                assignment = tracking.linear_sum_assignment(cost)
                matched = candidates[assignment]
                selected_f = shell_f_all[matched]
                selected_cost = cost[np.arange(args.mode_count), assignment]
                shell_conf = shell_data["tracking_confidence"][shell_hi, shell_ki, matched]
                valid = (solid_f >= low_hz) & (solid_f <= high_hz) & (selected_f >= low_hz) & (selected_f <= high_hz)
                valid &= np.isfinite(shell_conf) & (selected_cost <= 1.1)
                target = (hi, ki, ni)
                outputs["matched_shell_frequency_hz"][target] = selected_f
                outputs["matched_shell_branch_id"][target] = shell_data["branch_id"][shell_hi, shell_ki, matched]
                outputs["match_cost"][target] = selected_cost
                outputs["frequency_ratio_solid_over_shell"][target] = solid_f / selected_f
                outputs["calibration_confidence"][target] = np.where(valid, np.exp(-selected_cost) * shell_conf, 0.0)
                outputs["calibration_valid_mask"][target] = valid
    return outputs

def main() -> None:
    args = parse_args()
    solve_count = len(args.thickness_mm) * len(args.k_indices) * len(args.circumferential_orders)
    if solve_count > args.max_solves or args.mode_count < 1 or args.mode_count > 8:
        raise ValueError(f"Requested {solve_count} solves or invalid mode count; limit={args.max_solves}")
    shell_data = load_shell(args.shell_library, args.allow_smoke_library)
    h_indices = axis_indices(shell_data["thickness_mm"], list(args.thickness_mm))
    if any(index < 0 or index >= shell_data["kz_rad_m"].size for index in args.k_indices):
        raise ValueError("--k-indices are outside the Shell wave-number axis")
    k_indices = np.asarray(args.k_indices, dtype=int)
    axes = {
        "thickness_mm": shell_data["thickness_mm"][h_indices],
        "kz_rad_m": shell_data["kz_rad_m"][k_indices],
        "circumferential_orders": np.asarray(args.circumferential_orders, dtype=np.int32),
        "local_mode_index": np.arange(args.mode_count, dtype=np.int32),
    }
    shape = (len(h_indices), len(k_indices), len(args.circumferential_orders), args.mode_count)
    output_path = args.output_root / "calibration" / "solid_shell_calibration.npz"
    shell_sha = sha256(args.shell_library)
    payload = initial_payload(output_path, shape, axes, shell_sha)
    metadata_path = args.output_root / "metadata" / "solid_shell_calibration.json"
    if payload["completed_mask"].all() and metadata_path.exists():
        print(f"Solid calibration resumed: {output_path}")
        return
    client = solid.shell.start_client(cores=args.cores)
    trees = []
    try:
        for hi, thickness in enumerate(axes["thickness_mm"]):
            if payload["completed_mask"][hi].all():
                continue
            model = None
            try:
                config = solid.SolidCellConfig(thickness_mm=float(thickness), mesh_hmax_mm=args.mesh_hmax_mm, mode_count=args.mode_count, eigen_shift_hz=args.eigen_shift_khz * 1000.0)
                model = solid.build_solid_model(client, f"solid_calibration_h{thickness:g}", config)
                trees.append(solid.validate_model_tree(model))
                for ki, kz in enumerate(axes["kz_rad_m"]):
                    for ni, order in enumerate(axes["circumferential_orders"]):
                        if payload["completed_mask"][hi, ki, ni]:
                            continue
                        frequency, polarization = solid.solve_modes(model, float(kz), int(order))
                        if frequency.size < args.mode_count:
                            raise RuntimeError("Solid eigensolve returned too few modes")
                        target = (hi, ki, ni, slice(None))
                        payload["solid_frequency_hz"][target] = frequency[: args.mode_count].real
                        payload["solid_frequency_imag_hz"][target] = frequency[: args.mode_count].imag
                        payload["solid_polarization"][target] = polarization[: args.mode_count]
                        payload["completed_mask"][hi, ki, ni] = True
                        write_checkpoint(output_path, payload)
                        streaming.clear_solution_data(model)
                        print(f"[solid] h={thickness:g} k={kz:g} n={order} completed={payload['completed_mask'].sum()}/{solve_count}", flush=True)
            finally:
                if model is not None:
                    client.remove(model)
    finally:
        client.clear()
    payload.update(match_shell(payload, shell_data, h_indices, k_indices, args))
    write_checkpoint(output_path, payload)
    valid = payload["calibration_valid_mask"]
    metadata = {"schema_version": 1, "scientific_ready": False, "completed_points": int(payload["completed_mask"].sum()), "total_points": solve_count, "valid_mode_fraction": float(valid.mean()), "shell_library": str(args.shell_library), "shell_library_sha256": shell_sha, "config": asdict(solid.SolidCellConfig(mesh_hmax_mm=args.mesh_hmax_mm, mode_count=args.mode_count, eigen_shift_hz=args.eigen_shift_khz * 1000.0)), "model_trees": trees, "checkpoint": str(output_path)}
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"Solid calibration: {output_path}; valid_mode_fraction={valid.mean():.3f}")


if __name__ == "__main__":
    main()

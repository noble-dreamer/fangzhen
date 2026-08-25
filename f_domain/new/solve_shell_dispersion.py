"""Stream a guarded Shell dispersion smoke scan to checkpointed NPZ output."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

import axisymmetric_dispersion_common as axisymmetric
import dispersion_tracking as tracking
import shell_dispersion_common as common
import shell_mode_features as mode_features
import streaming_export_common as streaming


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "shell_dispersion"
RUN_ID = "shell_dispersion_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thickness-mm", type=float, nargs="+", default=[5.0, 7.5, 10.0])
    parser.add_argument("--formulation", choices=("axisymmetric", "full-ring"), default="axisymmetric")
    parser.add_argument("--circumferential-orders", type=int, nargs="+", default=[8, 10, 12])
    parser.add_argument("--k-count", type=int, default=3)
    parser.add_argument("--k-max-rad-m", type=float)
    parser.add_argument("--frequency-range-khz", type=float, nargs=2, default=[15.0, 110.0])
    parser.add_argument("--mode-count", type=int, default=12)
    parser.add_argument("--max-circumferential-order", type=int, default=12)
    parser.add_argument("--eigen-shift-khz", type=float, default=60.0)
    parser.add_argument("--cell-length-mm", type=float, default=5.0)
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Small integration scan.")
    mode.add_argument("--formal", action="store_true", help="Dense uncalibrated axisymmetric scan.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if any(not 1.0 <= value <= 10.0 for value in args.thickness_mm):
        raise SystemExit("Thickness values must be in [1, 10] mm.")
    if np.any(np.diff(np.asarray(args.thickness_mm, dtype=float)) <= 0.0):
        raise SystemExit("--thickness-mm must be strictly increasing.")
    if not 0 <= args.max_circumferential_order <= 32:
        raise SystemExit("--max-circumferential-order must be in [0, 32].")
    if not args.circumferential_orders or any(value < 0 for value in args.circumferential_orders):
        raise SystemExit("--circumferential-orders must contain nonnegative integers.")
    if len(set(args.circumferential_orders)) != len(args.circumferential_orders):
        raise SystemExit("--circumferential-orders must not contain duplicates.")
    if args.frequency_range_khz[1] <= args.frequency_range_khz[0]:
        raise SystemExit("Frequency range must be increasing.")
    if args.smoke:
        if not 1 <= args.k_count <= 11 or not 1 <= len(args.thickness_mm) <= 3:
            raise SystemExit("Smoke accepts 1--11 k points and one to three thickness values.")
        if not 1 <= args.mode_count <= 32:
            raise SystemExit("Smoke --mode-count must be in [1, 32].")
        return
    if args.formulation != "axisymmetric":
        raise SystemExit("--formal supports only the prescribed-order axisymmetric formulation.")
    if not 5 <= args.k_count <= 81 or not 3 <= len(args.thickness_mm) <= 21:
        raise SystemExit("Formal scan requires 5--81 k points and 3--21 thickness values.")
    if not 1 <= args.mode_count <= 16 or max(args.circumferential_orders) > 16:
        raise SystemExit("Formal scan permits at most 16 modes and circumferential order 16.")
    if "smoke" in str(args.output_root).lower():
        raise SystemExit("Formal output root must not contain 'smoke'.")


def axes(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    thickness = np.asarray(args.thickness_mm, dtype=float)
    default_kmax = np.pi / (args.cell_length_mm * 1e-3)
    kmax = default_kmax if args.k_max_rad_m is None else args.k_max_rad_m
    if not np.isfinite(kmax) or kmax < 0.0:
        raise SystemExit("--k-max-rad-m must be finite and nonnegative.")
    return thickness, np.linspace(0.0, kmax, args.k_count, dtype=float)


def checkpoint_path(output_root: Path) -> Path:
    return output_root / "dispersion" / "shell_dispersion_library.npz"


def load_checkpoint(
    path: Path,
    thickness_mm: np.ndarray,
    kz_rad_m: np.ndarray,
    mode_count: int,
    max_order: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    shape = (thickness_mm.size, kz_rad_m.size, mode_count)
    signature_shape = (*shape, (2 * max_order + 1) * 3)
    arrays = {
        "frequency_hz": np.full(shape, np.nan),
        "frequency_imag_hz": np.full(shape, np.nan),
        "polarization": np.full((*shape, 3), np.nan),
        "circumferential_order": np.full(shape, -1, dtype=np.int32),
        "order_confidence": np.full(shape, np.nan),
        "observability": np.full(shape, np.nan),
        "signature": np.zeros(signature_shape, dtype=complex),
        "surface_area_m2": np.full(shape[:2], np.nan),
        "edge_length_m": np.full(shape[:2], np.nan),
    }
    if not path.exists():
        return arrays, np.zeros(shape[:2], dtype=bool)
    with np.load(path, allow_pickle=False) as data:
        if int(data.get("schema_version", 0)) != 2:
            raise RuntimeError("Existing checkpoint is not modal schema v2; use a new --output-root")
        if not np.array_equal(data["thickness_mm"], thickness_mm):
            raise RuntimeError("Existing checkpoint thickness axis is incompatible")
        if not np.allclose(data["kz_rad_m"], kz_rad_m, rtol=0.0, atol=1e-12):
            raise RuntimeError("Existing checkpoint wave-number axis is incompatible")
        for name in arrays:
            if name == "signature":
                arrays[name] = data["raw_signature_real"] + 1j * data["raw_signature_imag"]
            else:
                arrays[name] = np.asarray(data[f"raw_{name}"], dtype=arrays[name].dtype)
        completed = np.asarray(data["completed_mask"], dtype=bool)
    expected = {name: value.shape for name, value in arrays.items()}
    if any(arrays[name].shape != shape for name, shape in expected.items()) or completed.shape != shape[:2]:
        raise RuntimeError("Existing modal checkpoint array shape is incompatible")
    return arrays, completed


def write_checkpoint(
    path: Path,
    thickness_mm: np.ndarray,
    kz_rad_m: np.ndarray,
    arrays: dict[str, np.ndarray],
    completed_mask: np.ndarray,
    derived: dict[str, np.ndarray] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    payload = {
        "schema_version": np.asarray(2, dtype=np.int32),
        "thickness_mm": thickness_mm,
        "kz_rad_m": kz_rad_m,
        "mode_index": np.arange(arrays["frequency_hz"].shape[2], dtype=np.int32),
        "completed_mask": completed_mask,
        "frequency_unit": np.asarray("Hz"),
        "wave_number_unit": np.asarray("rad/m"),
        "thickness_unit": np.asarray("mm"),
        "observability_definition": np.asarray("axial polarization times rectangular patch sinc squared"),
    }
    for name, value in arrays.items():
        if name == "signature":
            payload["raw_signature_real"] = value.real
            payload["raw_signature_imag"] = value.imag
        else:
            payload[f"raw_{name}"] = value
    if derived is not None:
        for name, value in derived.items():
            if name == "signature":
                payload["signature_real"] = value.real
                payload["signature_imag"] = value.imag
            else:
                payload[name] = value
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def emit(progress_path: Path, started: float, status: str, **fields) -> None:
    event = {
        "wall_time_s": time.time(),
        "run_id": progress_path.stem.removesuffix("_progress"),
        "status": status,
        "elapsed_s": time.monotonic() - started,
        **fields,
    }
    streaming.append_progress(progress_path, event)
    print(json.dumps(event, ensure_ascii=False, default=str), flush=True)


def axisymmetric_checkpoint_path(output_root: Path) -> Path:
    return output_root / "dispersion" / "axisymmetric_shell_dispersion_library.npz"


def load_axisymmetric_checkpoint(path, thickness_mm, kz_rad_m, orders, mode_count, run_kind):
    shape = (thickness_mm.size, kz_rad_m.size, orders.size, mode_count)
    arrays = {
        "frequency_hz": np.full(shape, np.nan),
        "frequency_imag_hz": np.full(shape, np.nan),
        "polarization": np.full((*shape, 3), np.nan),
        "observability": np.full(shape, np.nan),
        "signature": np.zeros((*shape, 3), dtype=complex),
    }
    expected_shapes = {name: value.shape for name, value in arrays.items()}
    completed = np.zeros(shape[:3], dtype=bool)
    if not path.exists():
        return arrays, completed
    with np.load(path, allow_pickle=False) as data:
        if int(data.get("schema_version", 0)) != 3:
            raise RuntimeError("Existing axisymmetric checkpoint is not schema v3")
        stored_kind = str(data["run_kind"]) if "run_kind" in data.files else "smoke"
        if stored_kind != run_kind:
            raise RuntimeError(f"Existing checkpoint run_kind={stored_kind!r} is incompatible")
        for name, expected in (
            ("thickness_mm", thickness_mm),
            ("kz_rad_m", kz_rad_m),
            ("circumferential_orders", orders),
        ):
            if not np.array_equal(data[name], expected):
                raise RuntimeError(f"Existing checkpoint {name} axis is incompatible")
        for name in arrays:
            if name == "signature":
                arrays[name] = data["raw_signature_real"] + 1j * data["raw_signature_imag"]
            else:
                arrays[name] = np.asarray(data[f"raw_{name}"], dtype=arrays[name].dtype)
        completed = np.asarray(data["completed_mask"], dtype=bool)
    if any(arrays[name].shape != expected_shapes[name] for name in arrays):
        raise RuntimeError("Existing axisymmetric checkpoint array shape is incompatible")
    if completed.shape != shape[:3]:
        raise RuntimeError("Existing axisymmetric completion mask shape is incompatible")
    return arrays, completed


def write_axisymmetric_checkpoint(path, thickness_mm, kz_rad_m, orders, arrays, completed, run_kind, derived=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    payload = {
        "schema_version": np.asarray(3, dtype=np.int32),
        "run_kind": np.asarray(run_kind),
        "scientific_ready": np.asarray(False),
        "thickness_mm": thickness_mm,
        "kz_rad_m": kz_rad_m,
        "circumferential_orders": orders,
        "local_mode_index": np.arange(arrays["frequency_hz"].shape[3], dtype=np.int32),
        "completed_mask": completed,
        "frequency_unit": np.asarray("Hz"),
        "wave_number_unit": np.asarray("rad/m"),
        "thickness_unit": np.asarray("mm"),
        "observability_definition": np.asarray("prescribed n; axial polarization times patch sinc squared"),
    }
    for name, value in arrays.items():
        if name == "signature":
            payload["raw_signature_real"] = value.real
            payload["raw_signature_imag"] = value.imag
        else:
            payload[f"raw_{name}"] = value
    if derived:
        payload.update(derived)
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def _axisymmetric_derived(arrays, thickness_mm, kz_rad_m, orders):
    chunks = []
    for order_index, circumferential_order in enumerate(orders):
        polarization = arrays["polarization"][:, :, order_index]
        tracked = tracking.track_modes(
            arrays["frequency_hz"][:, :, order_index],
            polarization,
            np.full(arrays["frequency_hz"][:, :, order_index].shape, circumferential_order),
            arrays["observability"][:, :, order_index],
            arrays["signature"][:, :, order_index],
            frequency_imag_hz=arrays["frequency_imag_hz"][:, :, order_index],
            order_confidence=np.ones(arrays["frequency_hz"][:, :, order_index].shape),
        )
        if thickness_mm.size >= 2 and kz_rad_m.size >= 2:
            tracked = tracking.derive_dispersion(tracked, thickness_mm, kz_rad_m)
        tracked["branch_id"] += order_index * arrays["frequency_hz"].shape[3]
        chunks.append(tracked)
    output = {}
    for name in chunks[0]:
        if name == "signature":
            output["signature_real"] = np.concatenate([chunk[name].real for chunk in chunks], axis=2)
            output["signature_imag"] = np.concatenate([chunk[name].imag for chunk in chunks], axis=2)
        else:
            output[name] = np.concatenate([chunk[name] for chunk in chunks], axis=2)
    return output


def run_axisymmetric(args: argparse.Namespace) -> None:
    thickness_mm, kz_rad_m = axes(args)
    orders = np.asarray(sorted(args.circumferential_orders), dtype=np.int32)
    run_kind = "formal" if args.formal else "smoke"
    output_root = args.output_root.resolve()
    progress_path = output_root / "progress" / f"axisymmetric_shell_dispersion_{run_kind}_progress.jsonl"
    metadata_path = output_root / "metadata" / f"axisymmetric_shell_dispersion_{run_kind}.json"
    npz_path = axisymmetric_checkpoint_path(output_root)
    arrays, completed = load_axisymmetric_checkpoint(
        npz_path, thickness_mm, kz_rad_m, orders, args.mode_count, run_kind
    )
    started = time.monotonic()
    total = int(completed.size)
    emit(progress_path, started, "run_start", formulation="axisymmetric", completed=int(completed.sum()), total=total)
    if completed.all() and metadata_path.exists():
        emit(progress_path, started, "run_done", formulation="axisymmetric", resumed=True)
        return
    config = axisymmetric.AxisymmetricCellConfig(
        cell_length_mm=args.cell_length_mm,
        mode_count=args.mode_count,
        eigen_shift_hz=args.eigen_shift_khz * 1000.0,
    )
    client = common.shell.start_client(cores=args.cores)
    model = None
    try:
        model = axisymmetric.build_axisymmetric_model(client, "axisymmetric_shell_dispersion_smoke", config)
        emit(progress_path, started, "model_ready", tree=axisymmetric.validate_model_tree(model))
        for h_index, thickness in enumerate(thickness_mm):
            for k_index, kz in enumerate(kz_rad_m):
                for n_index, circumferential_order in enumerate(orders):
                    if completed[h_index, k_index, n_index]:
                        continue
                    emit(
                        progress_path,
                        started,
                        "point_start",
                        thickness_mm=float(thickness),
                        kz_rad_m=float(kz),
                        circumferential_order=int(circumferential_order),
                    )
                    frequency, polarization = axisymmetric.solve_axisymmetric_modes(
                        model, float(thickness), float(kz), int(circumferential_order)
                    )
                    count = min(frequency.size, args.mode_count)
                    target = (h_index, k_index, n_index, slice(0, count))
                    arrays["frequency_hz"][target] = frequency[:count].real
                    arrays["frequency_imag_hz"][target] = frequency[:count].imag
                    arrays["polarization"][target] = polarization[:count]
                    theta = circumferential_order * 6e-3 / (2.0 * 155e-3)
                    axial = float(kz) * 27e-3 / 2.0
                    sinc = lambda value: 1.0 if abs(value) < 1e-12 else np.sin(value) / value
                    arrays["observability"][target] = polarization[:count, 2] * sinc(theta) ** 2 * sinc(axial) ** 2
                    signature = polarization[:count] / np.maximum(
                        np.linalg.norm(polarization[:count], axis=1, keepdims=True), 1e-30
                    )
                    arrays["signature"][target] = signature
                    completed[h_index, k_index, n_index] = True
                    write_axisymmetric_checkpoint(npz_path, thickness_mm, kz_rad_m, orders, arrays, completed, run_kind)
                    streaming.clear_solution_data(model)
                    emit(progress_path, started, "point_done", completed=int(completed.sum()), total=total)
    finally:
        if model is not None:
            client.remove(model)
        client.clear()
    derived = _axisymmetric_derived(arrays, thickness_mm, kz_rad_m, orders)
    low_hz, high_hz = np.asarray(args.frequency_range_khz) * 1000.0
    derived["frequency_in_range_mask"] = (
        (derived["frequency_hz"] >= low_hz) & (derived["frequency_hz"] <= high_hz)
    )
    write_axisymmetric_checkpoint(npz_path, thickness_mm, kz_rad_m, orders, arrays, completed, run_kind, derived)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps({
        "schema_version": 3,
        "run_kind": run_kind,
        "scientific_ready": False,
        "formulation": "2D axisymmetric Shell with prescribed circumferential order and axial Floquet k",
        "config": asdict(config),
        "circumferential_orders": orders.tolist(),
        "completed_points": int(completed.sum()),
        "total_points": total,
        "checkpoint": str(npz_path),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    emit(progress_path, started, "run_done", formulation="axisymmetric", completed=total, total=total)


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.formulation == "axisymmetric":
        run_axisymmetric(args)
        return
    thickness_mm, kz_rad_m = axes(args)
    config = common.DispersionCellConfig(
        cell_length_mm=args.cell_length_mm,
        mode_count=args.mode_count,
        eigen_shift_hz=args.eigen_shift_khz * 1000.0,
    )
    output_root = args.output_root.resolve()
    progress_path = output_root / "progress" / f"{RUN_ID}_progress.jsonl"
    metadata_path = output_root / "metadata" / f"{RUN_ID}.json"
    npz_path = checkpoint_path(output_root)
    arrays, completed = load_checkpoint(
        npz_path,
        thickness_mm,
        kz_rad_m,
        args.mode_count,
        args.max_circumferential_order,
    )
    total = int(completed.size)
    started = time.monotonic()
    emit(progress_path, started, "run_start", completed=int(completed.sum()), total=total)
    if completed.all() and metadata_path.exists():
        emit(progress_path, started, "run_done", completed=total, total=total, resumed=True)
        return

    client = common.shell.start_client(cores=args.cores)
    model = None
    try:
        model = common.build_periodic_shell_model(client, RUN_ID, config)
        tree = common.validate_model_tree(model)
        emit(progress_path, started, "model_ready", tree=tree)
        for h_index, thickness in enumerate(thickness_mm):
            for k_index, kz in enumerate(kz_rad_m):
                if completed[h_index, k_index]:
                    continue
                point = h_index * kz_rad_m.size + k_index + 1
                emit(
                    progress_path,
                    started,
                    "point_start",
                    point=point,
                    total=total,
                    thickness_mm=float(thickness),
                    kz_rad_m=float(kz),
                )
                features = mode_features.extract_mode_features(
                    model,
                    float(thickness),
                    float(kz),
                    max_order=args.max_circumferential_order,
                )
                low_hz, high_hz = np.asarray(args.frequency_range_khz) * 1000.0
                selected = np.flatnonzero(
                    (features.frequency_hz.real >= low_hz) & (features.frequency_hz.real <= high_hz)
                )[: args.mode_count]
                count = selected.size
                if count == 0:
                    raise RuntimeError("No eigenmodes lie inside the requested frequency range")
                target = (h_index, k_index, slice(0, count))
                arrays["frequency_hz"][target] = features.frequency_hz[selected].real
                arrays["frequency_imag_hz"][target] = features.frequency_hz[selected].imag
                arrays["polarization"][target] = features.polarization[selected]
                arrays["circumferential_order"][target] = features.circumferential_order[selected]
                arrays["order_confidence"][target] = features.order_confidence[selected]
                arrays["observability"][target] = features.observability[selected]
                arrays["signature"][target] = features.signature[selected]
                arrays["surface_area_m2"][h_index, k_index] = features.surface_area_m2
                arrays["edge_length_m"][h_index, k_index] = features.edge_length_m
                completed[h_index, k_index] = True
                write_checkpoint(npz_path, thickness_mm, kz_rad_m, arrays, completed)
                streaming.clear_solution_data(model)
                emit(progress_path, started, "point_done", point=point, total=total, modes=count)
    except Exception as error:
        emit(progress_path, started, "run_failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if model is not None:
            client.remove(model)
        client.clear()

    tracked = tracking.track_modes(
        arrays["frequency_hz"],
        arrays["polarization"],
        arrays["circumferential_order"],
        arrays["observability"],
        arrays["signature"],
        frequency_imag_hz=arrays["frequency_imag_hz"],
        order_confidence=arrays["order_confidence"],
    )
    if thickness_mm.size >= 2 and kz_rad_m.size >= 2:
        derived = tracking.derive_dispersion(tracked, thickness_mm, kz_rad_m)
    else:
        derived = tracked
        for name in (
            "phase_velocity_m_s",
            "group_velocity_m_s",
            "df_dh_hz_per_mm",
            "dk_dh_rad_m_per_mm",
        ):
            derived[name] = np.full_like(arrays["frequency_hz"], np.nan)
    write_checkpoint(npz_path, thickness_mm, kz_rad_m, arrays, completed, derived=derived)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_id": RUN_ID,
        "config": asdict(config),
        "frequency_range_khz": args.frequency_range_khz,
        "checkpoint": str(npz_path),
        "progress": str(progress_path),
        "completed_points": int(completed.sum()),
        "total_points": total,
        "schema_version": 2,
        "max_circumferential_order": args.max_circumferential_order,
        "observability": "axial-polarization rectangular-patch coupling proxy; calibration pending",
        "note": "Tracked modal smoke; finite-pipe and 3D calibration are not applied.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    emit(progress_path, started, "run_done", completed=int(completed.sum()), total=total)


if __name__ == "__main__":
    main()

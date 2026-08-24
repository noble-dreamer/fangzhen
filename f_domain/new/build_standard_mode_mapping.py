"""Import Dispersion Calculator curves and map Shell branches to standard modes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

import dispersion_tracking as tracking


HERE = Path(__file__).resolve().parent
THICKNESSES_MM = (5.0, 6.0, 7.0, 7.5, 8.0, 9.0, 10.0)
DC_VERSION = "3.2.0.0"
DC_SHA256 = "34BE241E1597EBFD3A450C8E91F3CFB837FBC3F9E84FF75AF6BE5FBC0D4D322B"
FIELDS = (
    ("f (kHz)", "frequency_hz", 1_000.0),
    ("Phase velocity (m/ms)", "phase_velocity_m_s", 1_000.0),
    ("Energy velocity 1 (m/ms)", "energy_velocity_1_m_s", 1_000.0),
    ("Energy velocity 2 (m/ms)", "energy_velocity_2_m_s", 1_000.0),
    ("Energy velocity absolute (m/ms)", "group_velocity_m_s", 1_000.0),
    ("Skew angle (deg)", "skew_angle_deg", 1.0),
    ("Propagation time (micsec)", "propagation_time_s", 1e-6),
    ("Coincidence angle (deg)", "coincidence_angle_deg", 1.0),
    ("Wavelength (mm)", "wavelength_m", 1e-3),
    ("Wavenumber (rad/mm)", "kz_rad_m", 1_000.0),
    ("Attenuation (Np/m)", "attenuation_np_m", 1.0),
)
MODE_SPECS = (
    [("F", n, m) for n in range(1, 9) for m in range(1, 4)]
    + [("L", 0, m) for m in range(1, 3)]
    + [("T", 0, 1)]
)
GATES = {
    "min_points": 20, "min_thicknesses": 5, "min_coverage": 0.60,
    "max_median_df": 0.12, "max_p95_df": 0.25, "max_cost": 0.30,
    "min_margin": 0.05, "min_consistency": 0.80, "min_confidence": 0.35,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def thickness_tag(value: float) -> str:
    return f"h{value:04.1f}".replace(".", "p")


def read_export(path: Path, family: str, order: int, mode_count: int) -> list[np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader([handle.readline()]))
        values = np.genfromtxt(handle, delimiter=",", dtype=float)
    values = np.atleast_2d(values)
    if len(header) != mode_count * len(FIELDS) or values.shape[1] != len(header):
        raise RuntimeError(f"{path}: expected {mode_count} horizontal 11-column blocks")
    curves = []
    for mode_index in range(mode_count):
        label = f"{family}({order},{mode_index + 1})"
        start = mode_index * len(FIELDS)
        for offset, (title, _, _) in enumerate(FIELDS):
            if header[start + offset] != f"{label} {title}":
                raise RuntimeError(f"{path}: invalid column {start + offset + 1}: {header[start + offset]!r}")
        block = values[:, start : start + len(FIELDS)]
        finite = np.isfinite(block)
        if np.any(finite.any(axis=1) & ~finite.all(axis=1)):
            raise RuntimeError(f"{path}: partial row in {label}")
        curve = block[finite.all(axis=1)] * np.asarray([field[2] for field in FIELDS])
        frequency, phase_velocity, kz = curve[:, 0], curve[:, 1], curve[:, 9]
        band = (frequency >= 15_000.0) & (frequency <= 110_000.0)
        if band.sum() < 2 or frequency[-1] < 110_000.0:
            raise RuntimeError(f"{path}: {label} has insufficient support through 110 kHz")
        if np.any(np.diff(frequency[band]) <= 0.0) or np.any(np.diff(kz[band]) <= 0.0):
            raise RuntimeError(f"{path}: {label} is not a single increasing branch in the mapping band")
        expected_k = 2.0 * np.pi * frequency[band] / phase_velocity[band]
        error = np.max(np.abs(kz[band] - expected_k) / np.maximum(np.abs(expected_k), 1e-12))
        if not np.isfinite(error) or error > 1e-6:
            raise RuntimeError(f"{path}: {label} k=2*pi*f/cp relative error {error:.3g}")
        curves.append(curve)
    return curves


def import_dc(args: argparse.Namespace) -> None:
    if sha256(args.dc_executable) != DC_SHA256:
        raise RuntimeError(f"Unexpected DC executable SHA-256: {args.dc_executable}")
    records: dict[tuple[int, int], np.ndarray] = {}
    source_hashes: dict[str, str] = {}
    unique_hashes: dict[tuple[str, int], set[str]] = {}
    max_points = 0
    for thickness_index, thickness in enumerate(THICKNESSES_MM):
        tag = thickness_tag(thickness)
        directory = args.input_root / tag
        expected = [directory / f"pipe_{tag}_F{n}.txt" for n in range(1, 9)]
        expected += [directory / f"pipe_{tag}_L.txt", directory / f"pipe_{tag}_T.txt"]
        actual = sorted(directory.glob("*.txt"))
        if set(actual) != set(expected):
            missing = sorted(str(path) for path in set(expected) - set(actual))
            extra = sorted(str(path) for path in set(actual) - set(expected))
            raise RuntimeError(f"{directory}: TXT set mismatch; missing={missing}, extra={extra}")
        mode_offset = 0
        for path, (family, order, count) in zip(
            expected, [("F", n, 3) for n in range(1, 9)] + [("L", 0, 2), ("T", 0, 1)], strict=True
        ):
            file_hash = sha256(path)
            relative = path.relative_to(args.input_root).as_posix()
            source_hashes[relative] = file_hash
            if family != "T":
                unique_hashes.setdefault((family, order), set()).add(file_hash)
            for curve in read_export(path, family, order, count):
                records[(thickness_index, mode_offset)] = curve
                max_points = max(max_points, curve.shape[0])
                mode_offset += 1
    duplicates = [key for key, values in unique_hashes.items() if len(values) != len(THICKNESSES_MM)]
    if duplicates:
        raise RuntimeError(f"Repeated F/L content across thicknesses: {duplicates}")
    shape = (len(THICKNESSES_MM), len(MODE_SPECS), max_points)
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(1, dtype=np.int32),
        "standard_reference_ready": np.asarray(True),
        "thickness_mm": np.asarray(THICKNESSES_MM),
        "family": np.asarray([item[0] for item in MODE_SPECS]),
        "circumferential_order": np.asarray([item[1] for item in MODE_SPECS], dtype=np.int32),
        "standard_m": np.asarray([item[2] for item in MODE_SPECS], dtype=np.int32),
        "standard_mode_label": np.asarray([f"{f}({n},{m})" for f, n, m in MODE_SPECS]),
        "valid_mask": np.zeros(shape, dtype=bool),
        "dc_version": np.asarray(DC_VERSION),
        "dc_executable_sha256": np.asarray(DC_SHA256),
    }
    for _, key, _ in FIELDS:
        payload[key] = np.full(shape, np.nan)
    for (thickness_index, mode_index), curve in records.items():
        count = curve.shape[0]
        payload["valid_mask"][thickness_index, mode_index, :count] = True
        for column, (_, key, _) in enumerate(FIELDS):
            payload[key][thickness_index, mode_index, :count] = curve[:, column]
    manifest_text = "".join(f"{name}:{value}\n" for name, value in sorted(source_hashes.items()))
    payload["source_txt_manifest_sha256"] = np.asarray(hashlib.sha256(manifest_text.encode()).hexdigest().upper())
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "dc_standard_pipe_dispersion.npz"
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(output)
    metadata = {
        "schema_version": 1, "standard_reference_ready": True, "dc_version": DC_VERSION,
        "dc_executable": str(args.dc_executable), "dc_executable_sha256": DC_SHA256,
        "material": {"E_pa": 70e9, "nu": 0.33, "rho_kg_m3": 2700.0},
        "geometry": {"inner_diameter_mm": 300.0, "thickness_mm": list(THICKNESSES_MM)},
        "mapping_band_hz": [15000.0, 110000.0], "source_root": str(args.input_root),
        "source_txt_sha256": source_hashes, "source_txt_manifest_sha256": str(payload["source_txt_manifest_sha256"]),
        "output": str(output), "output_sha256": sha256(output),
    }
    metadata_path = args.output_root / "dc_standard_pipe_dispersion.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"DC standard library: {output}; modes={len(MODE_SPECS)}, max_points={max_points}")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def validate_mapping_inputs(shell: dict[str, np.ndarray], dc: dict[str, np.ndarray]) -> None:
    if int(dc.get("schema_version", 0)) != 1 or not bool(dc.get("standard_reference_ready", False)):
        raise RuntimeError("DC library must be a ready schema-v1 standard reference")
    if int(shell.get("schema_version", 0)) != 3 or not bool(shell.get("scientific_ready", False)):
        raise RuntimeError("Shell library must be a complete calibrated schema-v3 library")
    if str(shell.get("readiness_scope", "")) != "shell_proxy_simulation_only":
        raise RuntimeError("Shell library has the wrong readiness scope")
    if str(shell.get("run_kind", "")) != "formal" or not np.asarray(shell["completed_mask"]).all():
        raise RuntimeError("Shell library must originate from a complete formal scan")
    required = {
        "frequency_hz", "group_velocity_m_s", "tracking_residual", "branch_id",
        "circumferential_order", "frequency_in_range_mask", "source_shell_sha256",
        "source_mesh_report_sha256", "source_solid_calibration_sha256",
    }
    missing = sorted(required - shell.keys())
    if missing:
        raise RuntimeError(f"Calibrated Shell library is missing fields: {missing}")
    if not np.array_equal(shell["thickness_mm"], dc["thickness_mm"]):
        raise RuntimeError("Shell and DC thickness axes differ")
    if not np.array_equal(shell["circumferential_orders"], np.arange(9)):
        raise RuntimeError("Standard mapping requires the dedicated n=0..8 formal scan")
    if np.asarray(shell["kz_rad_m"]).size != 41:
        raise RuntimeError("Standard mapping requires k-count=41")
    if np.asarray(shell["local_mode_index"]).size not in (8, 16):
        raise RuntimeError("Standard mapping requires mode-count 8 or the 16-mode fallback")
    expected = set(MODE_SPECS)
    actual = set(zip(dc["family"].tolist(), dc["circumferential_order"].tolist(), dc["standard_m"].tolist()))
    if actual != expected:
        raise RuntimeError("DC library does not contain exactly F(1..8,1..3), L(0,1..2), T(0,1)")
    for branch_index in range(shell["frequency_hz"].shape[2]):
        if np.unique(shell["branch_id"][:, :, branch_index]).size != 1:
            raise RuntimeError(f"Shell branch_id is not constant at branch index {branch_index}")
        if np.unique(shell["circumferential_order"][:, :, branch_index]).size != 1:
            raise RuntimeError(f"Shell n is not constant at branch index {branch_index}")


def score_branch(shell: dict[str, np.ndarray], dc: dict[str, np.ndarray], branch_index: int, order: int) -> list[dict]:
    shell_f = np.asarray(shell["frequency_hz"][:, :, branch_index], dtype=float)
    shell_g = np.asarray(shell["group_velocity_m_s"][:, :, branch_index], dtype=float)
    shell_k = np.asarray(shell["kz_rad_m"], dtype=float)
    base = np.asarray(shell["frequency_in_range_mask"][:, :, branch_index], dtype=bool)
    base &= np.asarray(shell["tracking_residual"][:, :, branch_index]) <= 1.1
    base &= np.isfinite(shell_f) & np.isfinite(shell_g)
    work = []
    for standard_m in range(1, 4):
        mode = np.flatnonzero(
            (dc["family"] == "F") & (dc["circumferential_order"] == order) & (dc["standard_m"] == standard_m)
        )
        if mode.size != 1:
            raise RuntimeError(f"DC mode F({order},{standard_m}) is not unique")
        df = np.full(shell_f.shape, np.nan)
        dg = np.full(shell_f.shape, np.nan)
        for thickness_index in range(shell_f.shape[0]):
            valid = np.asarray(dc["valid_mask"][thickness_index, mode[0]], dtype=bool)
            dc_f = dc["frequency_hz"][thickness_index, mode[0], valid]
            dc_k = dc["kz_rad_m"][thickness_index, mode[0], valid]
            dc_g = dc["group_velocity_m_s"][thickness_index, mode[0], valid]
            band = (dc_f >= 15_000.0) & (dc_f <= 110_000.0) & np.isfinite(dc_g)
            dc_f, dc_k, dc_g = dc_f[band], dc_k[band], dc_g[band]
            supported = base[thickness_index] & (shell_k >= dc_k[0]) & (shell_k <= dc_k[-1])
            if not supported.any():
                continue
            reference_f = np.interp(shell_k[supported], dc_k, dc_f)
            reference_g = np.interp(shell_k[supported], dc_k, dc_g)
            df[thickness_index, supported] = np.abs(np.log(shell_f[thickness_index, supported] / reference_f))
            denominator = np.maximum(np.abs(shell_g[thickness_index, supported]) + np.abs(reference_g), 200.0)
            dg[thickness_index, supported] = np.abs(shell_g[thickness_index, supported] - reference_g) / denominator
        work.append({"df": df, "dg": dg, "point_cost": df + 0.25 * dg})
    point_cost = np.stack([item["point_cost"] for item in work])
    finite_count = np.isfinite(point_cost).sum(axis=0)
    ordered = np.sort(np.where(np.isfinite(point_cost), point_cost, np.inf), axis=0)
    ambiguous = np.zeros(shell_f.shape, dtype=bool)
    enough = finite_count >= 2
    ambiguous[enough] = ordered[1][enough] - ordered[0][enough] < 0.02
    local_cost = np.full((shell_f.shape[0], 3), np.inf)
    masks = []
    for mode_index, item in enumerate(work):
        mask = np.isfinite(item["point_cost"]) & ~ambiguous
        masks.append(mask)
        for thickness_index in range(shell_f.shape[0]):
            if mask[thickness_index].any():
                local_cost[thickness_index, mode_index] = float(np.median(item["point_cost"][thickness_index, mask[thickness_index]]))
    metrics = []
    for mode_index, (item, mask) in enumerate(zip(work, masks, strict=True)):
        count = int(mask.sum())
        covered = np.flatnonzero(mask.any(axis=1))
        coverage = count / max(int(base.sum()), 1)
        median_df = float(np.median(item["df"][mask])) if count else 1_000.0
        median_dg = float(np.median(item["dg"][mask])) if count else 1_000.0
        p95_df = float(np.percentile(item["df"][mask], 95)) if count else 1_000.0
        cost = median_df + 0.25 * median_dg + 0.5 * (1.0 - coverage)
        consistency = float(np.mean(np.argmin(local_cost[covered], axis=1) == mode_index)) if covered.size else 0.0
        confidence = coverage * consistency * np.exp(-cost / 0.12)
        metrics.append({
            "standard_m": mode_index + 1, "point_count": count, "thickness_count": int(covered.size),
            "coverage": coverage, "median_df": median_df, "p95_df": p95_df,
            "median_dg": median_dg, "cost": cost, "consistency": consistency,
            "pre_margin_quality": float(confidence),
        })
    return metrics


def map_shell(args: argparse.Namespace) -> None:
    for label, path in (("calibrated Shell library", args.shell_library), ("DC standard library", args.dc_library)):
        if not path.is_file():
            raise SystemExit(f"Missing {label}: {path}")
    shell, dc = load_npz(args.shell_library), load_npz(args.dc_library)
    validate_mapping_inputs(shell, dc)
    branch_ids = shell["branch_id"][0, 0].astype(int)
    branch_orders = shell["circumferential_order"][0, 0].astype(int)
    results, diagnostics = [], []
    for order in range(1, 9):
        indices = np.flatnonzero(branch_orders == order)
        candidates = [{"branch_id": int(branch_ids[index]), "scores": score_branch(shell, dc, int(index), order)} for index in indices]
        size = len(indices) + 3
        assignment_cost = np.zeros((size, size))
        assignment_cost[: len(indices), :3] = [[score["cost"] for score in row["scores"]] for row in candidates]
        assignment_cost[: len(indices), 3:] = 0.30
        assignment_cost[len(indices) :, :3] = 0.30
        assignment = tracking.linear_sum_assignment(assignment_cost)
        for mode_index in range(3):
            hits = np.flatnonzero(assignment[: len(indices)] == mode_index)
            if hits.size:
                row = int(hits[0])
                result = dict(candidates[row]["scores"][mode_index])
                other_costs = [value["cost"] for i, value in enumerate(candidates[row]["scores"]) if i != mode_index]
                result["margin"] = float(min(other_costs) - result["cost"])
                result["branch_id"] = candidates[row]["branch_id"]
                checks = {
                    "points": result["point_count"] >= GATES["min_points"],
                    "thicknesses": result["thickness_count"] >= GATES["min_thicknesses"],
                    "coverage": result["coverage"] >= GATES["min_coverage"],
                    "median_df": result["median_df"] <= GATES["max_median_df"],
                    "p95_df": result["p95_df"] <= GATES["max_p95_df"], "cost": result["cost"] <= GATES["max_cost"],
                    "margin": result["margin"] >= GATES["min_margin"],
                    "consistency": result["consistency"] >= GATES["min_consistency"],
                }
                result["confidence"] = float(result["coverage"] * result["consistency"] * np.exp(-result["cost"] / 0.12) * (1.0 - np.exp(-max(result["margin"], 0.0) / 0.05)))
                checks["confidence"] = result["confidence"] >= GATES["min_confidence"]
            else:
                result = {"standard_m": mode_index + 1, "branch_id": -1, "cost": 1_000.0, "margin": 0.0, "coverage": 0.0, "consistency": 0.0, "confidence": 0.0, "median_df": 1_000.0, "p95_df": 1_000.0, "median_dg": 1_000.0, "point_count": 0, "thickness_count": 0}
                checks = {"assignment": False}
            result.update({"n": order, "family": "F", "valid": all(checks.values()), "failed_gates": [name for name, passed in checks.items() if not passed]})
            results.append(result)
        diagnostics.append({"n": order, "candidates": candidates})
    shell_sha, dc_sha = sha256(args.shell_library), sha256(args.dc_library)
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / "standard_mode_mapping_report.json"
    output = args.output_root / "standard_mode_mapping.npz"
    passed = sum(bool(row["valid"]) for row in results)
    report = {"schema_version": 1, "published": passed == 24, "valid_mappings": passed, "required_mappings": 24, "gates": GATES, "branch_source": str(args.shell_library), "branch_source_sha256": shell_sha, "reference_source": str(args.dc_library), "reference_source_sha256": dc_sha, "mappings": results, "diagnostics": diagnostics}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if passed != 24:
        output.unlink(missing_ok=True)
        raise SystemExit(f"Standard mapping failed ({passed}/24); diagnostics: {report_path}")
    payload = {"schema_version": np.asarray(1, dtype=np.int32), "standard_mapping_ready": np.asarray(True), "branch_source_sha256": np.asarray(shell_sha), "reference_source_sha256": np.asarray(dc_sha)}
    for key in ("branch_id", "n", "standard_m", "cost", "margin", "coverage", "consistency", "confidence", "valid"):
        payload[key] = np.asarray([row[key] for row in results])
    payload["family"] = np.asarray(["F"] * len(results))
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(output)
    print(f"Standard mode mapping: {output}; valid={passed}/24")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("import-dc", help="Validate horizontal DC TXT exports and build schema v1.")
    importer.add_argument("--input-root", type=Path, default=Path(r"D:\lab_ultr\fz\dc_exports"))
    importer.add_argument("--output-root", type=Path, default=HERE / "outputs" / "standard_mode_mapping")
    importer.add_argument("--dc-executable", type=Path, default=Path(r"C:\Program Files\DispersionCalculator32\application\DC_v32_Installer.exe"))
    mapper = commands.add_parser("map-shell", help="Map calibrated Shell branches to standard F(n,m).")
    mapper.add_argument("--shell-library", type=Path, required=True)
    mapper.add_argument("--dc-library", type=Path, default=HERE / "outputs" / "standard_mode_mapping" / "dc_standard_pipe_dispersion.npz")
    mapper.add_argument("--output-root", type=Path, default=HERE / "outputs" / "standard_mode_mapping")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "import-dc":
        import_dc(args)
    elif args.command == "map-shell":
        map_shell(args)


if __name__ == "__main__":
    main()

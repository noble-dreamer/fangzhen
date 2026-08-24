"""Validate finite-pipe complex receiver convergence across shell mesh levels."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
SIMPLE_ROOT = HERE.parents[1]
F_DOMAIN = HERE.parent
for path in (SIMPLE_ROOT, F_DOMAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import frequency_domain_common as fcommon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thickness-mm", type=float, nargs="+", default=(5.0, 7.5, 10.0))
    parser.add_argument("--frequencies-khz", type=float, nargs="+", default=(20.0, 52.5, 95.0))
    parser.add_argument("--elements-per-wavelength", type=float, nargs=3, default=(6.0, 8.0, 12.0))
    parser.add_argument("--tx", type=int, default=1)
    parser.add_argument("--max-complex-relative", type=float, default=0.05)
    parser.add_argument("--max-phase-rmse-rad", type=float, default=0.05)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--delete-case-outputs", action="store_true")
    parser.add_argument("--heartbeat-s", type=float, default=120.0)
    parser.add_argument("--cores", type=int, default=None)
    fcommon.add_frequency_solver_arguments(parser)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if any(not 1.0 <= value <= 10.0 for value in args.thickness_mm):
        raise ValueError("Thickness values must be in [1,10] mm")
    if np.any(np.diff(args.thickness_mm) <= 0.0) or np.any(np.diff(args.frequencies_khz) <= 0.0):
        raise ValueError("Thickness and frequency axes must be strictly increasing")
    if np.any(np.diff(args.elements_per_wavelength) <= 0.0) or min(args.elements_per_wavelength) < 4.0:
        raise ValueError("Mesh levels must be three increasing values >= 4")
    if not 1 <= args.tx <= 16:
        raise ValueError("--tx must be in [1,16]")


@contextmanager
def uniform_remaining_thickness(thickness_mm: float):
    original = fcommon.shell.thickness_expression
    fcommon.shell.thickness_expression = lambda _defects, _lobes: f"{thickness_mm:.12g}[mm]"
    try:
        yield
    finally:
        fcommon.shell.thickness_expression = original


def load_response(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        response = np.asarray(data["H_real"]) + 1j * np.asarray(data["H_imag"])
        completed = np.asarray(data["completed_mask"], dtype=bool)
    if response.shape[0:2] != (1, 16) or not completed.all() or not np.isfinite(response).all():
        raise RuntimeError(f"Invalid validation response: {path}")
    return response[0]


def assert_no_model_problems(path: Path) -> None:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    cases = metadata.get("model", {}).get("problems", {}).get("case_problems", [])
    for case in cases:
        if case.get("build_problems") or case.get("post_solve_problems") or case.get("nonzero_channels") != 16:
            raise RuntimeError(f"COMSOL problem or missing receiver in {path}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    frequencies = tuple(1000.0 * np.asarray(args.frequencies_khz, dtype=float))
    cases = fcommon.make_cases((args.tx,), frequencies)
    fcommon.configure_dataset_a_frequency(use_parametric_sweep=False)
    fcommon.apply_solver_arguments(args)
    cases_root = args.output_root / "cases"
    responses: dict[tuple[float, float], np.ndarray] = {}
    client = fcommon.shell.start_client(cores=args.cores)
    try:
        for thickness in args.thickness_mm:
            for epw in args.elements_per_wavelength:
                sample_id = f"uniform_h{thickness:g}mm_epw{epw:g}".replace(".", "p")
                fcommon.shell.MESH = replace(
                    fcommon.shell.MESH,
                    max_frequency_hz=max(frequencies),
                    elements_per_wavelength=float(epw),
                )
                with uniform_remaining_thickness(float(thickness)):
                    result = fcommon.solve_export_frequency_sample(
                        client=client,
                        dataset="A_frequency_uniform_mesh_validation",
                        sample_id=sample_id,
                        defect_state=f"uniform_remaining_thickness_{thickness:g}mm",
                        output_root=cases_root,
                        cases=cases,
                        defects=[],
                        lobes=[],
                        sample_metadata={"defects": [], "lobes": [], "remaining_thickness_mm": float(thickness), "elements_per_wavelength": float(epw)},
                        clear_each_case=True,
                        heartbeat_s=args.heartbeat_s,
                        reuse_sample_model=True,
                        write_label_preview=False,
                        keep_case_csv=False,
                        checkpoint_every_cases=len(cases),
                    )
                assert_no_model_problems(result.metadata_path)
                response_path = cases_root / "frequency_response" / f"{sample_id}_H_complex.npz"
                responses[(float(thickness), float(epw))] = load_response(response_path)
    finally:
        client.clear()

    medium, fine = map(float, args.elements_per_wavelength[-2:])
    metrics = []
    for thickness in map(float, args.thickness_mm):
        medium_h = responses[(thickness, medium)]
        fine_h = responses[(thickness, fine)]
        for frequency_index, frequency_hz in enumerate(frequencies):
            reference = fine_h[:, frequency_index]
            current = medium_h[:, frequency_index]
            relative = float(np.linalg.norm(current - reference) / max(np.linalg.norm(reference), 1e-30))
            phase_rmse = float(np.sqrt(np.mean(np.angle(current * np.conj(reference)) ** 2)))
            metrics.append({
                "thickness_mm": thickness,
                "frequency_hz": frequency_hz,
                "medium_epw": medium,
                "fine_epw": fine,
                "complex_relative_l2": relative,
                "phase_rmse_rad": phase_rmse,
                "passed": relative <= args.max_complex_relative and phase_rmse <= args.max_phase_rmse_rad,
            })
    passed = all(row["passed"] for row in metrics)
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / "finite_pipe_mesh_validation.json"
    report = {
        "schema_version": 1,
        "validation_scope": "finite-pipe patch-averaged axial receiver mesh convergence",
        "scientific_ready": False,
        "passed": passed,
        "thresholds": {"complex_relative_l2": args.max_complex_relative, "phase_rmse_rad": args.max_phase_rmse_rad},
        "mesh_elements_per_wavelength": list(map(float, args.elements_per_wavelength)),
        "metrics": metrics,
        "note": "Passing this mesh gate does not replace field-spectrum or 3D calibration.",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if passed and args.delete_case_outputs:
        shutil.rmtree(cases_root)
    print(f"Mesh validation passed={passed}: {report_path}")
    if not passed:
        raise SystemExit("Finite-pipe mesh convergence thresholds were not met")


if __name__ == "__main__":
    main()

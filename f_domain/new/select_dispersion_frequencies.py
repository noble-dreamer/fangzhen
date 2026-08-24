"""Select a new frequency set from current irregular pilot responses and dispersion."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from irregular_defect_common import GENERATOR_NAME

HERE = Path(__file__).resolve().parent
GENERATOR = GENERATOR_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--dispersion-library", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs" / "irregular_frequency_selection")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--band-quotas", type=int, nargs=3, default=(3, 4, 5))
    parser.add_argument("--band-edges-khz", type=float, nargs=2, default=(40.0, 70.0))
    parser.add_argument("--min-spacing-khz", type=float, default=2.5)
    parser.add_argument("--min-observability", type=float, default=0.05)
    parser.add_argument("--min-dispersion-coverage", type=float, default=2.0 / 3.0)
    parser.add_argument("--residual-threshold", type=float, default=1.1)
    parser.add_argument("--allow-unvalidated-dispersion", action="store_true", help="Code-smoke only.")
    return parser.parse_args()

def load_response(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {"H_real", "H_imag", "completed_mask", "frequencies_hz", "tx_indices", "rx_indices"}
        missing = required.difference(data.files)
        if missing:
            raise RuntimeError(f"{path} missing {sorted(missing)}")
        response = np.asarray(data["H_real"]) + 1j * np.asarray(data["H_imag"])
        completed = np.asarray(data["completed_mask"], dtype=bool)
        frequencies = np.asarray(data["frequencies_hz"], dtype=float)
        tx = np.asarray(data["tx_indices"], dtype=int)
        rx = np.asarray(data["rx_indices"], dtype=int)
    if response.shape != (tx.size, rx.size, frequencies.size) or completed.shape != (tx.size, frequencies.size):
        raise RuntimeError(f"Invalid response schema in {path}")
    if not completed.all() or not np.isfinite(response).all() or np.any(np.diff(frequencies) <= 0.0):
        raise RuntimeError(f"Incomplete, non-finite, or unordered response in {path}")
    return frequencies, response, tx, rx

def sample_class(root: Path, response_path: Path) -> tuple[str, str]:
    sample_id = response_path.name.removesuffix("_H_complex.npz")
    metadata = json.loads((root / "metadata" / f"{sample_id}.json").read_text(encoding="utf-8"))
    defect = metadata.get("sample", {}).get("irregular_defect", {})
    if defect.get("generator") != GENERATOR:
        raise RuntimeError(f"{sample_id} is not {GENERATOR}")
    size_class = str(defect.get("size_class", ""))
    if size_class not in {"small", "medium", "large"}:
        raise RuntimeError(f"{sample_id} has invalid size_class={size_class!r}")
    texture_sha = str(defect.get("texture_sha256", ""))
    if len(texture_sha) != 64:
        raise RuntimeError(f"{sample_id} has invalid texture SHA-256")
    return size_class, texture_sha

def rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(np.argsort(np.nan_to_num(values, nan=-np.inf)))
    return order.astype(float) / max(values.size - 1, 1)


def pilot_metrics(root: Path) -> tuple[np.ndarray, list[dict[str, float]], dict[str, int], str]:
    response_root = root / "frequency_response"
    healthy_paths = list(response_root.glob("*irregular_healthy_H_complex.npz"))
    if len(healthy_paths) != 1:
        raise RuntimeError(f"Expected one irregular healthy response, found {len(healthy_paths)}")
    frequencies, healthy, tx, rx = load_response(healthy_paths[0])
    grouped: dict[str, list[np.ndarray]] = {name: [] for name in ("small", "medium", "large")}
    texture_hashes: set[str] = set()
    phase_rows, participation_rows, all_rows = [], [], []
    for path in sorted(response_root.glob("*irregular_sample_*_H_complex.npz")):
        current_f, damaged, current_tx, current_rx = load_response(path)
        if not (np.array_equal(frequencies, current_f) and np.array_equal(tx, current_tx) and np.array_equal(rx, current_rx)):
            raise RuntimeError(f"Pilot axes do not match healthy: {path}")
        delta = damaged - healthy
        relative = np.sqrt(np.mean(np.abs(delta) ** 2, axis=(0, 1))) / np.sqrt(
            np.mean(np.abs(healthy) ** 2, axis=(0, 1)) + 1e-30
        )
        path_signal = np.abs(delta) / (np.abs(healthy) + 1e-30)
        participation = path_signal.sum(axis=(0, 1)) ** 2 / (
            tx.size * rx.size * np.sum(path_signal**2, axis=(0, 1)) + 1e-30
        )
        size_class, texture_sha = sample_class(root, path)
        grouped[size_class].append(relative)
        texture_hashes.add(texture_sha)
        all_rows.append(relative)
        phase_rows.append(np.mean(np.abs(np.angle(damaged * np.conj(healthy))), axis=(0, 1)))
        participation_rows.append(participation)
    counts = {name: len(values) for name, values in grouped.items()}
    if any(count == 0 for count in counts.values()):
        raise RuntimeError(f"Pilot must include small, medium, and large defects: {counts}")
    if len(texture_hashes) != 1:
        raise RuntimeError(f"Pilot must use one texture image SHA-256, found {sorted(texture_hashes)}")
    class_medians = np.stack([np.median(grouped[name], axis=0) for name in grouped])
    balanced = np.exp(np.mean(np.log(np.maximum(class_medians, 1e-30)), axis=0))
    sample_values = np.stack(all_rows)
    stability = sample_values.mean(axis=0) / (sample_values.mean(axis=0) + sample_values.std(axis=0) + 1e-30)
    health = np.median(np.abs(healthy), axis=(0, 1))
    rows = [
        {
            "frequency_hz": float(frequency),
            "balanced_relative_l2": float(balanced[index]),
            "sample_stability": float(stability[index]),
            "phase_change": float(np.median(phase_rows, axis=0)[index]),
            "participation": float(np.median(participation_rows, axis=0)[index]),
            "healthy_median_abs": float(health[index]),
        }
        for index, frequency in enumerate(frequencies)
    ]
    return frequencies, rows, counts, next(iter(texture_hashes))


def dispersion_metrics(path: Path, targets: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if int(data["schema_version"]) != 3 or not np.asarray(data["completed_mask"]).all():
            raise RuntimeError("Dispersion library must be complete schema v3")
        ready = bool(data["scientific_ready"]) if "scientific_ready" in data.files else False
        scope = str(data["readiness_scope"]) if "readiness_scope" in data.files else ""
        if not (ready and scope == "shell_proxy_simulation_only") and not args.allow_unvalidated_dispersion:
            raise RuntimeError("Dispersion library lacks the scientific-ready Shell-proxy scope")
        frequency = np.asarray(data["frequency_hz"], dtype=float)
        observability = np.asarray(data["observability"], dtype=float)
        confidence = np.asarray(data["tracking_confidence"], dtype=float)
        residual = np.asarray(data["tracking_residual"], dtype=float)
        dk_dh = np.asarray(data["dk_dh_rad_m_per_mm"], dtype=float)
        derivative_valid = np.asarray(data["derivative_valid_mask"], dtype=bool)
        order = np.asarray(data["circumferential_order"], dtype=int)
    valid = derivative_valid & np.isfinite(frequency) & np.isfinite(dk_dh)
    valid &= (observability >= args.min_observability) & (residual <= args.residual_threshold) & (order > 0)
    qualities, coverages = [], []
    for target in targets:
        per_thickness = []
        for h_index in range(frequency.shape[0]):
            candidates = []
            for branch in range(frequency.shape[2]):
                for k_index in range(frequency.shape[1] - 1):
                    edge = (h_index, slice(k_index, k_index + 2), branch)
                    f_edge = frequency[edge]
                    if valid[edge].all() and f_edge[0] != f_edge[1] and np.prod(f_edge - target) <= 0.0:
                        alpha = float((target - f_edge[0]) / (f_edge[1] - f_edge[0]))
                        quality_edge = np.abs(dk_dh[edge]) * observability[edge] * confidence[edge]
                        candidates.append(float((1.0 - alpha) * quality_edge[0] + alpha * quality_edge[1]))
            if candidates:
                per_thickness.append(max(candidates))
        coverages.append(len(per_thickness) / frequency.shape[0])
        qualities.append(float(np.median(per_thickness)) if per_thickness else 0.0)
    return np.asarray(qualities), np.asarray(coverages)


def choose(rows: list[dict[str, float]], args: argparse.Namespace) -> list[dict[str, float]]:
    if sum(args.band_quotas) != args.count:
        raise ValueError("--band-quotas must sum to --count")
    edges = [float("-inf"), *(1000.0 * np.asarray(args.band_edges_khz)), float("inf")]
    chosen: list[dict[str, float]] = []
    spacing = args.min_spacing_khz * 1000.0
    for band, quota in enumerate(args.band_quotas):
        candidates = sorted(
            (row for row in rows if row["eligible"] and edges[band] < row["frequency_hz"] <= edges[band + 1]),
            key=lambda row: (-row["score"], row["frequency_hz"]),
        )
        for row in candidates:
            if all(abs(row["frequency_hz"] - item["frequency_hz"]) >= spacing for item in chosen):
                chosen.append(row)
                if sum(edges[band] < item["frequency_hz"] <= edges[band + 1] for item in chosen) == quota:
                    break
        if sum(edges[band] < item["frequency_hz"] <= edges[band + 1] for item in chosen) != quota:
            raise RuntimeError(f"Insufficient eligible frequencies for band {band}")
    return sorted(chosen, key=lambda row: row["frequency_hz"])


def main() -> None:
    args = parse_args()
    frequencies, rows, counts, texture_sha = pilot_metrics(args.pilot_root)
    dispersion, coverage = dispersion_metrics(args.dispersion_library, frequencies, args)
    health = np.asarray([row["healthy_median_abs"] for row in rows])
    health_floor = float(np.quantile(health, 0.05))
    components = [
        rank01(np.log1p([row["balanced_relative_l2"] for row in rows])),
        rank01([row["sample_stability"] for row in rows]),
        rank01([row["phase_change"] for row in rows]),
        rank01([row["participation"] for row in rows]),
        rank01(np.log1p(dispersion)),
    ]
    for index, row in enumerate(rows):
        row.update(
            dispersion_quality=float(dispersion[index]),
            dispersion_coverage=float(coverage[index]),
            score=float(0.40 * components[0][index] + 0.15 * components[1][index] + 0.10 * components[2][index] + 0.10 * components[3][index] + 0.20 * components[4][index] + 0.05 * coverage[index]),
            eligible=bool(health[index] >= health_floor and coverage[index] >= args.min_dispersion_coverage),
        )
    selected = choose(rows, args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (args.output_root / "frequency_ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: -row["score"]))
    selected_hz = [int(round(row["frequency_hz"])) for row in selected]
    (args.output_root / "selected_frequencies.txt").write_text(",".join(map(str, selected_hz)) + "\n", encoding="utf-8")
    summary = {"generator": GENERATOR, "texture_sha256": texture_sha, "size_class_counts": counts, "selected_frequencies_hz": selected_hz, "pilot_root": str(args.pilot_root), "dispersion_library": str(args.dispersion_library)}
    (args.output_root / "frequency_selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Selected frequencies (Hz): {','.join(map(str, selected_hz))}")


if __name__ == "__main__":
    main()

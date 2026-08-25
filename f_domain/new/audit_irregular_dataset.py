"""Audit current irregular worker outputs and write the minimal EDM transfer list."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from irregular_defect_common import GENERATOR_NAME


GENERATOR = GENERATOR_NAME
MODEL_KEYS = (
    "pipe", "transducer", "material", "absorbing_layer", "defect_model", "solver", "mesh", "sweep",
    "receiver_indices", "receiver_model", "actuation_model", "analysis_type", "frequency_domain",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True, help="Non-overlapping defect worker roots.")
    parser.add_argument("--healthy-root", type=Path, required=True)
    parser.add_argument("--frequencies-file", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int)
    parser.add_argument("--shared-files", type=Path, nargs="*", default=())
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def selected_frequencies(path: Path) -> np.ndarray:
    tokens = path.read_text(encoding="utf-8").replace(",", " ").split()
    values = np.asarray([float(token) for token in tokens], dtype=float)
    if values.size == 0 or np.any(np.diff(values) <= 0.0) or np.unique(values).size != values.size:
        raise RuntimeError("Selected frequencies must be nonempty, unique, and increasing")
    return values


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def model_fingerprint(metadata: dict) -> tuple[str, dict]:
    model = metadata.get("model", {})
    missing = [key for key in MODEL_KEYS if key not in model]
    if missing:
        raise RuntimeError(f"Metadata model contract missing {missing}")
    contract = {key: model[key] for key in MODEL_KEYS}
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), contract


def audit_response(path: Path, expected_frequencies: np.ndarray) -> None:
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
    expected_shape = (16, 16, expected_frequencies.size)
    if response.shape != expected_shape or completed.shape != (16, expected_frequencies.size):
        raise RuntimeError(f"{path} has response/mask shape {response.shape}/{completed.shape}, expected {expected_shape}")
    if not np.array_equal(tx, np.arange(1, 17)) or not np.array_equal(rx, np.arange(17, 33)):
        raise RuntimeError(f"{path} must contain ordered TX 1..16 and RX 17..32")
    if not np.array_equal(frequencies, expected_frequencies) or not completed.all() or not np.isfinite(response).all():
        raise RuntimeError(f"{path} has mismatched frequency axis, incomplete mask, or non-finite response")
    if not ((np.abs(response) > 0.0).sum(axis=1) == 16).all():
        raise RuntimeError(f"{path} contains a completed case without 16 nonzero receivers")


def label_files(root: Path, sample_id: str) -> list[Path]:
    label_root = root / "labels"
    return [
        label_root / f"{sample_id}_defect_depth_mm.npy",
        label_root / f"{sample_id}_defect_depth_norm.npy",
        label_root / f"{sample_id}_defect_mask.npy",
        label_root / f"{sample_id}_defect_label_metadata.json",
    ]


def audit_label(paths: list[Path], defect: dict) -> None:
    if any(not path.exists() for path in paths):
        raise FileNotFoundError(next(path for path in paths if not path.exists()))
    depth_mm = np.asarray(np.load(paths[0]), dtype=float)
    depth_norm = np.asarray(np.load(paths[1]), dtype=float)
    mask = np.asarray(np.load(paths[2]))
    metadata = load_json(paths[3])
    if depth_mm.shape != (512, 512) or depth_norm.shape != depth_mm.shape or mask.shape != depth_mm.shape:
        raise RuntimeError(f"Invalid label shape for {paths[0]}")
    if not np.isfinite(depth_mm).all() or not np.isfinite(depth_norm).all():
        raise RuntimeError(f"Non-finite label values for {paths[0]}")
    if depth_mm.min() < 0.0 or depth_mm.max() > 5.0 + 1e-5:
        raise RuntimeError(f"Label depth outside [0,5] mm for {paths[0]}")
    if not np.allclose(depth_norm, depth_mm / 9.0, atol=1e-6) or not np.array_equal(mask > 0, depth_mm >= 0.01):
        raise RuntimeError(f"Millimeter, normalized, and mask labels disagree for {paths[0]}")
    if metadata.get("generator") != GENERATOR or metadata.get("size_class") != defect.get("size_class"):
        raise RuntimeError(f"Label metadata generator/size class mismatch for {paths[0]}")
    provenance = ("texture_source", "texture_sha256", "texture_rotation_deg")
    if any(metadata.get(key) != defect.get(key) for key in provenance):
        raise RuntimeError(f"Label/simulation texture provenance mismatch for {paths[0]}")


def relative_to_repo(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError as error:
        raise RuntimeError(f"Transfer path is outside repository: {path}") from error


def main() -> None:
    args = parse_args()
    frequencies = selected_frequencies(args.frequencies_file)
    healthy_id = "dataset_a_frequency_irregular_healthy"
    healthy_response = args.healthy_root / "frequency_response" / f"{healthy_id}_H_complex.npz"
    healthy_metadata_path = args.healthy_root / "metadata" / f"{healthy_id}.json"
    audit_response(healthy_response, frequencies)
    healthy_metadata = load_json(healthy_metadata_path)
    reference_fingerprint, contract = model_fingerprint(healthy_metadata)
    rows, transfer, seen = [], [healthy_response, healthy_metadata_path, args.frequencies_file], set()
    counts = {name: 0 for name in ("small", "medium", "large")}
    texture_hashes: set[str] = set()
    for root in args.roots:
        for response_path in sorted((root / "frequency_response").glob("*irregular_sample_*_H_complex.npz")):
            sample_id = response_path.name.removesuffix("_H_complex.npz")
            if sample_id in seen:
                raise RuntimeError(f"Duplicate sample ID across workers: {sample_id}")
            seen.add(sample_id)
            metadata_path = root / "metadata" / f"{sample_id}.json"
            metadata = load_json(metadata_path)
            defect = metadata.get("sample", {}).get("irregular_defect", {})
            size_class = str(defect.get("size_class", ""))
            if defect.get("generator") != GENERATOR or size_class not in counts:
                raise RuntimeError(f"{sample_id} is not a valid {GENERATOR} sample")
            fingerprint, _ = model_fingerprint(metadata)
            if fingerprint != reference_fingerprint:
                raise RuntimeError(f"Healthy/damaged model fingerprint mismatch: {sample_id}")
            audit_response(response_path, frequencies)
            labels = label_files(root, sample_id)
            audit_label(labels, defect)
            texture_hashes.add(str(defect.get("texture_sha256", "")))
            counts[size_class] += 1
            transfer.extend([response_path, metadata_path, *labels])
            rows.append({"sample_id": sample_id, "size_class": size_class, "worker_root": str(root), "frequency_count": frequencies.size})
    if args.expected_samples is not None and len(rows) != args.expected_samples:
        raise RuntimeError(f"Expected {args.expected_samples} samples, found {len(rows)}")
    if not rows or any(count == 0 for count in counts.values()):
        raise RuntimeError(f"Dataset must contain all three size classes: {counts}")
    if len(texture_hashes) != 1 or len(next(iter(texture_hashes), "")) != 64:
        raise RuntimeError(f"Dataset must use one valid texture image SHA-256: {sorted(texture_hashes)}")
    for path in args.shared_files:
        if not path.exists():
            raise FileNotFoundError(path)
        transfer.append(path)
    args.output_root.mkdir(parents=True, exist_ok=True)
    audit_csv = args.output_root / "dataset_audit.csv"
    with audit_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = args.output_root / "dataset_audit_summary.json"
    summary = {"status": "passed", "sample_count": len(rows), "size_class_counts": counts, "texture_sha256": next(iter(texture_hashes)), "frequencies_hz": frequencies.tolist(), "model_fingerprint_sha256": reference_fingerprint, "model_contract": contract}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    transfer_path = args.output_root / "transfer_manifest.txt"
    transfer.extend([audit_csv, summary_path, transfer_path])
    transfer_path.write_text("\n".join(sorted({relative_to_repo(path) for path in transfer})) + "\n", encoding="utf-8")
    print(f"Audit passed: {len(rows)} samples; transfer list: {transfer_path}")


if __name__ == "__main__":
    main()

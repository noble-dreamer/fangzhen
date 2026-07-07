"""Build an output_dataset tree by subsetting an existing frequency dataset.

This is intended for turning a full pilot sweep, for example output2, into a
top-N training dataset with the same directory structure.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = ROOT / "output2" / "streaming_dataset_a_frequency_shell"
DEFAULT_DEST_ROOT = ROOT / "output_dataset" / "streaming_dataset_a_frequency_shell"
DEFAULT_SELECTION_ROOT = ROOT / "output2" / "frequency_selection_physics_highfreq_quota_40samples"
DEFAULT_FREQUENCY_FILE = DEFAULT_SELECTION_ROOT / "physics_highfreq_quota_top15_frequencies.txt"
DEFAULT_DEST_SELECTION_ROOT = ROOT / "output_dataset" / "frequency_selection_physics_highfreq_quota_40samples"
DEFAULT_DEST_LEGACY_SELECTION_ROOT = ROOT / "output_dataset" / "frequency_selection"
SAMPLE_PATTERN = re.compile(r"dataset_a_frequency_sample_(\d{4})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy an existing streaming frequency dataset and keep only selected frequencies."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--dest-root", type=Path, default=DEFAULT_DEST_ROOT)
    parser.add_argument("--frequency-file", type=Path, default=DEFAULT_FREQUENCY_FILE)
    parser.add_argument(
        "--selection-source",
        type=Path,
        default=DEFAULT_SELECTION_ROOT,
        help="Frequency-selection result directory to copy beside the new dataset.",
    )
    parser.add_argument("--selection-dest", type=Path, default=DEFAULT_DEST_SELECTION_ROOT)
    parser.add_argument(
        "--legacy-selection-dest",
        type=Path,
        default=DEFAULT_DEST_LEGACY_SELECTION_ROOT,
        help="Compatibility directory for DEFAULT_SELECTION_TXT used by get_pic.",
    )
    parser.add_argument("--tolerance-hz", type=float, default=1e-6)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files in the destination tree. Existing files are not deleted.",
    )
    parser.add_argument(
        "--skip-progress",
        action="store_true",
        help="Do not copy progress JSONL files from the source tree.",
    )
    return parser.parse_args()


def parse_frequency_values(text: str) -> list[float]:
    values: list[float] = []
    for token in text.replace("\n", ",").replace(";", ",").split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    return values


def relative_to(path: Path, root: Path) -> Path:
    return path.resolve().relative_to(root.resolve())


def copy_file(src: Path, dst: Path, *, overwrite: bool, replacements: dict[str, str] | None = None) -> bool:
    if dst.exists() and not overwrite:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if replacements and src.suffix.lower() in {".json", ".jsonl", ".csv", ".md", ".txt"}:
        text = src.read_text(encoding="utf-8")
        for old, new in replacements.items():
            text = text.replace(old, new)
        dst.write_text(text, encoding="utf-8")
    else:
        shutil.copy2(src, dst)
    return True


def copy_tree_files(
    src_root: Path,
    dst_root: Path,
    *,
    overwrite: bool,
    replacements: dict[str, str] | None = None,
) -> tuple[int, int]:
    copied = 0
    skipped = 0
    if not src_root.exists():
        return copied, skipped
    for src in src_root.rglob("*"):
        if not src.is_file():
            continue
        dst = dst_root / relative_to(src, src_root)
        if copy_file(src, dst, overwrite=overwrite, replacements=replacements):
            copied += 1
        else:
            skipped += 1
    return copied, skipped


def scalar_string(data: np.lib.npyio.NpzFile, key: str, fallback: str) -> str:
    if key not in data.files:
        return fallback
    value = np.asarray(data[key])
    if value.shape == ():
        return str(value.item())
    return str(value.tolist())


def subset_npz(src: Path, dst: Path, requested: list[float], *, tolerance_hz: float, overwrite: bool) -> dict[str, Any]:
    if dst.exists() and not overwrite:
        with np.load(dst, allow_pickle=False) as data:
            frequencies = tuple(float(item) for item in np.asarray(data["frequencies_hz"]).tolist())
            completed = np.asarray(data["completed_mask"], dtype=bool)
        return {
            "source": str(src),
            "destination": str(dst),
            "status": "skipped_existing",
            "frequency_count": len(frequencies),
            "completed_cases": int(np.sum(completed)),
            "case_count": int(completed.size),
        }

    with np.load(src, allow_pickle=False) as data:
        required = {"H_real", "H_imag", "completed_mask", "tx_indices", "rx_indices", "frequencies_hz"}
        missing = required.difference(data.files)
        if missing:
            raise RuntimeError(f"{src} missing fields: {sorted(missing)}")
        frequencies = np.asarray(data["frequencies_hz"], dtype=float)
        indices: list[int] = []
        missing_freqs: list[float] = []
        for value in requested:
            matches = np.where(np.abs(frequencies - value) <= tolerance_hz)[0]
            if matches.size == 0:
                missing_freqs.append(value)
            else:
                indices.append(int(matches[0]))
        if missing_freqs:
            raise RuntimeError(
                f"{src} does not contain requested frequencies: "
                + ",".join(f"{value:g}" for value in missing_freqs)
            )
        if len(set(indices)) != len(indices):
            raise RuntimeError("Requested frequency list contains duplicates after source matching.")

        h_real = np.asarray(data["H_real"])[:, :, indices]
        h_imag = np.asarray(data["H_imag"])[:, :, indices]
        completed = np.asarray(data["completed_mask"], dtype=bool)[:, indices]
        sample_id = scalar_string(data, "sample_id", dst.stem.replace("_H_complex", ""))
        dataset = scalar_string(data, "dataset", "")
        defect_state = scalar_string(data, "defect_state", "")
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dst,
            H_real=h_real,
            H_imag=h_imag,
            completed_mask=completed,
            tx_indices=np.asarray(data["tx_indices"], dtype=np.int32),
            rx_indices=np.asarray(data["rx_indices"], dtype=np.int32),
            frequencies_hz=frequencies[indices],
            sample_id=np.asarray(sample_id),
            dataset=np.asarray(dataset),
            defect_state=np.asarray(defect_state),
        )
    return {
        "source": str(src),
        "destination": str(dst),
        "status": "written",
        "frequency_count": len(requested),
        "completed_cases": int(np.sum(completed)),
        "case_count": int(completed.size),
    }


def response_destination_name(src: Path) -> str | None:
    return src.name


def subset_all_npz(
    source_response_dir: Path,
    dest_response_dir: Path,
    requested: list[float],
    *,
    tolerance_hz: float,
    overwrite: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for src in sorted(source_response_dir.glob("*_H_complex.npz")):
        dst_name = response_destination_name(src)
        if dst_name is None:
            continue
        dst = dest_response_dir / dst_name
        rows.append(subset_npz(src, dst, requested, tolerance_hz=tolerance_hz, overwrite=overwrite))
    return rows


def frequency_is_selected(value: str, selected: list[float], tolerance_hz: float) -> bool:
    try:
        numeric = float(value)
    except ValueError:
        return False
    return any(abs(numeric - item) <= tolerance_hz for item in selected)


def subset_csv(src: Path, dst: Path, selected: list[float], *, tolerance_hz: float, overwrite: bool) -> dict[str, Any]:
    if dst.exists() and not overwrite:
        return {"source": str(src), "destination": str(dst), "status": "skipped_existing", "rows": None}
    dst.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with src.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        if not reader.fieldnames:
            return {"source": str(src), "destination": str(dst), "status": "empty", "rows": 0}
        if "frequency_hz" not in reader.fieldnames:
            shutil.copy2(src, dst)
            return {"source": str(src), "destination": str(dst), "status": "copied_no_frequency_column", "rows": None}
        with dst.open("w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if frequency_is_selected(row.get("frequency_hz", ""), selected, tolerance_hz):
                    writer.writerow(row)
                    rows_written += 1
    return {"source": str(src), "destination": str(dst), "status": "written", "rows": rows_written}


def subset_all_csv(
    source_csv_dir: Path,
    dest_csv_dir: Path,
    selected: list[float],
    *,
    tolerance_hz: float,
    overwrite: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not source_csv_dir.exists():
        return rows
    for src in sorted(source_csv_dir.glob("*.csv")):
        if "_tx" in src.stem and "_f" in src.stem:
            match = re.search(r"_f(\d+(?:\.\d+)?)Hz", src.stem)
            if not match or not frequency_is_selected(match.group(1), selected, tolerance_hz):
                continue
            dst = dest_csv_dir / src.name
            if copy_file(src, dst, overwrite=overwrite):
                rows.append({"source": str(src), "destination": str(dst), "status": "copied_case_csv", "rows": None})
            else:
                rows.append({"source": str(src), "destination": str(dst), "status": "skipped_existing", "rows": None})
        else:
            rows.append(subset_csv(src, dest_csv_dir / src.name, selected, tolerance_hz=tolerance_hz, overwrite=overwrite))
    return rows


def replace_paths(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [replace_paths(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: replace_paths(item, replacements) for key, item in value.items()}
    return value


def update_metadata(
    metadata_dir: Path,
    requested: list[float],
    *,
    replacements: dict[str, str],
    source_root: Path,
    overwrite: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(metadata_dir.glob("dataset_a_frequency_*.json")):
        if path.exists() and not overwrite:
            # The file has already been copied. Still rewrite frequency_export to
            # keep metadata coherent with the subset dataset.
            pass
        data = json.loads(path.read_text(encoding="utf-8"))
        data = replace_paths(data, replacements)
        tx = list(range(1, 17))
        with np.load(path.parent.parent / "frequency_response" / f"{path.stem}_H_complex.npz", allow_pickle=False) as response:
            tx = [int(item) for item in np.asarray(response["tx_indices"]).tolist()]
            completed = np.asarray(response["completed_mask"], dtype=bool)
        case_count = len(tx) * len(requested)
        case_problems = (
            data.get("model", {})
            .get("problems", {})
            .get("case_problems", [])
        )
        if isinstance(case_problems, list):
            selected_cases = []
            for item in case_problems:
                if isinstance(item, dict) and frequency_is_selected(str(item.get("frequency_hz", "")), requested, 1e-6):
                    selected_cases.append(item)
            data["model"]["problems"]["case_problems"] = selected_cases
        export = data.setdefault("frequency_export", {})
        subset_source_npz = source_root / "frequency_response" / f"{path.stem}_H_complex.npz"
        subset_source_csv = source_root / "csv" / "frequency_response" / f"{path.stem}_frequency_response.csv"
        export["case_count"] = case_count
        export["tx"] = [tx_item for tx_item in tx for _ in requested]
        export["frequencies_hz"] = [freq for _ in tx for freq in requested]
        export["frequency_axis_hz"] = list(requested)
        export["subset_source_npz"] = str(subset_source_npz)
        export["subset_source_csv"] = str(subset_source_csv)
        export["complex_response_npz"] = str(path.parent.parent / "frequency_response" / f"{path.stem}_H_complex.npz")
        export["cumulative_response_csv"] = str(path.parent.parent / "csv" / "frequency_response" / f"{path.stem}_frequency_response.csv")
        export["response_files"] = []
        export["note"] = (
            "Frequency-domain response was subset from the pilot sweep to the selected "
            "physics_highfreq_quota frequency set."
        )
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        rows.append({
            "metadata": str(path),
            "sample_id": path.stem,
            "case_count": case_count,
            "completed_cases": int(np.sum(completed)),
        })
    return rows


def write_manifest(dest_root: Path, metadata_rows: list[dict[str, Any]]) -> Path:
    rows: list[dict[str, Any]] = []
    for item in sorted(metadata_rows, key=lambda row: row["sample_id"]):
        metadata_path = Path(item["metadata"])
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        sample = data.get("sample", {})
        defects = sample.get("defects", []) if isinstance(sample, dict) else []
        lobes = sample.get("lobes", []) if isinstance(sample, dict) else []
        seed = sample.get("seed") if isinstance(sample, dict) else None
        rows.append({
            "sample_id": item["sample_id"],
            "dataset": data.get("dataset", "A_frequency"),
            "defect_state": data.get("defect_state", ""),
            "seed": "" if seed is None else seed,
            "case_count": item["case_count"],
            "response_file_count": 0,
            "feature_file_count": 2,
            "defect_count": len(defects) if isinstance(defects, list) else 0,
            "lobe_count": len(lobes) if isinstance(lobes, list) else 0,
            "metadata": str(metadata_path),
            "saved_mph": False,
            "analysis_type": "frequency_domain",
            "status": "subset_from_output2",
            "note": "top15 physics_highfreq_quota subset",
        })
    path = dest_root / "manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return path


def write_summary(dest_root: Path, summary: dict[str, Any]) -> Path:
    path = dest_root / "frequency_subset_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def write_selection_compatibility_files(args: argparse.Namespace, *, overwrite: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if args.selection_source.exists():
        copied, skipped = copy_tree_files(args.selection_source, args.selection_dest, overwrite=overwrite)
        result["selection_copy"] = {"copied": copied, "skipped": skipped, "dest": str(args.selection_dest)}
    args.legacy_selection_dest.mkdir(parents=True, exist_ok=True)
    source_txt = args.selection_source / "physics_highfreq_quota_top15_frequencies.txt"
    source_csv = args.selection_source / "physics_highfreq_quota_top15.csv"
    if source_txt.exists():
        copy_file(
            source_txt,
            args.legacy_selection_dest / "frequency_sensitivity_top15_frequencies.txt",
            overwrite=overwrite,
        )
        copy_file(
            source_txt,
            args.legacy_selection_dest / "physics_highfreq_quota_top15_frequencies.txt",
            overwrite=overwrite,
        )
    if source_csv.exists():
        copy_file(source_csv, args.legacy_selection_dest / "frequency_sensitivity_top15.csv", overwrite=overwrite)
        copy_file(source_csv, args.legacy_selection_dest / "physics_highfreq_quota_top15.csv", overwrite=overwrite)
    result["legacy_selection_dest"] = str(args.legacy_selection_dest)
    return result


def copy_auxiliary_output_root(
    source_output_root: Path,
    dest_output_root: Path,
    *,
    excluded_names: set[str],
    overwrite: bool,
    replacements: dict[str, str],
) -> dict[str, Any]:
    result: dict[str, Any] = {"directories": [], "files": []}
    if not source_output_root.exists():
        return result
    dest_output_root.mkdir(parents=True, exist_ok=True)
    for src in sorted(source_output_root.iterdir()):
        if src.name in excluded_names:
            continue
        dst = dest_output_root / src.name
        if src.is_dir():
            copied, skipped = copy_tree_files(src, dst, overwrite=overwrite, replacements=replacements)
            result["directories"].append({
                "source": str(src),
                "destination": str(dst),
                "copied": copied,
                "skipped": skipped,
            })
        elif src.is_file():
            copied = copy_file(src, dst, overwrite=overwrite, replacements=replacements)
            result["files"].append({
                "source": str(src),
                "destination": str(dst),
                "status": "copied" if copied else "skipped_existing",
            })
    return result


def main() -> None:
    args = parse_args()
    if not args.source_root.exists():
        raise RuntimeError(f"Source dataset root does not exist: {args.source_root}")
    if not args.frequency_file.exists():
        raise RuntimeError(f"Frequency file does not exist: {args.frequency_file}")
    requested = parse_frequency_values(args.frequency_file.read_text(encoding="utf-8"))
    if not requested:
        raise RuntimeError(f"No frequencies found in {args.frequency_file}")

    source_root = args.source_root.resolve()
    dest_root = args.dest_root.resolve()
    replacements = {
        str(source_root): str(dest_root),
        str(source_root.parent): str(dest_root.parent),
    }
    if str(source_root.parent).lower() != str(dest_root.parent).lower():
        replacements[str(source_root.parent).replace("/", "\\")] = str(dest_root.parent).replace("/", "\\")
        replacements[str(source_root.parent).replace("\\", "/")] = str(dest_root.parent).replace("\\", "/")
    replacements[str(source_root).replace("/", "\\")] = str(dest_root).replace("/", "\\")
    replacements[str(source_root).replace("\\", "/")] = str(dest_root).replace("\\", "/")

    label_copied, label_skipped = copy_tree_files(
        args.source_root / "labels",
        args.dest_root / "labels",
        overwrite=args.overwrite,
        replacements=replacements,
    )
    metadata_copied, metadata_skipped = copy_tree_files(
        args.source_root / "metadata",
        args.dest_root / "metadata",
        overwrite=args.overwrite,
        replacements=replacements,
    )
    progress_copied = progress_skipped = 0
    if not args.skip_progress:
        progress_copied, progress_skipped = copy_tree_files(
            args.source_root / "progress",
            args.dest_root / "progress",
            overwrite=args.overwrite,
            replacements=replacements,
        )

    npz_rows = subset_all_npz(
        args.source_root / "frequency_response",
        args.dest_root / "frequency_response",
        requested,
        tolerance_hz=args.tolerance_hz,
        overwrite=args.overwrite,
    )
    csv_rows = subset_all_csv(
        args.source_root / "csv" / "frequency_response",
        args.dest_root / "csv" / "frequency_response",
        requested,
        tolerance_hz=args.tolerance_hz,
        overwrite=args.overwrite,
    )
    metadata_rows = update_metadata(
        args.dest_root / "metadata",
        requested,
        replacements=replacements,
        source_root=args.source_root,
        overwrite=True,
    )
    manifest_path = write_manifest(args.dest_root, metadata_rows)
    auxiliary_result = copy_auxiliary_output_root(
        args.source_root.parent,
        args.dest_root.parent,
        excluded_names={args.source_root.name},
        overwrite=args.overwrite,
        replacements=replacements,
    )
    selection_result = write_selection_compatibility_files(args, overwrite=args.overwrite)
    summary = {
        "source_root": str(args.source_root),
        "dest_root": str(args.dest_root),
        "frequency_file": str(args.frequency_file),
        "frequencies_hz": requested,
        "labels": {"copied": label_copied, "skipped": label_skipped},
        "metadata": {"copied": metadata_copied, "skipped": metadata_skipped, "updated": len(metadata_rows)},
        "progress": {"copied": progress_copied, "skipped": progress_skipped},
        "npz": npz_rows,
        "csv": {
            "file_count": len(csv_rows),
            "rows_written": sum(item.get("rows") or 0 for item in csv_rows),
            "files": csv_rows,
        },
        "manifest": str(manifest_path),
        "auxiliary_output_root": auxiliary_result,
        "selection": selection_result,
    }
    summary_path = write_summary(args.dest_root, summary)
    print(f"Subset dataset root: {args.dest_root}")
    print(f"Frequency count: {len(requested)}")
    print(f"NPZ files: {len(npz_rows)}")
    print(f"CSV files: {len(csv_rows)}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

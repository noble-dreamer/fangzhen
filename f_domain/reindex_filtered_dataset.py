"""Copy selected frequency samples into output_dataset_new with contiguous IDs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_OUTPUT = ROOT / "output_dataset"
DEFAULT_DEST_OUTPUT = ROOT / "output_dataset_new"
STREAM_NAME = "streaming_dataset_a_frequency_shell"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, default=DEFAULT_SOURCE_OUTPUT)
    parser.add_argument("--dest-output", type=Path, default=DEFAULT_DEST_OUTPUT)
    parser.add_argument(
        "--selection",
        type=Path,
        default=DEFAULT_SOURCE_OUTPUT / STREAM_NAME / "balanced_defect_samples.csv",
    )
    return parser.parse_args()

def rewrite_string(
    value: str,
    old_id: str,
    new_id: str,
    source_output: Path,
    dest_output: Path,
) -> str:
    source_stream = source_output.resolve() / STREAM_NAME
    dest_stream = dest_output.resolve() / STREAM_NAME
    value = value.replace(str(source_output.resolve()), str(dest_output.resolve()))
    value = value.replace(str(source_stream), str(dest_stream))
    value = value.replace(old_id, new_id)
    normalized = value.replace("/", "\\")
    markers = (
        f"\\f_domain\\output_dataset\\{STREAM_NAME}",
        f"\\f_domain\\output2\\{STREAM_NAME}",
    )
    for marker in markers:
        index = normalized.lower().find(marker.lower())
        if index >= 0 and (normalized.startswith("\\") or normalized[1:2] == ":"):
            return str(dest_stream) + normalized[index + len(marker):]
    return value

def rewrite_value(value: Any, *args: Any) -> Any:
    if isinstance(value, str):
        return rewrite_string(value, *args)
    if isinstance(value, list):
        return [rewrite_value(item, *args) for item in value]
    if isinstance(value, dict):
        return {key: rewrite_value(item, *args) for key, item in value.items()}
    return value


def write_npz(src: Path, dst: Path, old_id: str, new_id: str) -> None:
    with np.load(src, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    sample_id = str(payload.get("sample_id", np.asarray(old_id)).item())
    payload["sample_id"] = np.asarray(sample_id.replace(old_id, new_id))
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **payload)


def write_json(
    src: Path,
    dst: Path,
    old_id: str,
    new_id: str,
    new_number: int | None,
    source_output: Path,
    dest_output: Path,
) -> None:
    data = json.loads(src.read_text(encoding="utf-8"))
    data = rewrite_value(data, old_id, new_id, source_output, dest_output)
    if isinstance(data, dict) and "sample_id" in data:
        data["sample_id"] = new_id
    if new_number is not None and src.parent.name == "metadata":
        sample = data.get("sample", {})
        if isinstance(sample, dict):
            sample["sample_id"] = new_number
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def copy_family(
    source_stream: Path,
    dest_stream: Path,
    old_id: str,
    new_id: str,
    new_number: int | None,
    source_output: Path,
    dest_output: Path,
) -> None:
    response_files = list((source_stream / "frequency_response").glob(f"{old_id}*_H_complex.npz"))
    exact_files = [
        source_stream / "metadata" / f"{old_id}.json",
        source_stream / "progress" / f"{old_id}_progress.jsonl",
    ]
    csv_files = list((source_stream / "csv" / "frequency_response").glob(f"{old_id}*.csv"))
    label_files = list((source_stream / "labels").glob(f"{old_id}_*"))
    required = response_files + exact_files + csv_files + label_files
    if not response_files or not csv_files or len(label_files) != 5 or any(not path.exists() for path in required):
        raise RuntimeError(f"Incomplete source family for {old_id}")
    for src in response_files:
        write_npz(src, dest_stream / "frequency_response" / src.name.replace(old_id, new_id), old_id, new_id)
    for src in exact_files + csv_files + label_files:
        relative = src.relative_to(source_stream)
        dst = dest_stream / Path(str(relative).replace(old_id, new_id))
        if src.suffix.lower() == ".json":
            write_json(src, dst, old_id, new_id, new_number, source_output, dest_output)
        elif src.suffix.lower() in {".jsonl", ".csv"}:
            text = rewrite_string(src.read_text(encoding="utf-8"), old_id, new_id, source_output, dest_output)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(text, encoding="utf-8")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source_output = args.source_output.resolve()
    dest_output = args.dest_output.resolve()
    source_stream = source_output / STREAM_NAME
    dest_stream = dest_output / STREAM_NAME
    if dest_output.exists() and any(dest_output.rglob("*")):
        raise RuntimeError(f"Destination is not empty: {dest_output}")
    with args.selection.open(newline="", encoding="utf-8") as file:
        selected = list(csv.DictReader(file))
    if not selected:
        raise RuntimeError(f"Selection is empty: {args.selection}")
    mapping = []
    for number, row in enumerate(selected, start=1):
        old_id = row["sample_id"]
        new_id = f"dataset_a_frequency_sample_{number:04d}"
        mapping.append({"new_sample_id": new_id, "source_sample_id": old_id,
                        "new_numeric_id": number, "source_numeric_id": int(old_id.rsplit("_", 1)[1])})

    dest_output.mkdir(parents=True, exist_ok=True)
    for src in source_output.iterdir():
        if src.is_dir() and src.name != STREAM_NAME:
            shutil.copytree(src, dest_output / src.name)
    copy_family(source_stream, dest_stream, "dataset_a_frequency_healthy",
                "dataset_a_frequency_healthy", None, source_output, dest_output)
    for item in mapping:
        copy_family(source_stream, dest_stream, item["source_sample_id"], item["new_sample_id"],
                    int(item["new_numeric_id"]), source_output, dest_output)

    source_manifest = list(csv.DictReader((source_stream / "manifest.csv").open(encoding="utf-8")))
    manifest_by_id = {row["sample_id"]: row for row in source_manifest}
    damaged_template = manifest_by_id["dataset_a_frequency_sample_0001"]
    manifest_rows = [dict(manifest_by_id["dataset_a_frequency_healthy"])]
    manifest_rows[0]["metadata"] = str(dest_stream / "metadata" / "dataset_a_frequency_healthy.json")
    for item in mapping:
        metadata_path = dest_stream / "metadata" / f"{item['new_sample_id']}.json"
        sample = json.loads(metadata_path.read_text(encoding="utf-8"))["sample"]
        row = dict(manifest_by_id.get(item["source_sample_id"], damaged_template))
        row.update({"sample_id": item["new_sample_id"], "seed": sample["seed"],
                    "defect_count": len(sample["defects"]), "lobe_count": len(sample["lobes"]),
                    "metadata": str(metadata_path)})
        manifest_rows.append(row)
    write_csv(dest_stream / "manifest.csv", manifest_rows, list(manifest_rows[0]))
    write_csv(dest_output / "sample_id_mapping.csv", mapping, list(mapping[0]))

    selected_by_id = {row["sample_id"]: row for row in selected}
    filtered_rows = []
    for item in mapping:
        row = dict(selected_by_id[item["source_sample_id"]])
        row["source_sample_id"] = item["source_sample_id"]
        row["sample_id"] = item["new_sample_id"]
        row["metadata"] = str(dest_stream / "metadata" / f"{item['new_sample_id']}.json")
        filtered_rows.append(row)
    fields = ["sample_id", "source_sample_id"] + [key for key in selected[0] if key != "sample_id"]
    write_csv(dest_stream / "balanced_defect_samples.csv", filtered_rows, fields)
    summary = {"source_output": str(source_output), "dest_output": str(dest_output),
               "sample_count": len(mapping), "healthy_included": True,
               "mapping": str(dest_output / "sample_id_mapping.csv")}
    (dest_output / "filtered_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Reindexed {len(mapping)} samples into {dest_output}")


if __name__ == "__main__":
    main()

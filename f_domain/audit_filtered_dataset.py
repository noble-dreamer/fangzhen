"""Audit one-to-one correspondence across original and reindexed filtered datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SIMPLE_ROOT = ROOT.parent
STREAM_NAME = "streaming_dataset_a_frequency_shell"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-frequency", type=Path, default=ROOT / "output_dataset" / STREAM_NAME)
    parser.add_argument("--new-frequency", type=Path, default=ROOT / "output_dataset_new" / STREAM_NAME)
    parser.add_argument("--source-pic", type=Path, default=SIMPLE_ROOT / "get_pic" / "output_dataset")
    parser.add_argument("--new-pic", type=Path, default=SIMPLE_ROOT / "get_pic" / "output_dataset_new")
    parser.add_argument("--mapping", type=Path, default=ROOT / "output_dataset_new" / "sample_id_mapping.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "output_dataset_new" / "correspondence_audit.json")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def compare_npz(source: Path, target: Path, ignored: set[str]) -> int:
    require(source.exists() and target.exists(), f"Missing NPZ pair: {source} -> {target}")
    compared = 0
    with np.load(source, allow_pickle=False) as left, np.load(target, allow_pickle=False) as right:
        require(left.files == right.files, f"NPZ fields differ: {source} -> {target}")
        for key in left.files:
            if key in ignored:
                continue
            require(np.array_equal(left[key], right[key]), f"NPZ array differs: {source.name}:{key}")
            compared += 1
    return compared


def main() -> None:
    args = parse_args()
    mapping = read_csv(args.mapping)
    pic_mapping = read_csv(args.new_pic / "sample_id_mapping.csv")
    require(mapping == pic_mapping, "f_domain and get_pic mapping tables differ")
    expected = [f"dataset_a_frequency_sample_{index:04d}" for index in range(1, len(mapping) + 1)]
    require([row["new_sample_id"] for row in mapping] == expected, "New IDs are not contiguous")
    frequency_manifest = read_csv(args.new_frequency / "manifest.csv")
    pic_manifest = read_csv(args.new_pic / "manifest.csv")
    require([row["sample_id"] for row in frequency_manifest[1:]] == expected, "Frequency manifest IDs differ")
    require([row["sample_id"] for row in pic_manifest] == expected, "get_pic manifest IDs differ")

    counters = {"frequency_arrays": 0, "label_arrays": 0, "coarse_arrays": 0, "x_arrays": 0}
    for item, pic_row in zip(mapping, pic_manifest, strict=True):
        old_id = item["source_sample_id"]
        new_id = item["new_sample_id"]
        new_number = int(item["new_numeric_id"])
        source_response = args.source_frequency / "frequency_response" / f"{old_id}_H_complex.npz"
        new_response = args.new_frequency / "frequency_response" / f"{new_id}_H_complex.npz"
        counters["frequency_arrays"] += compare_npz(source_response, new_response, {"sample_id"})
        with np.load(new_response, allow_pickle=False) as response:
            require(str(response["sample_id"].item()) == new_id, f"Response ID differs: {new_id}")

        source_meta = json.loads((args.source_frequency / "metadata" / f"{old_id}.json").read_text(encoding="utf-8"))
        new_meta = json.loads((args.new_frequency / "metadata" / f"{new_id}.json").read_text(encoding="utf-8"))
        require(new_meta["sample_id"] == new_id, f"Metadata ID differs: {new_id}")
        require(new_meta["sample"]["sample_id"] == new_number, f"Numeric metadata ID differs: {new_id}")
        for key in ("seed", "defects", "lobes"):
            require(source_meta["sample"][key] == new_meta["sample"][key], f"Metadata sample differs: {new_id}:{key}")
        require("_new_new" not in json.dumps(new_meta), f"Repeated new path found: {new_id}")

        for suffix in ("defect_depth_mm.npy", "defect_depth_norm.npy", "defect_mask.npy"):
            source_label = args.source_frequency / "labels" / f"{old_id}_{suffix}"
            new_label = args.new_frequency / "labels" / f"{new_id}_{suffix}"
            require(np.array_equal(np.load(source_label), np.load(new_label)), f"Label differs: {new_id}:{suffix}")
            counters["label_arrays"] += 1

        source_coarse = args.source_pic / "coarse_maps" / f"{old_id}_coarse_maps.npz"
        new_coarse = args.new_pic / "coarse_maps" / f"{new_id}_coarse_maps.npz"
        counters["coarse_arrays"] += compare_npz(source_coarse, new_coarse, {"algorithm_config_json"})
        with np.load(new_coarse, allow_pickle=False) as coarse:
            algorithm = json.loads(str(coarse["algorithm_config_json"].item()))
        require(algorithm["sample_id"] == new_id, f"Coarse-map ID differs: {new_id}")
        require("_new_new" not in json.dumps(algorithm), f"Repeated coarse-map path found: {new_id}")

        source_x = args.source_pic / "x_matrix" / f"{old_id}_x_matrix.npz"
        new_x = args.new_pic / "x_matrix" / f"{new_id}_x_matrix.npz"
        counters["x_arrays"] += compare_npz(source_x, new_x, {"source_healthy_npz", "source_damaged_npz"})
        with np.load(new_x, allow_pickle=False) as x_data:
            damaged_path = Path(str(x_data["source_damaged_npz"].item()))
        require(damaged_path == new_response.resolve() and damaged_path.exists(), f"x_matrix response path differs: {new_id}")
        for key in ("damaged_npz", "coarse_map_npz", "x_matrix_npz", "selected_frequency_file"):
            require(Path(pic_row[key]).exists(), f"get_pic manifest path missing: {new_id}:{key}")

    report = {
        "status": "passed",
        "one_to_one": True,
        "verified_samples": len(mapping),
        "first_new_id": expected[0],
        "last_new_id": expected[-1],
        "mapping": str(args.mapping.resolve()),
        "source_frequency": str(args.source_frequency.resolve()),
        "new_frequency": str(args.new_frequency.resolve()),
        "source_pic": str(args.source_pic.resolve()),
        "new_pic": str(args.new_pic.resolve()),
        "array_comparisons": counters,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

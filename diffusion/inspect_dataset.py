from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import build_dataset_from_config
from utils.config import ensure_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect diffusion dataset tensors and write a small summary.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def describe_array(values: np.ndarray) -> dict[str, float | list[int]]:
    finite = np.isfinite(values)
    if not np.any(finite):
        return {"shape": list(values.shape), "finite_fraction": 0.0}
    data = values[finite]
    return {
        "shape": list(values.shape),
        "finite_fraction": float(np.mean(finite)),
        "min": float(np.min(data)),
        "p01": float(np.percentile(data, 1.0)),
        "mean": float(np.mean(data)),
        "p99": float(np.percentile(data, 99.0)),
        "max": float(np.max(data)),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset = build_dataset_from_config(config, split=args.split)
    limit = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    summaries = []
    for index in tqdm(range(limit), desc="inspecting", dynamic_ncols=True):
        item = dataset[index]
        summaries.append(
            {
                "sample_id": item["sample_id"],
                "pic": describe_array(item["pic"].numpy()),
                "x_matrix": describe_array(item["x_matrix"].numpy()),
                "target": describe_array(item["target"].numpy()),
                "coarse_path": item["coarse_path"],
                "x_path": item["x_path"],
                "label_path": item["label_path"],
                "pic_channel_names": item["pic_channel_names"],
                "x_feature_names": item["x_feature_names"],
            }
        )
    output = {
        "sample_count": len(dataset),
        "inspected_count": len(summaries),
        "samples": summaries,
    }
    output_path = args.output or Path(config.get("run_dir", ROOT / "runs" / "inspect")) / "dataset_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()

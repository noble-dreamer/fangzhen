"""Evaluate coarse maps against defect labels.

Run with:
    conda run -n get_pic python simple/get_pic/evaluate_coarse_maps.py --coarse <file.npz> --label <label.npy>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import coarse_map_common as cm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate coarse maps against labels.")
    parser.add_argument("--coarse", type=Path, nargs="+", required=True)
    parser.add_argument("--label", type=Path, default=None, help="Label npy. If omitted, infer from standard label dir.")
    parser.add_argument("--output-dir", type=Path, default=cm.DEFAULT_OUTPUT_ROOT / "reports")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--raw", action="store_true", help="Evaluate pic_raw instead of normalized pic.")
    return parser.parse_args()


def infer_label_path(coarse_path: Path) -> Path:
    sample = coarse_path.name.replace("_coarse_maps.npz", "")
    return cm.DEFAULT_LABEL_DIR / f"{sample}_defect_depth_norm.npy"


def evaluate_one(coarse_path: Path, label_path: Path, threshold: float, raw: bool) -> dict:
    data = np.load(coarse_path, allow_pickle=False)
    key = "pic_raw" if raw and "pic_raw" in data.files else "pic"
    pic = np.asarray(data[key], dtype=np.float32)
    names = [str(item) for item in np.asarray(data["channel_names"]).tolist()]
    theta = np.asarray(data["theta_deg"], dtype=np.float32)
    z = np.asarray(data["z_mm"], dtype=np.float32)
    label = np.asarray(np.load(label_path), dtype=np.float32)
    if label.shape != pic.shape[1:]:
        raise RuntimeError(f"{label_path} shape {label.shape} != coarse map shape {pic.shape[1:]}")
    metrics = {}
    for index, name in enumerate(names):
        metrics[name] = {
            "pearson": cm.pearson(pic[index], label),
            "nrmse": cm.nrmse(pic[index], label),
            "mask_iou": cm.mask_iou(pic[index], label, threshold),
            "top5_hit_rate": cm.top_fraction_hit_rate(pic[index], label, threshold, top_fraction=0.05),
            "prediction_mass_in_label": cm.prediction_mass_in_target(pic[index], label, threshold),
            "centroid_error_mm": cm.centroid_error_mm(pic[index], label, theta, z),
        }
    return {
        "coarse": str(coarse_path),
        "label": str(label_path),
        "map_key": key,
        "threshold": threshold,
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for coarse_path in args.coarse:
        label_path = args.label or infer_label_path(coarse_path)
        if not label_path.exists():
            raise FileNotFoundError(label_path)
        report = evaluate_one(coarse_path, label_path, args.threshold, args.raw)
        output_path = args.output_dir / f"{coarse_path.stem}_metrics.json"
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)


if __name__ == "__main__":
    main()

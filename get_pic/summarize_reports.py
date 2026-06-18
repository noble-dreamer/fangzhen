"""Small helper to summarize coarse-map report JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize selected channels from coarse-map metrics reports.")
    parser.add_argument("--report", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--channels",
        nargs="*",
        default=[
            "ray_log_amp_loss",
            "ray_relative_delta",
            "ray_phase_change",
            "ray_delta_abs",
            "low_frequency_band_map",
            "mid_frequency_band_map",
            "high_frequency_band_map",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in args.report:
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"\n{path}")
        for channel in args.channels:
            metrics = data["metrics"][channel]
            print(
                f"{channel:24s} "
                f"pearson={metrics['pearson']:.4f} "
                f"nrmse={metrics['nrmse']:.4f} "
                f"iou={metrics['mask_iou']:.4f} "
                f"top5={metrics['top5_hit_rate']:.4f} "
                f"mass={metrics['prediction_mass_in_label']:.4f} "
                f"centroid={metrics['centroid_error_mm']:.1f}"
            )


if __name__ == "__main__":
    main()

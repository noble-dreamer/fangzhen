"""Batch preview coarse-map outputs for an output_dataset folder.

Run with:
    conda run -n get_pic python simple/get_pic/preview_output_dataset.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import preview_coarse_maps


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output_dataset"
DEFAULT_OVERVIEW_CHANNELS = (
    "ray_log_amp_loss",
    "ray_relative_delta",
    "ray_phase_change",
    "high_frequency_band_map",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create preview PNGs for generated coarse maps.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preview-dir", type=Path, default=None)
    parser.add_argument("--raw", action="store_true", help="Preview pic_raw instead of normalized pic.")
    parser.add_argument("--limit", type=int, default=None, help="Only preview the first N coarse maps.")
    parser.add_argument(
        "--overview-channels",
        nargs="*",
        default=list(DEFAULT_OVERVIEW_CHANNELS),
        help="Channel names to write as all-sample overview montages.",
    )
    return parser.parse_args()


def load_channel(path: Path, channel_name: str, *, raw: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    data = np.load(path, allow_pickle=False)
    key = "pic_raw" if raw and "pic_raw" in data.files else "pic"
    pic = np.asarray(data[key], dtype=np.float32)
    names = [str(item) for item in np.asarray(data["channel_names"]).tolist()]
    if channel_name not in names:
        return None
    return pic[names.index(channel_name)], np.asarray(data["theta_deg"], dtype=float), np.asarray(data["z_mm"], dtype=float)


def write_overview(
    coarse_paths: list[Path],
    output_path: Path,
    *,
    channel_name: str,
    raw: bool,
    columns: int = 8,
) -> bool:
    loaded = [(path, load_channel(path, channel_name, raw=raw)) for path in coarse_paths]
    loaded = [(path, item) for path, item in loaded if item is not None]
    if not loaded:
        return False

    rows = int(math.ceil(len(loaded) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.25 * columns, 1.8 * rows),
        dpi=160,
        squeeze=False,
        constrained_layout=True,
    )
    last_image = None
    for index, ax in enumerate(axes.ravel()):
        if index >= len(loaded):
            ax.axis("off")
            continue
        path, item = loaded[index]
        channel, theta, z = item
        theta_step = theta[1] - theta[0] if theta.size > 1 else 0.0
        extent = [float(theta.min()), float(theta.max() + theta_step), float(z.min()), float(z.max())]
        last_image = ax.imshow(
            channel,
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=None if raw else 0.0,
            vmax=None if raw else 1.0,
        )
        sample_label = path.stem.replace("_coarse_maps", "").replace("dataset_a_frequency_sample_", "")
        ax.set_title(sample_label, fontsize=6, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])

    if last_image is not None:
        colorbar = fig.colorbar(last_image, ax=axes.ravel().tolist(), fraction=0.016, pad=0.01)
        colorbar.set_label(channel_name, fontsize=7)
        colorbar.ax.tick_params(labelsize=6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    preview_dir = args.preview_dir or output_root / "preview"
    coarse_dir = output_root / "coarse_maps"
    coarse_paths = sorted(coarse_dir.glob("*_coarse_maps.npz"))
    if args.limit is not None:
        coarse_paths = coarse_paths[: args.limit]
    if not coarse_paths:
        raise RuntimeError(f"No *_coarse_maps.npz files found under {coarse_dir}")

    sample_dir = preview_dir / "samples"
    for coarse_path in coarse_paths:
        preview_path = sample_dir / f"{coarse_path.stem}_preview.png"
        preview_coarse_maps.write_preview(coarse_path, preview_path, raw=args.raw)
        print(preview_path)

    overview_dir = preview_dir / "overview"
    for channel_name in args.overview_channels:
        suffix = "_raw" if args.raw else ""
        output_path = overview_dir / f"{channel_name}{suffix}_overview.png"
        if write_overview(coarse_paths, output_path, channel_name=channel_name, raw=args.raw):
            print(output_path)


if __name__ == "__main__":
    main()

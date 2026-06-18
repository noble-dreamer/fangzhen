"""Write coarse-map preview PNGs.

Run with:
    conda run -n get_pic python simple/get_pic/preview_coarse_maps.py --coarse <file.npz>
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview generated coarse maps.")
    parser.add_argument("--coarse", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--raw", action="store_true", help="Preview pic_raw instead of normalized pic.")
    return parser.parse_args()


def write_preview(path: Path, output_path: Path, *, raw: bool = False) -> Path:
    data = np.load(path, allow_pickle=False)
    key = "pic_raw" if raw and "pic_raw" in data.files else "pic"
    pic = np.asarray(data[key], dtype=np.float32)
    names = [str(item) for item in np.asarray(data["channel_names"]).tolist()]
    theta = np.asarray(data["theta_deg"], dtype=float)
    z = np.asarray(data["z_mm"], dtype=float)
    count = pic.shape[0]
    cols = min(4, count)
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.6 * rows), dpi=160, squeeze=False)
    extent = [float(theta.min()), float(theta.max() + (theta[1] - theta[0] if theta.size > 1 else 0.0)), float(z.min()), float(z.max())]
    for index, ax in enumerate(axes.ravel()):
        if index >= count:
            ax.axis("off")
            continue
        image = ax.imshow(
            pic[index],
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
        )
        ax.set_title(names[index], fontsize=8)
        ax.set_xlabel("theta deg", fontsize=7)
        ax.set_ylabel("z mm", fontsize=7)
        ax.tick_params(labelsize=7)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    for coarse_path in args.coarse:
        output_dir = args.output_dir or coarse_path.parent.parent / "previews"
        output_path = output_dir / f"{coarse_path.stem}_preview.png"
        written = write_preview(coarse_path, output_path, raw=args.raw)
        print(written)


if __name__ == "__main__":
    main()

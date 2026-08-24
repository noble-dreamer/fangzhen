"""Render paged, common-scale montages for the reindexed filtered defect labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT / "output_dataset_new"
STREAM_NAME = "streaming_dataset_a_frequency_shell"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "label_previews")
    parser.add_argument("--pages", type=int, default=5)
    parser.add_argument("--columns", type=int, default=10)
    parser.add_argument("--vmax-mm", type=float, default=5.0)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(path, size=size) if path.exists() else ImageFont.load_default()


def main() -> None:
    args = parse_args()
    if args.pages <= 0 or args.columns <= 0 or args.vmax_mm <= 0:
        raise ValueError("pages, columns, and vmax-mm must be positive")
    stream_root = args.dataset_root / STREAM_NAME
    selection_path = stream_root / "balanced_defect_samples.csv"
    with selection_path.open(newline="", encoding="utf-8") as file:
        samples = list(csv.DictReader(file))
    if len(samples) < args.pages:
        raise RuntimeError(f"Cannot split {len(samples)} samples into {args.pages} non-empty pages")
    args.output.mkdir(parents=True, exist_ok=True)
    chunks = np.array_split(np.arange(len(samples)), args.pages)
    manifest = []
    page_paths = []
    title_font = load_font(15)
    header_font = load_font(24)
    scale_font = load_font(14)
    colormap = matplotlib.colormaps["viridis"]
    tile_width, tile_height = 380, 300
    for page_number, indices in enumerate(chunks, start=1):
        rows = math.ceil(len(indices) / args.columns)
        header_height, scale_width = 55, 80
        canvas = Image.new(
            "RGB",
            (tile_width * args.columns + scale_width, header_height + tile_height * rows),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        for slot, sample_index in enumerate(indices):
            sample = samples[int(sample_index)]
            new_id = sample["sample_id"]
            source_id = sample["source_sample_id"]
            depth_path = stream_root / "labels" / f"{new_id}_defect_depth_mm.npy"
            depth = np.load(depth_path, mmap_mode="r")
            peak = float(np.nanmax(depth))
            rgb = colormap(np.clip(np.asarray(depth) / args.vmax_mm, 0.0, 1.0), bytes=True)[:, :, :3]
            label_image = Image.fromarray(rgb, "RGB").resize(
                (tile_width - 12, tile_height - 58), Image.Resampling.BILINEAR
            )
            column, row = slot % args.columns, slot // args.columns
            left, top = column * tile_width, header_height + row * tile_height
            draw.rectangle((left + 3, top + 3, left + tile_width - 3, top + tile_height - 3), outline="#b0b0b0")
            draw.multiline_text(
                (left + 7, top + 5),
                f"{new_id[-4:]} <- {source_id[-4:]} | N={sample['defect_count']}\n"
                f"Dmin={float(sample['min_diameter_mm']):.0f}  "
                f"dmin={float(sample['min_depth_mm']):.2f}  "
                f"R={float(sample['strength_ratio']):.2f}  peak={peak:.2f}",
                fill="black",
                font=title_font,
                spacing=2,
            )
            canvas.paste(label_image, (left + 6, top + 54))
            manifest.append({
                "page": page_number,
                "slot": slot + 1,
                "sample_id": new_id,
                "source_sample_id": source_id,
                "defect_count": sample["defect_count"],
                "min_diameter_mm": sample["min_diameter_mm"],
                "min_depth_mm": sample["min_depth_mm"],
                "strength_ratio": sample["strength_ratio"],
                "peak_depth_mm": peak,
                "depth_npy": str(depth_path.resolve()),
            })
        first_id = samples[int(indices[0])]["sample_id"][-4:]
        last_id = samples[int(indices[-1])]["sample_id"][-4:]
        draw.text(
            (15, 12),
            f"Filtered defect labels {first_id}-{last_id} / {len(samples)} | common scale 0-{args.vmax_mm:g} mm",
            fill="black",
            font=header_font,
        )
        gradient = np.linspace(1.0, 0.0, 512, dtype=float)[:, None]
        gradient_rgb = colormap(gradient, bytes=True)[:, :, :3]
        colorbar = Image.fromarray(gradient_rgb, "RGB").resize((28, tile_height * rows - 80))
        scale_left, scale_top = tile_width * args.columns + 10, header_height + 35
        canvas.paste(colorbar, (scale_left, scale_top))
        draw.text((scale_left + 34, scale_top - 8), f"{args.vmax_mm:g}", fill="black", font=scale_font)
        draw.text((scale_left + 34, scale_top + colorbar.height // 2 - 8), f"{args.vmax_mm / 2:g}", fill="black", font=scale_font)
        draw.text((scale_left + 34, scale_top + colorbar.height - 14), "0", fill="black", font=scale_font)
        draw.text((scale_left - 5, scale_top + colorbar.height + 8), "mm", fill="black", font=scale_font)
        page_path = args.output / f"filtered_labels_page_{page_number:02d}_of_{args.pages:02d}.png"
        canvas.save(page_path, optimize=True, dpi=(args.dpi, args.dpi))
        page_paths.append(str(page_path.resolve()))
        print(f"Wrote {page_path} with {len(indices)} labels")
    write_csv(args.output / "preview_manifest.csv", manifest)
    summary = {"sample_count": len(samples), "page_count": args.pages, "columns": args.columns,
               "common_scale_mm": [0.0, args.vmax_mm], "pages": page_paths,
               "manifest": str((args.output / "preview_manifest.csv").resolve())}
    (args.output / "preview_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

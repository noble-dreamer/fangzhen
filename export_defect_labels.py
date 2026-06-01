"""Export unfolded theta-z defect labels from simple dataset metadata."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

import defect_label_common as labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create unfolded pipe-surface defect depth labels from simple metadata.'
    )
    parser.add_argument(
        '--metadata',
        type=Path,
        nargs='+',
        required=True,
        help='Metadata JSON file(s), metadata directory, or glob-like path(s).',
    )
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--theta-count', type=int, default=labels.DEFAULT_THETA_COUNT)
    parser.add_argument('--z-count', type=int, default=labels.DEFAULT_Z_COUNT)
    parser.add_argument('--h-min-mm', type=float, default=labels.DEFAULT_H_MIN_MM)
    parser.add_argument('--mask-threshold-mm', type=float, default=labels.DEFAULT_MASK_THRESHOLD_MM)
    parser.add_argument(
        '--preview-max-mm',
        default='auto',
        help='PNG color scale in mm, or "auto" to scale each sample by its own maximum.',
    )
    parser.add_argument(
        '--prediction',
        type=Path,
        default=None,
        help='Optional prediction .npy to compare against the generated depth_mm label. Use with one metadata file.',
    )
    parser.add_argument('--montage', action='store_true', help='Write a montage PNG for all generated previews.')
    parser.add_argument('--montage-columns', type=int, default=5)
    return parser.parse_args()


def metadata_files(inputs: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        if item.is_dir():
            files.extend(sorted(item.glob('*.json')))
        elif any(char in str(item) for char in '*?['):
            files.extend(Path(match) for match in sorted(glob.glob(str(item))))
        else:
            files.append(item)
    unique: list[Path] = []
    seen: set[Path] = set()
    for item in files:
        resolved = item.resolve()
        if resolved not in seen:
            unique.append(item)
            seen.add(resolved)
    if not unique:
        raise FileNotFoundError('no metadata JSON files matched the input')
    return unique


def default_output_dir(metadata_path: Path) -> Path:
    if metadata_path.parent.name.lower() == 'metadata':
        return metadata_path.parent.parent / 'labels'
    return metadata_path.parent / 'labels'


def parse_preview_max(value: str) -> float | None:
    if value.lower() == 'auto':
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError('--preview-max-mm must be positive or "auto"')
    return parsed


def resize_rgb_nearest(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    source = np.asarray(rgb, dtype=np.uint8)
    y_index = np.linspace(0, source.shape[0] - 1, height).round().astype(int)
    x_index = np.linspace(0, source.shape[1] - 1, width).round().astype(int)
    return source[y_index][:, x_index]


def label_metadata_path(preview_path: Path) -> Path:
    if preview_path.name.endswith('_defect_label.png'):
        sample_id = preview_path.name.removesuffix('_defect_label.png')
        return preview_path.with_name(f'{sample_id}_defect_label_metadata.json')
    return preview_path.with_suffix('.json')


def preview_rgb_from_metadata(preview_path: Path, tile_size: int) -> np.ndarray:
    meta_path = label_metadata_path(preview_path)
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    depth = np.load(metadata['files']['depth_mm_npy'])
    preview_max = max(float(metadata.get('preview_max_mm', np.nanmax(depth))), 1e-12)
    rgb = labels.parula_like_rgb(np.clip(depth / preview_max, 0.0, 1.0))
    return resize_rgb_nearest(rgb, tile_size, tile_size)


def write_montage_with_matplotlib(image_paths: list[Path], output_path: Path, columns: int) -> bool:
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return False

    metadata_items = [
        json.loads(label_metadata_path(path).read_text(encoding='utf-8'))
        for path in image_paths
    ]
    depths = [np.load(item['files']['depth_mm_npy']) for item in metadata_items]
    if not depths:
        return True
    columns = max(columns, 1)
    rows = math.ceil(len(depths) / columns)
    vmax = max(max(float(np.nanmax(depth)), 1e-12) for depth in depths)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.35 * columns, 2.15 * rows),
        dpi=180,
        squeeze=False,
        constrained_layout=True,
    )
    last_image = None
    for index, ax in enumerate(axes.flat):
        if index >= len(depths):
            ax.axis('off')
            continue
        meta = metadata_items[index]
        z_range = meta['z_axis']['range']
        last_image = ax.imshow(
            depths[index],
            origin='lower',
            extent=[0.0, 360.0, float(z_range[0]), float(z_range[1])],
            aspect='auto',
            interpolation='nearest',
            cmap=meta.get('preview_colormap', labels.DEFAULT_COLORMAP),
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(meta['sample_id'], fontsize=7, pad=3)
        ax.set_xticks([0, 180, 360])
        ax.set_yticks([float(z_range[0]), float(z_range[1])])
        ax.tick_params(labelsize=6, length=2)
        if index % columns == 0:
            ax.set_ylabel('z (mm)', fontsize=7)
        if index // columns == rows - 1:
            ax.set_xlabel('theta (deg)', fontsize=7)
    if last_image is not None:
        colorbar = fig.colorbar(last_image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.01)
        colorbar.set_label('Wall loss depth (mm)', fontsize=7)
        colorbar.ax.tick_params(labelsize=6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0.035)
    plt.close(fig)
    return True


def write_montage(image_paths: list[Path], output_path: Path, columns: int) -> None:
    if not image_paths:
        return
    if write_montage_with_matplotlib(image_paths, output_path, columns):
        return
    columns = max(columns, 1)
    tile = 140
    gap = 16
    rows = math.ceil(len(image_paths) / columns)
    canvas_w = columns * tile + (columns + 1) * gap
    canvas_h = rows * tile + (rows + 1) * gap
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    for index, path in enumerate(image_paths):
        row, col = divmod(index, columns)
        x = gap + col * (tile + gap)
        y = gap + row * (tile + gap)
        canvas[y:y + tile, x:x + tile] = preview_rgb_from_metadata(path, tile)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels.write_rgb_png(output_path, canvas)


def main() -> None:
    args = parse_args()
    paths = metadata_files(args.metadata)
    preview_max_mm = parse_preview_max(args.preview_max_mm)
    if args.prediction is not None and len(paths) != 1:
        raise ValueError('--prediction can only be used with one metadata file')

    preview_paths: list[Path] = []
    for metadata_path in paths:
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        sample = labels.sample_from_metadata(metadata)
        pipe = labels.pipe_from_metadata(metadata)
        sample_id = labels.label_stem_from_metadata(metadata, metadata_path.stem)
        output_dir = args.output_dir or default_output_dir(metadata_path)
        label_meta = labels.write_label_package(
            output_dir=output_dir,
            sample_id=sample_id,
            sample=sample,
            pipe=pipe,
            theta_count=args.theta_count,
            z_count=args.z_count,
            h_min_mm=args.h_min_mm,
            mask_threshold_mm=args.mask_threshold_mm,
            preview_max_mm=preview_max_mm,
        )
        preview_paths.append(Path(label_meta['files']['preview_png']))
        print(f'wrote {label_meta["files"]["metadata_json"]}')

        if args.prediction is not None:
            prediction = np.load(args.prediction)
            target = np.load(label_meta['files']['depth_mm_npy'])
            metrics = labels.compare_prediction(
                prediction,
                target,
                threshold=args.mask_threshold_mm,
            )
            print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.montage:
        montage_dir = args.output_dir or default_output_dir(paths[0])
        montage_path = montage_dir / 'defect_label_montage.png'
        write_montage(preview_paths, montage_path, columns=args.montage_columns)
        print(f'wrote {montage_path}')


if __name__ == '__main__':
    main()

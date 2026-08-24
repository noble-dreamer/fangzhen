"""Generate auditable previews for irregular single-defect fields."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SIMPLE_ROOT = HERE.parents[1]
if str(SIMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMPLE_ROOT))

import defect_label_common as labels
from irregular_defect_common import PIPE_MID_RADIUS_MM, generate_irregular_defect, unfolded_depth_map


DEFAULT_OUTPUT = HERE / 'output_dataset' / 'dataset_a_frequency_shell' / 'examples'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=9)
    parser.add_argument('--seed0', type=int, default=20260820)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def write_overview(rows: list[tuple[str, str, np.ndarray]], path: Path) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    columns = min(3, len(rows))
    row_count = int(np.ceil(len(rows) / columns))
    fig, axes = plt.subplots(row_count, columns, figsize=(4.2 * columns, 3.6 * row_count), squeeze=False)
    for axis, (name, size_class, depth) in zip(axes.flat, rows):
        image = axis.imshow(
            depth,
            origin='lower',
            extent=[0.0, 2.0 * np.pi * PIPE_MID_RADIUS_MM, 0.0, 1000.0],
            aspect='equal',
            cmap='viridis',
            vmin=0.0,
            vmax=5.0,
        )
        axis.set_title(f'{name} ({size_class})')
        axis.set_xlabel('unfolded circumference (mm)')
        axis.set_ylabel('axial z (mm)')
        fig.colorbar(image, ax=axis, label='wall loss (mm)', fraction=0.046)
    for axis in axes.flat[len(rows):]:
        axis.set_visible(False)
    fig.suptitle('Multiscale defects on the unfolded pipe surface')
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError('--count must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overview_rows: list[tuple[str, str, np.ndarray]] = []
    manifest = []
    for index in range(1, args.count + 1):
        seed = args.seed0 + index
        stem = f'irregular_sample_{index:04d}'
        field = generate_irregular_defect(seed)
        theta_deg, z_mm, label_depth = unfolded_depth_map(field)
        npz_path = args.output_dir / f'{stem}_defect_field.npz'
        json_path = args.output_dir / f'{stem}_defect_field.json'
        png_path = args.output_dir / f'{stem}_defect_label.png'
        np.savez_compressed(
            npz_path,
            local_arc_mm=field.arc_mm,
            local_z_mm=field.z_mm,
            local_depth_mm=field.depth_mm,
            theta_deg=theta_deg,
            label_z_mm=z_mm,
            label_depth_mm=label_depth,
        )
        metadata = field.metadata()
        metadata['files'] = {'field_npz': str(npz_path), 'preview_png': str(png_path)}
        json_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        labels.save_depth_png(label_depth, png_path, theta_deg, z_mm, stem, preview_max_mm=5.0)
        overview_rows.append((stem, field.size_class, label_depth))
        manifest.append({'sample_id': stem, **metadata})
        print(f'[example] {stem}: class={field.size_class}, max_depth={label_depth.max():.3f} mm, seed={seed}')
    overview_path = args.output_dir / 'irregular_defect_examples.png'
    write_overview(overview_rows, overview_path)
    (args.output_dir / 'example_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Wrote {overview_path}')


if __name__ == '__main__':
    main()

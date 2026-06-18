"""Generate 100 preview defect maps using the current smooth defect strategy."""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import defect_label_common as labels
import simple_defect_common as defects


OUTPUT_ROOT = Path(__file__).resolve().parent / 'defect_preview_100'
SEED0 = 810000
SAMPLE_COUNT = 100


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def try_write_montage(image_paths: list[Path], output_path: Path, columns: int = 10) -> bool:
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return False

    metadata_items = [
        json.loads((path.with_name(path.stem + '_metadata.json')).read_text(encoding='utf-8'))
        for path in image_paths
    ]
    depths = [np.load(item['files']['depth_mm_npy']) for item in metadata_items]
    vmax = 5.0
    rows = math.ceil(len(depths) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.15 * columns, 1.75 * rows),
        dpi=160,
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
        ax.set_title(meta['sample_id'].replace('defect_preview_', ''), fontsize=6, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
    if last_image is not None:
        colorbar = fig.colorbar(last_image, ax=axes.ravel().tolist(), fraction=0.016, pad=0.01)
        colorbar.set_label('Wall loss depth (mm)', fontsize=7)
        colorbar.ax.tick_params(labelsize=6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    return True


def main() -> None:
    preview_dir = OUTPUT_ROOT / 'images'
    label_dir = OUTPUT_ROOT / 'labels'
    metadata_dir = OUTPUT_ROOT / 'metadata'
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    config = defects.DefectSamplingConfig(
        min_defects=1,
        max_defects=3,
        aspect_ratio_range=(0.75, 1.45),
        irregular_lobes=True,
    )
    rows: list[dict] = []
    image_paths: list[Path] = []
    for sample_id in range(1, SAMPLE_COUNT + 1):
        seed = SEED0 + sample_id
        sample = defects.generate_sample(sample_id, seed, config)
        sample_name = f'defect_preview_{sample_id:04d}'
        sample_dict = defects.sample_to_dict(sample)
        label_meta = labels.write_label_package(
            label_dir,
            sample_name,
            sample_dict,
            preview_max_mm=5.0,
        )
        png_path = Path(label_meta['files']['preview_png'])
        target_png = preview_dir / f'{sample_name}.png'
        target_meta = preview_dir / f'{sample_name}_metadata.json'
        target_png.parent.mkdir(parents=True, exist_ok=True)
        png_path.replace(target_png)
        Path(label_meta['files']['metadata_json']).replace(target_meta)
        label_meta['files']['preview_png'] = str(target_png)
        label_meta['files']['metadata_json'] = str(target_meta)
        target_meta.write_text(json.dumps(label_meta, ensure_ascii=False, indent=2), encoding='utf-8')
        image_paths.append(target_png)
        depth = np.load(label_meta['files']['depth_mm_npy'])
        rows.append({
            'sample_id': sample_name,
            'seed': seed,
            'defect_count': len(sample.defects),
            'lobe_count': len(sample.lobes),
            'size_classes': '+'.join(item.size_class for item in sample.defects),
            'max_depth_mm': float(np.nanmax(depth)),
            'mean_depth_mm': float(np.nanmean(depth)),
            'area_over_0p1mm_px': int(np.sum(depth >= 0.1)),
            'area_over_1mm_px': int(np.sum(depth >= 1.0)),
            'defects_json': json.dumps([asdict(item) for item in sample.defects], ensure_ascii=False),
            'lobes_json': json.dumps([asdict(item) for item in sample.lobes], ensure_ascii=False),
            'preview_png': str(target_png),
            'depth_mm_npy': label_meta['files']['depth_mm_npy'],
        })

    write_csv(OUTPUT_ROOT / 'defect_preview_summary.csv', rows)
    summary = {
        'sample_count': SAMPLE_COUNT,
        'seed0': SEED0,
        'sampling_config': asdict(config),
        'max_depth_limit_mm': labels.DEFAULT_DEFECT_LOSS_MAX_MM,
        'window_power': labels.DEFAULT_DEFECT_WINDOW_POWER,
        'defect_count_range': [min(row['defect_count'] for row in rows), max(row['defect_count'] for row in rows)],
        'lobe_count_range': [min(row['lobe_count'] for row in rows), max(row['lobe_count'] for row in rows)],
        'size_class_counts': {
            size_class: sum(row['size_classes'].split('+').count(size_class) for row in rows)
            for size_class in ('large', 'medium', 'small')
        },
        'size_pattern_counts': {
            pattern: sum(1 for row in rows if row['size_classes'] == pattern)
            for pattern in sorted({row['size_classes'] for row in rows})
        },
        'max_depth_observed_mm': max(row['max_depth_mm'] for row in rows),
        'preview_dir': str(preview_dir),
        'label_dir': str(label_dir),
    }
    (OUTPUT_ROOT / 'defect_preview_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    montage_path = OUTPUT_ROOT / 'defect_preview_montage.png'
    montage_ok = try_write_montage(image_paths, montage_path, columns=10)
    print(f'Generated {len(image_paths)} defect preview images under {preview_dir}')
    print(f'Summary CSV: {OUTPUT_ROOT / "defect_preview_summary.csv"}')
    print(f'Summary JSON: {OUTPUT_ROOT / "defect_preview_summary.json"}')
    print(f'Montage: {montage_path if montage_ok else "not generated; matplotlib unavailable"}')


if __name__ == '__main__':
    main()

"""Select existing frequency samples whose metadata has detectable, balanced defects."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SIMPLE_ROOT = ROOT.parent
if str(SIMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMPLE_ROOT))

import simple_defect_common as defects


DEFAULT_DATASET_ROOT = ROOT / 'output_dataset' / 'streaming_dataset_a_frequency_shell'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-root', type=Path, default=DEFAULT_DATASET_ROOT / 'metadata')
    parser.add_argument('--output', type=Path, default=DEFAULT_DATASET_ROOT / 'balanced_defect_samples.csv')
    parser.add_argument('--min-diameter-mm', type=float, default=95.0)
    parser.add_argument('--min-depth-mm', type=float, default=1.5)
    parser.add_argument('--max-strength-ratio', type=float, default=3.0)
    parser.add_argument('--max-defects', type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = []
    rejected = Counter()
    paths = sorted(args.metadata_root.glob('dataset_a_frequency_sample_*.json'))
    for path in paths:
        data = json.loads(path.read_text(encoding='utf-8'))
        sample = data['sample']
        metrics = defects.defect_strength_metrics(sample['defects'])
        reasons = []
        if metrics['defect_count'] > args.max_defects:
            reasons.append('too_many_defects')
        if metrics['min_diameter_mm'] < args.min_diameter_mm:
            reasons.append('diameter_below_floor')
        if metrics['min_depth_mm'] < args.min_depth_mm:
            reasons.append('depth_below_floor')
        if metrics['strength_ratio'] > args.max_strength_ratio:
            reasons.append('strength_ratio_above_limit')
        if reasons:
            rejected.update(reasons)
            continue
        selected.append({
            'sample_id': path.stem,
            'seed': sample.get('seed'),
            **metrics,
            'metadata': str(path),
        })

    if not paths:
        raise FileNotFoundError(f'No damaged sample metadata found under {args.metadata_root}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(selected[0].keys()) if selected else [
            'sample_id', 'seed', 'defect_count', 'min_diameter_mm', 'min_depth_mm',
            'min_strength_proxy_mm3', 'max_strength_proxy_mm3', 'strength_ratio', 'metadata',
        ])
        writer.writeheader()
        writer.writerows(selected)

    counts = Counter(int(row['defect_count']) for row in selected)
    print(f'Scanned {len(paths)} metadata files; selected {len(selected)} -> {args.output}')
    print(f'Selected defect counts: {dict(sorted(counts.items()))}')
    print(f'Rejection flags: {dict(sorted(rejected.items()))}')


if __name__ == '__main__':
    main()

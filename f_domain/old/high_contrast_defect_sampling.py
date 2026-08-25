"""Archived high-contrast defect sampling used before balanced_detectable_v1."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path


SIMPLE_ROOT = Path(__file__).resolve().parents[2]
if str(SIMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMPLE_ROOT))

import simple_defect_common as defects


LEGACY_SEED0 = 710000
LEGACY_V1_LAST_SAMPLE_ID = 1040
LEGACY_HIGH_CONTRAST_V1_CONFIG = defects.DefectSamplingConfig(
    min_defects=1,
    max_defects=3,
    diameter_range_mm=(120.0, 240.0),
    depth_range_mm=(1.0, 4.2),
    max_total_depth_mm=5.0,
    aspect_ratio_range=(0.75, 1.45),
    z_margin_mm=260.0,
    clearance_mm=50.0,
    irregular_lobes=True,
    convex_lobes_range=(0, 2),
    lobe_radius_fraction=(0.30, 0.55),
    lobe_offset_fraction=(0.45, 0.80),
    lobe_depth_fraction=(0.06, 0.16),
    size_mixture=True,
    defect_count_weights=(0.30, 0.45, 0.25),
    size_classes=(
        defects.DefectSizeClass('small', (50.0, 95.0), (0.8, 2.6), (0, 0)),
        defects.DefectSizeClass('medium', (95.0, 170.0), (1.0, 3.4), (0, 1)),
        defects.DefectSizeClass('large', (170.0, 240.0), (1.5, 4.2), (0, 1)),
    ),
    single_size_weights=(('large', 0.45), ('medium', 0.35), ('small', 0.20)),
    two_defect_size_patterns=(
        (0.45, 'large', 'small'),
        (0.30, 'large', 'medium'),
        (0.20, 'medium', 'small'),
        (0.05, 'medium', 'medium'),
    ),
    three_defect_size_patterns=(
        (0.55, 'large', 'medium', 'small'),
        (0.25, 'large', 'small', 'small'),
        (0.15, 'medium', 'medium', 'small'),
        (0.05, 'large', 'medium', 'medium'),
    ),
)
LEGACY_HIGH_CONTRAST_V2_CONFIG = replace(
    LEGACY_HIGH_CONTRAST_V1_CONFIG,
    convex_lobes_range=(1, 4),
    lobe_radius_fraction=(0.20, 0.45),
    lobe_depth_fraction=(0.10, 0.25),
    size_classes=(
        defects.DefectSizeClass('small', (50.0, 95.0), (0.8, 2.6), (0, 1)),
        defects.DefectSizeClass('medium', (95.0, 170.0), (1.0, 3.4), (1, 2)),
        defects.DefectSizeClass('large', (170.0, 240.0), (1.5, 4.2), (1, 3)),
    ),
)


def generate_legacy_sample(
    sample_id: int,
    seed0: int = LEGACY_SEED0,
    profile: str = 'auto',
) -> defects.GeneratedSample:
    """Reproduce a former sample using v1, v2, or the existing dataset cutoff."""
    if profile == 'auto':
        profile = 'v1' if sample_id <= LEGACY_V1_LAST_SAMPLE_ID else 'v2'
    configs = {
        'v1': LEGACY_HIGH_CONTRAST_V1_CONFIG,
        'v2': LEGACY_HIGH_CONTRAST_V2_CONFIG,
    }
    if profile not in configs:
        raise ValueError(f'Unknown legacy profile {profile!r}; expected auto, v1, or v2.')
    seed = seed0 + sample_id
    return defects.generate_sample(sample_id, seed, configs[profile])

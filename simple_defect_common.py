"""Random defect helpers for the lightweight shell datasets."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import simple_shell_common as shell


@dataclass(frozen=True)
class GeneratedDefect:
    theta_deg: float
    z_mm: float
    diameter_mm: float
    depth_mm: float
    lobe_count: int
    diameter_theta_mm: float | None = None
    diameter_z_mm: float | None = None
    size_class: str = 'legacy'


@dataclass(frozen=True)
class GeneratedLobe:
    parent_index: int
    theta_deg: float
    z_mm: float
    radius_mm: float
    depth_mm: float
    radius_theta_mm: float | None = None
    radius_z_mm: float | None = None


@dataclass(frozen=True)
class GeneratedSample:
    sample_id: int
    seed: int
    defects: list[GeneratedDefect]
    lobes: list[GeneratedLobe]


@dataclass(frozen=True)
class DefectSizeClass:
    name: str
    diameter_range_mm: tuple[float, float]
    depth_range_mm: tuple[float, float]
    lobe_count_range: tuple[int, int]


@dataclass(frozen=True)
class DefectSamplingConfig:
    min_defects: int = 1
    max_defects: int = 3
    diameter_range_mm: tuple[float, float] = (120.0, 240.0)
    depth_range_mm: tuple[float, float] = (1.0, 4.2)
    max_total_depth_mm: float = 5.0
    aspect_ratio_range: tuple[float, float] = (1.0, 1.0)
    z_margin_mm: float = 260.0
    clearance_mm: float = 50.0
    irregular_lobes: bool = True
    convex_lobes_range: tuple[int, int] = (0, 2)
    lobe_radius_fraction: tuple[float, float] = (0.30, 0.55)
    lobe_offset_fraction: tuple[float, float] = (0.45, 0.80)
    lobe_depth_fraction: tuple[float, float] = (0.06, 0.16)
    size_mixture: bool = True
    defect_count_weights: tuple[float, ...] = (0.30, 0.45, 0.25)
    size_classes: tuple[DefectSizeClass, ...] = (
        DefectSizeClass('small', (50.0, 95.0), (0.8, 2.6), (0, 0)),
        DefectSizeClass('medium', (95.0, 170.0), (1.0, 3.4), (0, 1)),
        DefectSizeClass('large', (170.0, 240.0), (1.5, 4.2), (0, 1)),
    )
    single_size_weights: tuple[tuple[str, float], ...] = (
        ('large', 0.45),
        ('medium', 0.35),
        ('small', 0.20),
    )
    two_defect_size_patterns: tuple[tuple[object, ...], ...] = (
        (0.45, 'large', 'small'),
        (0.30, 'large', 'medium'),
        (0.20, 'medium', 'small'),
        (0.05, 'medium', 'medium'),
    )
    three_defect_size_patterns: tuple[tuple[object, ...], ...] = (
        (0.55, 'large', 'medium', 'small'),
        (0.25, 'large', 'small', 'small'),
        (0.15, 'medium', 'medium', 'small'),
        (0.05, 'large', 'medium', 'medium'),
    )


def wrapped_theta_delta_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def surface_distance_mm(a_theta: float, a_z: float, b_theta: float, b_z: float) -> float:
    ds = math.radians(wrapped_theta_delta_deg(a_theta, b_theta)) * shell.PIPE.mid_radius_mm
    dz = a_z - b_z
    return math.hypot(ds, dz)


def _weighted_choice(rng: random.Random, choices: list[tuple[object, float]]) -> object:
    total = sum(max(float(weight), 0.0) for _value, weight in choices)
    if total <= 0.0:
        raise ValueError('Weighted choice requires at least one positive weight.')
    threshold = rng.random() * total
    running = 0.0
    fallback = choices[-1][0]
    for value, weight in choices:
        weight = max(float(weight), 0.0)
        if weight <= 0.0:
            continue
        running += weight
        fallback = value
        if threshold <= running:
            return value
    return fallback


def _sample_range(rng: random.Random, values: tuple[float, float]) -> float:
    lower, upper = values
    if upper < lower:
        raise ValueError(f'Invalid range {values}: upper bound is smaller than lower bound.')
    return rng.uniform(lower, upper)


def _size_class_lookup(config: DefectSamplingConfig) -> dict[str, DefectSizeClass]:
    lookup: dict[str, DefectSizeClass] = {}
    for item in config.size_classes:
        if item.name in lookup:
            raise ValueError(f'Duplicate defect size class: {item.name}')
        lookup[item.name] = item
    return lookup


def _choose_defect_count(rng: random.Random, config: DefectSamplingConfig) -> int:
    if config.max_defects < config.min_defects:
        raise ValueError('max_defects must be >= min_defects')
    if config.max_defects == config.min_defects:
        return config.min_defects
    candidates = list(range(config.min_defects, config.max_defects + 1))
    if config.min_defects == 1 and len(config.defect_count_weights) >= config.max_defects:
        weights = config.defect_count_weights[:config.max_defects]
        return int(_weighted_choice(rng, list(zip(candidates, weights))))
    if len(config.defect_count_weights) == len(candidates):
        return int(_weighted_choice(rng, list(zip(candidates, config.defect_count_weights))))
    return rng.randint(config.min_defects, config.max_defects)


def _weighted_pattern(rng: random.Random, rows: tuple[tuple[object, ...], ...]) -> tuple[str, ...]:
    if not rows:
        raise ValueError('Size-mixture pattern list is empty.')
    choices = [(tuple(str(item) for item in row[1:]), float(row[0])) for row in rows]
    return tuple(_weighted_choice(rng, choices))


def _choose_size_plan(rng: random.Random, count: int, config: DefectSamplingConfig) -> list[str]:
    if count <= 0:
        return []
    if not config.size_mixture:
        return ['legacy'] * count

    lookup = _size_class_lookup(config)
    if count == 1:
        plan = [str(_weighted_choice(rng, [(name, weight) for name, weight in config.single_size_weights]))]
    elif count == 2:
        plan = list(_weighted_pattern(rng, config.two_defect_size_patterns))
    else:
        plan = list(_weighted_pattern(rng, config.three_defect_size_patterns))
        while len(plan) < count:
            plan.append(str(_weighted_choice(rng, [('medium', 0.65), ('small', 0.35)])))
        plan = plan[:count]

    unknown = [name for name in plan if name not in lookup]
    if unknown:
        raise ValueError(f'Unknown defect size class in sampling plan: {unknown}')
    order = {'large': 0, 'medium': 1, 'small': 2}
    return sorted(plan, key=lambda name: order.get(name, 99))


def _sample_defect_shape(
    rng: random.Random,
    config: DefectSamplingConfig,
    size_class: str,
    lookup: dict[str, DefectSizeClass],
) -> tuple[float, float, tuple[int, int]]:
    if size_class == 'legacy':
        return (
            _sample_range(rng, config.diameter_range_mm),
            _sample_range(rng, config.depth_range_mm),
            config.convex_lobes_range,
        )
    spec = lookup[size_class]
    return (
        _sample_range(rng, spec.diameter_range_mm),
        _sample_range(rng, spec.depth_range_mm),
        spec.lobe_count_range,
    )


def generate_sample(sample_id: int, seed: int, config: DefectSamplingConfig) -> GeneratedSample:
    rng = random.Random(seed)
    count = _choose_defect_count(rng, config)
    size_lookup = _size_class_lookup(config)
    size_plan = _choose_size_plan(rng, count, config)
    placed: list[GeneratedDefect] = []
    footprints: list[float] = []
    z_min = config.z_margin_mm
    z_max = shell.PIPE.length_mm - config.z_margin_mm

    for size_class in size_plan:
        diameter_mm, depth_mm, lobe_range = _sample_defect_shape(rng, config, size_class, size_lookup)
        depth_mm = min(depth_mm, config.max_total_depth_mm)
        aspect = _sample_range(rng, config.aspect_ratio_range)
        diameter_theta_mm = diameter_mm * math.sqrt(aspect)
        diameter_z_mm = diameter_mm / math.sqrt(aspect)
        radius_mm = diameter_mm / 2.0
        footprint = max(diameter_theta_mm, diameter_z_mm) * 0.8
        valid = False
        theta_deg = 0.0
        z_mm = shell.PIPE.length_mm / 2.0
        for _attempt in range(1000):
            z_mm = rng.uniform(z_min + footprint, z_max - footprint)
            theta_margin = math.degrees(footprint / shell.PIPE.mid_radius_mm)
            theta_deg = rng.uniform(theta_margin, 360.0 - theta_margin)
            valid = all(
                surface_distance_mm(theta_deg, z_mm, other.theta_deg, other.z_mm)
                > footprint + other_footprint + config.clearance_mm
                for other, other_footprint in zip(placed, footprints)
            )
            if valid:
                break
        if not valid:
            continue
        lobe_count = rng.randint(*lobe_range) if config.irregular_lobes else 0
        placed.append(
            GeneratedDefect(
                theta_deg,
                z_mm,
                diameter_mm,
                depth_mm,
                lobe_count,
                diameter_theta_mm,
                diameter_z_mm,
                size_class,
            )
        )
        footprints.append(footprint)

    lobes: list[GeneratedLobe] = []
    for parent_index, defect in enumerate(placed, start=1):
        base_radius = defect.diameter_mm / 2.0
        for _ in range(defect.lobe_count):
            remaining_depth = config.max_total_depth_mm - defect.depth_mm
            if remaining_depth <= 0.05:
                continue
            angle = rng.uniform(0.0, 2.0 * math.pi)
            offset = base_radius * rng.uniform(*config.lobe_offset_fraction)
            ds = offset * math.cos(angle)
            dz = offset * math.sin(angle)
            dtheta = math.degrees(ds / shell.PIPE.mid_radius_mm)
            lobes.append(
                GeneratedLobe(
                    parent_index=parent_index,
                    theta_deg=(defect.theta_deg + dtheta) % 360.0,
                    z_mm=max(z_min, min(z_max, defect.z_mm + dz)),
                    radius_mm=base_radius * rng.uniform(*config.lobe_radius_fraction),
                    depth_mm=min(defect.depth_mm * rng.uniform(*config.lobe_depth_fraction), remaining_depth),
                )
            )
    return GeneratedSample(sample_id=sample_id, seed=seed, defects=placed, lobes=lobes)


def to_shell_defects(sample: GeneratedSample) -> tuple[list[shell.DefectConfig], list[shell.DefectLobeConfig]]:
    defects = [
        shell.DefectConfig(
            theta_deg=item.theta_deg,
            z_mm=item.z_mm,
            radius_mm=item.diameter_mm / 2.0,
            depth_mm=item.depth_mm,
            radius_theta_mm=(item.diameter_theta_mm or item.diameter_mm) / 2.0,
            radius_z_mm=(item.diameter_z_mm or item.diameter_mm) / 2.0,
        )
        for item in sample.defects
    ]
    lobes = [
        shell.DefectLobeConfig(
            parent_index=item.parent_index,
            theta_deg=item.theta_deg,
            z_mm=item.z_mm,
            radius_mm=item.radius_mm,
            depth_mm=item.depth_mm,
            radius_theta_mm=item.radius_theta_mm or item.radius_mm,
            radius_z_mm=item.radius_z_mm or item.radius_mm,
        )
        for item in sample.lobes
    ]
    return defects, lobes


def sample_to_dict(sample: GeneratedSample) -> dict:
    return {
        'sample_id': sample.sample_id,
        'seed': sample.seed,
        'defects': [asdict(item) for item in sample.defects],
        'lobes': [asdict(item) for item in sample.lobes],
    }


def write_metadata(path: Path, sample: GeneratedSample, dataset: str, model_meta: dict, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'dataset': dataset,
        'sample': sample_to_dict(sample),
        'model': model_meta,
        'extra': extra,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

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
class DefectSamplingConfig:
    min_defects: int = 1
    max_defects: int = 4
    diameter_range_mm: tuple[float, float] = (80.0, 230.0)
    depth_range_mm: tuple[float, float] = (0.6, 3.0)
    aspect_ratio_range: tuple[float, float] = (1.0, 1.0)
    z_margin_mm: float = 260.0
    clearance_mm: float = 50.0
    irregular_lobes: bool = True
    convex_lobes_range: tuple[int, int] = (4, 8)
    lobe_radius_fraction: tuple[float, float] = (0.18, 0.35)
    lobe_offset_fraction: tuple[float, float] = (0.65, 0.95)
    lobe_depth_fraction: tuple[float, float] = (0.45, 1.0)


def wrapped_theta_delta_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def surface_distance_mm(a_theta: float, a_z: float, b_theta: float, b_z: float) -> float:
    ds = math.radians(wrapped_theta_delta_deg(a_theta, b_theta)) * shell.PIPE.mid_radius_mm
    dz = a_z - b_z
    return math.hypot(ds, dz)


def generate_sample(sample_id: int, seed: int, config: DefectSamplingConfig) -> GeneratedSample:
    rng = random.Random(seed)
    count = rng.randint(config.min_defects, config.max_defects)
    placed: list[GeneratedDefect] = []
    footprints: list[float] = []
    z_min = config.z_margin_mm
    z_max = shell.PIPE.length_mm - config.z_margin_mm

    for _ in range(count):
        diameter_mm = rng.uniform(*config.diameter_range_mm)
        depth_mm = rng.uniform(*config.depth_range_mm)
        aspect = rng.uniform(*config.aspect_ratio_range)
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
        lobe_count = rng.randint(*config.convex_lobes_range) if config.irregular_lobes else 0
        placed.append(
            GeneratedDefect(
                theta_deg,
                z_mm,
                diameter_mm,
                depth_mm,
                lobe_count,
                diameter_theta_mm,
                diameter_z_mm,
            )
        )
        footprints.append(footprint)

    lobes: list[GeneratedLobe] = []
    for parent_index, defect in enumerate(placed, start=1):
        base_radius = defect.diameter_mm / 2.0
        for _ in range(defect.lobe_count):
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
                    depth_mm=defect.depth_mm * rng.uniform(*config.lobe_depth_fraction),
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

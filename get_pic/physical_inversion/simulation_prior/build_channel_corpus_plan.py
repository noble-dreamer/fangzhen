"""Create an independent, distribution-matched COMSOL calibration corpus plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from channel_common import (
    ChannelPriorConfig,
    config_sha256,
    physical_sources_match,
    source_fingerprints,
)
from common import (
    load_geometry,
    model_semantic_sha256,
    sha256_file,
    validate_requested_axes,
)


HERE = Path(__file__).resolve().parent
SIMPLE_ROOT = HERE.parents[2]
if str(SIMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMPLE_ROOT))

import simple_defect_common as defects  # noqa: E402


DEFAULT_CONFIG = HERE / "configs" / "dataset_a_channel_prior.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan independent random COMSOL cases for the matched channel prior."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reuse-compatible", action="store_true")
    return parser.parse_args()


def corpus_sampling_config() -> defects.DefectSamplingConfig:
    """Exactly mirror the formal Dataset A distribution, with independent seeds."""

    return defects.DefectSamplingConfig(
        min_defects=1,
        max_defects=3,
        aspect_ratio_range=(0.75, 1.45),
        irregular_lobes=True,
    )


def build_plan(config: ChannelPriorConfig) -> dict:
    healthy, _geometry = load_geometry(config)
    validate_requested_axes(
        healthy,
        tx_indices=config.simulation_tx_indices,
        frequencies_hz=config.frequencies_hz,
    )
    healthy_metadata = json.loads(config.healthy_metadata_path.read_text(encoding="utf-8"))
    sampling = corpus_sampling_config()
    samples = []
    for index in range(1, config.corpus_sample_count + 1):
        seed = config.corpus_seed0 + index
        generated = defects.generate_sample(100000 + index, seed, sampling)
        sample_id = f"{config.corpus_sample_id_prefix}{index:04d}"
        sample_metadata = defects.sample_to_dict(generated)
        samples.append(
            {
                "sample_id": sample_id,
                "corpus_index": index,
                "synthetic_seed": seed,
                "split": "fit" if index <= config.corpus_fit_count else "validation",
                "sample_metadata": sample_metadata,
                "defect_count": len(generated.defects),
                "lobe_count": len(generated.lobes),
            }
        )
    return {
        "schema_version": 1,
        "artifact_kind": "comsol_distribution_matched_channel_corpus_plan",
        "purpose": (
            "simulation-only complex Rytov channel calibration using independent synthetic cases "
            "drawn from the formal Dataset A defect generator"
        ),
        "real_formal_sample_labels_used": False,
        "config": config.to_dict(),
        "config_sha256": config_sha256(config),
        "healthy_npz": str(config.healthy_path),
        "healthy_npz_sha256": sha256_file(config.healthy_path),
        "healthy_metadata": str(config.healthy_metadata_path),
        "healthy_model_semantic_sha256": model_semantic_sha256(healthy_metadata),
        "source_fingerprints": source_fingerprints(),
        "sampling": {
            "implementation": "simple_defect_common.DefectSamplingConfig",
            "formal_dataset_match": {
                "min_defects": 1,
                "max_defects": 3,
                "aspect_ratio_range": [0.75, 1.45],
                "irregular_lobes": True
            },
            "seed_range": [config.corpus_seed0 + 1, config.corpus_seed0 + config.corpus_sample_count],
            "disjoint_from_formal_dataset_seed_rule": "formal seeds are 710000 + sample_id"
        },
        "frequency_hz_ordered": [float(value) for value in config.frequencies_hz],
        "simulation_tx_indices": list(config.simulation_tx_indices),
        "samples": samples,
    }


def compatible(existing: dict, config: ChannelPriorConfig) -> bool:
    healthy_metadata = json.loads(config.healthy_metadata_path.read_text(encoding="utf-8"))
    return (
        existing.get("artifact_kind") == "comsol_distribution_matched_channel_corpus_plan"
        and existing.get("config_sha256") == config_sha256(config)
        and existing.get("healthy_npz_sha256") == sha256_file(config.healthy_path)
        and existing.get("healthy_model_semantic_sha256") == model_semantic_sha256(healthy_metadata)
        and physical_sources_match(existing.get("source_fingerprints"))
    )


def main() -> None:
    args = parse_args()
    config = ChannelPriorConfig.from_json(args.config.resolve())
    output = args.output.resolve() if args.output else config.plan_path
    if output.exists() and args.reuse_compatible and not args.force:
        existing = json.loads(output.read_text(encoding="utf-8"))
        if not compatible(existing, config):
            raise RuntimeError(f"Existing corpus plan is incompatible: {output}")
        print(f"Reusing compatible corpus plan: {output}")
        return
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --force")
    plan = build_plan(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    fit_count = sum(item["split"] == "fit" for item in plan["samples"])
    print(f"Channel corpus plan: {output}")
    print(
        f"samples={len(plan['samples'])}, fit={fit_count}, validation={len(plan['samples']) - fit_count}, "
        f"COMSOL cases={len(plan['samples']) * len(config.simulation_tx_indices) * len(config.frequencies_hz)}"
    )


if __name__ == "__main__":
    main()

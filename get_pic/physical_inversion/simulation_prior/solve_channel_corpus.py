"""Run local COMSOL frequency solves for the matched synthetic calibration corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from channel_common import (
    ChannelPriorConfig,
    canonical_sample_metadata,
    config_sha256,
    physical_sources_match,
    response_is_complete,
    source_fingerprints,
)
from common import model_semantic_sha256, sha256_file, validate_requested_axes


HERE = Path(__file__).resolve().parent
SIMPLE_ROOT = HERE.parents[2]
F_DOMAIN_ROOT = SIMPLE_ROOT / "f_domain"
if str(SIMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMPLE_ROOT))
if str(F_DOMAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(F_DOMAIN_ROOT))

import frequency_domain_common as fcommon  # noqa: E402
import simple_shell_common as shell  # noqa: E402
from simple.get_pic import coarse_map_common as cm  # noqa: E402


DEFAULT_CONFIG = HERE / "configs" / "dataset_a_channel_prior.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve independent distribution-matched COMSOL channel-prior samples."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--sample-id", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--resume-incomplete", action="store_true")
    parser.add_argument("--skip-label-preview", action="store_true")
    parser.add_argument("--checkpoint-final-only", action="store_true")
    parser.add_argument("--heartbeat-s", type=float, default=fcommon.streaming.DEFAULT_HEARTBEAT_S)
    parser.add_argument("--cores", type=int, default=None)
    fcommon.add_frequency_solver_arguments(parser)
    return parser.parse_args()


def validate_plan(plan: dict[str, Any], config: ChannelPriorConfig) -> None:
    if plan.get("artifact_kind") != "comsol_distribution_matched_channel_corpus_plan":
        raise RuntimeError(f"Unexpected corpus plan artifact: {plan.get('artifact_kind')}")
    if plan.get("config_sha256") != config_sha256(config):
        raise RuntimeError("Corpus plan does not match the current channel-prior config")
    if plan.get("healthy_npz_sha256") != sha256_file(config.healthy_path):
        raise RuntimeError("Corpus plan was generated from a different formal healthy response")
    healthy_metadata = json.loads(config.healthy_metadata_path.read_text(encoding="utf-8"))
    if plan.get("healthy_model_semantic_sha256") != model_semantic_sha256(healthy_metadata):
        raise RuntimeError("Corpus plan formal-model fingerprint mismatch")
    if not physical_sources_match(plan.get("source_fingerprints")):
        raise RuntimeError("Corpus plan physical-model fingerprint mismatch; regenerate it")


def select_samples(plan: dict[str, Any], requested: list[str] | None) -> list[dict[str, Any]]:
    samples = list(plan.get("samples", []))
    if requested is None:
        return samples
    requested_ids = {token for value in requested for token in str(value).split(",") if token}
    selected = [item for item in samples if item["sample_id"] in requested_ids]
    missing = sorted(requested_ids.difference(item["sample_id"] for item in selected))
    if missing:
        raise ValueError(f"Unknown corpus sample IDs: {missing}")
    return selected


def model_components(sample: dict[str, Any]) -> tuple[list[shell.DefectConfig], list[shell.DefectLobeConfig]]:
    metadata = sample["sample_metadata"]
    defects = []
    for item in metadata.get("defects", []):
        diameter = float(item["diameter_mm"])
        defects.append(
            shell.DefectConfig(
                theta_deg=float(item["theta_deg"]),
                z_mm=float(item["z_mm"]),
                radius_mm=diameter / 2.0,
                depth_mm=float(item["depth_mm"]),
                radius_theta_mm=float(item.get("diameter_theta_mm") or diameter) / 2.0,
                radius_z_mm=float(item.get("diameter_z_mm") or diameter) / 2.0,
            )
        )
    lobes = []
    for item in metadata.get("lobes", []):
        radius = float(item["radius_mm"])
        lobes.append(
            shell.DefectLobeConfig(
                parent_index=int(item["parent_index"]),
                theta_deg=float(item["theta_deg"]),
                z_mm=float(item["z_mm"]),
                radius_mm=radius,
                depth_mm=float(item["depth_mm"]),
                radius_theta_mm=float(item.get("radius_theta_mm") or radius),
                radius_z_mm=float(item.get("radius_z_mm") or radius),
            )
        )
    if not defects:
        raise RuntimeError(f"Corpus sample {sample['sample_id']} has no defects")
    return defects, lobes


def main() -> None:
    args = parse_args()
    config = ChannelPriorConfig.from_json(args.config.resolve())
    plan_path = args.plan.resolve() if args.plan else config.plan_path
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan, config)
    output_root = args.output_root.resolve() if args.output_root else config.corpus_output_path
    healthy = cm.load_frequency_response(config.healthy_path)
    validate_requested_axes(
        healthy,
        tx_indices=config.simulation_tx_indices,
        frequencies_hz=config.frequencies_hz,
    )
    samples = select_samples(plan, args.sample_id)
    if not samples:
        raise ValueError("No corpus samples selected")
    cases = fcommon.make_cases(config.simulation_tx_indices, config.frequencies_hz)
    expected_model_sha = plan["healthy_model_semantic_sha256"]
    pending: list[dict[str, Any]] = []
    collisions: list[str] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        response_path = output_root / "frequency_response" / f"{sample_id}_H_complex.npz"
        metadata_path = output_root / "metadata" / f"{sample_id}.json"
        expected_sample = canonical_sample_metadata(sample, plan_path, config)
        complete = response_is_complete(
            response_path,
            metadata_path,
            tx_indices=config.simulation_tx_indices,
            rx_indices=tuple(int(value) for value in healthy.rx_indices),
            frequencies_hz=config.frequencies_hz,
            expected_sample=expected_sample,
        )
        if complete:
            if args.overwrite_existing:
                pending.append(sample)
            else:
                print(f"[skip] complete matching corpus response: {response_path.name}")
        elif response_path.exists() or metadata_path.exists():
            if args.overwrite_existing or args.resume_incomplete:
                pending.append(sample)
            else:
                collisions.append(str(response_path))
        else:
            pending.append(sample)
    if collisions:
        raise RuntimeError(
            "Incomplete or incompatible corpus outputs exist; pass --resume-incomplete or use a new output root: "
            + ", ".join(collisions[:5])
        )
    print(f"Plan: {plan_path}")
    print(f"Output root: {output_root}")
    print(f"Selected samples: {len(samples)}, pending: {len(pending)}")
    print(f"COMSOL cases pending: {len(pending) * len(cases)}")
    if args.dry_run:
        print("Dry run complete; COMSOL client was not started.")
        return
    if not pending:
        print("All selected corpus responses already exist and are complete.")
        return

    fcommon.configure_dataset_a_frequency(use_parametric_sweep=False)
    fcommon.apply_solver_arguments(args)
    client = shell.start_client(cores=args.cores)
    rows: list[dict[str, Any]] = []
    try:
        for index, sample in enumerate(pending, start=1):
            sample_id = str(sample["sample_id"])
            defects, lobes = model_components(sample)
            print(f"[{index}/{len(pending)}] solving {sample_id}")
            result = fcommon.solve_export_frequency_sample(
                client=client,
                dataset="A_frequency_simulation_channel_prior",
                sample_id=sample_id,
                defect_state="independent_distribution_matched_outer_corrosion",
                output_root=output_root,
                cases=cases,
                defects=defects,
                lobes=lobes,
                sample_metadata=canonical_sample_metadata(sample, plan_path, config),
                clear_each_case=True,
                heartbeat_s=args.heartbeat_s,
                reuse_sample_model=not args.rebuild_each_case,
                write_label_preview=not args.skip_label_preview,
                keep_case_csv=False,
                checkpoint_every_cases=0 if args.checkpoint_final_only else len(config.frequencies_hz),
                clear_solution_after_each_export=True,
            )
            rows.append(
                {
                    "sample_id": result.sample_id,
                    "split": sample["split"],
                    "seed": sample["synthetic_seed"],
                    "defect_count": sample["defect_count"],
                    "lobe_count": sample["lobe_count"],
                    "case_count": result.case_count,
                    "response_npz": str(output_root / "frequency_response" / f"{sample_id}_H_complex.npz"),
                    "metadata": str(result.metadata_path),
                    "status": "solved",
                }
            )
    finally:
        client.clear()
    if rows:
        fcommon.write_manifest(output_root / "manifest.csv", rows)
    print(f"Corpus manifest: {output_root / 'manifest.csv'}")


if __name__ == "__main__":
    main()

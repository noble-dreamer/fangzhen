from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT.parent / "diffusion"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.append(str(SOURCE_ROOT))

from data.dataset import build_dataset_from_config
from physics.ray_operator import RayOperator
from sample_edm import (
    load_label_display_spec,
    resolve_sample_output_directory,
    save_physical_posterior_outputs,
    save_preview,
    save_uncertainty_preview,
)
from train_edm import build_model
from uncertainty import (
    load_posterior_outputs,
    posterior_metrics,
    sample_posterior,
    save_posterior_outputs,
)
from utils.config import ensure_dir, load_config
from utils.metrics import pearson_np, safe_float, ssim_torch
from utils.reproducibility import seed_everything
from utils.training import move_batch, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EDM predictions or checkpoints.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--pred-dir", type=Path, default=None, help="Evaluate existing *_prediction.npy files.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--sample-ids", nargs="*", default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--thresholds", type=float, nargs="*", default=[0.1, 0.2, 0.3])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-previews", action="store_true")
    parser.add_argument("--num-posterior-samples", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--lower-quantile", type=float, default=None)
    parser.add_argument("--upper-quantile", type=float, default=None)
    parser.add_argument("--defect-threshold", type=float, default=None)
    parser.add_argument("--consensus-probability-threshold", type=float, default=None)
    parser.add_argument("--save-all-samples", action="store_true", default=None)
    parser.add_argument("--physics-guidance-scale", type=float, default=None)
    parser.add_argument("--physics-guidance-start-fraction", type=float, default=None)
    parser.add_argument("--physics-feature-index", type=int, default=1)
    return parser.parse_args()


def regression_metrics(pred: np.ndarray, target: np.ndarray, thresholds: list[float]) -> dict[str, float]:
    valid = np.isfinite(pred) & np.isfinite(target)
    if not np.any(valid):
        return {"mae": float("nan"), "rmse": float("nan"), "nrmse": float("nan"), "pearson": float("nan")}
    diff = pred[valid] - target[valid]
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    scale = max(float(np.max(target[valid]) - np.min(target[valid])), 1e-6)
    result = {
        "mae": mae,
        "rmse": rmse,
        "nrmse": rmse / scale,
        "pearson": pearson_np(pred, target),
        "volume_error": float(abs(np.sum(pred[valid]) - np.sum(target[valid])) / max(abs(np.sum(target[valid])), 1e-6)),
    }
    for threshold in thresholds:
        pred_mask = pred >= threshold
        target_mask = target >= threshold
        union = np.logical_or(pred_mask, target_mask).sum()
        inter = np.logical_and(pred_mask, target_mask).sum()
        denom_dice = pred_mask.sum() + target_mask.sum()
        key = str(threshold).replace(".", "")
        result[f"iou_{key}"] = float(inter / union) if union > 0 else float("nan")
        result[f"dice_{key}"] = float(2 * inter / denom_dice) if denom_dice > 0 else float("nan")
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 1234)))
    if args.sample_ids is not None:
        config.setdefault("data", {}).setdefault(args.split, {})["sample_ids"] = args.sample_ids
    dataset = build_dataset_from_config(config, split=args.split)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    output_dir = ensure_dir(args.output_dir or Path(config.get("run_dir", ROOT / "runs" / "edm_debug")) / "eval")
    device = resolve_device(args.device)
    model = None
    if args.pred_dir is None:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when --pred-dir is not provided")
        model = build_model(config).to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_key = "ema" if args.use_ema and "ema" in checkpoint else "model"
        model.load_state_dict(checkpoint[state_key], strict=True)
        model.eval()
    sample_cfg = config.get("sample", {})
    uncertainty_cfg = sample_cfg.get("uncertainty", {})
    num_posterior_samples = int(
        args.num_posterior_samples
        if args.num_posterior_samples is not None
        else uncertainty_cfg.get("num_samples", 16)
    )
    sample_seed = int(
        args.sample_seed
        if args.sample_seed is not None
        else uncertainty_cfg.get("sample_seed", config.get("seed", 1234))
    )
    lower_quantile = float(
        args.lower_quantile
        if args.lower_quantile is not None
        else uncertainty_cfg.get("lower_quantile", 0.05)
    )
    upper_quantile = float(
        args.upper_quantile
        if args.upper_quantile is not None
        else uncertainty_cfg.get("upper_quantile", 0.95)
    )
    defect_threshold = float(
        args.defect_threshold
        if args.defect_threshold is not None
        else uncertainty_cfg.get("defect_threshold", 0.1)
    )
    consensus_probability_threshold = float(
        args.consensus_probability_threshold
        if args.consensus_probability_threshold is not None
        else uncertainty_cfg.get("consensus_probability_threshold", 0.5)
    )
    save_all_samples = bool(
        args.save_all_samples
        if args.save_all_samples is not None
        else uncertainty_cfg.get("save_all_samples", False)
    )
    physics_guidance_scale = float(
        args.physics_guidance_scale
        if args.physics_guidance_scale is not None
        else sample_cfg.get("physics_guidance_scale", 0.0)
    )
    physics_guidance_start_fraction = float(
        args.physics_guidance_start_fraction
        if args.physics_guidance_start_fraction is not None
        else sample_cfg.get("physics_guidance_start_fraction", 0.5)
    )
    physics_operator = None
    if args.pred_dir is None and physics_guidance_scale > 0.0:
        image_size = int(config.get("data", {}).get("image_size", 256))
        physics_operator = RayOperator(image_shape=(image_size, image_size)).to(device)
    edm_cfg = config.get("edm", {})
    steps = args.steps or int(sample_cfg.get("steps", 32))
    sample_kwargs = {
        "steps": steps,
        "sigma_min": float(sample_cfg.get("sigma_min", edm_cfg.get("sigma_min", model.sigma_min if model is not None else 0.002))),
        "sigma_max": float(sample_cfg.get("sigma_max", edm_cfg.get("sigma_max", model.sigma_max if model is not None else 80.0))),
        "rho": float(sample_cfg.get("rho", edm_cfg.get("rho", model.rho if model is not None else 7.0))),
        "s_churn": float(sample_cfg.get("s_churn", 0.0)),
        "s_tmin": float(sample_cfg.get("s_tmin", 0.0)),
        "s_tmax": float(sample_cfg.get("s_tmax", float("inf"))),
        "s_noise": float(sample_cfg.get("s_noise", 1.0)),
        "physics_operator": physics_operator,
        "physics_guidance_scale": physics_guidance_scale,
        "physics_guidance_start_fraction": physics_guidance_start_fraction,
        "physics_feature_index": args.physics_feature_index,
    }
    pic_channel_names = list(config.get("data", {}).get("pic_channels", []))
    coverage_index = pic_channel_names.index("path_coverage") if "path_coverage" in pic_channel_names else None
    reliability_index = pic_channel_names.index("reliability_mask") if "reliability_mask" in pic_channel_names else None
    rows = []
    for index, batch in enumerate(tqdm(loader, desc="edm evaluating", dynamic_ncols=True)):
        if args.max_samples is not None and index >= args.max_samples:
            break
        sample_id = batch["sample_id"][0]
        target_np = batch["target"][0, 0].numpy().astype(np.float32)
        coarse_np = batch["pic"][0, min(1, batch["pic"].shape[1] - 1)].numpy().astype(np.float32)
        display = load_label_display_spec(batch["label_path"][0], target_np)
        posterior = None
        if args.pred_dir is not None:
            prediction_dir = resolve_sample_output_directory(args.pred_dir, sample_id)
            pred_path = prediction_dir / f"{sample_id}_prediction.npy"
            if not pred_path.exists():
                raise FileNotFoundError(pred_path)
            pred_np = np.asarray(np.load(pred_path), dtype=np.float32)
            posterior = load_posterior_outputs(prediction_dir, sample_id)
            if posterior is not None:
                pred_np = posterior.mean[0, 0].numpy().astype(np.float32)
        else:
            batch = move_batch(batch, device)
            posterior_seed = sample_seed + index * num_posterior_samples
            posterior = sample_posterior(
                model,
                batch["pic"],
                batch["x_matrix"],
                frequency_hz=batch["frequency_hz"],
                tx_indices=batch["tx_indices"],
                rx_indices=batch["rx_indices"],
                num_samples=num_posterior_samples,
                sample_seed=posterior_seed,
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
                defect_threshold=defect_threshold,
                consensus_probability_threshold=consensus_probability_threshold,
                sample_kwargs=sample_kwargs,
            )
            save_posterior_outputs(
                output_dir,
                sample_id,
                posterior,
                save_all_samples=save_all_samples,
            )
            pred_np = posterior.mean[0, 0].numpy().astype(np.float32)
            target_np = batch["target"][0, 0].detach().cpu().numpy().astype(np.float32)
            coarse_np = batch["pic"][0, min(1, batch["pic"].shape[1] - 1)].detach().cpu().numpy().astype(np.float32)
        if pred_np.shape != target_np.shape:
            raise RuntimeError(f"Prediction shape {pred_np.shape} does not match target {target_np.shape} for {sample_id}")
        if posterior is None:
            np.save(output_dir / f"{sample_id}_prediction_mm.npy", display.depth_mm(pred_np))
        else:
            save_physical_posterior_outputs(output_dir, sample_id, posterior, display)
        metrics = regression_metrics(pred_np, target_np, args.thresholds)
        metrics["mae_mm"] = metrics["mae"] * display.normalization_denominator_mm
        metrics["rmse_mm"] = metrics["rmse"] * display.normalization_denominator_mm
        metrics["normalization_denominator_mm"] = display.normalization_denominator_mm
        metrics["physical_depth_limit_mm"] = display.depth_limit_mm
        physical_limit_normalized = display.depth_limit_mm / display.normalization_denominator_mm
        metrics["prediction_above_physical_limit_fraction"] = float(
            np.mean(pred_np > physical_limit_normalized)
        )
        try:
            metrics["ssim"] = safe_float(ssim_torch(torch.from_numpy(pred_np)[None, None], torch.from_numpy(target_np)[None, None]))
        except Exception:
            metrics["ssim"] = float("nan")
        if posterior is not None:
            source_pic = batch["pic"]
            coverage_np = None
            reliability_np = None
            if coverage_index is not None:
                coverage_np = source_pic[0, coverage_index].detach().cpu().numpy().astype(np.float32)
            if reliability_index is not None:
                reliability_np = source_pic[0, reliability_index].detach().cpu().numpy().astype(np.float32)
            metrics.update(
                posterior_metrics(
                    posterior,
                    target_np,
                    coverage=coverage_np,
                    reliability=reliability_np,
                )
            )
            consensus_np = posterior.consensus_prediction[0, 0].numpy().astype(np.float32)
            consensus_metrics = regression_metrics(consensus_np, target_np, args.thresholds)
            metrics.update({f"consensus_{key}": value for key, value in consensus_metrics.items()})
            metrics["consensus_mae_mm"] = (
                consensus_metrics["mae"] * display.normalization_denominator_mm
            )
            metrics["consensus_rmse_mm"] = (
                consensus_metrics["rmse"] * display.normalization_denominator_mm
            )
        rows.append({"sample_id": sample_id, **metrics})
        if args.save_previews:
            if posterior is None:
                save_preview(
                    output_dir / f"{sample_id}_preview.png",
                    pred_np,
                    target_np,
                    coarse_np,
                    display,
                )
            else:
                save_uncertainty_preview(
                    output_dir / f"{sample_id}_preview.png",
                    posterior,
                    target_np,
                    coarse_np,
                    display,
                )
    if not rows:
        raise RuntimeError("No samples evaluated")
    fields = ["sample_id"] + sorted({key for row in rows for key in row if key != "sample_id"})
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for field in fields:
        if field == "sample_id":
            continue
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        summary[field] = {
            "mean": float(np.mean(values)) if values.size else float("nan"),
            "std": float(np.std(values)) if values.size else float("nan"),
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

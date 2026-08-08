from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import build_dataset_from_config
from models import ConditionalRegressor
from physics.ray_operator import RayOperator
from train_diffusion import build_model as build_diffusion_model
from train_regressor import build_model as build_regressor_model
from utils.config import ensure_dir, load_config
from utils.metrics import pearson_np, safe_float, ssim_torch, tensor_iou
from utils.reproducibility import seed_everything
from utils.training import move_batch, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predictions or model checkpoints.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model-type", choices=["diffusion", "regressor"], default="diffusion")
    parser.add_argument("--pred-dir", type=Path, default=None, help="Evaluate existing *_prediction.npy files.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--sample-ids", nargs="*", default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--thresholds", type=float, nargs="*", default=[0.1, 0.2, 0.3])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--physics-guidance-scale", type=float, default=0.0)
    parser.add_argument("--physics-guidance-start-fraction", type=float, default=0.5)
    parser.add_argument("--physics-feature-index", type=int, default=1)
    return parser.parse_args()


def regression_metrics(pred: np.ndarray, target: np.ndarray, thresholds: list[float]) -> dict[str, float]:
    valid = np.isfinite(pred) & np.isfinite(target)
    if not np.any(valid):
        return {"mae": float("nan"), "rmse": float("nan"), "nrmse": float("nan"), "pearson": float("nan")}
    diff = pred[valid] - target[valid]
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
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


def predict_batch(
    model,
    model_type: str,
    batch: dict,
    steps: int | None,
    *,
    physics_operator=None,
    physics_guidance_scale: float = 0.0,
    physics_guidance_start_fraction: float = 0.5,
    physics_feature_index: int = 1,
) -> torch.Tensor:
    if model_type == "regressor":
        with torch.no_grad():
            return model(batch["pic"], batch["x_matrix"])
    return model.sample(
        batch["pic"],
        batch["x_matrix"],
        steps=steps,
        physics_operator=physics_operator,
        physics_guidance_scale=physics_guidance_scale,
        physics_guidance_start_fraction=physics_guidance_start_fraction,
        physics_feature_index=physics_feature_index,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 1234)))
    if args.sample_ids is not None:
        config.setdefault("data", {}).setdefault(args.split, {})["sample_ids"] = args.sample_ids
    dataset = build_dataset_from_config(config, split=args.split)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    output_dir = ensure_dir(args.output_dir or Path(config.get("run_dir", ROOT / "runs" / args.model_type)) / "eval")
    device = resolve_device(args.device)
    model = None
    if args.pred_dir is None:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when --pred-dir is not provided")
        model = build_regressor_model(config) if args.model_type == "regressor" else build_diffusion_model(config)
        model = model.to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_key = "ema" if args.use_ema and "ema" in checkpoint else "model"
        model.load_state_dict(checkpoint[state_key], strict=True)
        model.eval()
    physics_operator = None
    if args.pred_dir is None and args.model_type == "diffusion" and args.physics_guidance_scale > 0.0:
        image_size = int(config.get("data", {}).get("image_size", 256))
        physics_operator = RayOperator(image_shape=(image_size, image_size)).to(device)
    rows = []
    for index, batch in enumerate(tqdm(loader, desc="evaluating", dynamic_ncols=True)):
        if args.max_samples is not None and index >= args.max_samples:
            break
        sample_id = batch["sample_id"][0]
        target_np = batch["target"][0, 0].numpy().astype(np.float32)
        if args.pred_dir is not None:
            pred_path = args.pred_dir / f"{sample_id}_prediction.npy"
            if not pred_path.exists():
                raise FileNotFoundError(pred_path)
            pred_np = np.asarray(np.load(pred_path), dtype=np.float32)
        else:
            batch = move_batch(batch, device)
            pred = predict_batch(
                model,
                args.model_type,
                batch,
                args.steps or config.get("sample", {}).get("steps", 50),
                physics_operator=physics_operator,
                physics_guidance_scale=args.physics_guidance_scale,
                physics_guidance_start_fraction=args.physics_guidance_start_fraction,
                physics_feature_index=args.physics_feature_index,
            )
            pred_np = pred[0, 0].detach().cpu().numpy().astype(np.float32)
            target_np = batch["target"][0, 0].detach().cpu().numpy().astype(np.float32)
        if pred_np.shape != target_np.shape:
            raise RuntimeError(f"Prediction shape {pred_np.shape} does not match target {target_np.shape} for {sample_id}")
        metrics = regression_metrics(pred_np, target_np, args.thresholds)
        try:
            ssim_value = ssim_torch(
                torch.from_numpy(pred_np)[None, None],
                torch.from_numpy(target_np)[None, None],
            )
            metrics["ssim"] = safe_float(ssim_value)
        except Exception:
            metrics["ssim"] = float("nan")
        rows.append({"sample_id": sample_id, **metrics})
    if not rows:
        raise RuntimeError("No samples evaluated")
    fields = list(rows[0].keys())
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

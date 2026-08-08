from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import build_dataloaders
from models import ConditionalRegressor
from physics.losses import output_prior_losses
from physics.ray_operator import RayOperator
from utils.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from utils.config import dump_config, ensure_dir, load_config
from utils.ema import ModelEma
from utils.logging import CsvLogger, JsonlLogger
from utils.metrics import safe_float, ssim_torch, tensor_iou, tensor_mae, tensor_rmse
from utils.reproducibility import seed_everything
from utils.training import (
    autocast_context,
    build_optimizer,
    build_scheduler,
    grad_norm,
    make_scaler,
    move_batch,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train deterministic conditional U-Net baseline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint path. Use 'latest' to auto-detect.")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def build_model(config: dict) -> ConditionalRegressor:
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    return ConditionalRegressor(
        pic_channels=len(data_cfg.get("pic_channels", [])) or int(model_cfg.get("pic_channels", 8)),
        x_channels=int(model_cfg.get("x_channels", 7)),
        frequency_count=int(model_cfg.get("frequency_count", 15)),
        image_size=int(data_cfg.get("image_size", model_cfg.get("image_size", 256))),
        base_channels=int(model_cfg.get("base_channels", 48)),
        channel_mult=tuple(model_cfg.get("channel_mult", (1, 2, 4, 4))),
        num_res_blocks=int(model_cfg.get("num_res_blocks", 2)),
        attention_resolutions=tuple(model_cfg.get("attention_resolutions", (32,))),
        cond_dim=int(model_cfg.get("cond_dim", 256)),
        x_hidden_channels=int(model_cfg.get("x_hidden_channels", 64)),
        x_token_dim=int(model_cfg.get("x_token_dim", model_cfg.get("cond_dim", 256))),
        cross_attention_resolutions=tuple(model_cfg.get("cross_attention_resolutions", ())),
        cross_attention_heads=int(model_cfg.get("cross_attention_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.05)),
    )


@torch.no_grad()
def validate(model: torch.nn.Module, loader, device: torch.device, amp: bool, amp_dtype: str) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "mae": 0.0, "rmse": 0.0, "ssim": 0.0, "iou_01": 0.0}
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with autocast_context(device, amp, amp_dtype):
            pred = model(batch["pic"], batch["x_matrix"])
            target = batch["target"]
            loss = F.l1_loss(pred, target)
            mae = tensor_mae(pred, target)
            rmse = tensor_rmse(pred, target)
            ssim = ssim_torch(pred, target)
            iou = tensor_iou(pred, target, threshold=0.1)
        batch_count = int(target.shape[0])
        count += batch_count
        totals["loss"] += safe_float(loss) * batch_count
        totals["mae"] += safe_float(mae) * batch_count
        totals["rmse"] += safe_float(rmse) * batch_count
        totals["ssim"] += safe_float(ssim) * batch_count
        if math.isfinite(safe_float(iou)):
            totals["iou_01"] += safe_float(iou) * batch_count
    if count == 0:
        return {key: float("nan") for key in totals}
    return {key: value / count for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 1234)), deterministic=bool(config.get("deterministic", False)))
    device = resolve_device(args.device)
    train_loader, val_loader = build_dataloaders(config)
    run_dir = ensure_dir(args.run_dir or config.get("run_dir", ROOT / "runs" / "regressor_debug"))
    ensure_dir(run_dir / "checkpoints")
    dump_config(config, run_dir / "config_resolved.yaml")

    train_cfg = config.get("train", {})
    loss_cfg = config.get("loss", {})
    amp = bool(train_cfg.get("amp", True))
    amp_dtype = str(train_cfg.get("amp_dtype", "float16"))
    epochs = int(train_cfg.get("epochs", 20))
    total_steps = max(1, epochs * len(train_loader))
    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, total_steps=total_steps)
    scaler = make_scaler(device, amp, amp_dtype)
    ema = ModelEma(model, decay=float(train_cfg.get("ema_decay", 0.999))) if train_cfg.get("ema", True) else None
    ray_operator = None
    if float(loss_cfg.get("lambda_phys_v1", 0.0)) > 0.0:
        image_size = int(config.get("data", {}).get("image_size", 256))
        ray_operator = RayOperator(image_shape=(image_size, image_size)).to(device)

    start_epoch = 0
    global_step = 0
    best_metric = float("inf")
    resume_path = args.resume
    if resume_path is not None and str(resume_path).lower() == "latest":
        resume_path = latest_checkpoint(run_dir)
    if resume_path:
        checkpoint = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema.module if ema is not None else None,
            map_location=device,
        )
        start_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("step", 0))
        best_metric = float(checkpoint.get("best_metric", best_metric))

    jsonl = JsonlLogger(run_dir / "metrics.jsonl")
    csv = CsvLogger(
        run_dir / "loss_history.csv",
        [
            "epoch",
            "step",
            "lr",
            "loss",
            "loss_l1",
            "loss_ssim",
            "loss_tv",
            "loss_range",
            "loss_coverage_l1",
            "loss_phys_v1",
            "mae",
            "rmse",
            "ssim",
            "grad_norm",
        ],
    )

    save_every = int(train_cfg.get("save_every_steps", 500))
    val_every = int(train_cfg.get("val_every_steps", 500))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    for epoch in range(start_epoch, epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"regressor epoch {epoch + 1}/{epochs}", dynamic_ncols=True)
        for batch in progress:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp, amp_dtype):
                pred = model(batch["pic"], batch["x_matrix"])
                target = batch["target"]
                l1 = F.l1_loss(pred, target)
                ssim_value = ssim_torch(pred, target)
                loss_ssim = 1.0 - ssim_value
                prior = output_prior_losses(
                    pred,
                    target,
                    batch["pic"],
                    lambda_l1=0.0,
                    lambda_tv=float(loss_cfg.get("lambda_tv", 1e-4)),
                    lambda_range=float(loss_cfg.get("lambda_range", 0.01)),
                    lambda_coverage_l1=float(loss_cfg.get("lambda_coverage_l1", 0.05)),
                )
                phys = pred.new_tensor(0.0)
                if ray_operator is not None:
                    phys = ray_operator.consistency_loss(
                        pred,
                        batch["x_matrix"],
                        feature_index=int(loss_cfg.get("phys_feature_index", 1)),
                    )
                loss = (
                    float(loss_cfg.get("lambda_l1", 1.0)) * l1
                    + float(loss_cfg.get("lambda_ssim", 0.2)) * loss_ssim
                    + prior["loss_prior_total"]
                    + float(loss_cfg.get("lambda_phys_v1", 0.0)) * phys
                )
            scaler.scale(loss).backward()
            if grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            gnorm = grad_norm(model.parameters())
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            global_step += 1
            row = {
                "epoch": epoch + 1,
                "step": global_step,
                "lr": optimizer.param_groups[0]["lr"],
                "loss": safe_float(loss),
                "loss_l1": safe_float(l1),
                "loss_ssim": safe_float(loss_ssim),
                "loss_tv": safe_float(prior["loss_tv"]),
                "loss_range": safe_float(prior["loss_range"]),
                "loss_coverage_l1": safe_float(prior["loss_coverage_l1"]),
                "loss_phys_v1": safe_float(phys),
                "mae": safe_float(tensor_mae(pred, target)),
                "rmse": safe_float(tensor_rmse(pred, target)),
                "ssim": safe_float(ssim_value),
                "grad_norm": gnorm,
            }
            csv.write(row)
            if global_step % int(train_cfg.get("log_every_steps", 10)) == 0:
                jsonl.write({"type": "train", **row})
            progress.set_postfix(loss=f"{row['loss']:.4g}", mae=f"{row['mae']:.4g}")
            if val_loader is not None and val_every > 0 and global_step % val_every == 0:
                eval_model = ema.module if ema is not None else model
                metrics = validate(eval_model, val_loader, device, amp, amp_dtype)
                jsonl.write({"type": "val", "epoch": epoch + 1, "step": global_step, **metrics})
                model.train()
                if metrics["loss"] < best_metric:
                    best_metric = metrics["loss"]
                    save_checkpoint(
                        run_dir / "checkpoints" / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        ema=ema.module if ema is not None else None,
                        epoch=epoch + 1,
                        step=global_step,
                        best_metric=best_metric,
                    )
            if save_every > 0 and global_step % save_every == 0:
                save_checkpoint(
                    run_dir / "checkpoints" / f"step_{global_step:08d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    ema=ema.module if ema is not None else None,
                    epoch=epoch + 1,
                    step=global_step,
                    best_metric=best_metric,
                )
        save_checkpoint(
            run_dir / "checkpoints" / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema.module if ema is not None else None,
            epoch=epoch + 1,
            step=global_step,
            best_metric=best_metric,
        )


if __name__ == "__main__":
    main()

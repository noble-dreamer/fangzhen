from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT.parent / "diffusion"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.append(str(SOURCE_ROOT))

from data import build_dataloaders, dataset_sample_ids
from models import EDMDiffusion, PhysicalEncodingConfig
from physics.losses import output_prior_losses
from physics.ray_operator import RayOperator
from utils.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from utils.config import dump_config, ensure_dir, load_config
from utils.ema import ModelEma
from utils.logging import CsvLogger, JsonlLogger
from utils.metrics import safe_float, tensor_mae, tensor_rmse
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
    parser = argparse.ArgumentParser(description="Train EDM conditional defect-map denoiser.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint path. Use 'latest' to auto-detect.")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def build_model(config: dict) -> EDMDiffusion:
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    fusion_cfg = model_cfg.get("spatial_frequency_fusion", {})
    physical_cfg = fusion_cfg.get("physical_position", {})
    edm_cfg = config.get("edm", {})
    return EDMDiffusion(
        pic_channels=len(data_cfg.get("pic_channels", [])) or int(model_cfg.get("pic_channels", 8)),
        x_channels=int(model_cfg.get("x_channels", 7)),
        frequency_count=int(model_cfg.get("frequency_count", 15)),
        image_size=int(data_cfg.get("image_size", model_cfg.get("image_size", 256))),
        target_channels=int(model_cfg.get("target_channels", 1)),
        base_channels=int(model_cfg.get("base_channels", 48)),
        channel_mult=tuple(model_cfg.get("channel_mult", (1, 2, 4, 4))),
        num_res_blocks=int(model_cfg.get("num_res_blocks", 2)),
        attention_resolutions=tuple(model_cfg.get("attention_resolutions", (32,))),
        cond_dim=int(model_cfg.get("cond_dim", 256)),
        x_hidden_channels=int(model_cfg.get("x_hidden_channels", 64)),
        x_token_dim=int(model_cfg.get("x_token_dim", model_cfg.get("cond_dim", 256))),
        frequency_filter_kernel_size=int(fusion_cfg.get("filter_kernel_size", 5)),
        physical_encoding=PhysicalEncodingConfig(
            tx_count=int(physical_cfg.get("tx_count", 16)),
            rx_count=int(physical_cfg.get("rx_count", 16)),
            pipe_length_mm=float(physical_cfg.get("pipe_length_mm", 1000.0)),
            mid_radius_mm=float(physical_cfg.get("mid_radius_mm", 155.0)),
            tx_z_mm=float(physical_cfg.get("tx_z_mm", 100.0)),
            rx_z_mm=float(physical_cfg.get("rx_z_mm", 900.0)),
            frequency_reference_hz=float(physical_cfg.get("frequency_reference_hz", 100000.0)),
        ),
        fusion_resolutions=tuple(fusion_cfg.get("resolutions", ())),
        fusion_heads=int(fusion_cfg.get("heads", 4)),
        self_condition_prob=float(model_cfg.get("self_condition_prob", 0.5)),
        dropout=float(model_cfg.get("dropout", 0.05)),
        sigma_data=float(edm_cfg.get("sigma_data", 0.5)),
        sigma_min=float(edm_cfg.get("sigma_min", 0.002)),
        sigma_max=float(edm_cfg.get("sigma_max", 80.0)),
        p_mean=float(edm_cfg.get("p_mean", -1.2)),
        p_std=float(edm_cfg.get("p_std", 1.2)),
        rho=float(edm_cfg.get("rho", 7.0)),
    )


def spectral_magnitude_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction_spectrum = torch.fft.rfft2(prediction.float(), norm="ortho")
    target_spectrum = torch.fft.rfft2(target.float(), norm="ortho")
    prediction_log_magnitude = torch.log1p(prediction_spectrum.abs())
    target_log_magnitude = torch.log1p(target_spectrum.abs())
    return F.l1_loss(prediction_log_magnitude, target_log_magnitude)


@torch.no_grad()
def validate(model: EDMDiffusion, loader, device: torch.device, amp: bool, amp_dtype: str) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "x0_mae": 0.0, "x0_rmse": 0.0, "sigma_mean": 0.0}
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with autocast_context(device, amp, amp_dtype):
            losses = model.training_losses(
                batch["target"],
                batch["pic"],
                batch["x_matrix"],
                frequency_hz=batch["frequency_hz"],
                tx_indices=batch["tx_indices"],
                rx_indices=batch["rx_indices"],
            )
            x0 = losses["x0_pred"].clamp(0.0, 1.0)
            target = batch["target"]
            loss = losses["loss_edm"]
            mae = tensor_mae(x0, target)
            rmse = tensor_rmse(x0, target)
        batch_count = int(target.shape[0])
        count += batch_count
        totals["loss"] += safe_float(loss) * batch_count
        totals["x0_mae"] += safe_float(mae) * batch_count
        totals["x0_rmse"] += safe_float(rmse) * batch_count
        totals["sigma_mean"] += safe_float(losses["sigma"].float().mean()) * batch_count
    if count == 0:
        return {key: float("nan") for key in totals}
    return {key: value / count for key, value in totals.items()}


def physics_weight(loss_cfg: dict, global_step: int) -> float:
    target = float(loss_cfg.get("lambda_phys_v1", 0.0))
    if target <= 0.0:
        return 0.0
    warmup_start = int(loss_cfg.get("phys_start_step", 0))
    warmup_steps = int(loss_cfg.get("phys_warmup_steps", 0))
    if global_step < warmup_start:
        return 0.0
    if warmup_steps <= 0:
        return target
    alpha = min(1.0, max(0.0, (global_step - warmup_start) / warmup_steps))
    return target * alpha


def write_data_split_manifest(run_dir: Path, train_loader, val_loader, config: dict) -> None:
    """Persist the exact members of each split alongside a formal training run."""

    train_ids = dataset_sample_ids(train_loader.dataset)
    val_ids = [] if val_loader is None else dataset_sample_ids(val_loader.dataset)
    data_cfg = config.get("data", {})
    explicit_val = isinstance(data_cfg.get("val"), dict) and (
        data_cfg["val"].get("sample_ids") is not None
        or data_cfg["val"].get("sample_ids_file") is not None
    )
    payload = {
        "split_mode": "explicit" if explicit_val else "seeded_random",
        "source_config": str(config.get("_config_path", "")),
        "seed": int(config.get("seed", 1234)),
        "train": {"count": len(train_ids), "sample_ids": train_ids},
        "val": {"count": len(val_ids), "sample_ids": val_ids},
    }
    (run_dir / "data_split.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "validation_sample_ids.txt").write_text(
        "\n".join(val_ids) + ("\n" if val_ids else ""),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 1234)), deterministic=bool(config.get("deterministic", False)))
    device = resolve_device(args.device)
    train_loader, val_loader = build_dataloaders(config)
    run_dir = ensure_dir(args.run_dir or config.get("run_dir", ROOT / "runs" / "edm_debug"))
    ensure_dir(run_dir / "checkpoints")
    dump_config(config, run_dir / "config_resolved.yaml")
    write_data_split_manifest(run_dir, train_loader, val_loader, config)

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
            "loss_edm",
            "loss_x0_l1",
            "loss_tv",
            "loss_range",
            "loss_coverage_l1",
            "loss_spectral",
            "loss_phys_v1",
            "lambda_phys_v1",
            "x0_mae",
            "x0_rmse",
            "grad_norm",
            "sigma_mean",
            "sigma_min",
            "sigma_max",
        ],
    )

    save_every = int(train_cfg.get("save_every_steps", 500))
    val_every = int(train_cfg.get("val_every_steps", 500))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    for epoch in range(start_epoch, epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"edm epoch {epoch + 1}/{epochs}", dynamic_ncols=True)
        for batch in progress:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp, amp_dtype):
                losses = model.training_losses(
                    batch["target"],
                    batch["pic"],
                    batch["x_matrix"],
                    frequency_hz=batch["frequency_hz"],
                    tx_indices=batch["tx_indices"],
                    rx_indices=batch["rx_indices"],
                )
                x0_pred = losses["x0_pred"]
                prior = output_prior_losses(
                    x0_pred,
                    batch["target"],
                    batch["pic"],
                    lambda_l1=float(loss_cfg.get("lambda_x0_l1", 0.05)),
                    lambda_tv=float(loss_cfg.get("lambda_tv", 1e-4)),
                    lambda_range=float(loss_cfg.get("lambda_range", 0.01)),
                    lambda_coverage_l1=float(loss_cfg.get("lambda_coverage_l1", 0.0)),
                )
                phys_lambda = physics_weight(loss_cfg, global_step)
                spectral = spectral_magnitude_loss(x0_pred, batch["target"])
                spectral_lambda = float(loss_cfg.get("lambda_spectral", 0.0))
                phys = x0_pred.new_tensor(0.0)
                if ray_operator is not None and phys_lambda > 0.0:
                    phys = ray_operator.consistency_loss(
                        x0_pred.clamp(0.0, 1.0),
                        batch["x_matrix"],
                        feature_index=int(loss_cfg.get("phys_feature_index", 1)),
                    )
                loss = (
                    losses["loss_edm"]
                    + prior["loss_prior_total"]
                    + spectral_lambda * spectral
                    + phys_lambda * phys
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
            sigma = losses["sigma"].detach().float()
            row = {
                "epoch": epoch + 1,
                "step": global_step,
                "lr": optimizer.param_groups[0]["lr"],
                "loss": safe_float(loss),
                "loss_edm": safe_float(losses["loss_edm"]),
                "loss_x0_l1": safe_float(prior["loss_l1"]),
                "loss_tv": safe_float(prior["loss_tv"]),
                "loss_range": safe_float(prior["loss_range"]),
                "loss_coverage_l1": safe_float(prior["loss_coverage_l1"]),
                "loss_spectral": safe_float(spectral),
                "loss_phys_v1": safe_float(phys),
                "lambda_phys_v1": phys_lambda,
                "x0_mae": safe_float(tensor_mae(x0_pred.clamp(0.0, 1.0), batch["target"])),
                "x0_rmse": safe_float(tensor_rmse(x0_pred.clamp(0.0, 1.0), batch["target"])),
                "grad_norm": gnorm,
                "sigma_mean": safe_float(sigma.mean()),
                "sigma_min": safe_float(sigma.min()),
                "sigma_max": safe_float(sigma.max()),
            }
            csv.write(row)
            if global_step % int(train_cfg.get("log_every_steps", 10)) == 0:
                jsonl.write({"type": "train", **row})
            progress.set_postfix(loss=f"{row['loss']:.4g}", x0_mae=f"{row['x0_mae']:.4g}")
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

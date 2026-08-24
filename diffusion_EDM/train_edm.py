from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
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
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Load weights only and reset training state.")
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


class EDMTrainingForward(nn.Module):
    def __init__(self, model: EDMDiffusion) -> None:
        super().__init__()
        self.model = model

    def forward(self, target: torch.Tensor, pic: torch.Tensor, x_matrix: torch.Tensor, **coordinates):
        return self.model.training_losses(target, pic, x_matrix, **coordinates)


def setup_parallel(config: dict, requested_device: str) -> SimpleNamespace:
    cfg = config.get("parallel", {})
    mode = str(cfg.get("mode", "auto")).lower()
    ids = [int(value) for value in cfg.get("device_ids", [])]
    required = int(cfg.get("require_device_count", len(ids)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    visible = torch.cuda.device_count()
    if mode not in {"auto", "single", "dp", "ddp"}:
        raise ValueError(f"Unsupported parallel.mode: {mode}")
    if required and visible < required:
        raise RuntimeError(f"parallel requires {required} CUDA devices, but only {visible} are visible")
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA")
        expected = required or len(ids)
        if expected and world_size != expected:
            raise RuntimeError(f"WORLD_SIZE={world_size} does not match required device count {expected}")
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        backend = str(cfg.get("ddp_backend", "nccl"))
        if ids and local_rank not in ids:
            raise RuntimeError(f"LOCAL_RANK={local_rank} is not in parallel.device_ids={ids}")
        if os.name == "nt" and backend == "nccl":
            raise RuntimeError("NCCL DDP is unavailable on Windows; use normal Python for DataParallel")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)
        return SimpleNamespace(mode="ddp", device=torch.device("cuda", local_rank), rank=rank, count=world_size, main=rank == 0, ids=[local_rank])
    if mode == "ddp":
        raise RuntimeError("parallel.mode=ddp requires torchrun with WORLD_SIZE > 1")
    if mode == "dp" and not ids:
        ids = list(range(visible))
    if mode == "dp" or (mode == "auto" and len(ids) > 1):
        if len(ids) < 2 or not torch.cuda.is_available() or min(ids) < 0 or max(ids) >= visible:
            raise RuntimeError("DataParallel requires the configured visible CUDA devices")
        return SimpleNamespace(mode="dp", device=torch.device("cuda", ids[0]), rank=0, count=len(ids), main=True, ids=ids)
    return SimpleNamespace(mode="single", device=resolve_device(requested_device), rank=0, count=1, main=True, ids=[])


def wrap_training_forward(model: EDMDiffusion, runtime: SimpleNamespace) -> nn.Module:
    forward = EDMTrainingForward(model)
    if runtime.mode == "ddp":
        return DDP(forward, device_ids=[runtime.device.index], output_device=runtime.device.index)
    if runtime.mode == "dp":
        return nn.DataParallel(forward, device_ids=runtime.ids, output_device=runtime.ids[0])
    return forward


def reduce_train_row(row: dict[str, float | int], runtime: SimpleNamespace) -> dict[str, float | int]:
    if runtime.mode != "ddp":
        return row
    keys = (
        "loss", "loss_edm", "loss_x0_l1", "loss_tv", "loss_range",
        "loss_coverage_l1", "loss_spectral", "loss_phys_v1", "x0_mae",
        "x0_rmse", "grad_norm", "sigma_mean", "sigma_min", "sigma_max",
    )
    values = torch.tensor([float(row[key]) for key in keys], device=runtime.device)
    dist.all_reduce(values)
    values.div_(runtime.count)
    row.update(zip(keys, values.cpu().tolist()))
    return row


def barrier(runtime: SimpleNamespace) -> None:
    if runtime.mode == "ddp":
        dist.barrier()


def gradients_are_finite(model: nn.Module) -> bool:
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is not None and not bool(torch.isfinite(gradient.detach()).all().item()):
            return False
    return True


def _assert_finite_tensor(
    name: str,
    value: torch.Tensor,
    *,
    sample_ids: list[str] | None = None,
    report: bool = False,
) -> None:
    finite = torch.isfinite(value.detach())
    if not bool(finite.all().item()):
        bad_indices = (~finite.reshape(value.shape[0], -1).all(dim=1)).nonzero(as_tuple=False).flatten().tolist()
        bad_samples = [sample_ids[index] for index in bad_indices] if sample_ids is not None else bad_indices
        raise FloatingPointError(f"{name} contains non-finite values for samples {bad_samples}")
    if report:
        values = value.detach().float()
        print(f"[finite] {name}: shape={tuple(value.shape)} min={values.min().item():.6g} max={values.max().item():.6g}")


def spectral_magnitude_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    check_finite: bool = False,
    report_finite: bool = False,
    sample_ids: list[str] | None = None,
) -> torch.Tensor:
    if check_finite:
        _assert_finite_tensor("spectral prediction", prediction, sample_ids=sample_ids, report=report_finite)
        _assert_finite_tensor("spectral target", target, sample_ids=sample_ids, report=report_finite)
    prediction_spectrum = torch.fft.rfft2(prediction.float(), norm="ortho")
    target_spectrum = torch.fft.rfft2(target.float(), norm="ortho")
    prediction_log_magnitude = torch.log1p(prediction_spectrum.abs())
    target_log_magnitude = torch.log1p(target_spectrum.abs())
    loss = F.l1_loss(prediction_log_magnitude, target_log_magnitude)
    if check_finite:
        _assert_finite_tensor("spectral loss", loss.reshape(1), report=report_finite)
    return loss


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
    if args.resume is not None and args.init_checkpoint is not None:
        raise ValueError("Use either --resume or --init-checkpoint, not both")
    config = load_config(args.config)
    runtime = setup_parallel(config, args.device)
    seed_everything(int(config.get("seed", 1234)) + runtime.rank, deterministic=bool(config.get("deterministic", False)))
    device = runtime.device
    train_loader, val_loader = build_dataloaders(
        config, distributed=runtime.mode == "ddp", rank=runtime.rank, world_size=runtime.count,
        batch_size_multiplier=runtime.count if runtime.mode == "dp" else 1,
    )
    global_batch_size = int(config.get("loader", {}).get("batch_size", 2)) * runtime.count
    run_dir = Path(args.run_dir or config.get("run_dir", ROOT / "runs" / "edm_debug"))
    if runtime.main:
        ensure_dir(run_dir / "checkpoints")
        dump_config(config, run_dir / "config_resolved.yaml")
        write_data_split_manifest(run_dir, train_loader, val_loader, config)
    barrier(runtime)

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
        saved = checkpoint.get("extra", {}).get("parallel", {})
        if saved and (int(saved.get("world_size", runtime.count)) != runtime.count or int(saved.get("global_batch_size", global_batch_size)) != global_batch_size):
            raise RuntimeError("--resume requires matching world size and global batch size; use --init-checkpoint")
    elif args.init_checkpoint is not None:
        load_checkpoint(args.init_checkpoint, model=model, map_location=device)
        if ema is not None:
            ema.module.load_state_dict(model.state_dict(), strict=True)

    checkpoint_extra = {"parallel": {"mode": runtime.mode, "world_size": runtime.count, "global_batch_size": global_batch_size}}
    training_forward = wrap_training_forward(model, runtime)

    if runtime.main:
        jsonl = JsonlLogger(run_dir / "metrics.jsonl")
        csv = CsvLogger(run_dir / "loss_history.csv", [
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
        ])
    else:
        jsonl = csv = None

    save_every = int(train_cfg.get("save_every_steps", 500))
    val_every = int(train_cfg.get("val_every_steps", 500))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    max_consecutive_amp_overflows = max(1, int(train_cfg.get("max_consecutive_amp_overflows", 8)))
    consecutive_amp_overflows = 0
    for epoch in range(start_epoch, epochs):
        if runtime.mode == "ddp":
            train_loader.sampler.set_epoch(epoch)
        training_forward.train()
        progress = tqdm(train_loader, desc=f"edm epoch {epoch + 1}/{epochs}", dynamic_ncols=True) if runtime.main else train_loader
        for batch in progress:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp, amp_dtype):
                losses = training_forward(
                    batch["target"],
                    batch["pic"],
                    batch["x_matrix"],
                    frequency_hz=batch["frequency_hz"],
                    tx_indices=batch["tx_indices"],
                    rx_indices=batch["rx_indices"],
                )
                edm_loss = losses["loss_edm"].mean()
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
                    edm_loss
                    + prior["loss_prior_total"]
                    + spectral_lambda * spectral
                    + phys_lambda * phys
                )
            scaler.scale(loss).backward()
            if grad_clip > 0.0 or scaler.is_enabled():
                scaler.unscale_(optimizer)
            finite_gradients = gradients_are_finite(model)
            if not finite_gradients and not scaler.is_enabled():
                raise FloatingPointError("Non-finite gradients without an enabled GradScaler")
            if finite_gradients and grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            gnorm = grad_norm(model.parameters()) if finite_gradients else float("nan")
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if not finite_gradients:
                consecutive_amp_overflows += 1
                scale_after = scaler.get_scale()
                if runtime.main:
                    jsonl.write({
                        "type": "amp_overflow",
                        "epoch": epoch + 1,
                        "step": global_step,
                        "consecutive": consecutive_amp_overflows,
                        "scale_before": scale_before,
                        "scale_after": scale_after,
                    })
                if consecutive_amp_overflows >= max_consecutive_amp_overflows:
                    raise FloatingPointError(
                        f"{consecutive_amp_overflows} consecutive AMP overflows; "
                        f"scale {scale_before} -> {scale_after}"
                    )
                continue
            consecutive_amp_overflows = 0
            scheduler.step()
            if ema is not None and runtime.main:
                ema.update(model)
            global_step += 1
            sigma = losses["sigma"].detach().float()
            row = {
                "epoch": epoch + 1,
                "step": global_step,
                "lr": optimizer.param_groups[0]["lr"],
                "loss": safe_float(loss),
                "loss_edm": safe_float(edm_loss),
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
            row = reduce_train_row(row, runtime)
            if runtime.main:
                csv.write(row)
                if global_step % int(train_cfg.get("log_every_steps", 10)) == 0:
                    jsonl.write({"type": "train", **row})
                progress.set_postfix(loss=f"{row['loss']:.4g}", x0_mae=f"{row['x0_mae']:.4g}")
            if val_loader is not None and val_every > 0 and global_step % val_every == 0:
                barrier(runtime)
                if runtime.main:
                    eval_model = ema.module if ema is not None else model
                    metrics = validate(eval_model, val_loader, device, amp, amp_dtype)
                    jsonl.write({"type": "val", "epoch": epoch + 1, "step": global_step, **metrics})
                    training_forward.train()
                    if metrics["loss"] < best_metric:
                        best_metric = metrics["loss"]
                        save_checkpoint(
                            run_dir / "checkpoints" / "best.pt", model=model, optimizer=optimizer,
                            scheduler=scheduler, scaler=scaler, ema=ema.module if ema is not None else None,
                            epoch=epoch + 1, step=global_step, best_metric=best_metric, extra=checkpoint_extra,
                        )
                barrier(runtime)
            if runtime.main and save_every > 0 and global_step % save_every == 0:
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
                    extra=checkpoint_extra,
                )
        if runtime.main:
            save_checkpoint(
                run_dir / "checkpoints" / "last.pt", model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, ema=ema.module if ema is not None else None, epoch=epoch + 1, step=global_step,
                best_metric=best_metric, extra=checkpoint_extra,
            )
    barrier(runtime)
    if runtime.mode == "ddp" and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

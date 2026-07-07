from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import build_dataset_from_config
from models import GaussianDiffusion
from train_diffusion import build_model
from utils.checkpoint import load_checkpoint
from utils.config import ensure_dir, load_config
from utils.reproducibility import seed_everything
from utils.training import move_batch, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample trained conditional diffusion model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--sample-ids", nargs="*", default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def save_preview(path: Path, pred: np.ndarray, target: np.ndarray, coarse: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    items = [
        ("prediction", pred),
        ("label", target),
        ("coarse ray_relative_delta", coarse),
    ]
    for axis, (title, image) in zip(axes, items):
        axis.imshow(image, cmap="viridis", origin="lower", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.set_axis_off()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 1234)))
    device = resolve_device(args.device)
    if args.sample_ids is not None:
        config.setdefault("data", {}).setdefault(args.split, {})["sample_ids"] = args.sample_ids
    dataset = build_dataset_from_config(config, split=args.split)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_key = "ema" if args.use_ema and "ema" in checkpoint else "model"
    model.load_state_dict(checkpoint[state_key], strict=True)
    model.eval()
    output_dir = ensure_dir(args.output_dir or Path(config.get("run_dir", ROOT / "runs" / "diffusion_debug")) / "samples")
    rows = []
    with torch.no_grad():
        for index, batch in enumerate(tqdm(loader, desc="sampling", dynamic_ncols=True)):
            if args.max_samples is not None and index >= args.max_samples:
                break
            sample_id = batch["sample_id"][0]
            batch = move_batch(batch, device)
            pred = model.sample(
                batch["pic"],
                batch["x_matrix"],
                steps=args.steps or int(config.get("sample", {}).get("steps", 50)),
                eta=args.eta,
            )
            pred_np = pred[0, 0].detach().cpu().numpy().astype(np.float32)
            target_np = batch["target"][0, 0].detach().cpu().numpy().astype(np.float32)
            coarse_np = batch["pic"][0, min(1, batch["pic"].shape[1] - 1)].detach().cpu().numpy().astype(np.float32)
            npy_path = output_dir / f"{sample_id}_prediction.npy"
            png_path = output_dir / f"{sample_id}_preview.png"
            np.save(npy_path, pred_np)
            save_preview(png_path, pred_np, target_np, coarse_np)
            rows.append(
                {
                    "sample_id": sample_id,
                    "prediction_npy": str(npy_path),
                    "preview_png": str(png_path),
                    "checkpoint": str(args.checkpoint),
                    "state_key": state_key,
                }
            )
    if rows:
        with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()

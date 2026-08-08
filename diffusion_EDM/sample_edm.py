from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
from train_edm import build_model
from uncertainty import PosteriorSummary, sample_posterior, save_posterior_outputs
from utils.config import ensure_dir, load_config
from utils.reproducibility import seed_everything
from utils.training import move_batch, resolve_device


def sample_directory_name(sample_id: str) -> str:
    match = re.search(r"(?:^|[_-])sample[_-]?(\d+)$", str(sample_id), flags=re.IGNORECASE)
    if match is not None:
        digits = match.group(1)
        width = max(4, len(digits))
        return f"sample{int(digits):0{width}d}"
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sample_id)).strip("._")
    if not safe_name:
        raise ValueError(f"Cannot create an output directory from sample_id={sample_id!r}")
    return safe_name


def sample_output_directory(output_root: Path, sample_id: str) -> Path:
    return Path(output_root) / sample_directory_name(sample_id)


def resolve_sample_output_directory(output_root: Path, sample_id: str) -> Path:
    output_root = Path(output_root)
    filename = f"{sample_id}_prediction.npy"
    candidates = (
        sample_output_directory(output_root, sample_id),
        output_root / str(sample_id),
        output_root,
    )
    for candidate in candidates:
        if (candidate / filename).exists():
            return candidate
    return candidates[0]


@dataclass(frozen=True)
class LabelDisplaySpec:
    normalization_denominator_mm: float
    depth_limit_mm: float
    preview_vmax_mm: float
    theta_range_deg: tuple[float, float]
    z_range_mm: tuple[float, float]

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (*self.theta_range_deg, *self.z_range_mm)

    def depth_mm(self, normalized: np.ndarray, *, clip: bool = True) -> np.ndarray:
        values = np.asarray(normalized, dtype=np.float32) * self.normalization_denominator_mm
        if clip:
            values = np.clip(values, 0.0, self.depth_limit_mm)
        return values.astype(np.float32, copy=False)

    def delta_mm(self, normalized: np.ndarray) -> np.ndarray:
        values = np.asarray(normalized, dtype=np.float32) * self.normalization_denominator_mm
        return np.maximum(values, 0.0).astype(np.float32, copy=False)


def _positive_float(value: object, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if np.isfinite(parsed) and parsed > 0.0 else fallback


def load_label_display_spec(label_path: str | Path, target: np.ndarray) -> LabelDisplaySpec:
    label_path = Path(label_path)
    suffix = "_defect_depth_norm.npy"
    metadata_path = (
        label_path.with_name(label_path.name[: -len(suffix)] + "_defect_label_metadata.json")
        if label_path.name.endswith(suffix)
        else label_path.with_suffix(".json")
    )
    metadata: dict = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            metadata = {}

    denominator = _positive_float(metadata.get("normalization_denominator_mm"), 1.0)
    depth_limit = _positive_float(metadata.get("depth_limit_mm"), denominator)
    target_max_mm = float(np.nanmax(np.asarray(target, dtype=np.float32))) * denominator
    preview_vmax = _positive_float(metadata.get("preview_max_mm"), target_max_mm)
    if preview_vmax <= 1e-6:
        preview_vmax = depth_limit
    preview_vmax = min(preview_vmax, depth_limit)

    theta_axis = metadata.get("theta_axis", {})
    z_axis = metadata.get("z_axis", {})
    theta_values = theta_axis.get("range", [0.0, 360.0])
    z_values = z_axis.get("range", [0.0, 1000.0])
    try:
        theta_range = (float(theta_values[0]), float(theta_values[1]))
        z_range = (float(z_values[0]), float(z_values[1]))
    except (TypeError, ValueError, IndexError):
        theta_range = (0.0, 360.0)
        z_range = (0.0, 1000.0)
    return LabelDisplaySpec(
        normalization_denominator_mm=denominator,
        depth_limit_mm=depth_limit,
        preview_vmax_mm=max(preview_vmax, 1e-6),
        theta_range_deg=theta_range,
        z_range_mm=z_range,
    )


def _set_physical_axes(axis, display: LabelDisplaySpec) -> None:
    theta_min, theta_max = display.theta_range_deg
    z_min, z_max = display.z_range_mm
    axis.set_xlim(theta_min, theta_max)
    axis.set_ylim(z_min, z_max)
    axis.set_xticks(np.linspace(theta_min, theta_max, 5))
    axis.set_yticks(np.linspace(z_min, z_max, 6))
    axis.set_xlabel("theta (deg)")
    axis.set_ylabel("z (mm)")


def physical_posterior_arrays(
    summary: PosteriorSummary,
    display: LabelDisplaySpec,
) -> dict[str, np.ndarray]:
    def image(value: torch.Tensor) -> np.ndarray:
        return value[0, 0].detach().cpu().numpy().astype(np.float32)

    probability = image(summary.defect_probability)
    if summary.samples is not None:
        if summary.samples.shape[1:3] != (1, 1):
            raise RuntimeError(
                f"Physical posterior output requires samples (K,1,1,H,W), got "
                f"{tuple(summary.samples.shape)}"
            )
        samples_mm = np.clip(
            summary.samples[:, 0, 0].detach().cpu().numpy().astype(np.float32)
            * display.normalization_denominator_mm,
            0.0,
            display.depth_limit_mm,
        )
        quantiles = np.quantile(
            samples_mm,
            [summary.lower_quantile, 0.5, summary.upper_quantile],
            axis=0,
        ).astype(np.float32)
        lower, median, upper = quantiles
        mean = samples_mm.mean(axis=0, dtype=np.float32)
        std = samples_mm.std(axis=0, dtype=np.float32)
    else:
        mean = display.depth_mm(image(summary.mean))
        std = display.delta_mm(image(summary.std))
        median = display.depth_mm(image(summary.median))
        lower = display.depth_mm(image(summary.lower))
        upper = display.depth_mm(image(summary.upper))

    consensus = np.where(
        probability >= summary.consensus_probability_threshold,
        mean,
        np.zeros_like(mean),
    ).astype(np.float32)
    return {
        "prediction_mm": mean.astype(np.float32, copy=False),
        "uncertainty_mm": std.astype(np.float32, copy=False),
        "posterior_median_mm": median.astype(np.float32, copy=False),
        "posterior_lower_mm": lower.astype(np.float32, copy=False),
        "posterior_upper_mm": upper.astype(np.float32, copy=False),
        "posterior_interval_width_mm": np.maximum(upper - lower, 0.0).astype(
            np.float32,
            copy=False,
        ),
        "consensus_prediction_mm": consensus,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample trained EDM conditional defect-map model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--sample-ids", nargs="*", default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--sigma-min", type=float, default=None)
    parser.add_argument("--sigma-max", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--s-churn", type=float, default=None)
    parser.add_argument("--s-tmin", type=float, default=None)
    parser.add_argument("--s-tmax", type=float, default=None)
    parser.add_argument("--s-noise", type=float, default=None)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
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


def save_preview(
    path: Path,
    pred: np.ndarray,
    target: np.ndarray,
    coarse: np.ndarray,
    display: LabelDisplaySpec | None = None,
) -> None:
    display = display or LabelDisplaySpec(1.0, 1.0, 1.0, (0.0, 360.0), (0.0, 1000.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    items = [
        (
            "prediction wall loss",
            display.depth_mm(pred),
            "viridis",
            0.0,
            display.preview_vmax_mm,
            "Wall loss depth (mm)",
        ),
        (
            "label wall loss",
            display.depth_mm(target),
            "viridis",
            0.0,
            display.preview_vmax_mm,
            "Wall loss depth (mm)",
        ),
        ("coarse ray_relative_delta", coarse, "viridis", 0.0, 1.0, "Normalized response"),
    ]
    for axis, (title, image, cmap, vmin, vmax, colorbar_label) in zip(axes, items):
        rendered = axis.imshow(
            image,
            cmap=cmap,
            origin="lower",
            extent=display.extent,
            aspect="auto",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(title)
        _set_physical_axes(axis, display)
        colorbar = fig.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label(colorbar_label)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_uncertainty_preview(
    path: Path,
    summary: PosteriorSummary,
    target: np.ndarray,
    coarse: np.ndarray,
    display: LabelDisplaySpec | None = None,
    physical: dict[str, np.ndarray] | None = None,
) -> None:
    display = display or LabelDisplaySpec(1.0, 1.0, 1.0, (0.0, 360.0), (0.0, 1000.0))

    def image(value: torch.Tensor) -> np.ndarray:
        return value[0, 0].detach().cpu().numpy().astype(np.float32)

    physical = physical or physical_posterior_arrays(summary, display)
    mean = physical["prediction_mm"]
    std = physical["uncertainty_mm"]
    probability = image(summary.defect_probability)
    entropy = image(summary.defect_entropy)
    consensus = physical["consensus_prediction_mm"]
    target_mm = display.depth_mm(target)
    interval_width = physical["posterior_interval_width_mm"]
    std_vmax = max(float(np.quantile(std, 0.99)), 1e-6)
    width_vmax = max(float(np.quantile(interval_width, 0.99)), 1e-6)
    defect_threshold_mm = summary.defect_threshold * display.normalization_denominator_mm
    items = [
        ("posterior mean", mean, "viridis", 0.0, display.preview_vmax_mm, "Wall loss depth (mm)"),
        ("label wall loss", target_mm, "viridis", 0.0, display.preview_vmax_mm, "Wall loss depth (mm)"),
        ("coarse ray_relative_delta", coarse, "viridis", 0.0, 1.0, "Normalized response"),
        (
            "consensus prediction",
            consensus,
            "viridis",
            0.0,
            display.preview_vmax_mm,
            "Wall loss depth (mm)",
        ),
        ("uncertainty std", std, "magma", 0.0, std_vmax, "Std (mm)"),
        ("credible interval width", interval_width, "magma", 0.0, width_vmax, "Width (mm)"),
        (
            f"defect probability (depth >= {defect_threshold_mm:.2f} mm)",
            probability,
            "viridis",
            0.0,
            1.0,
            "Probability",
        ),
        ("defect entropy", entropy, "magma", 0.0, 1.0, "Binary entropy"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)
    for axis, (title, values, cmap, vmin, vmax, colorbar_label) in zip(axes.flat, items):
        rendered = axis.imshow(
            values,
            cmap=cmap,
            origin="lower",
            extent=display.extent,
            aspect="auto",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(title)
        _set_physical_axes(axis, display)
        colorbar = fig.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label(colorbar_label)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_physical_posterior_outputs(
    output_dir: Path,
    sample_id: str,
    summary: PosteriorSummary,
    display: LabelDisplaySpec,
    arrays: dict[str, np.ndarray] | None = None,
) -> dict[str, Path]:
    arrays = arrays or physical_posterior_arrays(summary, display)
    paths: dict[str, Path] = {}
    for name, values in arrays.items():
        path = output_dir / f"{sample_id}_{name}.npy"
        np.save(path, values)
        paths[name] = path
    return paths


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
    sample_cfg = config.get("sample", {})
    uncertainty_cfg = sample_cfg.get("uncertainty", {})
    edm_cfg = config.get("edm", {})
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
    if physics_guidance_scale > 0.0:
        image_size = int(config.get("data", {}).get("image_size", 256))
        physics_operator = RayOperator(image_shape=(image_size, image_size)).to(device)
    steps = args.steps or int(sample_cfg.get("steps", 32))
    sample_kwargs = {
        "steps": steps,
        "sigma_min": args.sigma_min if args.sigma_min is not None else float(sample_cfg.get("sigma_min", edm_cfg.get("sigma_min", model.sigma_min))),
        "sigma_max": args.sigma_max if args.sigma_max is not None else float(sample_cfg.get("sigma_max", edm_cfg.get("sigma_max", model.sigma_max))),
        "rho": args.rho if args.rho is not None else float(sample_cfg.get("rho", edm_cfg.get("rho", model.rho))),
        "s_churn": args.s_churn if args.s_churn is not None else float(sample_cfg.get("s_churn", 0.0)),
        "s_tmin": args.s_tmin if args.s_tmin is not None else float(sample_cfg.get("s_tmin", 0.0)),
        "s_tmax": args.s_tmax if args.s_tmax is not None else float(sample_cfg.get("s_tmax", float("inf"))),
        "s_noise": args.s_noise if args.s_noise is not None else float(sample_cfg.get("s_noise", 1.0)),
        "physics_operator": physics_operator,
        "physics_guidance_scale": physics_guidance_scale,
        "physics_guidance_start_fraction": physics_guidance_start_fraction,
        "physics_feature_index": args.physics_feature_index,
    }
    output_dir = ensure_dir(args.output_dir or Path(config.get("run_dir", ROOT / "runs" / "edm_debug")) / "samples")
    rows = []
    for index, batch in enumerate(tqdm(loader, desc="edm sampling", dynamic_ncols=True)):
        if args.max_samples is not None and index >= args.max_samples:
            break
        sample_id = batch["sample_id"][0]
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
        target_np = batch["target"][0, 0].detach().cpu().numpy().astype(np.float32)
        coarse_np = batch["pic"][0, min(1, batch["pic"].shape[1] - 1)].detach().cpu().numpy().astype(np.float32)
        display = load_label_display_spec(batch["label_path"][0], target_np)
        physical_arrays = physical_posterior_arrays(posterior, display)
        sample_dir = ensure_dir(sample_output_directory(output_dir, sample_id))
        output_paths = save_posterior_outputs(
            sample_dir,
            sample_id,
            posterior,
            save_all_samples=save_all_samples,
        )
        output_paths.update(
            save_physical_posterior_outputs(
                sample_dir,
                sample_id,
                posterior,
                display,
                physical_arrays,
            )
        )
        png_path = sample_dir / f"{sample_id}_preview.png"
        save_uncertainty_preview(
            png_path,
            posterior,
            target_np,
            coarse_np,
            display,
            physical_arrays,
        )
        uncertainty_np = posterior.std[0, 0].numpy()
        prediction_np = posterior.mean[0, 0].numpy()
        uncertainty_mm_np = physical_arrays["uncertainty_mm"]
        physical_limit_normalized = display.depth_limit_mm / display.normalization_denominator_mm
        rows.append(
            {
                "sample_id": sample_id,
                "sample_directory": str(sample_dir),
                "prediction_type": "posterior_mean",
                "prediction_npy": str(output_paths["prediction"]),
                "prediction_mm_npy": str(output_paths["prediction_mm"]),
                "uncertainty_npy": str(output_paths["uncertainty"]),
                "uncertainty_mm_npy": str(output_paths["uncertainty_mm"]),
                "defect_probability_npy": str(output_paths["defect_probability"]),
                "defect_entropy_npy": str(output_paths["defect_entropy"]),
                "consensus_prediction_npy": str(output_paths["consensus_prediction"]),
                "consensus_prediction_mm_npy": str(output_paths["consensus_prediction_mm"]),
                "posterior_summary_npz": str(output_paths["posterior_summary"]),
                "posterior_samples_npz": str(output_paths.get("posterior_samples", "")),
                "preview_png": str(png_path),
                "checkpoint": str(args.checkpoint),
                "state_key": state_key,
                "steps": steps,
                "num_posterior_samples": num_posterior_samples,
                "sample_seed_start": posterior_seed,
                "lower_quantile": lower_quantile,
                "upper_quantile": upper_quantile,
                "defect_threshold": defect_threshold,
                "defect_threshold_mm": defect_threshold * display.normalization_denominator_mm,
                "consensus_probability_threshold": consensus_probability_threshold,
                "normalization_denominator_mm": display.normalization_denominator_mm,
                "physical_depth_limit_mm": display.depth_limit_mm,
                "preview_vmax_mm": display.preview_vmax_mm,
                "prediction_above_physical_limit_fraction": float(
                    np.mean(prediction_np > physical_limit_normalized)
                ),
                "posterior_sample_above_physical_limit_fraction": float(
                    np.mean(
                        posterior.samples.detach().cpu().numpy()
                        > physical_limit_normalized
                    )
                    if posterior.samples is not None
                    else float("nan")
                ),
                "uncertainty_mean": float(np.mean(uncertainty_np)),
                "uncertainty_mean_mm": float(np.mean(uncertainty_mm_np)),
                "uncertainty_p95": float(np.quantile(uncertainty_np, 0.95)),
                "uncertainty_p95_mm": float(np.quantile(uncertainty_mm_np, 0.95)),
                "physics_guidance_scale": physics_guidance_scale,
            }
        )
    if rows:
        with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()

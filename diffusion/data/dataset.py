from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from .transforms import DEFAULT_PIC_CHANNELS, RandomCircularRoll, normalize_x_matrix, resize_label_nearest


SAMPLE_PATTERN = re.compile(r"dataset_a_frequency_sample_(\d{4,})")


@dataclass(frozen=True)
class DiffusionSample:
    sample_id: str
    coarse_path: Path
    x_path: Path
    label_path: Path


def parse_sample_id_ranges(values: list[str] | str | None) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        values = [values]
    output: list[str] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token and token.replace("-", "").isdigit():
                start_text, stop_text = token.split("-", 1)
                start = int(start_text)
                stop = int(stop_text)
                if stop < start:
                    raise ValueError(f"Invalid sample range: {token}")
                output.extend(f"dataset_a_frequency_sample_{index:04d}" for index in range(start, stop + 1))
            elif token.isdigit():
                output.append(f"dataset_a_frequency_sample_{int(token):04d}")
            else:
                output.append(token)
    return list(dict.fromkeys(output))


def infer_sample_id(path: Path) -> str:
    name = path.name
    if name.endswith("_coarse_maps.npz"):
        return name[: -len("_coarse_maps.npz")]
    match = SAMPLE_PATTERN.search(name)
    if match:
        return f"dataset_a_frequency_sample_{int(match.group(1)):04d}"
    return path.stem


def _read_npz_strings(data: np.lib.npyio.NpzFile, key: str) -> list[str]:
    if key not in data.files:
        return []
    return [str(item) for item in np.asarray(data[key]).tolist()]


class UltrasonicDiffusionDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        coarse_dir: str | Path,
        x_dir: str | Path,
        label_dir: str | Path,
        sample_ids: list[str] | str | None = None,
        pic_channels: list[str] | tuple[str, ...] = DEFAULT_PIC_CHANNELS,
        image_size: int | tuple[int, int] = 256,
        x_normalization: str = "robust_per_sample",
        use_raw_pic: bool = False,
        augment_circular_roll: bool = False,
        roll_max_fraction: float = 1.0,
        x_noise_std: float = 0.0,
        seed: int = 1234,
    ) -> None:
        self.coarse_dir = Path(coarse_dir)
        self.x_dir = Path(x_dir)
        self.label_dir = Path(label_dir)
        self.pic_channels = tuple(pic_channels)
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = (int(image_size[0]), int(image_size[1]))
        self.x_normalization = x_normalization
        self.use_raw_pic = bool(use_raw_pic)
        self.roll = RandomCircularRoll(enabled=augment_circular_roll, max_fraction=roll_max_fraction)
        self.x_noise_std = float(x_noise_std)
        self.seed = int(seed)
        self.samples = self._discover(parse_sample_id_ranges(sample_ids))
        if not self.samples:
            raise RuntimeError(
                "No diffusion samples found. Expected matching files under "
                f"{self.coarse_dir}, {self.x_dir}, and {self.label_dir}. "
                "Generate coarse maps first with simple/get_pic/generate_coarse_maps.py "
                "or pass the correct directories in the config."
            )

    def _discover(self, requested: list[str] | None) -> list[DiffusionSample]:
        if not self.coarse_dir.exists():
            raise FileNotFoundError(f"Missing coarse map directory: {self.coarse_dir}")
        if not self.x_dir.exists():
            raise FileNotFoundError(f"Missing x_matrix directory: {self.x_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Missing label directory: {self.label_dir}")
        sample_ids = requested
        if sample_ids is None:
            sample_ids = [infer_sample_id(path) for path in sorted(self.coarse_dir.glob("*_coarse_maps.npz"))]
        samples: list[DiffusionSample] = []
        missing: list[str] = []
        for sample_id in sample_ids:
            coarse_path = self.coarse_dir / f"{sample_id}_coarse_maps.npz"
            x_path = self.x_dir / f"{sample_id}_x_matrix.npz"
            label_path = self.label_dir / f"{sample_id}_defect_depth_norm.npy"
            if coarse_path.exists() and x_path.exists() and label_path.exists():
                samples.append(DiffusionSample(sample_id, coarse_path, x_path, label_path))
            else:
                missing.append(sample_id)
        if missing and requested is not None:
            preview = ", ".join(missing[:8])
            raise FileNotFoundError(f"Requested samples are missing one or more files: {preview}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)
# 从 coarse npz 读取 `pic` 或 `pic_raw`，按 `pic_channels` 选择通道。
    def _load_pic(self, path: Path) -> tuple[np.ndarray, list[str], dict[str, Any]]:
        data = np.load(path, allow_pickle=False)
        key = "pic_raw" if self.use_raw_pic and "pic_raw" in data.files else "pic"
        pic_all = np.asarray(data[key], dtype=np.float32)
        names = _read_npz_strings(data, "channel_names")
        if not names:
            names = [f"channel_{index}" for index in range(pic_all.shape[0])]
        name_to_index = {name: index for index, name in enumerate(names)}
        indices = []
        missing = []
        for name in self.pic_channels:
            if name in name_to_index:
                indices.append(name_to_index[name])
            else:
                missing.append(name)
        if missing:
            raise RuntimeError(f"{path} missing requested pic channels: {missing}")
        pic = pic_all[indices]
        metadata: dict[str, Any] = {}
        if "algorithm_config_json" in data.files:
            try:
                metadata = json.loads(str(np.asarray(data["algorithm_config_json"]).item()))
            except Exception:
                metadata = {}
        return pic, [names[index] for index in indices], metadata
    # 读取 `x`，并调用 `normalize_x_matrix()` 做 per-sample robust z-score。
    def _load_x(self, path: Path) -> tuple[np.ndarray, list[str]]:
        data = np.load(path, allow_pickle=False)
        if "x" not in data.files:
            raise RuntimeError(f"{path} missing x array")
        x = np.asarray(data["x"], dtype=np.float32)
        names = _read_npz_strings(data, "feature_names")
        return normalize_x_matrix(x, self.x_normalization), names

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        rng = np.random.default_rng(self.seed + index)
        pic, pic_channel_names, metadata = self._load_pic(sample.coarse_path)
        x_matrix, x_feature_names = self._load_x(sample.x_path)
        label = np.asarray(np.load(sample.label_path), dtype=np.float32)
        label = resize_label_nearest(label, self.image_size)
        if tuple(pic.shape[-2:]) != self.image_size:
            pic_tensor = torch.from_numpy(pic).unsqueeze(0)
            pic = (
                torch.nn.functional.interpolate(
                    pic_tensor,
                    size=self.image_size,
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .numpy()
                .astype(np.float32)
            )
        pic, label, x_matrix = self.roll(pic, label, x_matrix, rng)
        if self.x_noise_std > 0.0:
            noise = rng.normal(0.0, self.x_noise_std, size=x_matrix.shape).astype(np.float32)
            noise[-1] = 0.0
            x_matrix = x_matrix + noise
        pic = np.ascontiguousarray(pic, dtype=np.float32)
        x_matrix = np.ascontiguousarray(x_matrix, dtype=np.float32)
        label = np.ascontiguousarray(np.clip(label, 0.0, 1.0)[None, :, :], dtype=np.float32)
        return {
            "sample_id": sample.sample_id,
            "pic": torch.from_numpy(pic),
            "x_matrix": torch.from_numpy(x_matrix),
            "target": torch.from_numpy(label),
            "coarse_path": str(sample.coarse_path),
            "x_path": str(sample.x_path),
            "label_path": str(sample.label_path),
            "pic_channel_names": pic_channel_names,
            "x_feature_names": x_feature_names,
            "metadata": metadata,
        }


def build_dataset_from_config(config: dict[str, Any], *, split: str = "train") -> UltrasonicDiffusionDataset:
    data_cfg = config.get("data", {})
    split_cfg = data_cfg.get(split, {})
    sample_ids = split_cfg.get("sample_ids", data_cfg.get("sample_ids"))
    return UltrasonicDiffusionDataset(
        coarse_dir=split_cfg.get("coarse_dir", data_cfg["coarse_dir"]),
        x_dir=split_cfg.get("x_dir", data_cfg["x_dir"]),
        label_dir=split_cfg.get("label_dir", data_cfg["label_dir"]),
        sample_ids=sample_ids,
        pic_channels=data_cfg.get("pic_channels", list(DEFAULT_PIC_CHANNELS)),
        image_size=data_cfg.get("image_size", 256),
        x_normalization=data_cfg.get("x_normalization", "robust_per_sample"),
        use_raw_pic=data_cfg.get("use_raw_pic", False),
        augment_circular_roll=split_cfg.get("augment_circular_roll", data_cfg.get("augment_circular_roll", False)),
        roll_max_fraction=data_cfg.get("roll_max_fraction", 1.0),
        x_noise_std=split_cfg.get("x_noise_std", data_cfg.get("x_noise_std", 0.0)),
        seed=int(config.get("seed", 1234)),
    )


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader | None]:
    train_dataset = build_dataset_from_config(config, split="train")
    data_cfg = config.get("data", {})
    loader_cfg = config.get("loader", {})
    val_dataset: Dataset | None
    if "val" in data_cfg and data_cfg["val"].get("sample_ids") is not None:
        val_dataset = build_dataset_from_config(config, split="val")
    else:
        val_fraction = float(data_cfg.get("val_fraction", 0.0))
        if val_fraction > 0.0 and len(train_dataset) > 1:
            val_len = max(1, int(round(len(train_dataset) * val_fraction)))
            train_len = len(train_dataset) - val_len
            generator = torch.Generator().manual_seed(int(config.get("seed", 1234)))
            train_dataset, val_dataset = random_split(train_dataset, [train_len, val_len], generator=generator)
        else:
            val_dataset = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(loader_cfg.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(loader_cfg.get("num_workers", 0)),
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        drop_last=bool(loader_cfg.get("drop_last", False)),
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(loader_cfg.get("val_batch_size", loader_cfg.get("batch_size", 2))),
            shuffle=False,
            num_workers=int(loader_cfg.get("num_workers", 0)),
            pin_memory=bool(loader_cfg.get("pin_memory", True)),
            drop_last=False,
        )
    return train_loader, val_loader

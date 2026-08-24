from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

from models.unet import ResBlock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress CUDA kernels used by the EDM ResBlocks.")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--output", type=Path, default=Path("cuda_runtime_diagnostic.json"))
    return parser.parse_args()


def runtime_summary() -> dict[str, object]:
    device_index = torch.cuda.current_device()
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_cuda_runtime": str(torch.version.cuda),
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "device": torch.cuda.get_device_name(device_index),
        "capability": list(torch.cuda.get_device_capability(device_index)),
        "sdpa_flash": bool(torch.backends.cuda.flash_sdp_enabled()),
        "sdpa_mem_efficient": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "sdpa_cudnn": bool(torch.backends.cuda.cudnn_sdp_enabled()),
        "sdpa_math": bool(torch.backends.cuda.math_sdp_enabled()),
    }


def tensor_failure(name: str, values: torch.Tensor) -> dict[str, object] | None:
    detached = values.detach()
    finite = torch.isfinite(detached)
    if bool(finite.all().item()):
        return None
    return {
        "tensor": name,
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "finite": int(finite.sum().item()),
        "total": int(finite.numel()),
        "nan": int(torch.isnan(detached).sum().item()),
        "positive_inf": int(torch.isposinf(detached).sum().item()),
        "negative_inf": int(torch.isneginf(detached).sum().item()),
        "bad_indices": torch.nonzero(~finite, as_tuple=False)[:16].cpu().tolist(),
    }


def run_case(
    name: str,
    *,
    in_channels: int,
    out_channels: int,
    size: int,
    batch_size: int,
    iterations: int,
    device: torch.device,
) -> dict[str, object]:
    block = ResBlock(in_channels, out_channels, 256, dropout=0.0).to(device)
    optimizer = torch.optim.AdamW(block.parameters(), lr=1e-4)
    peak_allocated = 0
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        inputs = torch.randn(batch_size, in_channels, size, size, device=device)
        condition = torch.randn(batch_size, 256, device=device)
        output = block(inputs, condition)
        torch.cuda.synchronize(device)
        failure = tensor_failure("output", output)
        if failure is not None:
            return {"status": "failed", "iteration": iteration, "failure": failure}

        loss = output.float().square().mean()
        loss.backward()
        torch.cuda.synchronize(device)
        for parameter_name, parameter in block.named_parameters():
            if parameter.grad is None:
                continue
            failure = tensor_failure(f"gradient:{parameter_name}", parameter.grad)
            if failure is not None:
                return {"status": "failed", "iteration": iteration, "failure": failure}
        torch.nn.utils.clip_grad_norm_(block.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize(device)
        for parameter_name, parameter in block.named_parameters():
            failure = tensor_failure(f"parameter:{parameter_name}", parameter)
            if failure is not None:
                return {"status": "failed", "iteration": iteration, "failure": failure}
        peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated(device)))
    return {
        "status": "passed",
        "iterations": iterations,
        "peak_allocated_bytes": peak_allocated,
    }


def main() -> None:
    args = parse_args()
    if args.iterations <= 0 or args.batch_size <= 0:
        raise ValueError("iterations and batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnostic")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = not args.disable_cudnn
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    report: dict[str, object] = {
        "runtime": runtime_summary(),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "iterations": args.iterations,
        "cases": {},
    }
    cases = (
        ("down_256", 48, 48, 256),
        ("up_32", 384, 192, 32),
        ("up_256", 96, 48, 256),
    )
    for name, in_channels, out_channels, size in cases:
        print(
            f"[cuda-diagnostic] case={name} shape=({args.batch_size},{in_channels},{size},{size}) "
            f"cudnn={torch.backends.cudnn.enabled}",
            flush=True,
        )
        try:
            result = run_case(
                name,
                in_channels=in_channels,
                out_channels=out_channels,
                size=size,
                batch_size=args.batch_size,
                iterations=args.iterations,
                device=device,
            )
        except Exception as error:
            result = {"status": "error", "error_type": type(error).__name__, "error": str(error)}
        report["cases"][name] = result
        print(f"[cuda-diagnostic] case={name} result={result['status']}", flush=True)
        if result["status"] != "passed":
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[cuda-diagnostic] report={args.output}", flush=True)
    if any(result["status"] != "passed" for result in report["cases"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json

import torch

from train_edm import _assert_finite_tensor, spectral_magnitude_loss


def main() -> None:
    torch.manual_seed(20260809)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = torch.rand(2, 1, 256, 256, device=device)
    cases = {
        "zero_prediction": torch.zeros_like(target),
        "random_prediction": torch.randn_like(target),
    }
    losses: dict[str, float] = {}
    for name, values in cases.items():
        prediction = values.detach().requires_grad_(True)
        sample_ids = ["synthetic_0001", "synthetic_0002"]
        _assert_finite_tensor(
            "x0_pred",
            prediction,
            sample_ids=sample_ids,
            report=name == "zero_prediction",
        )
        loss = spectral_magnitude_loss(
            prediction,
            target,
            check_finite=True,
            report_finite=name == "zero_prediction",
            sample_ids=sample_ids,
        )
        loss.backward()
        assert torch.isfinite(loss), f"{name} loss is non-finite"
        assert prediction.grad is not None
        assert torch.isfinite(prediction.grad).all(), f"{name} gradient is non-finite"
        losses[name] = float(loss.detach())
    nonfinite_guard = False
    bad_prediction = torch.zeros_like(target)
    bad_prediction[0, 0, 0, 0] = float("nan")
    try:
        _assert_finite_tensor("x0_pred", bad_prediction, sample_ids=["bad_0001", "good_0002"])
    except FloatingPointError as exc:
        nonfinite_guard = "bad_0001" in str(exc)
    assert nonfinite_guard, "non-finite guard did not identify the bad sample"
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(
        json.dumps(
            {
                "status": "passed",
                "device": str(device),
                "losses": losses,
                "nonfinite_guard": nonfinite_guard,
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()

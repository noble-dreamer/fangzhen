---
name: ultrasonic-diffusion-inversion
description: Build, modify, diagnose, and validate the local ultrasonic conditional Diffusion and EDM inversion pipeline. Use for dataset loading, Pic/X encoders, physical losses, single/DataParallel/DDP FP16 training, AMP overflow recovery, checkpoint migration, posterior sampling, uncertainty and consensus outputs, normalized-to-millimeter postprocessing, evaluation, and run documentation in this repository.
---

# Ultrasonic Diffusion Inversion

Treat EDM output as a conditional posterior and preserve the repository's data, physical-unit, checkpoint, and parallel-training contracts.

## Start with the repository

1. Work from the repository root containing `diffusion/` and `diffusion_EDM/`.
2. Inspect the active branch, dirty worktree, task instructions, requested configs, and relevant entry points before editing.
3. Read [references/project-contract.md](references/project-contract.md) before changing data loading, training, checkpointing, sampling, or evaluation.
4. Reuse shared modules under `diffusion/data`, `diffusion/utils`, and `diffusion/physics`; preserve unrelated runs and user changes.
5. Keep each verified change within the repository's file and line limits, then commit it independently.

## Gate on data correctness

- Verify matching sample IDs, tensor shapes, channel names, finite values, frequency order, TX/RX axes, and fixed train/validation membership.
- Sort `frequency_hz` and gather the same x-matrix frequency axis. Never average away frequency, TX, or RX structure for convenience.
- Treat targets as continuous normalized wall-loss fields. Verify label metadata before converting to millimeters.
- Stop model expansion if a tiny deterministic baseline cannot overfit; fix alignment, scaling, or conditioning first.

## Gate the standard-dispersion decoder

- Read `f_domain/new/plan.md` before implementing the F-mode decoder. Require three SHA-linked inputs:
  calibrated Shell schema-v3, DC standard `F(n,m)` reference, and the `b_j -> m` mapping sidecar.
- Accept formal training only when the Shell library is scientifically ready in its declared proxy scope and the
  mapping publishes all 24 `F(1..8,1..3)` identities. Reject `standard_m=-1`, low-confidence, or extrapolated data.
- If a standard thickness/order/mode, c_E column, or DC provenance item is missing, use
  `../dispersion-calculator-standard-modes/SKILL.md` and list the exact TXT or screenshot to regenerate.
- If a numerical branch, derivative, observability value, or calibration SHA is missing, use
  `../comsol-frequency-streaming/SKILL.md` and name the formal, Solid, or publication step to rerun.
- Use DC `k(h,f,n,m)` as the standard relation and Shell observability/validity/mapping confidence as weights.
  Do not claim that assigning m replaces or improves the underlying Shell numerical values.

## Preserve EDM and parallel invariants

- Keep Pic spatial evidence, real-Hz frequency encoding, TX/RX position encoding, global FiLM conditioning, and configured spatial-frequency fusion responsibilities explicit.
- Route training through `EDMTrainingForward.forward()`. Never call `ddp.module.training_losses()` because that bypasses the DDP reducer lifecycle.
- Build optimizer, scheduler, scaler, and EMA from the raw model; load raw weights; only then wrap the training forward for DP or DDP.
- Save raw model and EMA state without `module.` prefixes. Use `--resume` only for matching world/global batch state; use `--init-checkpoint` for weight-only migration.
- Treat configured loader batch size as per-device in multi-GPU modes. Require the requested device count instead of silently degrading.

## Handle FP16 numerics correctly

- Keep compatible formal runs on FP16 unless the user explicitly defines an ablation.
- Distinguish finite forward/loss with non-finite unscaled gradients from a forward numerical failure.
- Let enabled `GradScaler` skip a transient overflow and reduce its scale. Do not advance scheduler, EMA, successful global step, validation, or checkpoint cadence for the skipped batch.
- Record rank-0 `amp_overflow` events and stop after the configured consecutive-overflow threshold. Never allow a plain optimizer step with non-finite gradients.
- Reduce DDP metrics before rank-0 logging; keep validation, logs, EMA, and checkpoint writing rank-0 only with barriers around validation.

## Produce and assess a posterior

- Sample the same trained model sequentially with independent seeds. Use K=2 or 4 only for smoke tests, K=16 for routine inspection, and K=32 for final stability checks.
- Save mean, population standard deviation, median, quantiles, defect probability, entropy, and conservative consensus while retaining posterior mean as the default prediction.
- Convert every posterior sample to clipped millimeters before recomputing physical statistics. Do not reinterpret the current `/9 mm` target as `/5 mm`.
- Evaluate the complete held-out split and report continuous, structural, segmentation, physical-limit, and calibration metrics. Do not infer quality from a smoke checkpoint.

## Verify changes

1. Run static compilation and focused tensor-shape/finite-gradient tests.
2. For training changes, exercise raw checkpoint strict load, `--init-checkpoint`, `--resume`, and controlled GradScaler overflow recovery.
3. Run one real sample through load, loss, backward, checkpoint reload, sampling, and evaluation when data is available.
4. Verify rank/sample ownership and single-writer outputs for parallel changes.
5. State what passed, what could not run in the current environment, and which scientific claims still require formal training.

# Project Contract

## Important paths

```text
diffusion/data/dataset.py
diffusion/utils/{checkpoint,ema,training}.py
diffusion/physics/
diffusion_EDM/train_edm.py
diffusion_EDM/{sample_edm,evaluate_edm,uncertainty}.py
diffusion_EDM/models/
diffusion_EDM/configs/
diffusion_EDM/README.md
get_pic/output_dataset/{coarse_maps,x_matrix}/
f_domain/output_dataset/streaming_dataset_a_frequency_shell/labels/
```

Do not change simulation or coarse-map generation for a model-only task. Treat `diffusion/data`, `diffusion/utils`, and `diffusion/physics` as shared APIs used by EDM.

## Data and physical units

Current 256x256 EDM tensors are:

```text
pic          [B,8,256,256]
x_matrix     [B,7,15,16,16]
frequency_hz [B,15]
tx_indices   [B,16]
rx_indices   [B,16]
target       [B,1,256,256]
```

The eight Pic channels are `ray_log_amp_loss`, `ray_relative_delta`, `ray_phase_change`, low/mid/high frequency maps, `path_coverage`, and `reliability_mask`. Sort real frequencies and gather the x-matrix frequency axis with the same order.

Labels are continuous wall-loss fields. Verify metadata, but current Dataset A uses:

```text
normalization_denominator_mm = 9
depth_limit_mm = 5
valid generated normalized depth = [0, 5/9]
```

Keep normalized arrays for compatibility. Convert each posterior sample with `clip(sample_norm * 9, 0, 5)` before recomputing physical mean, standard deviation, and quantiles.

## Training runtime

`parallel.mode: auto` selects by launch context:

- `WORLD_SIZE > 1`: DDP using the configured backend; Linux NCCL is the supported DDP path.
- Normal Python plus multiple configured devices: DataParallel. This is the preferred balanced path for the Windows A5000 and Linux 2080 Ti hosts.
- One visible/configured device: single GPU, including a future A6000 run.
- Fewer devices than `require_device_count`: fail explicitly.

Use `diffusion_EDM/configs/dataset_a_256_base48_edm_3gpu_fp16.yaml` for the fixed 1080/120 split with per-device batch 3, global batch 9, and 120 successful optimizer steps per epoch. DP multiplies the loader batch by device count. DDP keeps per-rank batch 3, uses `DistributedSampler`, calls `set_epoch()`, and requires a non-duplicating shard layout.

Only rank 0 creates run files, validates on its local device, updates EMA, logs, and saves. Other DDP ranks wait at validation barriers. Reduce training metrics before rank-0 logging.

## Checkpoint contract

Create and load the raw `EDMDiffusion` before wrapping `EDMTrainingForward` in DP/DDP. Always pass the raw model to checkpoint helpers so `model` and `ema` keys never contain `module.`.

- `--resume`: restore model, optimizer, scheduler, scaler, EMA, epoch, step, and best metric. Require matching world size and global batch.
- `--init-checkpoint`: strict-load only model weights, reset optimizer/scheduler/scaler/epoch/step, and initialize EMA from the loaded model.
- Sampling/evaluation: strict-load raw `model` or `ema`, independent of whether training used single, DP, or DDP.

## FP16 overflow contract

Keep the loss and model forward finite checks distinct from gradient checks. After scaled backward:

1. Call `scaler.unscale_(optimizer)`.
2. Check unscaled gradients and clip only when finite.
3. With enabled FP16 `GradScaler`, call `scaler.step()` and `scaler.update()` even for a transient overflow so the optimizer step is skipped and scale backs off.
4. On a skipped batch, do not advance scheduler, EMA, successful `global_step`, validation, or save cadence.
5. Log `amp_overflow` on rank 0 and fail at `train.max_consecutive_amp_overflows` (default 8).
6. Without an enabled scaler, treat non-finite gradients as fatal.

Finite input, sigma, prediction, target, and loss with only non-finite gradients is compatible with transient scale overflow. Do not misclassify it as corrupt data or alter the EDM objective without further evidence.

## Posterior and run outputs

Formal runs should contain resolved config, exact split manifests, step/validation logs, and raw/EMA checkpoints. Write posterior results under `samples/sampleNNNN/` with one root `manifest.csv`.

Retain posterior mean as the default prediction. Also save population standard deviation, median, lower/upper quantiles, thresholded defect probability, binary entropy, and consensus prediction. Consensus filters sample-inconsistent positives; it cannot detect a systematic false positive shared by all trajectories.

Use EMA `best.pt` for formal sampling and `last.pt` primarily for resume. Evaluate the full held-out split with normalized and millimeter MAE/RMSE, structural and segmentation metrics, physical-limit exceedance, CRPS/coverage/calibration, Brier score, and uncertainty-error correlation.

## Focused validation

Run from the parent directory of this repository when commands include the `simple/` prefix:

```powershell
conda run -n diffusion python -m compileall -q simple\diffusion_EDM
conda run -n diffusion python simple\diffusion_EDM\smoke_test_physical_encoding.py
conda run -n diffusion python simple\diffusion_EDM\smoke_test_uncertainty.py
```

For training-runtime changes, additionally use controlled numerical overflow inputs to prove that one overflow backs off the scale and the next finite batch updates, while repeated overflow reaches the configured guard. Verify strict checkpoint migration in both directions and ensure parallel runs create only one log/checkpoint set.

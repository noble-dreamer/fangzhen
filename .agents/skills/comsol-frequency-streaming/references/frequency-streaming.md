# Frequency Streaming Reference

## Contents

- [Important paths](#important-paths)
- [Construction traps](#construction-traps)
- [Minimal commands](#minimal-commands)
- [Pipe-dispersion smoke](#pipe-dispersion-smoke)
- [Model-tree checklist](#model-tree-checklist)
- [Result validation](#result-validation)
- [Healthy baseline compatibility](#healthy-baseline-compatibility)
- [Batch integrity](#batch-integrity)
- [Irregular-v3 selection and transfer](#irregular-v3-selection-and-transfer)

## Important paths

```text
simple_shell_common.py
f_domain/frequency_domain_common.py
f_domain/build_dataset_a_frequency_healthy.py
f_domain/solve_export_dataset_a_frequency.py
f_domain/select_sensitive_frequencies.py
f_domain/output_dataset/streaming_dataset_a_frequency_shell/
f_domain/new/solve_shell_dispersion.py
f_domain/new/axisymmetric_dispersion_common.py
f_domain/new/dispersion_tracking.py
f_domain/new/README.md
```

Use `output_dataset` for current selected-frequency training data. Treat `output2` as historical full-sweep or pilot data unless migration is explicitly requested.

## Construction traps

- `MeshConfig.max_frequency_hz` is a mesh input, not documentation. Update it before model construction whenever the requested band changes. A mesh designed for 60 kHz does not satisfy the declared eight-elements-per-wavelength rule at 95-100 kHz.
- The support geometry is a 3D cylinder, but Shell belongs only on the open cylindrical side selection. Confirm cap exclusion after geometry rebuilds.
- Dataset A absorbing layers are a `z`-dependent Rayleigh-beta field over both axial end bands. Verify reflection suppression per frequency instead of assuming one damping coefficient is broadband.
- For outer corrosion, keep the inner surface approximately fixed through the thickness offset and clamp cumulative loss with the same limits used by label generation.
- The smooth PZT window avoids tiny geometric patches, but `F0/pzt_A * window` is only approximately total-force normalized. Calibrate or normalize before interpreting absolute amplitudes.
- COMSOL 6.4 Shell ForceArea displays the load as `f_A`. Inspect the actual node property because compatibility setters may silently ignore obsolete property names.
- Use the integration coupling receiver average as the authoritative output. CutPoint evaluation may fail to reference the solution or return zero spatial points.
- Check the receiver denominator and returned complex array shape. Accept no implicit channel transpose or truncation.
- Frequency mode replaces shared builder functions in module state. Do not build a transient model later in the same interpreter without restoring those functions.
- A saved build-only MPH contains a configured sweep, not solved parameter points. Streaming intentionally disables COMSOL Parametric Sweep and loops over cases in Python.

## Minimal commands

Build an unsolved model for tree inspection:

```powershell
conda run --no-capture-output -n comsol python -u f_domain/build_dataset_a_frequency_healthy.py
```

Run one healthy case and exercise receiver export:

```powershell
conda run --no-capture-output -n comsol python -u f_domain/solve_export_dataset_a_frequency.py --only-healthy --tx 1 --frequencies 50000 --linear-solver pardiso --heartbeat-s 20 --skip-label-preview
```

For long Windows jobs, retain `--no-capture-output` and `python -u`. The Python heartbeat reports elapsed time and case-level ETA because `model.solve()` blocks access to reliable internal COMSOL progress.

On Windows, do not pipe a multiline here-string to `conda run ... python -`: it can enter the REPL, and
`conda run ... python -c` rejects newline-bearing arguments. Prefer a real script; for an ad-hoc multiline
smoke, first discover `sys.executable` with a one-line command and invoke that environment interpreter directly.

## Pipe-dispersion smoke

Run the local 4-point derivative-capable check from repository root `simple/`:

```powershell
conda run --no-capture-output -n comsol python -u `
  f_domain/new/solve_shell_dispersion.py `
  --thickness-mm 7.5 10 --k-count 2 --k-max-rad-m 25 `
  --circumferential-orders 10 --mode-count 4 `
  --output-root f_domain/new/outputs/axisymmetric_modal_quick_smoke --smoke
```

Run the current scientific smoke on the server:

```bash
conda run --no-capture-output -n comsol_lzx python -u \
  f_domain/new/solve_shell_dispersion.py \
  --thickness-mm 5 7.5 10 --k-count 5 --k-max-rad-m 100 \
  --circumferential-orders 8 10 12 --mode-count 4 \
  --output-root f_domain/new/outputs/axisymmetric_modal_science_smoke --smoke
```

Expect `axisymmetric_shell_dispersion_library.npz` schema v3. Require a complete `[H,K,N]` checkpoint mask, raw arrays shaped `[H,K,N,M]`, and tracked arrays shaped `[H,K,N*M]`. Check polarization sums near one, negligible imaginary frequency, observability in `[0,1]`, and derivative validity masks plus finite values. Invalid derived values must remain `NaN`.

Resume requires both the complete NPZ and metadata JSON. Repeat equivalent physical arguments with the same output root; compatibility checks do not compare every physical option. A valid resume returns `resumed=true` without `model_ready` or point events.

The local `3 thickness x 5 k x 3 n` validation completed 45/45 solve points in about 100 seconds and produced finite `c_g` and `dk/dh` at all 105 points in the 15--110 kHz band. Treat this as an integration benchmark, not production runtime or inversion accuracy.

Generate the uncalibrated dense raw library on the server with explicit formal mode:

```bash
conda run --no-capture-output -n comsol_lzx python -u \
  f_domain/new/solve_shell_dispersion.py --formal \
  --thickness-mm 5 6 7 7.5 8 9 10 --k-count 41 \
  --circumferential-orders 1 2 3 4 5 6 7 8 9 10 11 12 \
  --mode-count 8 --frequency-range-khz 15 110 --cores 16 \
  --output-root f_domain/new/outputs/axisymmetric_dispersion_formal
```

Formal mode rejects full-ring scans and smoke-named roots. Its checkpoint is resumable but deliberately records `scientific_ready=false`; do not pass this raw file to final frequency selection before the calibration gate promotes it.

Run the limited full-solid calibration on exact Shell axis points:

```bash
conda run --no-capture-output -n comsol_lzx python -u \
  f_domain/new/solve_solid_calibration.py \
  --shell-library f_domain/new/outputs/axisymmetric_dispersion_formal/dispersion/axisymmetric_shell_dispersion_library.npz \
  --thickness-mm 5 7.5 10 --k-indices 0 20 40 \
  --circumferential-orders 4 8 10 12 --mode-count 4 --max-solves 36 \
  --mesh-hmax-mm 1.25 --eigen-shift-khz 60 \
  --output-root f_domain/new/outputs/solid_calibration --cores 16
```

Requested thicknesses must exist exactly in the Shell axis; k values are selected by Shell indices to avoid floating-point axis mismatch. Span the full formal k axis rather than calibrating only its first few points. The runner rejects smoke Shell libraries unless an explicit code-smoke override is used, never saves MPH, and records `solid/shell` frequency ratio, matched branch, cost, validity, and confidence. Its output remains `scientific_ready=false`.

Run the implemented finite-pipe receiver mesh gate on the server:

```bash
conda run --no-capture-output -n comsol_lzx python -u \
  f_domain/new/validate_shell_dispersion.py \
  --thickness-mm 5 7.5 10 --frequencies-khz 20 52.5 95 \
  --elements-per-wavelength 6 8 12 --tx 1 \
  --max-complex-relative 0.05 --max-phase-rmse-rad 0.05 \
  --output-root f_domain/new/outputs/finite_pipe_mesh_validation \
  --delete-case-outputs --heartbeat-s 120 --cores 16
```

The script requires finite 16-channel responses and empty COMSOL build/post-solve problem lists. It compares EPW 8 against EPW 12; coarse EPW 6 is retained as context. Passing deletes only the intermediate `cases/` directory when requested and keeps `finite_pipe_mesh_validation.json`. Failure retains cases and exits nonzero. Do not describe this receiver-level check as periodic-cell field-spectrum validation.

Publish the calibrated Shell-proxy library only after both gates finish:

```bash
conda run --no-capture-output -n comsol_lzx python -u \
  f_domain/new/calibrate_dispersion_library.py \
  --shell-library f_domain/new/outputs/axisymmetric_dispersion_formal/dispersion/axisymmetric_shell_dispersion_library.npz \
  --mesh-report f_domain/new/outputs/finite_pipe_mesh_validation/finite_pipe_mesh_validation.json \
  --solid-calibration f_domain/new/outputs/solid_calibration/calibration/solid_shell_calibration.npz \
  --output-root f_domain/new/outputs/axisymmetric_dispersion_calibrated \
  --readiness-scope shell_proxy_simulation_only
```

The publisher verifies the source Shell SHA, strict mesh thresholds and actual metrics, complete Solid points,
valid-mode coverage, and calibrated derivative coverage. It re-derives corrected quantities and records all source
hashes. Its `scientific_ready=true` applies only to the current Shell-proxy simulation/EDM chain.

## Model-tree checklist

- Study type is `Frequency`; its frequency expression is `pzt_fc`.
- No transient pulse function appears in the harmonic load.
- Shell selection contains the side boundary and excludes caps.
- Shell elastic properties are explicit and have correct SI units.
- Healthy thickness is `h0`; damaged thickness and offset encode outer loss.
- End-band damping uses the intended axial ramp.
- Face load has one active transmitter gate and the intended axial-z direction.
- `intop_shell` is bound to the shell side selection.
- Sixteen receiver expressions use patch windows centered on PZT 17-32.
- Optional marker datasets do not participate in physics, mesh, or export.

## Result validation

Check every completed case for:

```text
channels == 16
all(real(H), imag(H)) are finite
at least 16 nonzero receiver magnitudes
H shape == (n_tx, 16, n_frequency)
completed_mask shape == (n_tx, n_frequency)
ordered frequency axis exactly matches the request
```

An empty `model.problems()` only proves that COMSOL found no reported model-tree error. Also verify nonzero loading, receiver values, healthy rotational symmetry, and a physically plausible response change for a known defect.

## Healthy baseline compatibility

Compare more than array shape. Require identical:

```text
pipe and shell dimensions
material and damping
absorbing-layer parameters
mesh design frequency and size
defect-free thickness and offset convention
transmitter positions, window, force scale, and drive direction
receiver positions, window, displacement component, and units
ordered tx, rx, and frequency axes
```

Use complex comparison features such as `Hd-H0` and `angle(Hd*conj(H0))`. Do not compare an axial response with historical radial-response data even if the arrays have the same shape.

## Batch integrity

- Checkpoint only after the current `H` slice and `completed_mask` are both updated.
- Clear solution data only after successful receiver evaluation and checkpointing.
- Treat NPZ, metadata, and labels as authoritative completion artifacts; shared `manifest.csv` can race between workers.
- Use non-overlapping `--start-id` ranges for concurrent workers.
- Preserve partial NPZ files with an incomplete mask for diagnosis or resume; never present them as complete training samples.

## Irregular-v3 selection and transfer

Run the dense irregular pilot from repository root `simple/`; four evenly spaced transmitters are sufficient for ranking, while formal EDM samples still require all 16:

```bash
PILOT_FREQS=$(seq -s, 20000 2500 100000)
conda run --no-capture-output -n comsol_lzx python -u \
  f_domain/new/solve_irregular_dataset_a_frequency.py \
  --include-healthy --samples 12 --start-id 1 --seed0 20260820 \
  --tx 1,5,9,13 --frequencies "$PILOT_FREQS" \
  --output-root f_domain/new/outputs/irregular_frequency_pilot \
  --checkpoint-every-cases 33 --skip-label-preview --heartbeat-s 120
```

Select 12 frequencies only after the calibrated dispersion library exists:

```bash
conda run --no-capture-output -n comsol_lzx python -u \
  f_domain/new/select_dispersion_frequencies.py \
  --pilot-root f_domain/new/outputs/irregular_frequency_pilot \
  --dispersion-library f_domain/new/outputs/axisymmetric_dispersion_calibrated/dispersion/axisymmetric_shell_dispersion_library.npz \
  --output-root f_domain/new/outputs/irregular_frequency_selection \
  --count 12 --band-quotas 3 4 5
```

The selector combines class-balanced complex sensitivity, sample stability, phase change, path participation, `|dk/dh|`, axial observability, and tracking confidence. It retains per-thickness dispersion masks and never fills unsupported frequencies with nearest values.

After selected-frequency healthy and defect workers finish, run the audit with every worker root, the healthy root, the selected-frequency file, and the dispersion/selection provenance as shared files. From repository root, archive exactly its output list:

```bash
tar -czf f_domain/new/outputs/irregular_edm_transfer.tar.gz \
  -T f_domain/new/outputs/irregular_dataset_audit/transfer_manifest.txt
```

The archive is sufficient to regenerate `pic` and `x_matrix` after transfer. Per-case CSV, progress logs, interpolation tables, preview PNG, MPH, and smoke directories are intentionally absent.

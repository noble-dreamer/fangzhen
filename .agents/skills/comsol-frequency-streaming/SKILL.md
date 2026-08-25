---
name: comsol-frequency-streaming
description: Build, audit, run, and debug COMSOL 6.4 frequency-domain streaming and pipe-dispersion simulations for the simple ultrasonic Shell project. Use for shell geometry, outer-surface corrosion, absorbing end layers, equivalent PZT loads, patch-averaged receivers, complex H(tx,rx,f) export, healthy baselines, frequency selection, model reuse, convergence checks, calibrated dispersion publication, COMSOL/mph batch workflows, axisymmetric pipe dispersion, modal tracking, b_j-to-standard-m mapping, polarization, observability, group velocity, or dk/dh.
---

# COMSOL Frequency Streaming

Work in the project containing `simple_shell_common.py` and `f_domain/`. Preserve its shell-model assumptions unless the user explicitly requests a higher-fidelity model.

## Start with an audit

1. Inspect `simple_shell_common.py`, `f_domain/frequency_domain_common.py`, the entry script, and `f_domain/README.md` before changing behavior. For dispersion work, also inspect `f_domain/new/README.md` and the three dispersion modules listed in the reference.
2. Read [references/frequency-streaming.md](references/frequency-streaming.md) before modifying geometry, physics, receiver export, frequency range, or batch output.
3. Confirm the active COMSOL installation is 6.4 and `mph` is imported from the intended conda environment.
4. Print or record the requested transmitters, ordered frequencies, output root, mesh limit, receiver component, and model family before an expensive solve.

## Preserve model invariants

- Model the pipe as a cylindrical midsurface and apply Shell only to the cylindrical side boundary; exclude support-cylinder caps.
- Treat the shell result as a lightweight proxy for coarse imaging and training data, not as a replacement for 3D piezoelectric validation.
- Set `E`, `nu`, and `rho` explicitly on the Shell elastic node. Do not rely only on a domain material lookup for a boundary Shell selection.
- Represent outer-surface corrosion with both local thickness loss and the configured negative thickness offset. Thickness reduction about an unchanged midsurface is not equivalent to outer corrosion.
- Generate labels from the same defect expressions, clipping, angular wrapping, and `(z, theta)` convention used by COMSOL. Do not duplicate stale defect formulas.
- Treat Dataset A end treatment as axial Rayleigh-damped absorbing bands, not as damping attached to a few end boundaries and not as a true low-reflecting boundary.

## Match frequency physics

- Use a `Frequency` study with `plist=pzt_fc` and a harmonic face-load amplitude. Never leave `pztpulse(t)` in a frequency-domain load.
- Keep the excitation direction and receiver component paired. The current model uses axial-z excitation and patch-weighted axial displacement; do not relabel it as radial response.
- In COMSOL 6.4, verify the Shell face load through `forceType=ForceArea` and `forceReferenceArea` (`f_A` in the GUI). A successful `set_if_possible()` call sequence or empty `model.problems()` is not proof that the physical load is active.
- Export receivers through `intop_shell(w_rx*w)/intop_shell(w_rx)`. Keep CutPoint datasets optional and visual-only.
- Treat `H(tx,rx,f)` as complex displacement amplitude, not voltage and not a time trace. Do not derive TOF directly from sparse frequency responses.

## Enforce mesh and solver-tree checks

- Set the mesh design frequency to at least the largest requested frequency before building the model. Compute `hmax` from the shortest relevant modal wavelength, not from a historical default.
- Run mesh convergence on complex amplitude, phase, and healthy-damaged differences at the highest retained frequency.
- Inspect the generated COMSOL solver tree. Do not assume a Python `relative_tolerance` value affects a frequency study unless the active frequency/stationary solver node contains the intended setting.
- Quantify absorbing-layer reflection across the full frequency band because Rayleigh damping is frequency dependent.
- Run `f_domain/new/validate_shell_dispersion.py` at representative thicknesses and low/mid/high frequencies before accepting a formal library. Its current gate compares finite-pipe patch-averaged complex response and phase between medium/fine meshes; it does not replace modal field-spectrum or real-pipe certification.
- Delete mesh-validation case outputs only after the report passes and `--delete-case-outputs` was explicit. Preserve failed cases for diagnosis; the report must remain `scientific_ready=false` until the remaining calibration gates pass.

## Stream in the correct order

1. Build one geometry, mesh, and solver tree per physical sample.
2. Reuse that model only across `tx` and `pzt_fc`; rebuild when the defect thickness field or other sample physics changes.
3. Set `tx` and `pzt_fc`, solve, evaluate all 16 complex receiver channels, validate them, checkpoint, then call `clearSolutionData()`.
4. Remove the sample model and clear the client after the sample. Never clear solution data before receiver extraction.
5. Run time-domain and frequency-domain builders in separate Python processes because frequency mode patches shared builder functions for the process lifetime.

## Build pipe dispersion by prescribed order

- Use `f_domain/new/solve_shell_dispersion.py`; prefer its default 2D axisymmetric Shell formulation with `Mode2Daxi=1`, prescribed `n_circ`, and axial Floquet `kz`.
- Keep `--formulation full-ring` diagnostic-only. Mixed circumferential orders around a fixed eigenvalue shift do not provide reliable branch candidates.
- Solve and track each circumferential order independently, then concatenate branches into schema v3.
- Preserve raw `[thickness,k,n,mode]` arrays and canonical `[thickness,k,branch]` arrays.
- Derive `c_g`, `df/dh`, and `dk/dh` only across reliable links. Use the calibrated residual gate `1.1`, keep rejected values as `NaN`, and consume derivatives only when the validity mask is true and the value is finite.
- Treat current observability as an axial-polarization and patch-aperture proxy, not a calibrated transducer transfer function.
- Use `--smoke` for integration checks. `--formal` may generate only a dense prescribed-order axisymmetric raw library; it must remain `scientific_ready=false` until finite-pipe and limited full-solid calibration pass.
- Keep `7.5 mm` explicitly on the formal Shell thickness axis. Run `solve_solid_calibration.py` at no more than 36 `(h,k,n)` points and span the formal k axis; for `k-count=41`, use indices `0 20 40`, not low-k-only `0 3 6`.
- Promote only with `calibrate_dispersion_library.py` after strict mesh and Solid gates pass. Scientific selection requires both `scientific_ready=true` and `readiness_scope=shell_proxy_simulation_only`; this scope is not real-pipe certification.
- Run short checks locally from repository root `simple/` in conda environment `comsol`; run server jobs from the same relative root in `comsol_lzx`.
- Use a new output root when physical arguments change. Reuse a root only with equivalent arguments to verify checkpoint resume.

## Map standard pipe modes

- Read [the DC standard-mode skill](../dispersion-calculator-standard-modes/SKILL.md) before requesting or
  auditing Dispersion Calculator exports. Do not duplicate its GUI instructions here.
- For standard mapping, run a new formal library on `h=5,6,7,7.5,8,9,10 mm`, `k-count=41`, `n=0..8`, and
  `mode-count=8`; use n=0 only for L/T anchors and n=1..8 for F. Never resume the historical n=1..12 root.
- Calibrate representative orders `n=0,1,4,8` at h `5,7.5,10` and k indices `0,20,40`, then map only from
  the published calibrated library. Preserve standard-reference, formal, calibrated, and mapping source SHAs.
- Require all 24 `F(1..8,1..3)` identities to pass the gates in `f_domain/new/plan.md`. Keep every failed or
  swapped branch `unknown`; never infer `m=j+1`.
- If DC thickness/order/mode/c_E data are missing, report the exact TXT and route the user to the DC skill.
  If a Shell candidate is missing, rerun formal with `mode-count=16` in a new root and repeat dependent Solid
  calibration/publication. If source SHAs differ, rebuild the mapping instead of relabeling it.
- Transfer calibrated Shell NPZ/metadata, DC standard NPZ/metadata, and mapping NPZ/report to EDM. Do not
  transfer raw TXT, MPH, progress, case CSV, or smoke directories for training.

## Protect dataset compatibility

- Reuse a healthy baseline only when ordered tx/frequency axes, receiver indices, mesh, material, damping, load direction, receiver expression, units, and geometry all match.
- Require `H_real/H_imag` shape `(n_tx, 16, n_f)`, `completed_mask` shape `(n_tx, n_f)`, finite values, and 16 nonzero channels for completed cases.
- Keep healthy and damaged pairs identical in all non-defect parameters.
- Preserve existing outputs unless overwrite is explicit. For multiple workers, assign non-overlapping sample ranges and do not use a shared manifest as the sole completion record.

## Select and transfer irregular-v3 data

- For the current multiscale irregular campaign, do not reuse legacy super-Gaussian frequency rankings, healthy responses, or training samples. Generate a dense pilot with `solve_irregular_dataset_a_frequency.py` and the generator `irregular_polygon_texture_v3_multiscale`.
- Pass pilot frequencies in strictly increasing order. Keep exactly one compatible healthy response in the pilot root and include small, medium, and large defects; `select_dispersion_frequencies.py` rejects missing classes, legacy generators, incomplete axes, and weak dispersion coverage.
- Use a complete calibrated schema-v3 dispersion library with the Shell-proxy readiness scope for scientific selection. A smoke library may validate code paths only and must not define the formal EDM frequency set.
- Generate the selected-frequency healthy response separately, then use all 16 transmitters for non-overlapping defect workers. Keep the selected frequency count and order identical across every worker.
- Before transfer, run `audit_irregular_dataset.py`. Require `(16,16,F)` finite responses, complete masks, 16 nonzero receivers per completed case, identical healthy/damaged model fingerprints, `/9 mm` labels, and all three size classes.
- Transfer only files listed by the audit manifest: complex responses, model metadata, four label artifacts, selection/dispersion provenance, and audit reports. Do not transfer case CSV, progress logs, defect tables, previews, MPH, or smoke outputs for EDM training.
- After a smoke passes and its concise result is recorded, delete only the output directory created for that smoke. Never clean production, partial, checkpoint, or resume roots as part of smoke cleanup.

## Validate before scaling

Run, in order: build-only model-tree inspection; one healthy tx/frequency solve with export; healthy symmetry checks; one known-defect sensitivity check; highest-frequency mesh and absorbing-layer convergence; then the full batch. For dispersion, run the 4-point integration smoke before the 45-point scientific smoke and audit schema, masks, finite derivatives, and resume. Heartbeat output proves the blocking solve is alive, not COMSOL's internal percentage.

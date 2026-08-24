# Physical Baselines

## Choose the baseline

`physical_inversion` is a geometric helical-ray baseline. An independent 48-sample COMSOL corpus fits fixed nonnegative order weights and robust complex-Rytov channel calibration. The current prior uses `alpha[-1,0,+1] = [0.35,0.30,0.35]`, then solves bounded SIRT/TV on a 64x64 grid and renders 256x256 millimeter maps.

`rytov` is a stronger full-wave linearized baseline. Weak known wall-loss probes produce a complex COMSOL Jacobian around the healthy state. `rotational_tx1` solves one TX and rotates it; strict `all_tx` independently solves all 16 TX. Neither method is iterative nonlinear FWI.

## Run the completed ray prior

From the parent directory containing `simple/`:

```powershell
conda run --no-capture-output -n get_pic python -u simple/get_pic/physical_inversion/run_simulation_channel_prior_inversion.py
conda run --no-capture-output -n get_pic python -u simple/get_pic/physical_inversion/validate_outputs.py
```

Use `--no-evaluate-labels` when predictions must be label-blind. Existing prior artifacts live below `physical_inversion/simulation_prior/output_matched_corpus`; formal predictions belong in `output_dataset_matched_corpus`. Do not regenerate COMSOL corpus merely to rerun formal inversion.

## Run the full-wave operator

Build/freeze/validate in this order: `build_training_plan.py`, `solve_training_corpus.py`, `fit_operator.py`, `validate_operator.py`, `run_inversion.py`, then `validate_outputs.py`. Use the matching rotational or strict config throughout; their plans, fingerprints, operators, and output roots are not interchangeable.

For interrupted strict COMSOL training, first run `solve_training_corpus.py ... --dry-run --resume-incomplete`. `complete=128`, `pending=0`, and `incompatible=0` means training is finished. Otherwise rerun with `--resume-incomplete`, never `--overwrite-existing`. An incomplete probe is recomputed as a unit; complete signed probes are reused.

Training ranges use zero-based `--start-basis-index/--end-basis-index` over `0..127`. Formal inversion ranges use actual response `--start-id/--end-id`; do not mix these namespaces. Formal batches automatically need distinct output roots/manifests.

## Artifacts and failure modes

- Ray kernels create mirror/ghost intersections because several helical paths explain similar pixels; sparse angular diversity and periodic theta worsen ambiguity.
- Boundary artifacts arise from low path coverage, endpoint sensitivity, incomplete regularization support, and theta/z boundary handling. Theta wraps; z must not wrap.
- Peak depth is biased low by path averaging, coarse basis support, TV shrinkage, nonnegative bounds, and weak-scattering linearization.
- Multiple defects violate simple additive scattering, so responses can merge, cancel in complex phase, or generate false positives.
- Principal complex log is discontinuous near `+-pi`; reject branch-contaminated observations rather than silently unwrapping unrelated channels.
- Formal labels must never select alpha, channel slopes, Jacobian weights, regularization, or stopping criteria. Record `formal_labels_used_for_inversion=false`.
- A 128-dimensional Jacobian rendered at 256x256 does not provide 65,536 independent resolved parameters.
- Strict `all_tx` removes rotational approximation error but not first-order Rytov error, model mismatch, mode conversion, or multiple scattering.
- Simulation/experiment mismatch in material, damping, transducer coupling, or boundaries can be reconstructed as false wall loss.
- Compare all methods in millimeters, on identical IDs/splits and coordinates, with continuous, localization, segmentation, data-residual, and multi-defect metrics.

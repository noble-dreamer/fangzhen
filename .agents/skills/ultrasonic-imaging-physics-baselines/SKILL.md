---
name: ultrasonic-imaging-physics-baselines
description: Generate, inspect, diagnose, and validate this repository's ultrasonic coarse-map/x_matrix conditions and traditional physical inversion baselines. Use for get_pic data flow, V1 helical ray backprojection, simulation-derived complex Rytov channel priors, full-wave linearized Rytov operators, millimeter outputs, COMSOL corpus continuation, physical-baseline comparison, and known artifact or path/frequency/sample-alignment failures.
---

# Ultrasonic Imaging Physics Baselines

Preserve the separation between an EDM condition, a traditional physical estimate, and a supervised label.

## Start with the repository

1. Work in the `simple` Git repository and inspect its branch, dirty files, `agent.md`, and the requested workflow before editing.
2. Use the formal `f_domain/output_dataset/streaming_dataset_a_frequency_shell`; never substitute historical `output2`.
3. Read [references/coarse-conditioning.md](references/coarse-conditioning.md) for coarse maps or `x_matrix` work.
4. Read [references/physical-baselines.md](references/physical-baselines.md) for ray-Rytov, full-wave Rytov, COMSOL corpus, or baseline comparison work.
5. Inspect the referenced implementation and README before relying on a command because local and server working-directory conventions differ.

## Select the correct product

- Generate `get_pic/output_dataset/{coarse_maps,x_matrix}` only when building EDM conditions. A coarse map is a spatial hint, not a calibrated final inversion.
- Use `get_pic/physical_inversion` for the frozen, simulation-derived helical-ray complex-Rytov baseline.
- Use `get_pic/rytov` for the frozen COMSOL full-wave linearized Rytov baseline. Treat `rotational_tx1` as an approximation and `all_tx` as the strict comparison.
- Use `ultrasonic-diffusion-inversion` after these products exist when the task concerns EDM training, sampling, or evaluation.

## Enforce physical and data boundaries

- Join healthy response, damaged response, metadata, coarse map, x-matrix, and label by the same canonical sample ID.
- Preserve complex `H[TX,RX,F]`, TX/RX indices, completion masks, and real `frequency_hz`. If frequencies are sorted, gather every frequency-dependent tensor with the same permutation.
- Keep theta periodic and z nonperiodic. Verify plot orientation as horizontal theta `0..360 deg` and vertical z `0..1000 mm`.
- Do not fit a physical prior, scale, kernel, regularizer, or inversion coefficient from formal labels. Read labels only after frozen prediction for descriptive metrics.
- Keep `prediction_mm` and `prediction_norm = prediction_mm / 9`; clip physical loss to the configured `0..5 mm` range.
- Do not modify coarse generation, physical inversion, Rytov, EDM, or formal simulation code outside the workflow explicitly requested.

## Validate before claiming success

1. Check exact file counts and IDs, NPZ keys, shapes, finite complex values, axes, frequencies, and signed plan/config fingerprints.
2. Run the workflow's dry-run or smoke test before COMSOL and its validator after generation.
3. Inspect representative single- and multi-defect previews plus aggregate metrics. Do not infer quality from one attractive image.
4. Report whether labels were read, which operator mode was used, which samples were processed, and which artifacts remain expected limitations.

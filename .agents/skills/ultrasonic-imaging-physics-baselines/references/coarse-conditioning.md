# Coarse Conditioning

## Data contract

Read formal inputs from:

```text
f_domain/output_dataset/streaming_dataset_a_frequency_shell/
  frequency_response/<id>_H_complex.npz
  metadata/<id>.json
  labels/<id>_defect_depth_norm.npy
```

`generate_coarse_maps.py` compares each damaged `H` with the shared healthy response, filters incomplete/nonfinite cases and selected frequencies, builds helical orders `-1,0,+1`, and coverage-normalizes ray-tube backprojections. It writes:

```text
get_pic/output_dataset/coarse_maps/<id>_coarse_maps.npz
get_pic/output_dataset/x_matrix/<id>_x_matrix.npz
```

The stored coarse NPZ has ten channels. Current EDM selects eight by name, excluding `ray_delta_abs` and `valid_case_count`, so its tensor is `[B,8,256,256]`. The x-matrix has seven features and normally loads as `[B,7,15,16,16]` after top-15 selection.

The seven x features are log absolute difference, log relative difference, phase cosine, phase sine, healthy log amplitude, damaged log amplitude, and validity mask. Preserve their names and axes.

## Run and inspect

From the parent directory containing `simple/`:

```powershell
conda run --no-capture-output -n get_pic python -u simple/get_pic/generate_coarse_maps.py --sample-ids 1-40 --preview --skip-existing
conda run --no-capture-output -n get_pic python -u simple/get_pic/evaluate_coarse_maps.py --coarse simple/get_pic/output_dataset/coarse_maps/dataset_a_frequency_sample_0001_coarse_maps.npz
```

Before a large run, inspect `get_pic/configs/dataset_a_v1.json`, the selected-frequency file, healthy metadata, and one source NPZ. For parallel chunks, pass `--no-manifest` and merge deliberately; multiple writers must not race on one manifest.

## Interpretation and known failures

- The first seven image channels are ray evidence; `path_coverage`, `valid_case_count`, and `reliability_mask` describe observability, not defects.
- Bright line crossings, axial tails, periodic mirror-like lobes, and broad spots follow sparse helical path coverage. They are expected V1 artifacts, not precise defect boundaries.
- Endpoint brightness near the TX/RX rings usually indicates inadequate endpoint masking, normalization, or frequency selection.
- A strong absolute-difference channel without matching relative/amplitude/phase evidence is often source-amplitude domination.
- Per-channel robust normalization makes previews readable but prevents comparing brightness numerically across panels. Use raw arrays for amplitude comparisons.
- `frequency_hz` may be ranked rather than ascending. Never sort frequencies without applying the identical permutation to x and masks.
- Running a config containing `simple/...` from inside `simple/` can create `simple/simple/...` paths in scripts that resolve against CWD. Follow the script README and inspect resolved paths.
- Coarse generation must not read labels. Labels belong to post-generation evaluation and EDM targets only.
- Do not compare a normalized coarse channel directly with a millimeter EDM prediction. Compare EDM against an independent calibrated physical baseline.

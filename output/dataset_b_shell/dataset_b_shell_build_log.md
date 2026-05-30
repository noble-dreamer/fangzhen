# Simple Shell Model Build Log

## Generated files

- `D:\lab_ultr\fz\simple\output\dataset_b_shell\pipe_shell_healthy.mph`

## Model family

Dataset B simple shell: realistic healthy baseline

## Notes

- Random seed: 20260528.
- No-defect shell model with material, transmitter, and receiver perturbations.
- PZT-domain uncertainty is represented as equivalent tx/rx position and amplitude error.

## Simplifications

- Pipe is a cylindrical shell midsurface at `Rm = 155.000 mm`.
- Wall loss defects are represented by spatially varying shell thickness, not Boolean corrosion cuts.
- PZT solids are removed. Excitation is an equivalent face load with a smooth transducer window.
- Receivers are 16 shell displacement points on the receiver ring.
- Mesh is controlled by wavelength: `hmax = 5.208 mm`, not by PZT block dimensions.

## Receiver points

| Channel | theta_deg | x_mm | y_mm | z_mm |
| --- | ---: | ---: | ---: | ---: |
| 17 | -0.7 | 154.990 | -1.762 | 899.056 |
| 18 | 22.1 | 143.575 | 58.406 | 899.169 |
| 19 | 44.1 | 111.283 | 107.894 | 900.497 |
| 20 | 68.5 | 56.846 | 144.200 | 899.425 |
| 21 | 89.5 | 1.431 | 154.993 | 900.909 |
| 22 | 112.9 | -60.301 | 142.789 | 899.465 |
| 23 | 134.3 | -108.290 | 110.897 | 900.019 |
| 24 | 158.0 | -143.726 | 58.033 | 899.251 |
| 25 | 179.6 | -154.997 | 0.979 | 899.700 |
| 26 | 203.0 | -142.686 | -60.544 | 899.415 |
| 27 | 225.6 | -108.458 | -110.734 | 900.979 |
| 28 | 246.8 | -60.950 | -142.513 | 900.448 |
| 29 | 270.1 | 0.239 | -155.000 | 900.526 |
| 30 | 292.2 | 58.565 | -143.510 | 899.537 |
| 31 | 314.6 | 108.786 | -110.411 | 899.039 |
| 32 | 336.5 | 142.167 | -61.754 | 900.302 |

## COMSOL self-check

```json
{
  "pipe_shell_healthy": []
}
```

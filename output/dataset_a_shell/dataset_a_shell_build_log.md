# Simple Shell Model Build Log

## Generated files

- `D:\lab_ultr\fz\simple\output\dataset_a_shell\pipe_shell_healthy.mph`

## Model family

Dataset A simple shell: ideal healthy baseline

## Notes

- Ideal no-defect shell model.
- Dataset A uses shell end absorbing layers to suppress end reflections.
- No transducer position or amplitude perturbation.
- PZT solids are replaced by equivalent shell face-load windows.

## Simplifications

- Pipe is a cylindrical shell midsurface at `Rm = 155.000 mm`.
- Wall loss defects are represented by spatially varying shell thickness, not Boolean corrosion cuts.
- PZT solids are removed. Excitation is an equivalent face load with a smooth transducer window.
- Receivers are 16 shell displacement points on the receiver ring.
- Mesh is controlled by wavelength: `hmax = 5.208 mm`, not by PZT block dimensions.

## Where to find the important settings in COMSOL Model Builder

- Thickness: `Component 1 > Shell Mechanics > shell thickness and defect wall loss`.
- Explicit shell material: `Component 1 > Shell Mechanics > explicit aluminum shell elastic material`.
- Equivalent excitation: `Component 1 > Shell Mechanics > equivalent transducer face load`.
- Active transmitter/frequency: `Global Definitions > Parameters`, then `tx` and `pzt_fc`.
- Excitation pulse: `Global Definitions > Functions > five-cycle Hanning sine`.
- Receiver points: `Results > Datasets > receiver PZT 17 point` through `receiver PZT 32 point`.

The excitation patches are not separate geometric PZT faces. Their positions are encoded in the face-load expression as smooth spatial windows so they do not force local mesh refinement.

## Receiver points

| Channel | theta_deg | x_mm | y_mm | z_mm |
| --- | ---: | ---: | ---: | ---: |
| 17 | 0.0 | 155.000 | 0.000 | 900.000 |
| 18 | 22.5 | 143.201 | 59.316 | 900.000 |
| 19 | 45.0 | 109.602 | 109.602 | 900.000 |
| 20 | 67.5 | 59.316 | 143.201 | 900.000 |
| 21 | 90.0 | 0.000 | 155.000 | 900.000 |
| 22 | 112.5 | -59.316 | 143.201 | 900.000 |
| 23 | 135.0 | -109.602 | 109.602 | 900.000 |
| 24 | 157.5 | -143.201 | 59.316 | 900.000 |
| 25 | 180.0 | -155.000 | 0.000 | 900.000 |
| 26 | 202.5 | -143.201 | -59.316 | 900.000 |
| 27 | 225.0 | -109.602 | -109.602 | 900.000 |
| 28 | 247.5 | -59.316 | -143.201 | 900.000 |
| 29 | 270.0 | -0.000 | -155.000 | 900.000 |
| 30 | 292.5 | 59.316 | -143.201 | 900.000 |
| 31 | 315.0 | 109.602 | -109.602 | 900.000 |
| 32 | 337.5 | 143.201 | -59.316 | 900.000 |

## COMSOL self-check

```json
{
  "pipe_shell_healthy": []
}
```

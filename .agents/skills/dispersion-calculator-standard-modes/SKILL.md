---
name: dispersion-calculator-standard-modes
description: Configure, export, audit, and supplement Dispersion Calculator 3.2 standard isotropic-pipe modes for the local ultrasonic project. Use when setting DC material or pipe geometry, generating standard F(n,m), L(0,m), or T(0,m) curves, validating horizontal TXT exports, diagnosing missing or duplicated thickness/order/mode data, or preparing the authoritative reference for b_j-to-m mapping.
---

# Dispersion Calculator Standard Modes

Use DC as the standard mode-name authority. Keep its numerical curves separate from COMSOL Shell branches and
Solid calibration values.

## Fix the project contract

- Use DC `3.2.0.0` at
  `C:\Program Files\DispersionCalculator32\application\DC_v32_Installer.exe`.
- Record executable SHA-256 `34BE241E1597EBFD3A450C8E91F3CFB837FBC3F9E84FF75AF6BE5FBC0D4D322B`.
- Treat DC as a GUI-only application. Do not invent a CLI or coordinate-based click automation.
- Read `f_domain/new/plan.md` before changing axes or mapping thresholds.
- Remember that TXT does not embed material or geometry metadata; record them beside every import.

Use this fixed reference configuration:

```text
material name: COMSOL_Al_E70
E = 70 GPa, nu = 0.33, rho = 2700 kg/m3
inner diameter = 300 mm
thickness h = 5, 6, 7, 7.5, 8, 9, 10 mm
outer diameter = 310, 312, 314, 315, 316, 318, 320 mm
n = 1..8 for F; n = 0 is supplied by L/T
calculation range = 0.001..125 kHz
mapping range = 15..110 kHz
```

## Configure DC

1. Open `Material editor`, set density `2700`, real Young's modulus `70`, imaginary modulus `0`, real Poisson
   ratio `0.33`, and imaginary ratio `0`; save as `COMSOL_Al_E70`.
2. Verify longitudinal and transverse bulk velocities near `6197.8243` and `3121.9527 m/s`.
3. On `Isotropic`, choose `Pipe`, the custom material, the required outer diameter, and inner diameter `300`.
   Leave outer fluid, inner fluid, `Sink at center`, and `Force 2-D tracing` unchecked.
4. Set frequency limit `125 kHz`, frequency step `0.125 kHz`, phase-velocity limit `21 m/ms`, phase-velocity
   accuracy `1e-6 m/s`, higher-mode search step `0.0125 kHz`, and samples r `50`; then enable `Fix`.
5. Enable higher-order, torsional, longitudinal, and flexural modes. Enter `8` in `Flexural mode orders`;
   this field is the maximum n and automatically computes `n=1..8`, so never enter `1:8`.
6. Run `Search`, then `Trace modes`, then `Calculate c_E` after every diameter change.
7. For visual inspection choose wavenumber, frequency x-axis `[0 125]`, and y-axis `[0 0.7] rad/mm`.

## Export and audit

- Check `Dispersion curves`, choose `Frequency (kHz)` and `Horizontal arrangement`, and export `*.txt`.
- Do not require MAT files; they contain MATLAB table objects. XLSX is optional for human inspection.
- Use `D:\lab_ultr\fz\dc_exports\hXXpX\pipe_hXXpX_{F1..F8,L,T}.txt` and exactly 10 TXT files per thickness.
- Parse each standard m as one 11-column block: f, phase velocity, two energy-velocity components, energy-velocity
  magnitude, skew, propagation time, coincidence angle, wavelength, wavenumber, and attenuation.
- Require F headers `F(n,1..3)` with the filename n, plus `L(0,1..2)` and `T(0,1)`. Canonicalize DC's
  second order p as project symbol m.
- Check finite single segments and increasing k inside 15--110 kHz, and verify `k=2*pi*f/cp` after SI conversion.
- Reject repeated F/L content across thicknesses. Allow identical `T(0,1)` files because this lossless torsional
  fundamental is nondispersive and independent of wall thickness.
- Keep TXT and derived NPZ/JSON outside Git; preserve them until the standard reference SHA is published.

## Request supplemental data

When an audit fails, report the exact missing artifact and the corresponding GUI action:

| Missing or failed data | Required action |
|---|---|
| One `hXXpX` directory or wrong prefix | Set inner diameter 300 and outer diameter `300+2h`, rerun all three calculation buttons, export all 10 TXT files. |
| `Fn.txt` absent | Set `Flexural mode orders` to at least n and rerun Search/Trace/c_E; never rename another order. |
| Higher m absent | Enable `Higher order modes`, use search step 0.0125 kHz, rerun Search before Trace. |
| Energy velocity blank or NaN | Run `Calculate c_E` after tracing, then overwrite only that thickness export. |
| Curve ends below 125 kHz | Restore frequency limit/step to 125/0.125 kHz and recompute, not extrapolate. |
| F/L hash duplicates another thickness | Re-enter the correct outer diameter and recompute; changing the filename is invalid. |
| Geometry/material provenance missing | Supply screenshots of Material editor, Specimen, Computational settings, Mode selection, and DC version. |
| Mapping remains ambiguous at named points | Export displacement through-thickness profiles only for the reported `(h,n,m,f)` points; do not export an unbounded grid. |

Hand the complete DC reference to `comsol-frequency-streaming` for numerical-branch mapping. Recompute DC only
when material, inner radius, thickness/frequency/n range, or the DC source changes; defect geometry and seed do
not require new standard curves.

# Frequency-Domain Simple Shell Build Log

## Generated files

- `D:\lab_ultr\fz\simple\f_domain\output\dataset_a_frequency_shell\pipe_shell_frequency_healthy.mph`

## Model family

Dataset A simple shell frequency-domain solve/export

## Notes

- Dataset A frequency-domain shell model.
- Shell end absorbing layers are enabled to suppress end reflections.
- No transducer position, amplitude, or material perturbation.
- PZT solids are replaced by equivalent shell face-load windows.
- Frequency-domain excitation uses harmonic load amplitude, not pztpulse(t).
- Receivers are patch-weighted radial displacement averages using intop_shell.

## Frequency-domain changes

- Study type: `Frequency`, with frequency list expression `pzt_fc`.
- Time pulse `pztpulse(t)` is not used in the face load.
- The equivalent load is a harmonic shell face-load amplitude `F0/pzt_A * window_tx`.
- Dataset A absorbing layers are enabled through the same axial Rayleigh damping ramp as the time-domain Dataset A model.
- Receivers are the same patch-weighted averages: `intop_shell(w_rx*u_r)/intop_shell(w_rx)`.

## COMSOL Model Builder checks

- Equivalent excitation: `Component 1 > Shell Mechanics > equivalent transducer face load`.
- Active transmitter/frequency: `Global Definitions > Parameters`, then `tx` and `pzt_fc`.
- Frequency study: `Study > simple shell displacement frequency domain`.
- Receiver weighted averages: `Results > Derived Values > receiver patch weighted average radial displacement`.
- Optional marker datasets: `Results > Datasets > transmitter PZT marker points` and `receiver PZT marker points`.

## COMSOL self-check

```json
{
  "pipe_shell_frequency_healthy": []
}
```

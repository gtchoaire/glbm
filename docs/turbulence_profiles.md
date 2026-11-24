# Centerline turbulence profiles

The optional `bmbc_turbulence_profiles` mode samples the raw moments required
for the requested article profiles without allocating volumetric Reynolds
stresses:

- along the central y line: `ux`, `uy`, `ux^2`, and `ux*uy`;
- along the central x line: `ux`, `uy`, `uy^2`, and `ux*uy`.

All moments use the same stationary window and the same geometric-center
interpolation. The post-processor computes

```text
ux_rms(y) = sqrt(<ux^2> - <ux>^2)
uy_rms(x) = sqrt(<uy^2> - <uy>^2)
<ux'uy'> = <ux uy> - <ux><uy>
```

The uncertainty is not obtained by independently combining scalar error bars.
For each spatial point, the BMBC lag-zero and lag-one cross-covariance matrices
of the raw block moments are propagated through the nonlinear expression with
the delta method. This preserves correlations between first and second
moments.

The digitized Prasad-Koseff turbulence curves retain the plotting
amplifications used in the comparison figures: RMS values are multiplied by
10 and Reynolds shear by 500. The post-processor removes these factors before
computing errors or drawing the experimental points.

Sidecars using this extended layout are BMBC format v2. The option is disabled
in all existing mean-velocity cases, so their v1 checkpoints and restarts keep
the validated layout.

## 256-cubed pilot

The first pilot uses two GPUs, Re=3200, 120 t-star of transient and 100 t-star
of sampling. It stores profiles and energy but not the ten volumetric averages.

```bash
cd /home/geoen/glbm
./scripts/run_re3200_turbulence_256_pilot.sh
```

If interrupted after a checkpoint, resume with:

```bash
./scripts/run_re3200_turbulence_256_pilot.sh \
  /absolute/path/to/checkpoint_re3200_turbulence_256_pilot.glbm
```

The script runs the multivariate BMBC and experimental comparison after the
simulation. The 100 t-star window is an initial diagnostic, not an assumption
that RMS and Reynolds shear have converged.

## Extending the validated pilot

After checking the 100 t-star pilot, extend the same BMBC history from the
saved checkpoint to 600 t-star of total sampling (ending at t-star = 720):

```bash
cd /home/geoen/glbm
./scripts/extend_re3200_turbulence_256.sh \
  /absolute/path/to/checkpoint_re3200_turbulence_256_pilot.glbm
```

The script reuses the pilot build directory, appends the solver log, preserves
the existing run directory, and regenerates the turbulence and BMBC summaries
after the extension.

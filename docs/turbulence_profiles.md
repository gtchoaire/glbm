# Statistics

Only the quantities required by the figures are stored.

| Figure | Statistic | Source |
|---|---|---|
| Kinetic energy, dissipation rate | volume averages in time | `global_statistics.csv` |
| Probe velocity and spectrum | `u_i(t)`, `E_ii(f)` | `global_statistics.csv` |
| Energy balance (`energy_balance.dat`) | `P_lid`, `<2 nu S_ij S_ij>_V` | `local_fields.bin` |
| Velocity and Reynolds stress profiles | `<u_i>`, `r_ij`, rms | `bmbc_blocks.bin` |
| Velocity and vorticity maps | `<u_i>`, `curl <u>` | `local_fields.bin` |
| Lumley triangle | `xi`, `eta` | both |
| Kolmogorov map | `min_z(eta_K)/dx` in the x-y plane | `local_fields.bin` |

Each sample evaluates 16 quantities per node: `u_i` (3), `u_i u_j` (6), `S_ij`
(6) and `S2 = S_ij S_ij` (1). `<S_ij>` and `<S2>` serve the Kolmogorov scale.
The stored state is post-collision, `m* = (1 - omega) m + omega u u`, so the
strain rate follows from the moments without finite differences:

```text
S_ij = (omega / (2 cs2)) (u_i u_j - m*_ij) / (1 - omega)
cs2 = 1/3, omega = 1/tau, tau = 1/2 + 3 nu, nu = U L / Re, L = N - 1
```

`<.>` is the arithmetic mean over the samples in `(start_t_star, end_t_star]`,
taken every `sample_interval` steps, with `t* = t U / L`.
`local_fields.bin` keeps the 16 means at every node and `bmbc_blocks.bin` the
block means on `y_center` (x = z = 1/2) and `x_center` (y = z = 1/2). With an
even number of nodes a center value is the mean of the surrounding nodes, with
products evaluated at the nodes.

```text
<f>_V = sum(w f) / ((Nx - 1)(Ny - 1)(Nz - 1))
<E>_V  = <u_i u_i>_V / 2                                     / U^2
<2 nu S_ij S_ij>_V                                           L / U^3

u'_i  = u_i - <u_i>
r_ij  = <u'_i u'_j> = <u_i u_j> - <u_i><u_j>                 / U^2
rms_i = sqrt(r_ii)                                           / U

omega = curl <u>                                             L / U

b_ij  = r_ij / r_kk - delta_ij / 3
6 eta^2 = b_ij b_ji,  6 xi^3 = b_ij b_jk b_ki

s_ij    = S_ij - <S_ij>
epsilon = 2 nu <s_ij s_ij> = 2 nu (<S2> - <S_ij><S_ij>)      L / U^3
eta_K   = (nu^3 / epsilon)^(1/4)                             / L
```

Notation follows Pope (2000): `epsilon` is the dissipation rate of turbulent
kinetic energy. The global series `2 nu S_ij S_ij` is the viscous dissipation of
the total kinetic energy `E`, mean flow included, so that
`d<E>_V/dt = P_lid - <2 nu S_ij S_ij>_V`. Volume integrals use the trapezoidal
rule: `w` is 1/2 on faces, 1/4 on edges and 1/8 on corners, because the wall
nodes lie on the walls (see also Luo et al., PRE 83, 056710, 2011).

```text
P_lid = U integral(2 nu <S_xy> dA) / V  on the lid             L / U^3
E_ii(f): one-sided spectrum, integral(E_ii df) = <u'_i u'_i>   U L
```

The probe is the point of Prasad and Koseff's time traces and spectra, on the
vertical centerline, 7 mm above the floor (`y/L = 7/150`), or 5 mm for
`Re = 10000`, by trilinear interpolation. The spectrum is a Welch average of 7
Hann segments with 50% overlap over the sampling window.

The vorticity uses fourth-order differences in the plane and a centered
difference across it. The planes are listed in `PLANES` in
`process_statistics.py` and can be changed after the run.

`wall_shear.dat` holds `<S_xy>` in lattice units on the upstream (`x = 0`),
downstream (`x = L`) and bottom (`y = 0`) walls at mid-span, from the moments and
as `1/2 (d<u_x>/dy + d<u_y>/dx)` with second- and fourth-order differences of the
mean velocity. `vortex_sizes.dat` holds the height and length, over `L`, of the
upstream (USV) and downstream (DSV) corner vortices: the point where `<S_xy>`
changes sign on each wall, by linear interpolation between nodes.

The Kolmogorov map takes, at each `(x, y)`, the smallest `eta_K` over all `z`,
divided by the grid spacing; nodes with a ratio below 1 are red.

Global averages, velocities, `r_ij`, rms, dissipation and the Kolmogorov scale
on the two lines carry BMBC intervals. The maps and the Lumley invariants do
not.

The digitized Prasad-Koseff RMS values are amplified by 10 and the Reynolds
shear by 500; the post-processor removes both factors.

## 256-cubed pilot

```bash
./scripts/run_re3200_turbulence_256_pilot.sh
./scripts/extend_re3200_turbulence_256.sh \
  /absolute/path/to/run/checkpoint/checkpoint_re3200_turbulence_256_pilot.glbm
```

The first command samples 100 t-star after 120 t-star of transient; the second
extends the same history to 600 t-star. An interrupted pilot resumes with its
checkpoint path as the first argument.

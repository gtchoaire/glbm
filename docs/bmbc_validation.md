# BMBC and mean-profile validation

The temporal accumulator and BMBC answer different questions:

- `TemporalAverages` updates the ten stored mean fields with the recurrent
  formula `mean += (sample - mean) / n`. The accumulators are always float64,
  even when the LBM state is float32, to prevent long-window stagnation.
- BMBC stores non-overlapping means only for the two validation centerlines and
  estimates the sampling variance of those means.
- The Prasad-Koseff comparison measures validation error, which also includes
  spatial discretization, modeling, boundary-condition, and experimental
  differences.

The implemented finite-sample estimator is

```
variance(mean) = (S0 + 2 S1) / ((K - 1) (K - 2))
S0 = sum_k (Xbar_k^2 - mean^2)
S1 = sum_k (Xbar_k Xbar_{k+1} - mean^2)
```

Following Russo and Luchini (2017), the post-processing selects, separately
for each profile, the available batch size whose spatial median is closest to
`S1/S0 = 0.5`. A single batch size cannot generally meet this condition at
every spatial point, so the CSV retains the full pointwise diagnostics.

Checkpoints v4 store `stats_precision float64`; the physical LBM state keeps
the configured solver precision. The loader and Python extractor remain
compatible with v3 checkpoints.

Reference: S. Russo and P. Luchini, *A fast algorithm for the estimation of
statistical error in DNS (or experimental) time averages*, JCP 347 (2017),
https://doi.org/10.1016/j.jcp.2017.07.005.

The production cases use a transient of `120 t*`, followed by a `100 t*`
sampling window. They are intentionally separate so Re=3200 can be assessed
before committing the GPU time to Re=10000.

For the longer one-GPU experiment, `re3200_long_mean_profiles.toml` uses the
same `120 t*` transient followed by `600 t*` of averaging. The ten float64
fields are sampled every eight solver steps to avoid making the statistics
kernel the dominant cost. A basic BMBC block has 128 sampled states, preserving
the previous physical length of 1024 solver steps. BMBC console reports occur
every 256 basic blocks and full profile CSVs every 512 blocks (about `105.7 t*`
between profile exports). Experimental L2 is never evaluated inside the solver
loop; it is computed offline from these CSVs.

```bash
cmake -S . -B build-re3200 \
  -DLBM_CONFIG_FILE="$PWD/cases/re3200_mean_profiles.toml"
cmake --build build-re3200 -j
ctest --test-dir build-re3200 --output-on-failure
./build-re3200/glbm

python post/verify_bmbc_averages.py \
  build-re3200/output/<run>/checkpoint_re3200_mean_profiles.glbm

python post/analyze_bmbc_profiles.py \
  build-re3200/output/<run>/bmbc_profiles_step_1091407.csv \
  --halfwidth-tol-over-u 0.01
```

Long run from the initial condition:

```bash
cmake -S . -B build-re3200-long \
  -DLBM_CONFIG_FILE="$PWD/cases/re3200_long_mean_profiles.toml"
cmake --build build-re3200-long -j
./build-re3200-long/glbm
```

After completion, analyze the evolution of L2 and BMBC uncertainty across all
periodic profile exports:

```bash
python post/analyze_bmbc_sequence.py \
  build-re3200-long/output/<run-directory>
```

The reported relative L2 error is evaluated at the irregular experimental
coordinates after linear interpolation of the numerical profile:

```
relative_L2 = sqrt(integral(error^2 ds) / integral(reference^2 ds))
```

Use the BMBC half-width and its stability across batch sizes to decide whether
more samples are needed. Use the experimental L2 to compare grids, Reynolds
numbers, and numerical choices; do not use it alone as a stopping rule for the
time average.

If the first `100 t*` window is insufficient, it can be continued without
discarding samples. The new end must extend the stored end, and the restart
must occur before any unsampled gap. The extension case reuses the checkpoint
filename, so completed checkpoints are replaced atomically instead of
accumulating large files on disk:

```bash
cmake -S . -B build-re3200-extended \
  -DLBM_CONFIG_FILE="$PWD/cases/re3200_extend_mean_profiles.toml" \
  -DLBM_RESTART_PATH_OVERRIDE="$PWD/build-re3200/output/<run>/checkpoint_re3200_mean_profiles.glbm"
cmake --build build-re3200-extended -j
./build-re3200-extended/glbm
```

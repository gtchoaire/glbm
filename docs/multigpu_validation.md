# Multi-GPU validation

Validation performed on 2026-11-19 under WSL with CUDA 13.4 and two
GeForce RTX 4090 GPUs (compute capability 8.9, 23028 MiB each).

The CUDA topology probe reports mapped host memory and unified virtual
addressing on both devices, but CUDA peer access is unavailable in both
directions. Halo exchange therefore uses pinned host memory between GPUs.

## Numerical equivalence

`cases/multigpu_validation.toml` is a short deterministic regression case,
not a production physics case. It runs a 64-cubed domain for 4096 steps with
one statistics sample per step, 64-sample BMBC blocks, periodic energy output,
binary fields and checkpoints.

Comparing the one- and two-GPU executions:

- all four final macroscopic fields are bitwise identical;
- total kinetic energy output is bitwise identical;
- the canonical checkpoint payload contains only the 10 solver moments and is
  bitwise identical;
- the independent local field statistics contain 16 float64 temporal means;
- after 64 complete BMBC blocks, the maximum differences are
  `1.73e-18` in the reported mean and `2.96e-21` in the variance;
- cross-decomposition restart succeeds in both directions (one to two GPUs
  and two to one GPU), and the continued solver/checkpoint state remains
  bitwise identical.

The BMBC block file records global dimensions, so its metadata does not depend
on whether the run uses one or two GPUs.

## Solver-only strong scaling

The benchmark cases disable statistics, checkpoints and field output. Each
performs approximately 17.18 billion lattice updates. Values below are the
best observed wall-clock measurements; two repetitions were used except for
the one-GPU 512-cubed run, which was executed once.

| Domain | Steps | 1 GPU MLUPS | 2 GPU MLUPS | Speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|
| 128 cubed | 8192 | 8510 | 4482 | 0.53 | 26% |
| 256 cubed | 1024 | 9086 | 10461 | 1.15 | 58% |
| 512 cubed | 128 | 8739 | 14831 | 1.70 | 85% |

Two GPUs should therefore not be used for a 128-cubed domain on this machine.
They become marginally useful at 256 cubed and clearly useful at 512 cubed.

Packing halos in device memory followed by asynchronous device-to-host and
host-to-device copies improved the 512-cubed two-GPU result from 13638 to
14831 MLUPS (8.7%) compared with having the packing kernel write directly to
mapped host memory.

## Reproduction

Use a fresh build directory for each GPU count. For example:

```bash
cmake -S . -B build-mgpu-validation-1 \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DLBM_CONFIG_FILE="$PWD/cases/multigpu_validation.toml" \
  -DLBM_NUM_GPUS_OVERRIDE=1
cmake --build build-mgpu-validation-1 -j"$(nproc)"
(cd build-mgpu-validation-1 && ./glbm)

cmake -S . -B build-mgpu-validation-2 \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DLBM_CONFIG_FILE="$PWD/cases/multigpu_validation.toml" \
  -DLBM_NUM_GPUS_OVERRIDE=2
cmake --build build-mgpu-validation-2 -j"$(nproc)"
(cd build-mgpu-validation-2 && ./glbm)
```

Replace the validation case with `multigpu_benchmark_128.toml`,
`multigpu_benchmark_256.toml` or `multigpu_benchmark_512.toml` for the
solver-only measurements.

The topology diagnostic is available when testing is enabled:

```bash
cmake --build build-mgpu-validation-1 --target cuda_topology_probe
./build-mgpu-validation-1/cuda_topology_probe
```

## Next validation stage

Profile BMBC sampling and the local field statistics have
independent intervals. The four 256-cubed cases prefixed with
`multigpu_stats_256_` isolate solver, profile, volumetric and combined costs.
Run all four on two GPUs with:

```bash
./scripts/run_multigpu_statistics_benchmark.sh 2
```

The command writes a compact `summary.csv` plus the individual logs under a
timestamped directory in `benchmark-results/`. RMS and Reynolds-stress
profiles should only be added after that baseline is recorded, so their
numerical and performance costs remain separable.

### Statistics baseline on 2026-11-19

The first 256-cubed, two-GPU run used 2048 steps and sampled both profile and
volumetric statistics every 16 steps:

| Mode | Wall MLUPS | Device MLUPS | Device cost vs. solver |
|---|---:|---:|---:|
| Solver | 10937 | 9844 | baseline |
| BMBC profiles | 10208 | 9560 | 2.9% |
| 10 local field means | 9069 | 8888 | 9.7% |
| Combined | 9186 | 8939 | 9.2% |

The small difference between volumetric and combined results is within the
variation of one short execution. These historical values predate the current
16-field layout and should not be used as its final performance measurement.

#!/usr/bin/env python3
"""Cross-check BMBC profile means against the 10-field recurrent averages.

This includes completed BMBC blocks and the partial block, so both sides use
exactly the same samples. No CUDA runtime is required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from extract_glbm_mean_profiles import FIELDS, get_means
from read_bmbc_snapshot import read_snapshot


def recurrent_profiles(means, dimensions):
    nx, ny, nz = dimensions
    x0, x1 = (nx - 1) // 2, nx // 2
    y0, y1 = (ny - 1) // 2, ny // 2
    z0, z1 = (nz - 1) // 2, nz // 2
    ux = np.asarray(means[FIELDS.index("ux_avg")], dtype=np.float64)
    uy = np.asarray(means[FIELDS.index("uy_avg")], dtype=np.float64)
    ux_y = 0.25 * (
        ux[z0, :, x0] + ux[z0, :, x1] + ux[z1, :, x0] + ux[z1, :, x1]
    )
    uy_x = 0.25 * (
        uy[z0, y0, :] + uy[z0, y1, :] + uy[z1, y0, :] + uy[z1, y1, :]
    )
    return np.concatenate((ux_y, uy_x))


def bmbc_all_sample_mean(metadata, partial_sum, batch_means):
    if metadata["total_samples"] == 0:
        raise ValueError("BMBC snapshot contains no samples")
    completed_sum = batch_means.sum(axis=0) * metadata["block_samples"]
    return (completed_sum + partial_sum) / metadata["total_samples"]


def verify(checkpoint, tolerance=None):
    header, means, dimensions = get_means(checkpoint)
    snapshot = Path(str(checkpoint) + ".bmbc")
    metadata, partial_sum, batch_means = read_snapshot(snapshot)
    if metadata["step"] != int(header["step"]):
        raise ValueError("Checkpoint and BMBC snapshot steps differ")
    if metadata["total_samples"] != int(header["stats_samples"]):
        raise ValueError("Checkpoint and BMBC snapshot sample counts differ")
    if (metadata["nx"], metadata["ny"]) != dimensions[:2]:
        raise ValueError("Checkpoint and BMBC profile dimensions differ")

    recurrent = recurrent_profiles(means, dimensions)
    bmbc = bmbc_all_sample_mean(metadata, partial_sum, batch_means)
    difference = recurrent - bmbc
    nx, ny, _ = dimensions
    slices = {"ux_y": slice(0, ny), "uy_x": slice(ny, ny + nx)}
    metrics = {}
    for profile, profile_slice in slices.items():
        error = difference[profile_slice]
        metrics[profile] = {
            "max_abs": float(np.max(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error * error))),
        }

    if tolerance is None:
        tolerance = (
            1e-12 if header.get("stats_precision") == "float64"
            else (1e-6 if header["precision"] == "float32" else 1e-12)
        )
    maximum = max(metric["max_abs"] for metric in metrics.values())
    return metadata, metrics, tolerance, maximum <= tolerance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path,
                        help="GLBM v3 checkpoint; its .bmbc sidecar is read automatically")
    parser.add_argument("--tolerance", type=float, default=None,
                        help="Absolute solver-unit tolerance (precision-dependent default)")
    args = parser.parse_args()

    metadata, metrics, tolerance, passed = verify(args.checkpoint, args.tolerance)
    print(f"Samples compared: {metadata['total_samples']}")
    for profile, values in metrics.items():
        print(f"{profile}: max|difference|={values['max_abs']:.6e}, "
              f"RMSE={values['rmse']:.6e}")
    print(f"Tolerance: {tolerance:.3e} | {'PASS' if passed else 'FAIL'}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()

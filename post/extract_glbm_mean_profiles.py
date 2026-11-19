#!/usr/bin/env python3
"""Extract actual time-averaged profiles from a GLBM v3/v4 checkpoint.

The payload stores 10 LBM moments per node (node-major), followed by 10
statistical means in field-major, global z/y/x (x-fastest) order. Version 4
stores statistics in float64 independently of solver precision.

Example:
    python3 extract_glbm_mean_profiles.py build/output/DATE/checkpoint_stats_128.glbm
    python3 extract_glbm_mean_profiles.py checkpoint.glbm --out post/means --save-fields
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

FIELDS = ("rho_avg", "ux_avg", "uy_avg", "uz_avg", "Sxx_avg",
          "Sxy_avg", "Sxz_avg", "Syy_avg", "Syz_avg", "Szz_avg")


def read_header(path: Path) -> tuple[dict[str, str], int]:
    fields: dict[str, str] = {}
    with path.open("rb") as file:
        while True:
            line = file.readline()
            if not line:
                raise ValueError("Checkpoint without a complete 'data' header marker")
            line = line.decode("utf-8").strip()
            if line == "data":
                return fields, file.tell()
            key, separator, value = line.partition(" ")
            if not separator:
                raise ValueError(f"Malformed header line: {line!r}")
            fields[key] = value


def get_means(path: Path):
    header, payload_offset = read_header(path)
    version = int(header.get("GLBM_CHECKPOINT", "0"))
    if version not in (3, 4):
        raise ValueError("This script requires GLBM_CHECKPOINT 3 or 4")
    if header.get("statistics") != "1":
        raise ValueError("Checkpoint contains no time-averaged fields (statistics 0)")
    if int(header["stats_fields"]) != len(FIELDS):
        raise ValueError("Unexpected statistics field count")

    nx, ny, nz = map(int, header["dimensions"].split())
    if min(nx, ny, nz) < 3:
        raise ValueError("Grid dimensions must each be at least 3 for edge_order=2")
    num_moments = int(header["moments"])
    dtype = {"float32": np.dtype("<f4"), "float64": np.dtype("<f8")}.get(
        header["precision"])
    if dtype is None:
        raise ValueError(f"Unsupported precision: {header['precision']}")

    num_nodes = nx * ny * nz
    means_offset = payload_offset + num_nodes * num_moments * dtype.itemsize
    stats_dtype = (
        np.dtype("<f8") if header.get("stats_precision") == "float64" else dtype
    )
    expected_size = means_offset + num_nodes * len(FIELDS) * stats_dtype.itemsize
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"Unexpected checkpoint size: {actual_size} bytes, "
            f"expected {expected_size} bytes; check format and file completeness"
        )
    # Memory mapping avoids loading the entire 128^3 / 256^3 dataset into RAM.
    means = np.memmap(
        path, mode="r", dtype=stats_dtype, offset=means_offset,
        shape=(len(FIELDS), nz, ny, nx), order="C"
    )
    return header, means, (nx, ny, nz)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="GLBM checkpoint v3/v4 with statistics")
    parser.add_argument("--out", type=Path, default=Path("post/mean_profiles"),
                        help="Directory for CSV outputs")
    parser.add_argument("--save-fields", action="store_true",
                        help="Also export each complete mean field to .npy (z,y,x order)")
    args = parser.parse_args()

    header, means, (nx, ny, nz) = get_means(args.checkpoint)
    args.out.mkdir(parents=True, exist_ok=True)
    z = nz // 2
    length_x = nx - 1
    length_y = ny - 1
    ystar = np.arange(ny, dtype=np.float64) / length_y

    # Global z-middle slice (shape: ny, nx).
    ux = np.asarray(means[FIELDS.index("ux_avg"), z], dtype=np.float64)
    uy = np.asarray(means[FIELDS.index("uy_avg"), z], dtype=np.float64)
    sxy = np.asarray(means[FIELDS.index("Sxy_avg"), z], dtype=np.float64)
    # Lattice unit spacing is 1.  No extra factor L, U, or sqrt(3) is applied.
    half_dux_dy = 0.5 * np.gradient(ux, 1.0, axis=0, edge_order=2)
    half_duy_dx = 0.5 * np.gradient(uy, 1.0, axis=1, edge_order=2)
    sxy_fd = half_dux_dy + half_duy_dx

    x_profiles = sorted({0, 1, 2, nx // 2, nx - 3, nx - 2, nx - 1})
    fieldnames = ["x", "x_star", "y", "y_star", "ux_avg", "uy_avg",
                  "half_dux_dy", "half_duy_dx", "sxy_avg", "sxy_fd", "difference"]
    for x in x_profiles:
        out = args.out / f"sxy_mean_x{x:03d}.csv"
        with out.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(fieldnames)
            for y in range(ny):
                writer.writerow([
                    x, x / length_x, y, ystar[y], ux[y, x], uy[y, x],
                    half_dux_dy[y, x], half_duy_dx[y, x], sxy[y, x],
                    sxy_fd[y, x], sxy[y, x] - sxy_fd[y, x]
                ])
        error = sxy[:, x] - sxy_fd[:, x]
        interior = (ystar >= 0.05) & (ystar <= 0.80)
        rmse = np.sqrt(np.mean(error[interior] ** 2))
        print(f"x={x:3d} | interior RMSE = {rmse:.6e} | {out}")

    if args.save_fields:
        for i, name in enumerate(FIELDS):
            dest = args.out / f"{name}.npy"
            np.save(dest, means[i])
            print(f"Exported full mean field {dest} (z,y,x)")

    print(f"Checkpoint step: {header['step']}")
    print(f"Statistical window: [{header['stats_start_step']}, "
          f"{header['stats_end_step']}] steps | samples: {header['stats_samples']}")
    print(f"Central z slice: z={z} / {nz - 1}")


if __name__ == "__main__":
    main()

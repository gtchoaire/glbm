"""Inspect a GLBM BMBC v1 companion checkpoint without running CUDA.

Usage:
  python post/read_bmbc_snapshot.py output/.../checkpoint.glbm.bmbc
  python post/read_bmbc_snapshot.py output/.../checkpoint.glbm.bmbc --npz blocks.npz

The file contains a complete *partial* block accumulator followed by
completed block means, each of shape [NY+NX]. Positions 0:NY hold ux(y),
positions NY:NY+NX hold uy(x). Solver units, not normalized by U_MAX.
"""
import argparse
from pathlib import Path
import numpy as np

FIELDS = [
    "GLBM_BMBC", "nx", "ny", "nz", "step", "block_samples",
    "completed_blocks", "samples_in_block", "total_samples"
]


def read_snapshot(path: Path):
    metadata = {}
    with path.open("rb") as f:
        for field in FIELDS:
            line = f.readline().decode("ascii").strip().split()
            if len(line) != 2 or line[0] != field:
                raise ValueError(f"Invalid BMBC field: expected {field}")
            metadata[field] = int(line[1])
        if metadata["GLBM_BMBC"] != 1:
            raise ValueError("Unsupported BMBC format version")
        if f.readline() != b"data\n":
            raise ValueError("Missing BMBC data marker")
        points = metadata["nx"] + metadata["ny"]
        partial_sum = np.fromfile(f, dtype="<f8", count=points)
        if partial_sum.size != points:
            raise ValueError("Truncated BMBC partial block")
        count = metadata["completed_blocks"] * points
        batch_means = np.fromfile(f, dtype="<f8", count=count)
        if batch_means.size != count:
            raise ValueError("Truncated BMBC completed blocks")
        if f.read(1):
            raise ValueError("Unexpected trailing BMBC payload")
        batch_means = batch_means.reshape(metadata["completed_blocks"], points)
    if (metadata["completed_blocks"] * metadata["block_samples"]
            + metadata["samples_in_block"] != metadata["total_samples"]):
        raise ValueError("Inconsistent BMBC counters")
    return metadata, partial_sum, batch_means


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--npz", type=Path, default=None,
                        help="Save block history and partial sum to compressed NumPy archive")
    args = parser.parse_args()
    metadata, partial, blocks = read_snapshot(args.file)
    for field in FIELDS:
        print(f"{field}: {metadata[field]}")
    print(f"Profiles: ux(y) {metadata['ny']} points, uy(x) {metadata['nx']} points")
    print(f"Completed block means array: {blocks.shape}")
    if args.npz is not None:
        np.savez_compressed(args.npz, batch_means=blocks,
                            partial_sum=partial, **metadata)
        print(f"Wrote: {args.npz}")


if __name__ == "__main__":
    main()

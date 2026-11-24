"""Inspect a GLBM BMBC v1 companion checkpoint without running CUDA.

Usage:
  python post/read_bmbc_snapshot.py output/.../checkpoint.glbm.bmbc
  python post/read_bmbc_snapshot.py output/.../checkpoint.glbm.bmbc --npz blocks.npz

The file contains a complete *partial* block accumulator followed by
completed block means. Version 1 stores the two mean-velocity profiles;
version 2 stores the optional raw turbulence-profile moments. Solver units,
not normalized by U_MAX.
"""
import argparse
from pathlib import Path
import numpy as np

BASE_FIELDS = [
    "nx", "ny", "nz", "step", "block_samples",
    "completed_blocks", "samples_in_block", "total_samples"
]


def read_snapshot(path: Path):
    metadata = {}
    with path.open("rb") as f:
        line = f.readline().decode("ascii").strip().split()
        if len(line) != 2 or line[0] != "GLBM_BMBC":
            raise ValueError("Invalid BMBC magic/version")
        metadata["GLBM_BMBC"] = int(line[1])
        if metadata["GLBM_BMBC"] not in (1, 2):
            raise ValueError("Unsupported BMBC format version")
        for field in BASE_FIELDS:
            line = f.readline().decode("ascii").strip().split()
            if len(line) != 2 or line[0] != field:
                raise ValueError(f"Invalid BMBC field: expected {field}")
            metadata[field] = int(line[1])
        if metadata["GLBM_BMBC"] >= 2:
            line = f.readline().decode("ascii").strip().split()
            if len(line) != 2 or line[0] != "profile_points":
                raise ValueError("Invalid BMBC field: expected profile_points")
            metadata["profile_points"] = int(line[1])
        if f.readline() != b"data\n":
            raise ValueError("Missing BMBC data marker")
        points = metadata.get("profile_points", metadata["nx"] + metadata["ny"])
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
    fields = ["GLBM_BMBC", *BASE_FIELDS]
    if "profile_points" in metadata:
        fields.append("profile_points")
    for field in fields:
        print(f"{field}: {metadata[field]}")
    print(f"Stored profile values per block: {partial.size}")
    print(f"Completed block means array: {blocks.shape}")
    if args.npz is not None:
        np.savez_compressed(args.npz, batch_means=blocks,
                            partial_sum=partial, **metadata)
        print(f"Wrote: {args.npz}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Derive centerline RMS and Reynolds-shear profiles from BMBC raw moments.

This stage computes the cumulative profile values. Statistical uncertainty of
the nonlinear derived quantities is intentionally handled separately from the
scalar raw-moment BMBC estimates.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_bmbc_profiles import aspect_ratio_suffix, percentile, relative_l2
from read_bmbc_snapshot import read_snapshot


RAW_PROFILES = (
    "ux_y", "uy_y", "ux2_y", "uxuy_y",
    "ux_x", "uy_x", "uy2_x", "uxuy_x",
)

DERIVED = {
    # The digitized Prasad-Koseff figures amplify RMS by 10 and Reynolds shear
    # by 500 so they can share the mean-velocity plot scale.
    "ux_rms_y": ("uxRMS_cy", r"$u_{x,rms}/U_{lid}$", r"$y^*$", 10.0),
    "uxuy_y": ("uxuy_cy", r"$\overline{u'_x u'_y}/U_{lid}^2$", r"$y^*$", 500.0),
    "uy_rms_x": ("uyRMS_cx", r"$u_{y,rms}/U_{lid}$", r"$x^*$", 10.0),
    "uxuy_x": ("uxuy_cx", r"$\overline{u'_x u'_y}/U_{lid}^2$", r"$x^*$", 500.0),
}


def load_raw_profiles(path: Path):
    grouped = {}
    metadata = {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"profile", "M", "K", "index", "position_star", "mean"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        for row in reader:
            profile = row["profile"]
            if profile not in RAW_PROFILES:
                continue
            m = int(row["M"])
            grouped.setdefault((profile, m), []).append(row)
            for key in ("reynolds_number", "aspect_ratio", "u_max"):
                if key in row:
                    value = float(row[key])
                    if key in metadata and metadata[key] != value:
                        raise ValueError(f"Inconsistent {key}")
                    metadata[key] = value
    if not grouped:
        raise ValueError("No turbulence raw moments in BMBC CSV")

    common_m = sorted(set.intersection(*[
        {m for current, m in grouped if current == profile}
        for profile in RAW_PROFILES
    ]))
    if not common_m:
        raise ValueError("Raw profiles have no common BMBC grouping")
    m = common_m[0]
    profiles = {}
    k_value = None
    for profile in RAW_PROFILES:
        rows = sorted(grouped[(profile, m)], key=lambda row: int(row["index"]))
        indices = [int(row["index"]) for row in rows]
        if indices != list(range(len(rows))):
            raise ValueError(f"Incomplete indices for {profile}")
        ks = {int(row["K"]) for row in rows}
        if len(ks) != 1:
            raise ValueError(f"Inconsistent K for {profile}")
        current_k = ks.pop()
        if k_value is not None and current_k != k_value:
            raise ValueError("Raw profiles use different numbers of blocks")
        k_value = current_k
        profiles[profile] = {
            "position": np.asarray([float(row["position_star"]) for row in rows]),
            "mean": np.asarray([float(row["mean"]) for row in rows]),
        }
    metadata.update({"M": m, "K": k_value})
    return profiles, metadata


def derive_profiles(raw, u_lid):
    ux_y, uy_y = raw["ux_y"]["mean"], raw["uy_y"]["mean"]
    ux_x, uy_x = raw["ux_x"]["mean"], raw["uy_x"]["mean"]
    ux_variance = raw["ux2_y"]["mean"] - ux_y * ux_y
    uy_variance = raw["uy2_x"]["mean"] - uy_x * uy_x
    tolerance = 64.0 * np.finfo(float).eps * u_lid * u_lid
    if np.any(ux_variance < -tolerance) or np.any(uy_variance < -tolerance):
        raise ValueError("Negative resolved variance exceeds roundoff tolerance")

    return {
        "ux_rms_y": {
            "position": raw["ux_y"]["position"],
            "value": np.sqrt(np.maximum(ux_variance, 0.0)) / u_lid,
        },
        "uxuy_y": {
            "position": raw["ux_y"]["position"],
            "value": (raw["uxuy_y"]["mean"] - ux_y * uy_y) / (u_lid * u_lid),
        },
        "uy_rms_x": {
            "position": raw["uy_x"]["position"],
            "value": np.sqrt(np.maximum(uy_variance, 0.0)) / u_lid,
        },
        "uxuy_x": {
            "position": raw["uy_x"]["position"],
            "value": (raw["uxuy_x"]["mean"] - ux_x * uy_x) / (u_lid * u_lid),
        },
    }


def unpack_block_history(metadata, blocks):
    nx, ny = metadata["nx"], metadata["ny"]
    expected = 4 * (nx + ny)
    if metadata.get("GLBM_BMBC") != 2 or blocks.shape[1] != expected:
        raise ValueError("BMBC snapshot does not contain turbulence profile moments")
    return {
        "ux_y": blocks[:, 0:ny],
        "uy_y": blocks[:, ny:2 * ny],
        "ux2_y": blocks[:, 2 * ny:3 * ny],
        "uxuy_y": blocks[:, 3 * ny:4 * ny],
        "ux_x": blocks[:, 4 * ny:4 * ny + nx],
        "uy_x": blocks[:, 4 * ny + nx:4 * ny + 2 * nx],
        "uy2_x": blocks[:, 4 * ny + 2 * nx:4 * ny + 3 * nx],
        "uxuy_x": blocks[:, 4 * ny + 3 * nx:4 * ny + 4 * nx],
    }


def delta_bmbc(values, quantity):
    """Multivariate BMBC plus delta-method propagation at every line point."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 3 or values.shape[0] < 3:
        raise ValueError("Expected [K, points, moments] with K >= 3")
    k = values.shape[0]
    mean = np.mean(values, axis=0)

    if quantity.endswith("rms"):
        resolved_variance = mean[:, 1] - mean[:, 0] ** 2
        derived = np.sqrt(np.maximum(resolved_variance, 0.0))
        gradient = np.full_like(mean, np.nan)
        valid = derived > 0.0
        gradient[valid, 0] = -mean[valid, 0] / derived[valid]
        gradient[valid, 1] = 0.5 / derived[valid]
    elif quantity == "covariance":
        derived = mean[:, 2] - mean[:, 0] * mean[:, 1]
        gradient = np.column_stack((-mean[:, 1], -mean[:, 0], np.ones(mean.shape[0])))
    else:
        raise ValueError(f"Unknown derived quantity: {quantity}")

    centered = values - mean[None, :, :]
    s0 = np.einsum("kni,knj->nij", centered, centered)
    forward = np.einsum("kni,knj->nij", values[:-1], values[1:])
    forward -= (k - 1) * np.einsum("ni,nj->nij", mean, mean)
    s1_symmetric = forward + np.swapaxes(forward, 1, 2)
    covariance_of_mean = (s0 + s1_symmetric) / ((k - 1) * (k - 2))

    variance = np.einsum("ni,nij,nj->n", gradient, covariance_of_mean, gradient)
    s0_projected = np.einsum("ni,nij,nj->n", gradient, s0, gradient)
    s1_projected = 0.5 * np.einsum(
        "ni,nij,nj->n", gradient, s1_symmetric, gradient
    )
    ratio = np.divide(
        s1_projected, s0_projected,
        out=np.full_like(s1_projected, np.nan),
        where=s0_projected > 0.0,
    )
    standard_error = np.sqrt(np.maximum(variance, 0.0))
    standard_error[~np.isfinite(variance) | (variance < 0.0)] = np.nan
    return derived, standard_error, ratio, variance


def analyze_snapshot(snapshot, u_lid, output):
    metadata, _, blocks = read_snapshot(snapshot)
    history = unpack_block_history(metadata, blocks)
    definitions = {
        "ux_rms_y": ("rms", ["ux_y", "ux2_y"], metadata["ny"], 1),
        "uxuy_y": ("covariance", ["ux_y", "uy_y", "uxuy_y"], metadata["ny"], 2),
        "uy_rms_x": ("rms", ["uy_x", "uy2_x"], metadata["nx"], 1),
        "uxuy_x": ("covariance", ["ux_x", "uy_x", "uxuy_x"], metadata["nx"], 2),
    }
    results = {}
    group_size = 1
    while blocks.shape[0] // group_size >= 16:
        k = blocks.shape[0] // group_size
        for quantity, (kind, fields, points, power) in definitions.items():
            moment_values = []
            for field in fields:
                raw = history[field][:k * group_size]
                grouped = raw.reshape(k, group_size, points).mean(axis=1)
                moment_values.append(grouped)
            values = np.stack(moment_values, axis=2)
            derived, standard_error, ratio, variance = delta_bmbc(values, kind)
            normalization = u_lid ** power
            results[(quantity, metadata["block_samples"] * group_size)] = {
                "K": k,
                "position": np.linspace(-1.0, 1.0, points),
                "value": derived / normalization,
                "standard_error": standard_error / normalization,
                "S1_over_S0": ratio,
                "variance": variance / (normalization * normalization),
            }
        group_size *= 2
    if not results:
        raise ValueError("At least 16 completed BMBC blocks are required")

    output.mkdir(parents=True, exist_ok=True)
    detail_path = output / "turbulence_bmbc_profiles.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow((
            "quantity", "M", "K", "index", "position_star", "value_normalized",
            "standard_error_normalized", "S1_over_S0", "variance_normalized",
        ))
        for (quantity, m), values in sorted(results.items()):
            for index in range(values["position"].size):
                writer.writerow((
                    quantity, m, values["K"], index, values["position"][index],
                    values["value"][index], values["standard_error"][index],
                    values["S1_over_S0"][index], values["variance"][index],
                ))

    summary_path = output / "turbulence_bmbc_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow((
            "quantity", "selected_M", "selected_K", "valid_fraction",
            "median_S1_over_S0", "p95_95pct_halfwidth_normalized",
        ))
        for quantity in definitions:
            candidates = []
            for (current, m), values in results.items():
                if current != quantity:
                    continue
                ratio = values["S1_over_S0"]
                valid = ratio[np.isfinite(ratio)]
                if valid.size:
                    candidates.append((float(np.median(np.abs(valid - 0.5))), m))
            if not candidates:
                continue
            selected_m = min(candidates)[1]
            values = results[(quantity, selected_m)]
            halfwidth = 1.96 * values["standard_error"]
            writer.writerow((
                quantity, selected_m, values["K"],
                float(np.isfinite(halfwidth).mean()),
                percentile(values["S1_over_S0"], 50),
                percentile(halfwidth, 95),
            ))
    return detail_path, summary_path


def load_reference(directory, stem, reynolds, aspect_ratio, plot_amplification):
    path = directory / (
        f"{stem}_{int(round(reynolds))}_{aspect_ratio_suffix(aspect_ratio)}.csv"
    )
    data = np.genfromtxt(path, delimiter=",", names=True)
    first = np.atleast_1d(data["x"]).astype(float)
    second = np.atleast_1d(data["y"]).astype(float)
    if stem.endswith("cy"):
        position, value = second, first
    else:
        position, value = first, second
    order = np.argsort(position)
    return path, position[order], value[order] / plot_amplification


def write_outputs(derived, metadata, output, reference_dir):
    output.mkdir(parents=True, exist_ok=True)
    values_path = output / "turbulence_profiles.csv"
    with values_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("quantity", "M", "K", "index", "position_star", "value_normalized"))
        for quantity, values in derived.items():
            for index, (position, value) in enumerate(zip(values["position"], values["value"])):
                writer.writerow((quantity, metadata["M"], metadata["K"], index, position, value))

    comparison_path = output / "experimental_comparison_summary.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow((
            "quantity", "reference_file", "experimental_plot_amplification",
            "relative_L2", "RMSE", "max_abs_error",
        ))
        for quantity, (stem, ylabel, xlabel, amplification) in DERIVED.items():
            values = derived[quantity]
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(values["position"], values["value"], label="GLBM")
            try:
                reference_path, position, reference = load_reference(
                    reference_dir, stem, metadata["reynolds_number"],
                    metadata["aspect_ratio"], amplification,
                )
            except (FileNotFoundError, ValueError) as error:
                print(f"Skipping experimental comparison for {quantity}: {error}")
            else:
                numerical = np.interp(position, values["position"], values["value"])
                difference = numerical - reference
                writer.writerow((
                    quantity, reference_path, amplification,
                    relative_l2(position, difference, reference),
                    float(np.sqrt(np.mean(difference * difference))),
                    float(np.max(np.abs(difference))),
                ))
                ax.scatter(position, reference, facecolors="none", edgecolors="black",
                           label="Prasad-Koseff")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output / f"{quantity}.png", dpi=180)
            plt.close(fig)
    return values_path, comparison_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--reference-dir", type=Path,
                        default=Path(__file__).with_name("Prasad_Koseff"))
    parser.add_argument("--snapshot", type=Path, default=None,
                        help="BMBC v2 sidecar for multivariate derived uncertainty")
    args = parser.parse_args()
    raw, metadata = load_raw_profiles(args.csv)
    derived = derive_profiles(raw, metadata["u_max"])
    output = args.out or args.csv.with_name(args.csv.stem + "_turbulence_analysis")
    values, comparison = write_outputs(derived, metadata, output, args.reference_dir)
    print(f"Wrote: {values}")
    print(f"Wrote: {comparison}")
    if args.snapshot is not None:
        detail, summary = analyze_snapshot(args.snapshot, metadata["u_max"], output)
        print(f"Wrote: {detail}")
        print(f"Wrote: {summary}")
    else:
        print("Pass --snapshot to compute multivariate BMBC uncertainty.")


if __name__ == "__main__":
    main()

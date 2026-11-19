#!/usr/bin/env python3
"""Analyze BMBC centerline profiles and optionally compare with experiment.

The BMBC diagnostic estimates sampling uncertainty. The experimental L2
diagnostic measures validation error; neither one substitutes for the other.

Examples:
    python post/analyze_bmbc_profiles.py output/.../bmbc_profiles_step_XXXX.csv
    python post/analyze_bmbc_profiles.py output/.../bmbc_profiles_step_XXXX.csv \
        --reynolds 3200 --aspect-ratio 1
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROFILES = {
    "ux_y": (r"$\overline{u_x}/U_{lid}$", r"$y^*$"),
    "uy_x": (r"$\overline{u_y}/U_{lid}$", r"$x^*$"),
}
FIELDS = (
    "M", "K", "index", "position_star", "mean_over_U", "S1_over_S0",
    "variance", "standard_error_over_U",
)


def load_profiles(path: Path):
    groups = {}
    metadata = {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = set(FIELDS) | {"profile"}
        columns = set(reader.fieldnames or [])
        if not required.issubset(columns):
            raise ValueError(f"Missing CSV columns: {sorted(required - columns)}")
        for row in reader:
            profile = row["profile"]
            if profile not in PROFILES:
                raise ValueError(f"Unknown profile: {profile}")
            m = int(row["M"])
            groups.setdefault((profile, m), []).append({
                key: int(row[key]) if key in ("M", "K", "index")
                else float(row[key]) for key in FIELDS
            })
            for key in ("reynolds_number", "aspect_ratio", "u_max", "sample_interval"):
                if key in row:
                    value = float(row[key])
                    if key in metadata and metadata[key] != value:
                        raise ValueError(f"Inconsistent {key} in BMBC CSV")
                    metadata[key] = value
    if not groups:
        raise ValueError("CSV is empty")

    series = {}
    for (profile, m), records in groups.items():
        records.sort(key=lambda record: record["index"])
        indices = [record["index"] for record in records]
        if indices != list(range(len(records))):
            raise ValueError(f"Incomplete/duplicate indices in {profile}, M={m}")
        if len({record["K"] for record in records}) != 1:
            raise ValueError(f"Inconsistent block count in {profile}, M={m}")
        series[(profile, m)] = {
            field: np.asarray([record[field] for record in records], dtype=float)
            for field in FIELDS
        }
    return series, metadata


def percentile(values, p):
    valid = np.asarray(values)[np.isfinite(values)]
    return float(np.percentile(valid, p)) if valid.size else float("nan")


def variance_change(previous, current):
    """Pointwise relative change between two variance estimates."""
    a, b = previous["variance"], current["variance"]
    if a.size != b.size:
        raise ValueError("M groupings have different spatial sizes")
    result = np.full(b.shape, np.nan, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (a >= 0) & (b >= 0)
    result[mask] = np.abs(b[mask] - a[mask]) / np.maximum(
        np.maximum(a[mask], b[mask]), 1e-30
    )
    return result


def select_profile_batch_size(series, profile, target=0.5):
    """Choose M whose spatial median is closest to the BMBC target."""
    candidates = []
    for current_profile, m in series:
        if current_profile != profile:
            continue
        ratio = series[(profile, m)]["S1_over_S0"]
        valid = ratio[np.isfinite(ratio)]
        if valid.size:
            candidates.append((float(np.median(np.abs(valid - target))), m))
    if not candidates:
        raise ValueError(f"No finite S1/S0 values for {profile}")
    return min(candidates)[1]


def plot_bmbc(series, out_dir):
    for profile, (velocity_label, position_label) in PROFILES.items():
        ms = sorted(m for current, m in series if current == profile)
        if not ms:
            continue
        for name in ("mean", "correlation", "standard_error"):
            fig, ax = plt.subplots(figsize=(8, 5))
            for m in ms:
                values = series[(profile, m)]
                x = values["position_star"]
                if name == "mean":
                    y, ylabel = values["mean_over_U"], velocity_label
                elif name == "correlation":
                    y, ylabel = values["S1_over_S0"], r"$S_1/S_0$"
                else:
                    y = 1.96 * values["standard_error_over_U"]
                    ylabel = r"$1.96\,\widehat{SE}/U_{lid}$"
                ax.plot(x, y, label=f"M={m}, K={int(values['K'][0])}")
            if name == "correlation":
                ax.axhline(0.5, color="black", linestyle="--", linewidth=1,
                           label="BMBC target 0.5")
            ax.set_xlabel(position_label)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / f"{profile}_{name}.png", dpi=180)
            plt.close(fig)


def write_bmbc_summary(series, out_dir, halfwidth_tol):
    summary_path = out_dir / "bmbc_summary.csv"
    columns = (
        "profile", "selected_M", "selected_K", "points", "samples_used",
        "valid_se_fraction", "median_S1_over_S0", "p95_abs_S1_over_S0_minus_0.5",
        "p95_95pct_halfwidth_over_U", "max_95pct_halfwidth_over_U",
        "fraction_below_halfwidth_tolerance", "halfwidth_tolerance_over_U",
    )
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for profile in PROFILES:
            if not any(current == profile for current, _ in series):
                continue
            m = select_profile_batch_size(series, profile)
            values = series[(profile, m)]
            ratio = values["S1_over_S0"]
            halfwidth = 1.96 * values["standard_error_over_U"]
            valid_halfwidth = halfwidth[np.isfinite(halfwidth)]
            fraction_below = (
                float(np.mean(valid_halfwidth <= halfwidth_tol))
                if halfwidth_tol is not None and valid_halfwidth.size else float("nan")
            )
            record = {
                "profile": profile,
                "selected_M": m,
                "selected_K": int(values["K"][0]),
                "points": len(halfwidth),
                "samples_used": m * int(values["K"][0]),
                "valid_se_fraction": float(np.isfinite(halfwidth).mean()),
                "median_S1_over_S0": percentile(ratio, 50),
                "p95_abs_S1_over_S0_minus_0.5": percentile(np.abs(ratio - 0.5), 95),
                "p95_95pct_halfwidth_over_U": percentile(halfwidth, 95),
                "max_95pct_halfwidth_over_U": percentile(halfwidth, 100),
                "fraction_below_halfwidth_tolerance": fraction_below,
                "halfwidth_tolerance_over_U": (
                    halfwidth_tol if halfwidth_tol is not None else float("nan")
                ),
            }
            writer.writerow(record)
            tolerance_text = (
                f" | fraction <= {halfwidth_tol:g} U: {fraction_below:.1%}"
                if halfwidth_tol is not None else ""
            )
            print(
                f"{profile:5s} selected M={m:>7}, K={record['selected_K']:>5} | "
                f"median S1/S0={record['median_S1_over_S0']:.3f} | "
                f"p95 95% halfwidth/U={record['p95_95pct_halfwidth_over_U']:.3e}"
                f"{tolerance_text}"
            )
    return summary_path


def aspect_ratio_suffix(aspect_ratio):
    known = {0.25: "025", 0.5: "05", 0.75: "075", 1.0: "1"}
    for value, suffix in known.items():
        if np.isclose(aspect_ratio, value):
            return suffix
    raise ValueError(f"No Prasad-Koseff filename convention for SAR={aspect_ratio}")


def load_reference(reference_dir, profile, reynolds, aspect_ratio):
    component = {"ux_y": "ux_cy", "uy_x": "uy_cx"}[profile]
    re_text = str(int(round(reynolds)))
    path = reference_dir / f"{component}_{re_text}_{aspect_ratio_suffix(aspect_ratio)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Experimental profile not found: {path}")
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size < 2:
        raise ValueError(f"Experimental profile has too few points: {path}")
    first = np.atleast_1d(data["x"]).astype(float)
    second = np.atleast_1d(data["y"]).astype(float)
    position, velocity = (second, first) if profile == "ux_y" else (first, second)
    order = np.argsort(position)
    position, velocity = position[order], velocity[order]
    if np.any(np.diff(position) <= 0):
        raise ValueError(f"Experimental positions are not unique: {path}")
    return path, position, velocity


def relative_l2(position, error, reference):
    numerator = np.trapezoid(error * error, position)
    denominator = np.trapezoid(reference * reference, position)
    return float(np.sqrt(numerator / denominator)) if denominator > 0 else float("nan")


def compare_with_experiment(series, out_dir, reference_dir, reynolds, aspect_ratio):
    summary_path = out_dir / "experimental_comparison_summary.csv"
    point_path = out_dir / "experimental_comparison_points.csv"
    summary_columns = (
        "profile", "reference_file", "M", "K", "experimental_points",
        "relative_L2", "RMSE_over_U", "max_abs_error_over_U",
    )
    point_columns = (
        "profile", "position_star", "experimental_over_U", "numerical_over_U",
        "error_over_U",
    )
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file, \
            point_path.open("w", newline="", encoding="utf-8") as point_file:
        summary_writer = csv.DictWriter(summary_file, fieldnames=summary_columns)
        point_writer = csv.DictWriter(point_file, fieldnames=point_columns)
        summary_writer.writeheader()
        point_writer.writeheader()
        for profile, (velocity_label, position_label) in PROFILES.items():
            ms = sorted(m for current, m in series if current == profile)
            if not ms:
                continue
            # Smallest M uses the greatest number of complete basic blocks.
            m = ms[0]
            values = series[(profile, m)]
            reference_path, position, experimental = load_reference(
                reference_dir, profile, reynolds, aspect_ratio
            )
            numerical = np.interp(
                position, values["position_star"], values["mean_over_U"]
            )
            error = numerical - experimental
            record = {
                "profile": profile,
                "reference_file": reference_path.name,
                "M": m,
                "K": int(values["K"][0]),
                "experimental_points": len(position),
                "relative_L2": relative_l2(position, error, experimental),
                "RMSE_over_U": float(np.sqrt(np.mean(error * error))),
                "max_abs_error_over_U": float(np.max(np.abs(error))),
            }
            summary_writer.writerow(record)
            for x, ref, num, difference in zip(position, experimental, numerical, error):
                point_writer.writerow({
                    "profile": profile,
                    "position_star": x,
                    "experimental_over_U": ref,
                    "numerical_over_U": num,
                    "error_over_U": difference,
                })
            print(
                f"{profile:5s} experiment | relative L2={record['relative_L2']:.4e} | "
                f"RMSE/U={record['RMSE_over_U']:.4e} | "
                f"max|e|/U={record['max_abs_error_over_U']:.4e}"
            )

            selected_m = select_profile_batch_size(series, profile)
            selected = series[(profile, selected_m)]
            halfwidth = 1.96 * selected["standard_error_over_U"]
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(values["position_star"], values["mean_over_U"],
                    label=f"GLBM mean (M={m})")
            ax.fill_between(
                selected["position_star"],
                selected["mean_over_U"] - halfwidth,
                selected["mean_over_U"] + halfwidth,
                alpha=0.2, label=f"BMBC 95% band (M={selected_m})",
            )
            ax.scatter(position, experimental, facecolors="none", edgecolors="black",
                       label=f"Prasad-Koseff Re={int(round(reynolds))}")
            ax.set_xlabel(position_label)
            ax.set_ylabel(velocity_label)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / f"{profile}_experiment.png", dpi=180)
            plt.close(fig)
    return summary_path, point_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV created by exportBMBCProfilesCSV")
    parser.add_argument("--out", type=Path, default=None, help="Output directory")
    parser.add_argument("--reference-dir", type=Path,
                        default=Path(__file__).with_name("Prasad_Koseff"))
    parser.add_argument("--reynolds", type=float, default=None,
                        help="Reynolds number; inferred from new BMBC CSVs")
    parser.add_argument("--aspect-ratio", type=float, default=None,
                        help="Spanwise aspect ratio; inferred from new BMBC CSVs")
    parser.add_argument("--halfwidth-tol-over-u", type=float, default=None,
                        help="Optional 95%% half-width tolerance normalized by U_lid")
    args = parser.parse_args()

    output = args.out or args.csv.with_suffix("").with_name(args.csv.stem + "_analysis")
    output.mkdir(parents=True, exist_ok=True)
    series, metadata = load_profiles(args.csv)
    reynolds = args.reynolds if args.reynolds is not None else metadata.get("reynolds_number")
    aspect_ratio = (
        args.aspect_ratio if args.aspect_ratio is not None else metadata.get("aspect_ratio")
    )

    print("BMBC FULL-PROFILE ANALYSIS")
    summary = write_bmbc_summary(series, output, args.halfwidth_tol_over_u)
    plot_bmbc(series, output)
    print(f"\nBMBC summary: {summary}")

    if (reynolds is None) != (aspect_ratio is None):
        parser.error("Reynolds number and aspect ratio must be provided together")
    if reynolds is not None:
        comparison, points = compare_with_experiment(
            series, output, args.reference_dir, reynolds, aspect_ratio
        )
        print(f"Experimental summary: {comparison}\nExperimental points: {points}")
    else:
        print("Experimental comparison skipped (Re/SAR unavailable).")
    print(f"Figures: {output}")
    print("Reminder: BMBC assumes a stationary sampled interval; experimental L2 also "
          "contains discretization, modeling, and measurement differences.")


if __name__ == "__main__":
    main()

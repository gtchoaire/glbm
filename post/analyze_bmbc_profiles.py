"""Inspect full central-line BMBC profiles exported by GLBM.

Usage:
    python post/analyze_bmbc_profiles.py path/to/bmbc_profiles_step_XXXX.csv

Requires numpy and matplotlib. Does not decide statistical convergence: the
underlying signal must be stationary and the estimated variance must be
reasonably insensitive to the grouping size M.
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROFILES = {"ux_y": (r"$\overline{u_x}/U_{lid}$", r"$y^*$"),
            "uy_x": (r"$\overline{u_y}/U_{lid}$", r"$x^*$")}
FIELDS = ("M", "K", "index", "position_star", "mean_over_U",
          "S1_over_S0", "variance", "standard_error_over_U")


def load_profiles(path):
    groups = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = set(FIELDS) | {"profile"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing CSV columns: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            profile = row["profile"]
            if profile not in PROFILES:
                raise ValueError(f"Unknown profile: {profile}")
            m = int(row["M"])
            groups.setdefault((profile, m), []).append({
                key: int(row[key]) if key in ("M", "K", "index")
                else float(row[key]) for key in FIELDS
            })
    if not groups:
        raise ValueError("CSV is empty")
    series = {}
    for (profile, m), records in groups.items():
        records.sort(key=lambda r: r["index"])
        indices = [r["index"] for r in records]
        if indices != list(range(len(records))):
            raise ValueError(f"Incomplete/duplicate indices in {profile}, M={m}")
        if len({r["K"] for r in records}) != 1:
            raise ValueError(f"Inconsistent block count in {profile}, M={m}")
        series[(profile, m)] = {
            field: np.asarray([r[field] for r in records], dtype=float)
            for field in FIELDS
        }
    return series


def percentile(values, p):
    valid = np.asarray(values)[np.isfinite(values)]
    return float(np.percentile(valid, p)) if valid.size else float("nan")


def variance_change(previous, current):
    """Pointwise relative difference in variance estimates (not in their means)."""
    a, b = previous["variance"], current["variance"]
    if a.size != b.size:
        raise ValueError("M groupings have different spatial sizes")
    result = np.full(b.shape, np.nan, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (a >= 0) & (b >= 0)
    result[mask] = np.abs(b[mask] - a[mask]) / np.maximum(
        np.maximum(a[mask], b[mask]), 1e-30
    )
    return result


def plot_all(series, out_dir):
    for profile, (velocity_label, position_label) in PROFILES.items():
        ms = sorted(m for p, m in series if p == profile)
        if not ms:
            continue
        for name in ("mean", "correlation", "standard_error"):
            fig, ax = plt.subplots(figsize=(8, 5))
            for m in ms:
                s = series[(profile, m)]
                x = s["position_star"]
                if name == "mean":
                    y = s["mean_over_U"]
                    ylabel = velocity_label
                elif name == "correlation":
                    y = s["S1_over_S0"]
                    ylabel = r"$S_1/S_0$"
                else:
                    y = 1.96 * s["standard_error_over_U"]
                    ylabel = r"$1.96\,\widehat{SE}/U_{lid}$ (provisório)"
                ax.plot(x, y, label=f"M={m}, K={int(s['K'][0])}")
            ax.set_xlabel(position_label)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / f"{profile}_{name}.png", dpi=180)
            plt.close(fig)


def write_summary(series, out_dir):
    summary_path = out_dir / "bmbc_summary.csv"
    columns = ("profile", "points", "M", "K", "samples_used", "valid_se_fraction",
               "p95_abs_S1_over_S0", "p95_approx_95pct_halfwidth_over_U",
               "max_approx_95pct_halfwidth_over_U", "p95_variance_change_from_previous_M")
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for profile in PROFILES:
            ms = sorted(m for p, m in series if p == profile)
            previous = None
            for m in ms:
                s = series[(profile, m)]
                halfwidth = 1.96 * s["standard_error_over_U"]
                n = len(halfwidth)
                change = (variance_change(previous, s) if previous is not None
                          else np.full(n, np.nan))
                record = {
                    "profile": profile, "points": n, "M": m, "K": int(s["K"][0]),
                    "samples_used": m * int(s["K"][0]),
                    "valid_se_fraction": float(np.isfinite(halfwidth).mean()),
                    "p95_abs_S1_over_S0": percentile(np.abs(s["S1_over_S0"]), 95),
                    "p95_approx_95pct_halfwidth_over_U": percentile(halfwidth, 95),
                    "max_approx_95pct_halfwidth_over_U": percentile(halfwidth, 100),
                    "p95_variance_change_from_previous_M": percentile(change, 95),
                }
                writer.writerow(record)
                print(f"{profile:5s} M={m:>7} K={record['K']:>5} | "
                      f"p95 |S1/S0|={record['p95_abs_S1_over_S0']:.3f} | "
                      f"p95 1.96SE/U={record['p95_approx_95pct_halfwidth_over_U']:.3e} | "
                      f"p95 ΔVar={record['p95_variance_change_from_previous_M']:.3f}")
                previous = s
    return summary_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV created by exportBMBCProfilesCSV")
    parser.add_argument("--out", type=Path, default=None, help="Output directory")
    args = parser.parse_args()
    output = args.out or args.csv.with_suffix("").with_name(args.csv.stem + "_analysis")
    output.mkdir(parents=True, exist_ok=True)
    series = load_profiles(args.csv)
    print("BMBC FULL-PROFILE ANALYSIS (not an automatic convergence decision)")
    summary = write_summary(series, output)
    plot_all(series, output)
    print(f"\nSummary: {summary}\nFigures: {output}")
    print("WARNING: these errors are meaningful only for a stationary series "
          "and sufficiently stable estimates as M changes.")


if __name__ == "__main__":
    main()

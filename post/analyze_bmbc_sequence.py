#!/usr/bin/env python3
"""Summarize BMBC uncertainty and experimental L2 across periodic exports."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_bmbc_profiles import (
    PROFILES,
    load_profiles,
    load_reference,
    percentile,
    relative_l2,
    select_profile_batch_size,
)


STEP_PATTERN = re.compile(r"bmbc_profiles_step_(\d+)\.csv$")


def step_from_path(path: Path) -> int:
    match = STEP_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Unexpected BMBC profile filename: {path}")
    return int(match.group(1))


def analyze_file(path, reference_dir, reynolds=None, aspect_ratio=None):
    series, metadata = load_profiles(path)
    reynolds = reynolds if reynolds is not None else metadata.get("reynolds_number")
    aspect_ratio = (
        aspect_ratio if aspect_ratio is not None else metadata.get("aspect_ratio")
    )
    if reynolds is None or aspect_ratio is None:
        raise ValueError(f"Reynolds number/SAR unavailable in {path}")

    rows = []
    for profile in PROFILES:
        ms = sorted(m for current, m in series if current == profile)
        if not ms:
            continue
        base_m = ms[0]
        base = series[(profile, base_m)]
        selected_m = select_profile_batch_size(series, profile)
        selected = series[(profile, selected_m)]
        halfwidth = 1.96 * selected["standard_error_over_U"]
        _, position, experimental = load_reference(
            reference_dir, profile, reynolds, aspect_ratio
        )
        numerical = np.interp(
            position, base["position_star"], base["mean_over_U"]
        )
        error = numerical - experimental
        rows.append({
            "step": step_from_path(path),
            "profile": profile,
            "complete_samples": base_m * int(base["K"][0]),
            "complete_solver_steps": (
                base_m * int(base["K"][0]) * int(metadata.get("sample_interval", 1))
            ),
            "averaging_t_star": (
                base_m * int(base["K"][0]) * int(metadata.get("sample_interval", 1))
                * metadata.get("u_max", 1.0) / (len(base["position_star"]) - 1)
            ),
            "selected_M": selected_m,
            "selected_M_steps": selected_m * int(metadata.get("sample_interval", 1)),
            "selected_K": int(selected["K"][0]),
            "median_S1_over_S0": percentile(selected["S1_over_S0"], 50),
            "p95_95pct_halfwidth_over_U": percentile(halfwidth, 95),
            "relative_L2": relative_l2(position, error, experimental),
            "RMSE_over_U": float(np.sqrt(np.mean(error * error))),
        })
    return rows


def plot_history(rows, output):
    for metric, ylabel in (
        ("relative_L2", r"relative $L_2$"),
        ("p95_95pct_halfwidth_over_U", r"p95 $1.96\,SE/U_{lid}$"),
        ("median_S1_over_S0", r"median $S_1/S_0$"),
    ):
        fig, ax = plt.subplots(figsize=(7, 5))
        for profile in PROFILES:
            selected = sorted(
                (row for row in rows if row["profile"] == profile),
                key=lambda row: row["averaging_t_star"],
            )
            if selected:
                ax.plot(
                    [row["averaging_t_star"] for row in selected],
                    [row[metric] for row in selected],
                    marker="o", label=profile,
                )
        if metric == "median_S1_over_S0":
            ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel(r"averaging time $t^*$ (complete blocks)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"sequence_{metric}.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path,
                        help="Run directory containing bmbc_profiles_step_*.csv")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--reference-dir", type=Path,
                        default=Path(__file__).with_name("Prasad_Koseff"))
    parser.add_argument("--reynolds", type=float, default=None)
    parser.add_argument("--aspect-ratio", type=float, default=None)
    args = parser.parse_args()

    files = sorted(
        args.directory.glob("bmbc_profiles_step_*.csv"), key=step_from_path
    )
    if not files:
        parser.error(f"No BMBC profile CSVs found in {args.directory}")
    output = args.out or args.directory / "bmbc_sequence_analysis"
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in files:
        rows.extend(analyze_file(
            path, args.reference_dir, args.reynolds, args.aspect_ratio
        ))
    # A restart can export a new step before completing another basic block.
    # Keep only the latest row for an identical profile/sample count.
    unique = {
        (row["profile"], row["complete_samples"]): row
        for row in rows
    }
    rows = sorted(
        unique.values(),
        key=lambda row: (row["averaging_t_star"], row["profile"]),
    )
    columns = tuple(rows[0])
    summary = output / "bmbc_sequence_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    plot_history(rows, output)
    print(f"Analyzed {len(files)} exports")
    print(f"Summary: {summary}")
    print(f"Figures: {output}")


if __name__ == "__main__":
    main()

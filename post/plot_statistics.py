#!/usr/bin/env python3
"""Generate article figures from processed GLBM text data."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt
import numpy as np


DATA_DIRECTORY = Path("statistics/derived/data")
FIGURE_DIRECTORY = Path("statistics/derived/figures")

ERROR_BARS = 21
ERROR_STYLE = {"elinewidth": 0.7, "capsize": 1.8, "capthick": 0.7, "zorder": 3}

BLACK, RED, BLUE, GREEN, ORANGE, PURPLE = (
    "#000000", "#C1272D", "#267BCF", "#2E8B57", "#D98200", "#7B4EA3",
)

mpl.rcParams.update({
    "figure.figsize":       (3.9, 2.6),
    "figure.dpi":           600,
    "savefig.dpi":          600,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.02,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
    "font.family":          "STIXGeneral",
    "mathtext.fontset":     "stix",
    "font.size":            8,
    "axes.labelsize":       9,
    "legend.fontsize":      7.5,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "axes.linewidth":       0.6,
    "lines.linewidth":      1.0,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "xtick.top":            True,
    "ytick.right":          True,
    "xtick.major.width":    0.6,
    "ytick.major.width":    0.6,
    "xtick.minor.width":    0.45,
    "ytick.minor.width":    0.45,
    "xtick.major.size":     3.2,
    "ytick.major.size":     3.2,
    "xtick.minor.size":     1.8,
    "ytick.minor.size":     1.8,
    "xtick.minor.visible":  True,
    "ytick.minor.visible":  True,
    "axes.formatter.limits": (-2, 4),
    "axes.formatter.use_mathtext": True,
    "legend.frameon":       False,
    "legend.handlelength":  1.7,
    "legend.handletextpad": 0.5,
    "legend.labelspacing":  0.35,
    "legend.borderaxespad": 0.6,
    "axes.prop_cycle":      mpl.cycler(
        color=(BLACK, RED, BLUE, GREEN, ORANGE, PURPLE),
        linestyle=("-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1)), (0, (3, 1))),
    ),
})

LABELS = {
    "ux": r"$\overline{u}_x/U$",
    "uy": r"$\overline{u}_y/U$",
    "ux_rms": r"$u_{x,\mathrm{rms}}/U$",
    "uy_rms": r"$u_{y,\mathrm{rms}}/U$",
    "r_xx": r"$\overline{u_x^{\prime}u_x^{\prime}}/U^2$",
    "r_yy": r"$\overline{u_y^{\prime}u_y^{\prime}}/U^2$",
    "r_zz": r"$\overline{u_z^{\prime}u_z^{\prime}}/U^2$",
    "r_xy": r"$\overline{u_x^{\prime}u_y^{\prime}}/U^2$",
    "r_xz": r"$\overline{u_x^{\prime}u_z^{\prime}}/U^2$",
    "r_yz": r"$\overline{u_y^{\prime}u_z^{\prime}}/U^2$",
}

POSITIONS = {"y_center": r"$2y/L-1$", "x_center": r"$2x/L-1$"}
LINES = {"y_center": r"$x=L/2,\ z=L_z/2$", "x_center": r"$y=L/2,\ z=L_z/2$"}

PROFILES = (
    ("reynolds_stress", r"$\overline{u_i^{\prime}u_j^{\prime}}/U^2$",
     ("r_xx", "r_yy", "r_zz", "r_xy", "r_xz", "r_yz")),
)


EXPERIMENTS = (
    ("velocity", r"$2x/L-1,\ \overline{u}_x/U$", r"$2y/L-1,\ \overline{u}_y/U$", (
        ("y_center", "ux", 1, BLUE, "x", LABELS["ux"]),
        ("x_center", "uy", 1, RED, "x", LABELS["uy"]),
    )),
    ("fluctuations", r"$2x/L-1,\ 10\,u_{x,\mathrm{rms}}/U,\ 500\,\overline{u_x^{\prime}u_y^{\prime}}/U^2$",
     r"$2y/L-1,\ 10\,u_{y,\mathrm{rms}}/U,\ 500\,\overline{u_x^{\prime}u_y^{\prime}}/U^2$", (
        ("y_center", "ux_rms", 10, BLUE, "x", LABELS["ux_rms"]),
        ("x_center", "uy_rms", 10, RED, "x", LABELS["uy_rms"]),
        ("y_center", "r_xy", 500, ORANGE, "+", LABELS["r_xy"]),
        ("x_center", "r_xy", 500, ORANGE, "+", None),
    )),
)


def read_dat(name: str):
    path = DATA_DIRECTORY / name
    if not path.exists():
        return None
    return np.atleast_1d(np.genfromtxt(path, names=True, dtype=None, encoding="utf-8"))


def save(figure, axis, name: str, xlabel: str, ylabel: str, **legend) -> None:
    axis.set(xlabel=xlabel, ylabel=ylabel)
    if axis.get_legend_handles_labels()[0]:
        axis.legend(**legend)
    path = FIGURE_DIRECTORY / f"{name}.pdf"
    figure.savefig(path)
    plt.close(figure)
    print(f"Wrote: {path}")


def plot_errorbars(
    axis, data, label: str, error_label: str | None = None, offset: float = 0.5,
    scale: float = 1, swap: bool = False, positions=None, **style,
) -> None:
    order = -1 if swap else 1
    curve, = axis.plot(*(data["position_star"], scale * data["value"])[::order], label=label, **style)
    if positions is None:
        angles = np.pi * (np.arange(ERROR_BARS) + offset) / ERROR_BARS
        positions = data["position_star"][np.rint((1 - np.cos(angles)) / 2 * (data.size - 1)).astype(int)]
    value, lower, upper = (
        scale * np.interp(positions, data["position_star"], data[name]) for name in ("value", "lower_95", "upper_95")
    )
    error = np.maximum(0, (value - lower, upper - value))
    axis.errorbar(
        *(positions, value)[::order], **{"xerr" if swap else "yerr": error},
        color=curve.get_color(), linestyle="none", label=error_label, **ERROR_STYLE,
    )


def select(data, line: str, quantity: str):
    return data[(data["line"] == line) & (data["quantity"] == quantity)]


def plot_global_statistics() -> None:
    data = read_dat("global_statistics.dat")
    if data is None:
        return
    window = read_dat("sampling.dat")
    summary = read_dat("global_summary.dat")
    for field, label in (
        ("kinetic_energy_over_U2", r"$\langle E\rangle_V/U^2$"),
        ("dissipation_star", r"$\langle 2\nu S_{ij}S_{ij}\rangle_V\,L/U^3$"),
    ):
        figure, axis = plt.subplots()
        axis.plot(data["t_star"], data[field], label="instantaneous")
        axis.set_xlim(data["t_star"][0], data["t_star"][-1])
        if window is not None and summary is not None:
            row = summary[summary["quantity"] == field][0]
            start, end = window["start_t_star"][0], window["end_t_star"][0]
            axis.plot([start, end], [row["mean"]] * 2, label="ensemble average")
            axis.errorbar(
                (start + end) / 2, row["mean"], yerr=row["half_width_95"],
                color=RED, linestyle="none", label="95% CI (BMBC)", **ERROR_STYLE,
            )
        save(figure, axis, f"global_{field}", r"$tU/L$", label, loc="best")


def plot_probe() -> None:
    data = read_dat("global_statistics.dat")
    spectrum = read_dat("probe_spectrum.dat")
    if spectrum is None:
        return
    figure, axis = plt.subplots()
    axis.plot(data["t_star"], data["probe_ux"], label=r"$u_x/U$", linewidth=0.6)
    axis.plot(data["t_star"], 4 * data["probe_uy"], label=r"$4u_y/U$", linewidth=0.6, linestyle="-")
    axis.set_xlim(data["t_star"][0], data["t_star"][-1])
    save(figure, axis, "probe_velocity", r"$tU/L$", r"$u_i/U$", loc="best", ncols=2)

    figure, axis = plt.subplots()
    for name in ("x", "y", "z"):
        axis.loglog(spectrum["frequency_star"], spectrum[f"u{name}"], label=rf"$u_{name}$")
    save(figure, axis, "probe_spectrum", r"$fL/U$", r"$E_{ii}(f)/(UL)$", loc="best")


def plot_profiles() -> None:
    data = read_dat("line_statistics.dat")
    if data is None:
        return
    for line in np.unique(data["line"]):
        for name, ylabel, quantities in PROFILES:
            figure, axis = plt.subplots()
            for number, quantity in enumerate(quantities):
                plot_errorbars(
                    axis, select(data, line, quantity), LABELS[quantity],
                    offset=(number + 0.5) / len(quantities),
                )
            axis.set_xlim(-1, 1)
            save(figure, axis, f"profile_{line}_{name}", POSITIONS[line], ylabel, loc="best", ncols=2)

    experiment = read_dat("experiment.dat")
    for name, xlabel, ylabel, curves in EXPERIMENTS:
        figure, axis = plt.subplots(figsize=(3.9, 3.9))
        axis.axhline(0, color=BLACK, linewidth=0.4)
        axis.axvline(0, color=BLACK, linewidth=0.4)
        for number, (line, quantity, scale, color, marker, label) in enumerate(curves):
            swap = line == "y_center"
            reference = None if experiment is None else select(experiment, line, quantity)
            plot_errorbars(
                axis, select(data, line, quantity), label,
                "95% CI (BMBC)" if number == 0 else None, scale=scale, swap=swap,
                positions=None if reference is None else reference["position_star"],
                color=color, linestyle="-",
            )
            if reference is None:
                continue
            points = (reference["position_star"], scale * reference["experiment"])
            axis.plot(
                *points[::-1 if swap else 1], color=BLACK, linestyle="none", marker=marker,
                markersize=3.2, markeredgewidth=0.6, zorder=4,
                label="Prasad & Koseff (1989)" if number == len(curves) - 1 else None,
            )
        axis.set(xlim=(-1, 1), ylim=(-1, 1))
        axis.set_aspect("equal")
        save(figure, axis, f"experiment_{name}", xlabel, ylabel, loc="upper left")


def plot_wall_shear() -> None:
    data = read_dat("wall_shear.dat")
    if data is None:
        return
    sizes = read_dat("vortex_sizes.dat")
    sizes = sizes[sizes["method"] == "moments"][0]
    limits = {
        "upstream": {"usv": sizes["usv_height"]}, "downstream": {"dsv": sizes["dsv_height"]},
        "bottom": {"usv": sizes["usv_length"], "dsv": 1 - sizes["dsv_length"]},
    }
    for wall in np.unique(data["wall"]):
        values = data[data["wall"] == wall]
        vertical = wall != "bottom"
        figure, axis = plt.subplots(figsize=(2.6, 3.4) if vertical else (3.9, 2.6))
        for name, label in (("moments", "moments"), ("fd2", "FD2"), ("fd4", "FD4")):
            pair = (values["position"], values[name])
            axis.plot(*(pair[::-1] if vertical else pair), label=label)
        for name, limit in limits[wall].items():
            if np.isfinite(limit):
                (axis.axhline if vertical else axis.axvline)(
                    limit, color={"usv": BLUE, "dsv": RED}[name], linestyle=":", linewidth=1.2,
                    label=f"{name.upper()} limit",
                )
        if vertical:
            axis.set(xlim=(-0.05, 0.05), ylim=(0, 1))
            save(figure, axis, f"wall_shear_{wall}", r"$\overline{S}_{xy}$", r"$y/L$", loc="center left")
        else:
            axis.set(xlim=(0, 1), ylim=(-0.01, 0.01))
            save(figure, axis, f"wall_shear_{wall}", r"$x/L$", r"$\overline{S}_{xy}$", loc="best")


def plot_planes() -> None:
    for path in sorted(DATA_DIRECTORY.glob("plane_*.npz")):
        data = np.load(path)
        normal = path.stem.split("_")[1]
        first, second = {"x": "zy", "y": "zx", "z": "xy"}[normal]
        horizontal, vertical = data[first], data[second]
        order = [name for name in "zyx" if name != normal]
        oriented = {
            name: data[name] if order == [second, first] else data[name].T
            for name in data.files if data[name].ndim == 2
        }
        speed = np.sqrt(sum(oriented[f"u{name}"] ** 2 for name in "xyz"))
        vorticity = oriented[f"w{normal}"]
        limit = np.nanpercentile(np.abs(vorticity), 99)
        width = 3.1
        size = (width, width * (vertical[-1] - vertical[0]) / (horizontal[-1] - horizontal[0]) * 0.85)
        for name, values, label, options in (
            ("velocity", speed, r"$|\overline{\mathbf{u}}|/U$", {"cmap": "viridis"}),
            ("vorticity", vorticity, rf"$\overline{{\omega}}_{normal}\,L/U$",
             {"cmap": "RdBu_r", "vmin": -limit, "vmax": limit}),
        ):
            figure, axis = plt.subplots(figsize=size)
            mesh = axis.pcolormesh(horizontal, vertical, values, shading="auto", rasterized=True, **options)
            figure.colorbar(mesh, ax=axis, label=label, extend="both" if name == "vorticity" else "neither")
            if name == "velocity":
                axis.streamplot(
                    np.linspace(horizontal[0], horizontal[-1], horizontal.size),
                    np.linspace(vertical[0], vertical[-1], vertical.size),
                    oriented[f"u{first}"], oriented[f"u{second}"],
                    color="white", linewidth=0.4, arrowsize=0.4,
                )
            axis.set_aspect("equal")
            axis.set(xlim=(horizontal[0], horizontal[-1]), ylim=(vertical[0], vertical[-1]))
            save(figure, axis, f"{path.stem}_{name}", rf"${first}/L$", rf"${second}/L$")


def plot_kolmogorov() -> None:
    path = DATA_DIRECTORY / "kolmogorov_minimum.npz"
    if not path.exists():
        return
    data = np.load(path)
    figure, axis = plt.subplots(figsize=(3.1, 2.635))
    mesh = axis.pcolormesh(
        data["x"], data["y"], data["eta_over_dx"], shading="auto", rasterized=True,
        cmap=mpl.colormaps["viridis"].with_extremes(under="red"),
        vmin=1, vmax=max(2, np.nanpercentile(data["eta_over_dx"], 99)),
    )
    figure.colorbar(mesh, ax=axis, label=r"$\min_z(\eta_K)/\Delta x$", extend="both")
    axis.set_aspect("equal")
    axis.set(xlim=(0, 1), ylim=(0, 1))
    save(figure, axis, "kolmogorov_minimum", r"$x/L$", r"$y/L$")


def plot_lumley() -> None:
    data = read_dat("lumley.dat")
    if data is None:
        return
    figure, axis = plt.subplots(figsize=(3.9, 2.9))
    xi = np.linspace(-1 / 6, 1 / 3, 400)
    axis.plot(xi, np.sqrt(1 / 27 + 2 * xi ** 3), color=BLACK, linestyle="-", linewidth=0.8)
    axis.plot(xi, np.abs(xi), color=BLACK, linestyle="-", linewidth=0.8)
    for line, marker in zip(np.unique(data["line"]), ("o", "^")):
        values = data[data["line"] == line]
        points = axis.scatter(
            values["xi"], values["eta"], c=values["position_star"], cmap="viridis",
            s=6, marker=marker, vmin=-1, vmax=1, linewidths=0,
        )
        axis.scatter([], [], color=BLACK, s=8, marker=marker, linewidths=0, label=LINES[line])
    for text, position in (
        ("two-component", (-0.06, 0.205)), ("axisymmetric", (0.16, 0.12)), ("isotropic", (0.015, 0.003)),
    ):
        axis.text(*position, text, fontsize=7)
    figure.colorbar(points, ax=axis, label=r"$2s/L-1$")
    axis.set(xlim=(-0.2, 0.36), ylim=(-0.01, 0.36))
    save(figure, axis, "lumley_triangle", r"$\xi$", r"$\eta$", loc="upper left")


def main() -> None:
    if not DATA_DIRECTORY.is_dir():
        raise FileNotFoundError("Run process_statistics.py before plotting")
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.png", "*.pdf"):
        for path in FIGURE_DIRECTORY.glob(pattern):
            path.unlink()
    plot_global_statistics()
    plot_probe()
    plot_profiles()
    plot_wall_shear()
    plot_planes()
    plot_kolmogorov()
    plot_lumley()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert raw GLBM statistics into article-ready text data."""

from __future__ import annotations

from pathlib import Path

import numpy as np


STATISTICS_DIRECTORY = Path("statistics")
DATA_DIRECTORY = STATISTICS_DIRECTORY / "derived" / "data"
REFERENCE_DIRECTORY = Path(__file__).resolve().with_name("Prasad_Koseff")

LOCAL_FIELD_NAMES = (
    "ux_mean", "uy_mean", "uz_mean",
    "Sxx_mean", "Sxy_mean", "Sxz_mean", "Syy_mean", "Syz_mean", "Szz_mean",
    "ux2_mean", "uy2_mean", "uz2_mean", "uxuy_mean", "uxuz_mean", "uyuz_mean",
    "S2_mean",
)
FIELD = {name.removesuffix("_mean"): index for index, name in enumerate(LOCAL_FIELD_NAMES)}

REYNOLDS_COMPONENTS = (
    ("r_xx", "ux2", "ux", "ux"),
    ("r_yy", "uy2", "uy", "uy"),
    ("r_zz", "uz2", "uz", "uz"),
    ("r_xy", "uxuy", "ux", "uy"),
    ("r_xz", "uxuz", "ux", "uz"),
    ("r_yz", "uyuz", "uy", "uz"),
)
STRAIN_COMPONENTS = (
    ("Sxx", 1), ("Syy", 1), ("Szz", 1), ("Sxy", 2), ("Sxz", 2), ("Syz", 2),
)

LINES = (("y_center", "y"), ("x_center", "x"))
PLANES = (("x", 0.5), ("y", 0.5), ("z", 0.5))

EXPERIMENTS = (
    ("ux", "y_center", "ux_cy", 1.0),
    ("uy", "x_center", "uy_cx", 1.0),
    ("ux_rms", "y_center", "uxRMS_cy", 10.0),
    ("r_xy", "y_center", "uxuy_cy", 500.0),
    ("uy_rms", "x_center", "uyRMS_cx", 10.0),
    ("r_xy", "x_center", "uxuy_cx", 500.0),
)

CORRELATION_LIMIT = 0.5
BLOCK_MULTIPLE = 60
SPECTRUM_SEGMENTS = 7


def format_value(value) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.17g}"
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    return str(value)


def write_dat(name: str, columns, rows) -> None:
    path = DATA_DIRECTORY / name
    with path.open("w", encoding="utf-8", newline="") as file:
        file.write(" ".join(columns) + "\n")
        for row in rows:
            file.write(" ".join(format_value(value) for value in row) + "\n")
    print(f"Wrote: {path}")


def read_text_header(path: Path) -> tuple[dict[str, str], int]:
    header = {}
    with path.open("rb") as file:
        while True:
            line = file.readline()
            if not line:
                raise ValueError(f"Incomplete header: {path}")
            text = line.decode("ascii").strip()
            if text == "data":
                return header, file.tell()
            key, separator, value = text.partition(" ")
            if not separator:
                raise ValueError(f"Malformed header line: {text!r}")
            header[key] = value


def read_local_fields(path: Path):
    header, offset = read_text_header(path)
    if header.get("GLBM_LOCAL_FIELD_STATISTICS") != "2":
        raise ValueError("Unsupported local field statistics format")
    if header.get("precision") != "float64" or header.get("layout") != "field_major_zyx":
        raise ValueError("Unsupported local field storage")
    if tuple(header["field_names"].split()) != LOCAL_FIELD_NAMES:
        raise ValueError("Unexpected local field layout")
    nx, ny, nz = map(int, header["dimensions"].split())
    expected_size = offset + nx * ny * nz * len(LOCAL_FIELD_NAMES) * 8
    if path.stat().st_size != expected_size:
        raise ValueError("Unexpected local field file size")
    fields = np.memmap(
        path, mode="r", dtype="<f8", offset=offset,
        shape=(len(LOCAL_FIELD_NAMES), nz, ny, nx), order="C",
    )
    return header, fields, (nx, ny, nz)


def line_sizes(dimensions) -> list[int]:
    return [dict(zip("xyz", dimensions))[axis] for _, axis in LINES]


def read_bmbc_blocks(path: Path):
    header, offset = read_text_header(path)
    if header.get("GLBM_BMBC") != "4":
        raise ValueError("Unsupported BMBC format")
    integers = (
        "nx", "ny", "nz", "step", "start_step", "sample_interval", "block_samples",
        "completed_blocks", "samples_in_block", "total_samples", "fields", "profile_points",
    )
    metadata = {name: int(header[name]) for name in integers}
    metadata |= {name: float(header[name]) for name in ("reynolds", "u_max", "aspect_ratio")}
    sizes = line_sizes((metadata["nx"], metadata["ny"], metadata["nz"]))
    points = len(LOCAL_FIELD_NAMES) * sum(sizes)
    if metadata["fields"] != len(LOCAL_FIELD_NAMES) or metadata["profile_points"] != points:
        raise ValueError("Unexpected BMBC layout")
    data = np.fromfile(path, dtype="<f8", offset=offset)
    if data.size != points * (metadata["completed_blocks"] + 1):
        raise ValueError("Invalid BMBC payload")
    samples = metadata["completed_blocks"] * metadata["block_samples"]
    if samples + metadata["samples_in_block"] != metadata["total_samples"]:
        raise ValueError("Inconsistent BMBC counters")
    blocks = data[points:].reshape(-1, points)
    lines = {}
    start = 0
    for (name, _), size in zip(LINES, sizes):
        stop = start + len(LOCAL_FIELD_NAMES) * size
        lines[name] = blocks[:, start:stop].reshape(-1, len(LOCAL_FIELD_NAMES), size)
        start = stop
    return metadata, lines


def batch_statistics(series):
    series = np.asarray(series, dtype=np.float64)
    count = series.shape[0]
    centered = series - series.mean(axis=0)
    s0 = np.sum(centered * centered, axis=0)
    s1 = np.sum(centered[:-1] * centered[1:], axis=0)
    variance = (s0 + 2 * s1) / ((count - 1) * (count - 2))
    ratio = np.divide(s1, s0, out=np.full_like(s1, np.nan), where=s0 > 0)
    return variance, ratio


def usable_blocks(count: int) -> int:
    return count - count % BLOCK_MULTIPLE if count >= BLOCK_MULTIPLE else count


def batch_uncertainty(series):
    total, points = series.shape
    group = np.zeros(points, dtype=int)
    error, ratio = np.full(points, np.nan), np.full(points, np.nan)
    pending = np.arange(points)
    for size in range(1, total // 3 + 1):
        if total % size or not pending.size:
            continue
        grouped = series[:, pending].reshape(total // size, size, -1).mean(axis=1)
        variance, correlation = batch_statistics(grouped)
        accepted = ~(correlation > CORRELATION_LIMIT)
        selected = pending[accepted]
        group[selected], ratio[selected] = size, correlation[accepted]
        positive = variance[accepted] >= 0
        error[selected[positive]] = np.sqrt(variance[accepted][positive])
        pending = pending[~accepted]
    return group, error, ratio


def fluctuation_product(fields, mean, second, first, other):
    a, b = mean[FIELD[first]], mean[FIELD[other]]
    return (
        fields[..., FIELD[second], :] - a * fields[..., FIELD[other], :]
        - b * fields[..., FIELD[first], :] + a * b
    )


def turbulent_dissipation(fields, mean, viscosity):
    value = fields[..., FIELD["S2"], :]
    for name, weight in STRAIN_COMPONENTS:
        average = mean[FIELD[name]]
        value = value + weight * average * (average - 2 * fields[..., FIELD[name], :])
    return 2 * viscosity * value


def line_series(blocks, mean, viscosity, u_lid, length):
    series = {name: blocks[:, FIELD[name]] / u_lid for name in ("ux", "uy", "uz")}
    for name, *fields in REYNOLDS_COMPONENTS:
        series[name] = fluctuation_product(blocks, mean, *fields) / u_lid ** 2
    series["dissipation"] = turbulent_dissipation(blocks, mean, viscosity) * length / u_lid ** 3
    return series


def transformed(result, function):
    values = {name: function(result[name]) for name in ("value", "lower", "upper")}
    return {**result, **values}


def kolmogorov_scale(dissipation, reynolds):
    return np.divide(
        reynolds ** -0.75, np.sqrt(np.sqrt(np.maximum(dissipation, 0))),
        out=np.full_like(dissipation, np.nan), where=dissipation > 0,
    )


def lumley_invariants(stress):
    trace = stress["r_xx"] + stress["r_yy"] + stress["r_zz"]
    tensor = np.empty(trace.shape + (3, 3))
    for (i, j), name in zip(
        ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)),
        (component[0] for component in REYNOLDS_COMPONENTS),
    ):
        tensor[..., i, j] = tensor[..., j, i] = np.divide(
            stress[name], trace, out=np.full_like(trace, np.nan), where=trace > 0,
        )
    tensor -= np.eye(3) / 3
    square = np.einsum("...ij,...ji->...", tensor, tensor)
    cube = np.einsum("...ij,...jk,...ki->...", tensor, tensor, tensor)
    return np.cbrt(cube / 6), np.sqrt(square / 6)


def flow_scales(metadata):
    length = metadata["nx"] - 1
    return metadata["u_max"] * length / metadata["reynolds"], metadata["u_max"], length


def block_t_star(metadata) -> float:
    steps = metadata["block_samples"] * metadata["sample_interval"]
    return steps * metadata["u_max"] / (metadata["nx"] - 1)


def analyze_lines(metadata, lines):
    results = {}
    for line, blocks in lines.items():
        blocks = blocks[:usable_blocks(blocks.shape[0])]
        mean = blocks.mean(axis=0)
        for name, series in line_series(blocks, mean, *flow_scales(metadata)).items():
            group, error, ratio = batch_uncertainty(series)
            value = series.mean(axis=0)
            results[(line, name)] = {
                "position": np.linspace(-1, 1, value.size), "value": value,
                "lower": value - 1.96 * error, "upper": value + 1.96 * error, "ratio": ratio,
                "count": np.divide(blocks.shape[0], group, out=np.zeros(value.size), where=group > 0).astype(int),
                "block_t_star": group * block_t_star(metadata),
            }
        for name, source in (("ux_rms", "r_xx"), ("uy_rms", "r_yy"), ("uz_rms", "r_zz")):
            results[(line, name)] = transformed(
                results[(line, source)], lambda values: np.sqrt(np.maximum(values, 0)),
            )
        scale = transformed(
            results[(line, "dissipation")],
            lambda values: kolmogorov_scale(values, metadata["reynolds"]),
        )
        results[(line, "kolmogorov")] = {**scale, "lower": scale["upper"], "upper": scale["lower"]}
    return results


def aspect_ratio_suffix(aspect_ratio):
    for value, suffix in {0.25: "025", 0.5: "05", 0.75: "075", 1.0: "1"}.items():
        if np.isclose(aspect_ratio, value):
            return suffix
    raise ValueError(f"Unsupported experimental aspect ratio: {aspect_ratio}")


def relative_l2(position, error, reference):
    numerator = np.trapezoid(error * error, position)
    denominator = np.trapezoid(reference * reference, position)
    return float(np.sqrt(numerator / denominator)) if denominator > 0 else float("nan")


def load_reference(directory, stem, reynolds, aspect_ratio, coordinate, amplification=1.0):
    path = directory / f"{stem}_{int(round(reynolds))}_{aspect_ratio_suffix(aspect_ratio)}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.genfromtxt(path, delimiter=",", names=True)
    first = np.atleast_1d(data["x"]).astype(float)
    second = np.atleast_1d(data["y"]).astype(float)
    position, value = (second, first) if coordinate == "y" else (first, second)
    order = np.argsort(position)
    return path, position[order], value[order] / amplification


def process_lines(metadata, lines) -> None:
    results = analyze_lines(metadata, lines)
    rows = []
    for (line, quantity), result in results.items():
        columns = ("position", "value", "lower", "upper", "ratio", "block_t_star", "count")
        rows.extend((line, quantity, *row) for row in zip(*(result[name] for name in columns)))
    write_dat(
        "line_statistics.dat",
        ("line", "quantity", "position_star", "value", "lower_95", "upper_95", "s1_over_s0",
         "block_t_star", "block_count"), rows,
    )

    rows = []
    for line in lines:
        stress = {name: results[(line, name)]["value"] for name, *_ in REYNOLDS_COMPONENTS}
        position = results[(line, "r_xx")]["position"]
        rows.extend((line, *row) for row in zip(position, *lumley_invariants(stress)))
    write_dat("lumley.dat", ("line", "position_star", "xi", "eta"), rows)

    axes = dict(LINES)
    rows, summary = [], []
    for quantity, line, stem, amplification in EXPERIMENTS:
        try:
            path, position, reference = load_reference(
                REFERENCE_DIRECTORY, stem, metadata["reynolds"],
                metadata["aspect_ratio"], axes[line], amplification,
            )
        except (FileNotFoundError, ValueError):
            continue
        result = results[(line, quantity)]
        simulation = np.interp(position, result["position"], result["value"])
        error = simulation - reference
        rows.extend((line, quantity, *row) for row in zip(position, reference, simulation, error))
        summary.append((
            line, quantity, path.name, relative_l2(position, error, reference),
            float(np.sqrt(np.mean(error * error))), float(np.max(np.abs(error))),
        ))
    if rows:
        write_dat(
            "experiment.dat",
            ("line", "quantity", "position_star", "experiment", "simulation", "error"), rows,
        )
        write_dat(
            "experiment_summary.dat",
            ("line", "quantity", "reference_file", "relative_l2", "rmse", "max_abs_error"),
            summary,
        )


def welch_spectrum(series, interval):
    size = 2 * series.size // (SPECTRUM_SEGMENTS + 1)
    window = np.hanning(size)
    segments = np.stack([
        series[start:start + size] for start in np.arange(SPECTRUM_SEGMENTS) * (size // 2)
    ])
    segments = (segments - segments.mean(axis=1, keepdims=True)) * window
    power = np.mean(np.abs(np.fft.rfft(segments, axis=1)) ** 2, axis=0)
    return np.fft.rfftfreq(size, interval)[1:], 2 * interval * power[1:] / np.sum(window ** 2)


def process_probe(data) -> None:
    if "probe_ux" not in data.dtype.names:
        return
    interval = float(np.median(np.diff(data["t_star"])))
    spectra = [welch_spectrum(data[f"probe_{name}"], interval) for name in ("ux", "uy", "uz")]
    write_dat(
        "probe_spectrum.dat", ("frequency_star", "ux", "uy", "uz"),
        zip(spectra[0][0], *(density for _, density in spectra)),
    )


def process_global_statistics(path: Path, window) -> None:
    data = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    names = ("kinetic_energy_over_U2", "dissipation_star")
    write_dat("global_statistics.dat", data.dtype.names, (tuple(row) for row in data))
    if window is None:
        return
    selected = data[(data["step"] > window[0]) & (data["step"] <= window[1])]
    selected = selected[:usable_blocks(selected.size)]
    if selected.size < BLOCK_MULTIPLE:
        return
    process_probe(selected)
    series = np.column_stack([selected[name] for name in names])
    interval = float(np.median(np.diff(selected["t_star"])))
    rows = []
    for column, name in enumerate(names):
        group, error, ratio = batch_uncertainty(series[:, column:column + 1])
        rows.append((
            name, series[:, column].mean(), 1.96 * error[0], ratio[0],
            group[0] * interval, selected.size // group[0] if group[0] else 0,
        ))
    write_dat(
        "global_summary.dat",
        ("quantity", "mean", "half_width_95", "s1_over_s0",
         "block_t_star", "block_count"), rows,
    )


def derivative_fd4(field: np.ndarray, axis: int) -> np.ndarray:
    values = np.moveaxis(np.asarray(field, dtype=np.float64), axis, 0)
    if values.shape[0] < 5:
        raise ValueError("Fourth-order derivatives require at least five nodes")
    derivative = np.empty_like(values)
    derivative[0] = (-25 * values[0] + 48 * values[1] - 36 * values[2] + 16 * values[3] - 3 * values[4]) / 12
    derivative[1] = (-3 * values[0] - 10 * values[1] + 18 * values[2] - 6 * values[3] + values[4]) / 12
    derivative[2:-2] = (values[:-4] - 8 * values[1:-3] + 8 * values[3:-1] - values[4:]) / 12
    derivative[-2] = (-values[-5] + 6 * values[-4] - 18 * values[-3] + 10 * values[-2] + 3 * values[-1]) / 12
    derivative[-1] = (3 * values[-5] - 16 * values[-4] + 36 * values[-3] - 48 * values[-2] + 25 * values[-1]) / 12
    return np.moveaxis(derivative, 0, axis)


def plane_slice(field, axis: str, index: int):
    return np.asarray(field[{"x": np.s_[:, :, index], "y": np.s_[:, index], "z": np.s_[index]}[axis]])


def process_planes(fields, dimensions, u_lid) -> None:
    length = dimensions[0] - 1
    sizes = dict(zip("xyz", dimensions))
    for axis, fraction in PLANES:
        coordinate = fraction * (sizes[axis] - 1)
        low, high = int(np.floor(coordinate)), int(np.ceil(coordinate))
        weight = coordinate - low
        mean = np.stack([
            (1 - weight) * plane_slice(field, axis, low) + weight * plane_slice(field, axis, high)
            for field in fields
        ])
        if low == high:
            low, high = max(low - 1, 0), min(high + 1, sizes[axis] - 1)
        velocity = {name[1]: mean[FIELD[name]] for name in ("ux", "uy", "uz")}
        in_plane = [name for name in "zyx" if name != axis]

        def derivative(component, direction):
            if direction != axis:
                return derivative_fd4(velocity[component], in_plane.index(direction))
            field = fields[FIELD["u" + component]]
            return (plane_slice(field, axis, high) - plane_slice(field, axis, low)) / (high - low)

        flat = mean.reshape(len(LOCAL_FIELD_NAMES), -1)
        stress = {
            name: fluctuation_product(flat, flat, *names).reshape(mean.shape[1:])
            for name, *names in REYNOLDS_COMPONENTS
        }
        xi, eta = lumley_invariants(stress)
        path = DATA_DIRECTORY / f"plane_{axis}_{fraction:g}.npz"
        np.savez(
            path,
            **{name: np.arange(sizes[name]) / length for name in in_plane},
            **{f"u{name}": values / u_lid for name, values in velocity.items()},
            wx=(derivative("z", "y") - derivative("y", "z")) * length / u_lid,
            wy=(derivative("x", "z") - derivative("z", "x")) * length / u_lid,
            wz=(derivative("y", "x") - derivative("x", "y")) * length / u_lid,
            xi=xi, eta=eta,
        )
        print(f"Wrote: {path}")


def process_kolmogorov(fields, dimensions, u_lid, reynolds) -> None:
    nx, ny, nz = dimensions
    length = nx - 1
    viscosity = u_lid * length / reynolds
    dissipation = np.zeros(nx * ny)
    for z in range(nz):
        plane = np.asarray(fields[:, z]).reshape(len(LOCAL_FIELD_NAMES), -1)
        np.maximum(dissipation, turbulent_dissipation(plane, plane, viscosity), out=dissipation)
    scale = kolmogorov_scale(dissipation * length / u_lid ** 3, reynolds)
    path = DATA_DIRECTORY / "kolmogorov_minimum.npz"
    np.savez(
        path, x=np.arange(nx) / length, y=np.arange(ny) / length,
        eta_over_dx=scale.reshape(ny, nx) * length,
    )
    print(f"Wrote: {path}")


def trapezoidal_weights(size):
    weights = np.ones(size)
    weights[[0, -1]] = 0.5
    return weights


def process_energy_balance(fields, dimensions, u_lid, reynolds) -> None:
    length = dimensions[0] - 1
    scale = 2 * length ** 2 / (reynolds * u_lid ** 2 * np.prod(np.array(dimensions) - 1.0))
    x, y, z = (trapezoidal_weights(size) for size in dimensions)
    dissipation = scale * np.einsum("kji,k,j,i", fields[FIELD["S2"]], z, y, x)
    power = scale * u_lid * np.einsum("ki,k,i", fields[FIELD["Sxy"]][:, -1], z, x)
    write_dat(
        "energy_balance.dat", ("lid_power_star", "dissipation_star", "relative_difference"),
        ((power, dissipation, dissipation / power - 1),),
    )


def sign_change(position, values, direction, lower, upper):
    index = np.flatnonzero((direction * values[:-1] < 0) & (direction * values[1:] >= 0))
    crossing = position[index] - values[index] * np.diff(position)[index] / np.diff(values)[index]
    crossing = crossing[(crossing > lower) & (crossing < upper)]
    return crossing if crossing.size else np.array([np.nan])


def vortex_sizes(x, y, shear):
    return (
        sign_change(y, shear[:, 0], 1, 0, 0.5)[-1], sign_change(x, shear[0], -1, 0, 0.5)[-1],
        sign_change(y, shear[:, -1], 1, 0, 0.5)[-1], 1 - sign_change(x, shear[0], 1, 0.5, 1)[0],
    )


def process_wall_shear(fields, dimensions) -> None:
    middle = [(dimensions[2] - 1) // 2, dimensions[2] // 2]
    ux, uy, shear = (fields[FIELD[name], middle].mean(axis=0) for name in ("ux", "uy", "Sxy"))
    values = {
        "moments": shear,
        "fd2": (np.gradient(ux, axis=0, edge_order=2) + np.gradient(uy, axis=1, edge_order=2)) / 2,
        "fd4": (derivative_fd4(ux, 0) + derivative_fd4(uy, 1)) / 2,
    }
    walls = {"upstream": np.s_[:, 0], "downstream": np.s_[:, -1], "bottom": np.s_[0, :]}
    position = {wall: np.arange(dimensions[0 if wall == "bottom" else 1]) / (dimensions[0] - 1) for wall in walls}
    write_dat(
        "wall_shear.dat", ("wall", "position", *values),
        ((wall, *row) for wall, index in walls.items()
         for row in zip(position[wall], *(field[index] for field in values.values()))),
    )
    x, y = position["bottom"], position["upstream"]
    write_dat(
        "vortex_sizes.dat", ("method", "usv_height", "usv_length", "dsv_height", "dsv_length"),
        ((name, *vortex_sizes(x, y, field)) for name, field in values.items()),
    )


def main() -> None:
    if not STATISTICS_DIRECTORY.is_dir():
        raise FileNotFoundError("Run this script from a run directory containing statistics/")
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.dat", "*.npz"):
        for path in DATA_DIRECTORY.glob(pattern):
            path.unlink()

    window = None
    block_path = STATISTICS_DIRECTORY / "bmbc_blocks.bin"
    if block_path.exists():
        metadata, lines = read_bmbc_blocks(block_path)
        if metadata["completed_blocks"] > 0:
            process_lines(metadata, lines)
            samples = usable_blocks(metadata["completed_blocks"]) * metadata["block_samples"]
            start = metadata["start_step"]
            window = (start, start + metadata["total_samples"] * metadata["sample_interval"])
            scale = metadata["u_max"] / (metadata["nx"] - 1)
            end = start + samples * metadata["sample_interval"]
            write_dat(
                "sampling.dat",
                ("start_step", "start_t_star", "end_step", "end_t_star", "sample_interval", "samples"),
                ((start, start * scale, end, end * scale, metadata["sample_interval"], samples),),
            )

    global_path = STATISTICS_DIRECTORY / "global_statistics.csv"
    if global_path.exists():
        process_global_statistics(global_path, window)

    local_path = STATISTICS_DIRECTORY / "local_fields.bin"
    if local_path.exists():
        header, fields, dimensions = read_local_fields(local_path)
        u_lid, reynolds = float(header["u_max"]), float(header["reynolds"])
        process_planes(fields, dimensions, u_lid)
        process_kolmogorov(fields, dimensions, u_lid, reynolds)
        process_energy_balance(fields, dimensions, u_lid, reynolds)
        process_wall_shear(fields, dimensions)


if __name__ == "__main__":
    main()

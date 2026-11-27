import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "post"))
SPEC = importlib.util.spec_from_file_location(
    "process_statistics", ROOT / "post" / "process_statistics.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TurbulenceProfilesTest(unittest.TestCase):
    def test_experimental_plot_amplification_is_removed(self):
        reference_dir = ROOT / "post" / "Prasad_Koseff"
        _, _, rms = MODULE.load_reference(reference_dir, "uxRMS_cy", 3200, 1, "y", 10.0)
        _, _, shear = MODULE.load_reference(reference_dir, "uxuy_cy", 3200, 1, "y", 500.0)
        self.assertLess(np.max(np.abs(rms)), 0.1)
        self.assertLess(np.max(np.abs(shear)), 0.01)

    def test_fluctuation_series_average_to_the_reynolds_stress(self):
        rng = np.random.default_rng(1)
        velocity = rng.standard_normal((240, 3, 4)) + np.array([1.0, -2.0, 0.5])[:, None]
        blocks = np.zeros((240, len(MODULE.LOCAL_FIELD_NAMES), 4))
        for index, name in enumerate(("ux", "uy", "uz")):
            blocks[:, MODULE.FIELD[name]] = velocity[:, index]
        for _, second, first, other in MODULE.REYNOLDS_COMPONENTS:
            i, j = "xyz".index(first[1]), "xyz".index(other[1])
            blocks[:, MODULE.FIELD[second]] = velocity[:, i] * velocity[:, j]
        mean = blocks.mean(axis=0)
        fluctuation = velocity - velocity.mean(axis=0)
        for _, second, first, other in MODULE.REYNOLDS_COMPONENTS:
            i, j = "xyz".index(first[1]), "xyz".index(other[1])
            series = MODULE.fluctuation_product(blocks, mean, second, first, other)
            np.testing.assert_allclose(series, fluctuation[:, i] * fluctuation[:, j], atol=1e-12)

    def test_turbulent_dissipation_is_the_mean_squared_strain_fluctuation(self):
        rng = np.random.default_rng(4)
        blocks = np.zeros((50, len(MODULE.LOCAL_FIELD_NAMES), 3))
        expected = 0.0
        for name, weight in MODULE.STRAIN_COMPONENTS:
            strain = rng.standard_normal((50, 3)) + 2.0
            blocks[:, MODULE.FIELD[name]] = strain
            blocks[:, MODULE.FIELD["S2"]] += weight * strain ** 2
            expected = expected + 2 * 0.01 * weight * (strain - strain.mean(axis=0)) ** 2
        series = MODULE.turbulent_dissipation(blocks, blocks.mean(axis=0), 0.01)
        np.testing.assert_allclose(series, expected, atol=1e-12)

    def test_couette_energy_balance(self):
        rows = []
        original, MODULE.write_dat = MODULE.write_dat, lambda name, columns, values: rows.extend(values)
        try:
            fields = np.zeros((len(MODULE.LOCAL_FIELD_NAMES), 9, 17, 17))
            fields[MODULE.FIELD["Sxy"]] = 0.05 / (2 * 16)
            fields[MODULE.FIELD["S2"]] = 2 * (0.05 / (2 * 16)) ** 2
            MODULE.process_energy_balance(fields, (17, 17, 9), 0.05, 100.0)
        finally:
            MODULE.write_dat = original
        np.testing.assert_allclose(rows[0], [0.01, 0.01, 0.0], atol=1e-12)

    def test_wall_shear_of_a_quadratic_profile(self):
        rows = []
        original, MODULE.write_dat = MODULE.write_dat, lambda name, columns, values: rows.extend(values)
        try:
            x = np.arange(17) / 16
            fields = np.zeros((len(MODULE.LOCAL_FIELD_NAMES), 8, 17, 17))
            fields[MODULE.FIELD["uy"]] = 0.05 * x * (1 - x)
            fields[MODULE.FIELD["Sxy"]] = 0.05 * (1 - 2 * x) / (2 * 16)
            MODULE.process_wall_shear(fields, (17, 17, 8))
        finally:
            MODULE.write_dat = original
        shear = [row for row in rows if len(row) == 5]
        for wall, sign in (("upstream", 1), ("downstream", -1), ("bottom", 0)):
            values = np.array([row[2:] for row in shear if row[0] == wall], dtype=float)
            expected = sign * 0.05 / 32 if sign else (0.05 * (1 - 2 * x) / 32)[:, None]
            np.testing.assert_allclose(values, np.broadcast_to(expected, values.shape), atol=1e-12)

    def test_vortex_sizes_ignore_the_corner_eddies(self):
        position = np.linspace(0, 1, 201)
        shear = np.zeros((201, 201))
        shear[:, 0] = np.interp(position, [0, 0.03, 0.2, 0.5, 1], [0.1, 0, -0.1, 0.4, 1])
        shear[:, -1] = np.interp(position, [0, 0.02, 0.3, 0.5, 1], [0.1, 0, -0.2, 0.3, 1])
        shear[0] = np.interp(position, [0, 0.02, 0.2, 0.4, 0.7, 0.9, 1], [-0.1, 0, 0.2, -0.3, -0.2, 0.3, 0.2])
        sizes = MODULE.vortex_sizes(position, position, shear)
        np.testing.assert_allclose(sizes, [0.26, 0.28, 0.38, 0.22], atol=1e-12)

    def test_spectrum_integrates_to_the_variance(self):
        series = np.random.default_rng(2).standard_normal(40000)
        frequency, density = MODULE.welch_spectrum(series, 0.1)
        self.assertAlmostEqual(np.sum(density) * frequency[0], series.var(), delta=0.03)

    def test_lumley_limits(self):
        def invariants(xx, yy, zz):
            names = ("r_xx", "r_yy", "r_zz", "r_xy", "r_xz", "r_yz")
            values = (xx, yy, zz, 0.0, 0.0, 0.0)
            stress = {name: np.array([value]) for name, value in zip(names, values)}
            return [float(value[0]) for value in MODULE.lumley_invariants(stress)]
        np.testing.assert_allclose(invariants(1.0, 1.0, 1.0), [0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(invariants(1.0, 0.0, 0.0), [1 / 3, 1 / 3])
        np.testing.assert_allclose(invariants(1.0, 1.0, 0.0), [-1 / 6, 1 / 6])

    def test_batch_statistics_formula_and_mean_invariance(self):
        series = np.array([1.0, 2.0, 4.0, 8.0])[:, None]
        variance, ratio = MODULE.batch_statistics(series)
        self.assertAlmostEqual(variance[0], 39.625 / 6.0)
        self.assertAlmostEqual(ratio[0], 5.4375 / 28.75)
        shifted, _ = MODULE.batch_statistics(series + 1000.0)
        self.assertAlmostEqual(variance[0], shifted[0])

    def test_batch_statistics_matches_correlation_matrix(self):
        rng = np.random.default_rng(2)
        for samples, size in ((24, 3), (60, 5), (48, 12)):
            values = 7.0 + np.cumsum(rng.standard_normal(samples))
            centered = values - values.mean()
            block = np.arange(samples) // size
            near = np.abs(block[:, None] - block[None, :]) <= 1
            self.assertEqual(np.sum(~near), (samples - size) * (samples - 2 * size))
            expected = np.outer(centered, centered)[near].sum() / np.sum(~near)
            means = values.reshape(-1, size).mean(axis=1)[:, None]
            self.assertAlmostEqual(MODULE.batch_statistics(means)[0][0], expected)

    def test_batch_size_is_the_smallest_below_the_correlation_limit(self):
        rng = np.random.default_rng(3)
        samples = 100800
        noise = 0.1 * rng.uniform(size=samples)
        values = np.empty(samples)
        values[0] = 0.5
        for index in range(1, samples):
            values[index] = 0.9 * values[index - 1] + noise[index]
        group, error, ratio = MODULE.batch_uncertainty(values[:, None])
        self.assertLessEqual(ratio[0], 0.5)
        for size in range(1, group[0]):
            if samples % size == 0:
                means = values.reshape(-1, size).mean(axis=1)[:, None]
                self.assertGreater(MODULE.batch_statistics(means)[1][0], 0.5)
        means = values.reshape(-1, 100).mean(axis=1)[:, None]
        variance, _ = MODULE.batch_statistics(means)
        self.assertLess(abs(samples * variance[0] * 12 - 1), 0.15)


if __name__ == "__main__":
    unittest.main()

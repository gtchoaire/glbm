import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "post"))
SPEC = importlib.util.spec_from_file_location(
    "analyze_turbulence_profiles", ROOT / "post" / "analyze_turbulence_profiles.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TurbulenceProfilesTest(unittest.TestCase):
    def test_experimental_plot_amplification_is_removed(self):
        reference_dir = ROOT / "post" / "Prasad_Koseff"
        _, _, rms = MODULE.load_reference(reference_dir, "uxRMS_cy", 3200, 1, 10.0)
        _, _, shear = MODULE.load_reference(reference_dir, "uxuy_cy", 3200, 1, 500.0)
        self.assertLess(np.max(np.abs(rms)), 0.1)
        self.assertLess(np.max(np.abs(shear)), 0.01)

    def test_derived_rms_and_covariance(self):
        position = np.array([-1.0, 1.0])
        raw = {
            "ux_y": {"position": position, "mean": np.array([1.0, 2.0])},
            "uy_y": {"position": position, "mean": np.array([0.5, -1.0])},
            "ux2_y": {"position": position, "mean": np.array([1.25, 5.0])},
            "uxuy_y": {"position": position, "mean": np.array([0.75, -1.5])},
            "ux_x": {"position": position, "mean": np.array([0.25, -0.5])},
            "uy_x": {"position": position, "mean": np.array([1.0, 2.0])},
            "uy2_x": {"position": position, "mean": np.array([2.0, 4.25])},
            "uxuy_x": {"position": position, "mean": np.array([0.5, -0.75])},
        }
        result = MODULE.derive_profiles(raw, 1.0)
        np.testing.assert_allclose(result["ux_rms_y"]["value"], [0.5, 1.0])
        np.testing.assert_allclose(result["uy_rms_x"]["value"], [1.0, 0.5])
        np.testing.assert_allclose(result["uxuy_y"]["value"], [0.25, 0.5])
        np.testing.assert_allclose(result["uxuy_x"]["value"], [0.25, 0.25])

    def test_delta_bmbc_matches_scalar_projected_formula(self):
        base = np.array([1.0, 2.0, 4.0, 8.0])[:, None]
        values = np.concatenate((base, 2.0 * base, 3.0 * base), axis=1)
        values = values[:, None, :]
        derived, _, _, variance = MODULE.delta_bmbc(values, "covariance")
        mean = values[:, 0, :].mean(axis=0)
        gradient = np.array([-mean[1], -mean[0], 1.0])
        projected = values[:, 0, :] @ gradient
        projected_mean = projected.mean()
        s0 = np.sum((projected - projected_mean) ** 2)
        s1 = np.sum(projected[:-1] * projected[1:] - projected_mean ** 2)
        expected_variance = (s0 + 2.0 * s1) / 6.0
        self.assertAlmostEqual(derived[0], mean[2] - mean[0] * mean[1])
        self.assertAlmostEqual(variance[0], expected_variance)


if __name__ == "__main__":
    unittest.main()

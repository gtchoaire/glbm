import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "post"))
MODULE_PATH = ROOT / "post" / "process_statistics.py"
SPEC = importlib.util.spec_from_file_location("process_statistics", MODULE_PATH)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


class BMBCAnalysisTest(unittest.TestCase):
    def test_relative_l2_uses_profile_coordinate_weighting(self):
        position = np.array([-1.0, -0.5, 1.0])
        reference = np.array([1.0, 1.0, 1.0])
        error = np.array([0.5, 0.5, 0.5])
        self.assertAlmostEqual(
            ANALYSIS.relative_l2(position, error, reference), 0.5
        )

    def test_prasad_koseff_column_orientation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, rows in {
                "ux_cy_3200_1.csv": [(-0.2, -1.0), (0.8, 1.0)],
                "uy_cx_3200_1.csv": [(-1.0, 0.1), (1.0, -0.3)],
            }.items():
                with (root / name).open("w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["x", "y"])
                    writer.writerows(rows)

            _, ux_position, ux_velocity = ANALYSIS.load_reference(
                root, "ux_cy", 3200, 1, "y"
            )
            _, uy_position, uy_velocity = ANALYSIS.load_reference(
                root, "uy_cx", 3200, 1, "x"
            )
            np.testing.assert_allclose(ux_position, [-1.0, 1.0])
            np.testing.assert_allclose(ux_velocity, [-0.2, 0.8])
            np.testing.assert_allclose(uy_position, [-1.0, 1.0])
            np.testing.assert_allclose(uy_velocity, [0.1, -0.3])


if __name__ == "__main__":
    unittest.main()

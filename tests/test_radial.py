import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from co2_h2_pes.data import sha256_file
from co2_h2_pes.radial import (
    coefficient_curve_diagnostics,
    leave_one_radius_out_diagnostic,
)


class RadialDiagnosticTests(unittest.TestCase):
    def test_identical_coefficients_have_zero_curve_error(self):
        radii = np.array([4.4, 4.6, 4.8, 5.0])
        coefficients = np.vstack([radii, 2.0 * radii])
        design = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        by_radius, summary = coefficient_curve_diagnostics(
            radii, coefficients, coefficients.copy(), design
        )
        np.testing.assert_allclose(by_radius["projection_deviation_rmse"], 0.0)
        self.assertAlmostEqual(summary.iloc[0]["global_relative_coefficient_error"], 0.0)

    def test_linear_radial_coefficients_are_exact_under_holdout_spline(self):
        radii = np.array([4.4, 4.6, 4.8, 5.0, 5.2])
        coefficients = np.vstack([radii, 2.0 * radii])
        design = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        truth = design @ coefficients
        detail, summary = leave_one_radius_out_diagnostic(
            radii,
            truth,
            design,
            {"linear": coefficients},
            minimum_R=4.4,
            maximum_R=5.2,
        )
        self.assertGreater(len(detail), 0)
        self.assertLess(summary["worst_individual_normalized_error"].max(), 1.0e-10)

    def test_frozen_real_data_holdout_localizes_the_shared_R48_failure(self):
        root = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "reference"
            / "radial_diagnostics_v1"
        )
        summary = pd.read_csv(root / "radial_holdout_summary.csv")
        failures = summary.loc[summary["worst_individual_normalized_error"] > 1.0]
        self.assertEqual(set(failures["heldout_R"]), {4.8})
        self.assertEqual(len(failures), 2)
        manifest = json.loads((root / "manifest.json").read_text())
        for key, filename in manifest["files"].items():
            self.assertEqual(sha256_file(root / filename), manifest["file_sha256"][key])


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from co2_h2_pes.design_study import evaluate_fixed_subset
from co2_h2_pes.data import sha256_file


class DesignStudyTests(unittest.TestCase):
    def test_exact_full_rank_subset_passes(self):
        design = np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]]
        )
        coefficients = np.array([[2.0, 3.0], [-1.0, 0.5]])
        truth = design @ coefficients
        result = evaluate_fixed_subset(
            design, truth, np.array([4.4, 4.6]), np.array([0, 1])
        )
        self.assertEqual(result["retained_rank"], 2)
        self.assertTrue(result["strict_pass"])
        self.assertAlmostEqual(result["worst_normalized_error"], 0.0)

    def test_duplicate_subset_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_fixed_subset(
                np.eye(2), np.eye(2), np.array([4.4, 4.6]), np.array([0, 0])
            )

    def test_out_of_bounds_subset_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_fixed_subset(
                np.eye(2), np.eye(2), np.array([4.4, 4.6]), np.array([-1, 0])
            )

    def test_frozen_real_data_summary_records_the_180_decision(self):
        root = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "reference"
            / "doptimal_design_study_v1"
        )
        sweep = pd.read_csv(root / "count_sweep.csv")
        substitutions = pd.read_csv(root / "single_substitution_summary.csv")
        sensitivity = pd.read_csv(root / "criterion_sensitivity_summary.csv")
        reoptimized = pd.read_csv(root / "reoptimized_recovery.csv")
        self.assertEqual(int(sweep.loc[sweep["strict_pass"], "training_points"].min()), 161)
        row = sweep.loc[sweep["training_points"] == 180].iloc[0]
        self.assertAlmostEqual(row["tolerance_safety_factor"], 3.027594, places=5)
        row = substitutions.loc[substitutions["training_points"] == 180].iloc[0]
        self.assertAlmostEqual(row["strict_pass_rate_percent"], 97.222222, places=5)
        row = sensitivity.loc[sensitivity["training_points"] == 180].iloc[0]
        self.assertEqual(int(row["specifications"]), 36)
        self.assertAlmostEqual(row["minimum_safety_factor"], 1.513797, places=5)
        self.assertEqual(int(reoptimized["reoptimized_strict_pass"].sum()), 16)
        self.assertEqual(len(reoptimized), 16)
        manifest = json.loads((root / "manifest.json").read_text())
        for key, filename in manifest["files"].items():
            self.assertEqual(sha256_file(root / filename), manifest["file_sha256"][key])


if __name__ == "__main__":
    unittest.main()

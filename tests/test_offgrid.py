import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from co2_h2_pes.data import sha256_file
from co2_h2_pes.offgrid import make_offgrid_angles


class OffGridTests(unittest.TestCase):
    def test_sobol_angles_are_deterministic_and_in_domain(self):
        first = make_offgrid_angles("coordinate-uniform", sample_size=16)
        second = make_offgrid_angles("coordinate-uniform", sample_size=16)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
            self.assertTrue(np.all(left > 0.0))
            self.assertTrue(np.all(left < np.pi))

    def test_two_sampling_measures_are_distinct(self):
        coordinate = make_offgrid_angles("coordinate-uniform", sample_size=16)
        solid = make_offgrid_angles("solid-angle", sample_size=16)
        self.assertFalse(np.array_equal(coordinate[0], solid[0]))

    def test_non_power_of_two_is_rejected(self):
        with self.assertRaises(ValueError):
            make_offgrid_angles("coordinate-uniform", sample_size=12)

    def test_frozen_real_data_diagnostic_preserves_evidence_boundary(self):
        root = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "reference"
            / "offgrid_diagnostics_v1"
        )
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertIn("not independent validation", manifest["evidence_type"])
        for key, filename in manifest["files"].items():
            self.assertEqual(sha256_file(root / filename), manifest["file_sha256"][key])
        summary = pd.read_csv(root / manifest["files"]["summary"])
        self.assertAlmostEqual(
            summary["maximum_absolute_disagreement"].max(), 108.388428, places=5
        )
        self.assertEqual(
            int(summary["reference_high_candidate_below_warning"].sum()), 0
        )
        self.assertEqual(
            int(summary["reference_above_1000_candidate_below_1000"].sum()), 1
        )


if __name__ == "__main__":
    unittest.main()

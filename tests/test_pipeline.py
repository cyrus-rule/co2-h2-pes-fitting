import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from co2_h2_pes.pipeline import run_doptimal_candidate, run_full_grid_reference


class PipelineTests(unittest.TestCase):
    def test_full_grid_run_writes_self_reconstructing_artifact(self):
        rng = np.random.default_rng(20260716)
        orientation_count = 180
        angles = pd.DataFrame(
            {
                "Theta_1": rng.uniform(0.0, 180.0, orientation_count),
                "Theta_2": rng.uniform(0.0, 180.0, orientation_count),
                "Phi": rng.uniform(0.0, 180.0, orientation_count),
            }
        )
        rows = []
        for radius in (6.4, 7.75, 9.0):
            block = angles.copy()
            block.insert(0, "R", radius)
            block["V"] = (
                12.0 * np.cos(np.radians(block["Theta_1"]))
                + 3.0 * np.cos(2.0 * np.radians(block["Phi"]))
                - 40.0 / radius
            )
            rows.append(block)
        data = pd.concat(rows, ignore_index=True)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "synthetic.dat"
            data.to_csv(source, sep=" ", header=False, index=False, float_format="%.12g")
            output = run_full_grid_reference(
                source,
                root / "run",
                expected_points_per_radius=orientation_count,
            )
            manifest = json.loads((output / "manifest.json").read_text())
            coefficients = pd.read_csv(output / "coefficients.csv")

            self.assertEqual(manifest["input"]["rows"], 3 * orientation_count)
            self.assertEqual(manifest["basis"]["term_count"], 158)
            self.assertEqual(manifest["method"]["parameters"]["rcond"], None)
            self.assertEqual(manifest["schema_version"], "1.1")
            self.assertEqual(
                manifest["design_matrix"]["evaluation_shape"],
                [orientation_count, 158],
            )
            self.assertEqual(
                manifest["design_matrix"]["fit_shape"],
                [orientation_count, 158],
            )
            self.assertLessEqual(
                manifest["validation"]["reconstruction_reload_max_abs_delta"],
                manifest["validation"]["reconstruction_reload_tolerance"],
            )
            self.assertEqual(len(coefficients), 3 * 158)
            for name in (
                "metrics.csv",
                "guardrail.csv",
                "evaluation_orientations.csv",
                "selected_orientations.csv",
                "v000.csv",
                "v000.png",
            ):
                self.assertTrue((output / name).is_file())
            with self.assertRaises(FileExistsError):
                run_full_grid_reference(
                    source,
                    output,
                    expected_points_per_radius=orientation_count,
                )

    def test_doptimal_run_records_fit_and_complement_separately(self):
        rng = np.random.default_rng(20260721)
        orientation_count = 200
        point_count = 180
        angles = pd.DataFrame(
            {
                "Theta_1": rng.uniform(0.0, 180.0, orientation_count),
                "Theta_2": rng.uniform(0.0, 180.0, orientation_count),
                "Phi": rng.uniform(0.0, 180.0, orientation_count),
            }
        )
        rows = []
        for radius in (4.4, 7.75):
            block = angles.copy()
            block.insert(0, "R", radius)
            block["V"] = (
                8.0 * np.cos(2.0 * np.radians(block["Theta_1"]))
                + 2.0 * np.cos(2.0 * np.radians(block["Phi"]))
                - 20.0 / radius
            )
            rows.append(block)
        data = pd.concat(rows, ignore_index=True)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "synthetic.dat"
            data.to_csv(source, sep=" ", header=False, index=False, float_format="%.12g")
            output = run_doptimal_candidate(
                source,
                root / "run",
                point_count=point_count,
                expected_points_per_radius=orientation_count,
            )
            manifest = json.loads((output / "manifest.json").read_text())
            metrics = pd.read_csv(output / "metrics.csv")
            selected = pd.read_csv(output / "selected_orientations.csv")

        self.assertEqual(manifest["method"]["id"], "doptimal180_candidate")
        self.assertEqual(
            manifest["design_matrix"]["evaluation_shape"],
            [orientation_count, 158],
        )
        self.assertEqual(
            manifest["design_matrix"]["fit_shape"],
            [point_count, 158],
        )
        self.assertEqual(manifest["design_matrix"]["fit_rank"], 158)
        self.assertFalse(
            manifest["method"]["parameters"]["selection_uses_potential_energies"]
        )
        self.assertEqual(len(selected), point_count)
        self.assertEqual(selected["design_position"].tolist(), list(range(1, 181)))
        self.assertEqual(
            set(metrics["evaluation_subset"]),
            {"all", "selected", "unselected"},
        )


if __name__ == "__main__":
    unittest.main()

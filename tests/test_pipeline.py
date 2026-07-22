import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from co2_h2_pes.pipeline import run_full_grid_reference


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
            self.assertLessEqual(
                manifest["validation"]["reconstruction_reload_max_abs_delta"],
                manifest["validation"]["reconstruction_reload_tolerance"],
            )
            self.assertEqual(len(coefficients), 3 * 158)
            for name in (
                "metrics.csv",
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


if __name__ == "__main__":
    unittest.main()

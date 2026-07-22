import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from co2_h2_pes.basis import BASIS_ID, CANDIDATE_BASIS_V1
from co2_h2_pes.coefficients import coefficient_matrix_to_table
from co2_h2_pes.comparison import compare_runs


class ComparisonTests(unittest.TestCase):
    def test_comparison_aligns_by_tuple_not_row_position(self):
        radii = np.array([7.75])
        left_values = np.zeros((len(CANDIDATE_BASIS_V1), 1))
        right_values = left_values.copy()
        right_values[CANDIDATE_BASIS_V1.index((0, 0, 0)), 0] = 2.0
        angles = pd.DataFrame(
            {
                "orientation_id": [0, 1, 2],
                "Theta_1": [0.0, 45.0, 90.0],
                "Theta_2": [15.0, 60.0, 120.0],
                "Phi": [0.0, 30.0, 90.0],
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, values in (("left", left_values), ("right", right_values)):
                run = root / name
                run.mkdir()
                (run / "manifest.json").write_text(
                    json.dumps({"basis": {"id": BASIS_ID}, "method": {"id": name}})
                )
                table = coefficient_matrix_to_table(name, values, radii)
                if name == "right":
                    table = table.sample(frac=1.0, random_state=7)
                table.to_csv(run / "coefficients.csv", index=False)
                angles.to_csv(run / "selected_orientations.csv", index=False)
                angles.to_csv(run / "evaluation_orientations.csv", index=False)
            output = compare_runs(root / "left", root / "right", root / "comparison")
            differences = pd.read_csv(output / "coefficient_differences.csv")
            potential = pd.read_csv(output / "potential_differences.csv")

        changed = differences.loc[differences["abs_delta"] > 0.0]
        self.assertEqual(len(changed), 1)
        self.assertEqual(tuple(changed.iloc[0][["l1", "l2", "L"]]), (0, 0, 0))
        expected = 2.0 / (4.0 * np.sqrt(np.pi))
        self.assertAlmostEqual(potential.iloc[0]["rmse_difference"], expected)


if __name__ == "__main__":
    unittest.main()

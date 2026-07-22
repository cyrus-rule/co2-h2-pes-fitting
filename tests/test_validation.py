import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd

from co2_h2_pes.data import sha256_file
from co2_h2_pes.basis import CANDIDATE_BASIS_V1
from co2_h2_pes.validation import (
    ValidationRequest,
    make_distributional_request,
    score_returned_validation_energies,
    validate_request,
)


class ValidationRequestTests(unittest.TestCase):
    @staticmethod
    def _frozen_root():
        return (
            Path(__file__).resolve().parents[1]
            / "data"
            / "reference"
            / "validation_request_v3"
        )

    def test_distributional_request_reproduces_v3_prefix(self):
        request = make_distributional_request()
        self.assertEqual(len(request), 24)
        self.assertEqual(
            request["Category"].value_counts().to_dict(),
            {
                "Distributional: coordinate-uniform": 12,
                "Distributional: solid-angle": 12,
            },
        )
        np.testing.assert_allclose(
            request.loc[0, ["Theta_1", "Theta_2", "Phi"]].to_numpy(dtype=float),
            [71.948981, 36.717370, 145.179469],
            atol=5.0e-7,
        )
        np.testing.assert_allclose(
            request.loc[12, ["Theta_1", "Theta_2", "Phi"]].to_numpy(dtype=float),
            [67.707816, 113.838162, 133.952857],
            atol=5.0e-7,
        )

    def test_frozen_v3_artifact_is_self_consistent(self):
        root = self._frozen_root()
        manifest = json.loads((root / "manifest.json").read_text())
        request = ValidationRequest(
            orientations=pd.read_csv(root / manifest["files"]["orientations"]),
            primary_energies=pd.read_csv(root / manifest["files"]["primary"]),
            optional_radial_energies=pd.read_csv(
                root / manifest["files"]["optional_radial"]
            ),
            complete_energies=pd.read_csv(root / manifest["files"]["complete"]),
        )
        validate_request(request)
        for key, filename in manifest["files"].items():
            self.assertEqual(
                sha256_file(root / filename),
                manifest["file_sha256"][key],
            )
        generated = make_distributional_request().round(
            {"Theta_1": 6, "Theta_2": 6, "Phi": 6}
        )
        pd.testing.assert_frame_equal(
            request.orientations.iloc[:24].drop(columns="Validation ID").reset_index(
                drop=True
            ),
            generated,
            check_dtype=False,
        )

    def test_scoring_requires_the_complete_frozen_primary_request(self):
        root = self._frozen_root()
        returned = pd.read_csv(root / "primary_validation_energies_144.csv")
        returned["V"] = 0.0
        radii = np.array([4.4, 4.8, 5.0])
        coefficients = np.zeros((len(CANDIDATE_BASIS_V1), len(radii)))
        pointwise, _, decision = score_returned_validation_energies(
            returned,
            radii,
            {"reference": coefficients, "candidate": coefficients},
        )
        self.assertEqual(len(pointwise), 288)
        self.assertTrue(decision["overall_pass"])

        altered = returned.copy()
        altered.loc[altered["Validation ID"] == "V001", "Theta_1"] += 0.01
        with self.assertRaisesRegex(ValueError, "frozen v3"):
            score_returned_validation_energies(
                altered,
                radii,
                {"reference": coefficients, "candidate": coefficients},
            )
        with self.assertRaisesRegex(ValueError, "all 144"):
            score_returned_validation_energies(
                returned.iloc[:-1],
                radii,
                {"reference": coefficients, "candidate": coefficients},
            )


if __name__ == "__main__":
    unittest.main()

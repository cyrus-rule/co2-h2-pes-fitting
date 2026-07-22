import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from co2_h2_pes.yumi import fill_yumi_template, parse_yumi_template


class YumiTests(unittest.TestCase):
    def test_masked_template_fill_zero_and_round_trip(self):
        source_text = (
            "SYNTHETIC CO2-H2\n"
            "2\n"
            "4.4 7.75 40.0\n"
            "0 0 0 0\n"
            "MASKED MASKED MASKED\n"
            "2 0 0 2\n"
            "MASKED MASKED MASKED\n"
            "0 0 2 2\n"
            "MASKED MASKED MASKED\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "template.txt"
            source.write_text(source_text, encoding="utf-8")
            template = parse_yumi_template(source, radial_count=3)
            self.assertTrue(np.isnan(template.coefficients).all())

            table = pd.DataFrame(
                {
                    "method": ["test"] * 3,
                    "R": [4.4, 7.75, 40.0],
                    "l1": [0, 0, 0],
                    "l2": [0, 0, 0],
                    "L": [0, 0, 0],
                    "coefficient": [1.0, -2.0, 3.0],
                }
            )
            filled = fill_yumi_template(template, table)
            output = Path(directory) / "filled.txt"
            output.write_text(filled.to_text(), encoding="utf-8")
            reparsed = parse_yumi_template(output, radial_count=3)

        np.testing.assert_allclose(reparsed.coefficients[0], [1.0, -2.0, 3.0])
        np.testing.assert_array_equal(reparsed.coefficients[1:], 0.0)
        self.assertIn("0.0 0.0 0.0", filled.to_text())

    def test_nan_cannot_be_serialized(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "template.txt"
            path.write_text("T\nH\n4.4\n0 0 0 0\nMASKED\n", encoding="utf-8")
            template = parse_yumi_template(path, radial_count=1)
            with self.assertRaises(ValueError):
                template.to_text()


if __name__ == "__main__":
    unittest.main()

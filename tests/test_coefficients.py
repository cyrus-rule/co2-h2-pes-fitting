import unittest

import numpy as np

from co2_h2_pes.basis import CANONICAL_BASIS
from co2_h2_pes.coefficients import (
    coefficient_matrix_to_table,
    coefficient_table_to_matrix,
)


class CoefficientTests(unittest.TestCase):
    def test_tuple_keyed_round_trip_ignores_row_order(self):
        radial_values = np.array([4.4, 6.6, 7.75])
        original = np.arange(
            len(CANONICAL_BASIS) * len(radial_values),
            dtype=float,
        ).reshape(len(CANONICAL_BASIS), len(radial_values))

        table = coefficient_matrix_to_table(
            "synthetic",
            original,
            radial_values,
        )
        scrambled = table.sample(frac=1.0, random_state=42).reset_index(drop=True)
        reconstructed = coefficient_table_to_matrix(scrambled, radial_values)

        np.testing.assert_array_equal(reconstructed, original)


if __name__ == "__main__":
    unittest.main()

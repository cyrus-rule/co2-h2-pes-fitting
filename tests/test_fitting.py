import unittest

import numpy as np

from co2_h2_pes.fitting import (
    CURRENT_SWITCH_T1,
    CURRENT_SWITCH_T2,
    apply_current_switch,
    fit_coefficients,
)


class FittingTests(unittest.TestCase):
    def test_switch_preserves_low_energies_and_caps_high_energies(self):
        values = np.array(
            [-100.0, CURRENT_SWITCH_T1, 3000.0, CURRENT_SWITCH_T2, 9000.0]
        )
        switched = apply_current_switch(values)
        cap = CURRENT_SWITCH_T1 + (2.0 / np.pi) * (
            CURRENT_SWITCH_T2 - CURRENT_SWITCH_T1
        )

        np.testing.assert_array_equal(switched[:2], values[:2])
        self.assertTrue(np.isclose(switched[-1], cap))
        self.assertTrue(np.all(np.diff(switched) >= 0.0))

    def test_least_squares_recovers_exact_coefficients(self):
        design = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        expected = np.array([2.0, -0.5])
        potential = design @ expected

        fitted, rank, _ = fit_coefficients(design, potential, rcond=None)

        np.testing.assert_allclose(fitted, expected)
        self.assertEqual(rank, 2)


if __name__ == "__main__":
    unittest.main()

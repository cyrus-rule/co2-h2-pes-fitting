import unittest

import numpy as np
from scipy.integrate import quad

from co2_h2_pes.basis import (
    evaluate_angular_basis,
    generate_basis_indices,
    normalized_associated_legendre,
)


class BasisTests(unittest.TestCase):
    def test_canonical_basis_contract(self):
        basis = generate_basis_indices()

        self.assertEqual(len(basis), 158)
        self.assertEqual(len(set(basis)), 158)
        self.assertTrue(all(l1 % 2 == 0 and 0 <= l1 <= 24 for l1, _, _ in basis))
        self.assertTrue(all(l2 in {0, 2, 4, 6} for _, l2, _ in basis))
        self.assertTrue(all(l2 == 0 for l1, l2, _ in basis if l1 >= 22))
        self.assertTrue(
            all(abs(l1 - l2) <= L <= l1 + l2 for l1, l2, L in basis)
        )
        self.assertTrue(all((l1 + l2 + L) % 2 == 0 for l1, l2, L in basis))

    def test_associated_legendre_normalization(self):
        normal, _ = quad(
            lambda x: normalized_associated_legendre(4, 2, x) ** 2,
            -1.0,
            1.0,
        )
        orthogonal, _ = quad(
            lambda x: (
                normalized_associated_legendre(2, 2, x)
                * normalized_associated_legendre(4, 2, x)
            ),
            -1.0,
            1.0,
        )

        self.assertTrue(np.isclose(normal, 1.0, atol=1.0e-10))
        self.assertTrue(np.isclose(orthogonal, 0.0, atol=1.0e-10))

    def test_representative_angular_basis_values_are_finite(self):
        values = evaluate_angular_basis(
            4,
            2,
            4,
            np.radians([15.0, 60.0, 120.0]),
            np.radians([30.0, 90.0, 150.0]),
            np.radians([0.0, 45.0, 90.0]),
        )

        self.assertEqual(values.shape, (3,))
        self.assertTrue(np.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()

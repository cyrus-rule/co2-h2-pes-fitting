import unittest

import numpy as np

from co2_h2_pes.methods import (
    QRGreedyDOptimalLeastSquares,
    qr_seeded_greedy_d_optimal_order,
    whiten_candidate_design,
)


class DOptimalMethodTests(unittest.TestCase):
    def test_whitening_produces_identity_information(self):
        rng = np.random.default_rng(11)
        design = rng.normal(size=(40, 8))
        whitened = whiten_candidate_design(design)
        np.testing.assert_allclose(
            whitened.T @ whitened,
            np.eye(8),
            rtol=1.0e-11,
            atol=1.0e-11,
        )

    def test_order_is_deterministic_unique_and_full_rank(self):
        rng = np.random.default_rng(12)
        design = rng.normal(size=(40, 8))
        first = qr_seeded_greedy_d_optimal_order(design, point_count=12)
        second = qr_seeded_greedy_d_optimal_order(design, point_count=12)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 12)
        self.assertEqual(len(set(first)), 12)
        self.assertEqual(np.linalg.matrix_rank(design[first]), 8)

    def test_method_selection_is_energy_blind(self):
        rng = np.random.default_rng(13)
        design = rng.normal(size=(40, 8))
        first_potentials = rng.normal(size=(40, 3))
        second_potentials = rng.normal(size=(40, 3)) * 1000.0
        method = QRGreedyDOptimalLeastSquares(point_count=12)

        first = method.fit(design, first_potentials)
        second = method.fit(design, second_potentials)

        np.testing.assert_array_equal(
            first.selected_orientation_ids,
            second.selected_orientation_ids,
        )
        self.assertEqual(first.rank, 8)
        self.assertEqual(first.method_id, "doptimal12_candidate")

    def test_point_count_cannot_underdetermine_candidate_basis(self):
        design = np.eye(8)
        with self.assertRaises(ValueError):
            qr_seeded_greedy_d_optimal_order(design, point_count=7)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from co2_h2_pes.evaluation import (
    DEFAULT_EVALUATION_SPEC,
    EvaluationSpec,
    energy_band_masks,
    hybrid_tolerance,
)


class EvaluationSpecTests(unittest.TestCase):
    def test_default_hybrid_tolerance_and_bands(self):
        values = np.array([-5.0, 0.0, 999.0, 1000.0, 3000.0, 5000.0])
        np.testing.assert_allclose(
            hybrid_tolerance(values),
            [1.0, 1.0, 9.99, 10.0, 30.0, 50.0],
        )
        bands = dict(energy_band_masks(values))
        self.assertTrue(bands["attractive_V<0"][0])
        self.assertTrue(bands["low_repulsive_0<=V<1000"][2])
        self.assertTrue(bands["lower_wall_1000<=V<3000"][3])
        self.assertTrue(bands["upper_wall_3000<=V<5000"][4])
        self.assertTrue(bands["guardrail_V>=5000"][5])

    def test_manifest_contains_every_working_threshold(self):
        manifest = DEFAULT_EVALUATION_SPEC.to_manifest()
        self.assertEqual(manifest["physical_ceiling_cm1"], 5000.0)
        self.assertEqual(manifest["guardrail_warning_cm1"], 3000.0)
        self.assertIn("hybrid_tolerance", manifest)

    def test_custom_band_names_record_custom_thresholds(self):
        spec = EvaluationSpec(
            id="custom",
            absolute_tolerance_floor_cm1=2.0,
            relative_tolerance_fraction=0.02,
            attractive_upper_cm1=0.0,
            low_repulsive_upper_cm1=800.0,
            lower_wall_upper_cm1=2400.0,
            physical_ceiling_cm1=4000.0,
            guardrail_warning_cm1=2500.0,
            guardrail_dangerous_cm1=500.0,
        )
        names = [name for name, _ in energy_band_masks(np.array([0.0]), spec)]
        self.assertIn("low_repulsive_0<=V<800", names)
        self.assertIn("guardrail_V>=4000", names)

    def test_invalid_guardrail_order_is_rejected(self):
        with self.assertRaises(ValueError):
            EvaluationSpec(
                id="bad",
                absolute_tolerance_floor_cm1=1.0,
                relative_tolerance_fraction=0.01,
                attractive_upper_cm1=0.0,
                low_repulsive_upper_cm1=1000.0,
                lower_wall_upper_cm1=3000.0,
                physical_ceiling_cm1=5000.0,
                guardrail_warning_cm1=6000.0,
                guardrail_dangerous_cm1=1000.0,
            )


if __name__ == "__main__":
    unittest.main()

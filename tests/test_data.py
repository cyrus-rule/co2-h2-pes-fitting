import tempfile
import unittest
from pathlib import Path

from co2_h2_pes.data import load_ab_initio_data, reference_orientations


class DataTests(unittest.TestCase):
    def test_loader_assigns_shared_orientation_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tiny.dat"
            source.write_text(
                "4.4 0 0 0 10\n"
                "4.4 90 45 30 20\n"
                "6.4 0 0 0 1\n"
                "6.4 90 45 30 2\n"
            )

            frame = load_ab_initio_data(source, expected_points_per_radius=2)
            reference = reference_orientations(frame)

        self.assertTrue(frame.groupby("R")["orientation_id"].nunique().eq(2).all())
        self.assertEqual(reference["orientation_id"].tolist(), [0, 1])
        self.assertEqual(
            list(
                reference[["Theta_1", "Theta_2", "Phi"]].itertuples(index=False)
            ),
            [(0, 0, 0), (90, 45, 30)],
        )


if __name__ == "__main__":
    unittest.main()


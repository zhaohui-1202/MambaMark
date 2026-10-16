from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.evaluate_distinguishability import nc_matrix, summarise

class GalleryMetricsTest(unittest.TestCase):

    def test_identity_and_pair_count(self):
        for count, pairs in [(4, 6), (5, 10), (17, 136)]:
            matrix = nc_matrix(np.eye(count))
            stats = summarise(matrix, 0.9)
            self.assertEqual(stats['pairs'], pairs)
            self.assertEqual(stats['false_accepts'], 0)
            np.testing.assert_allclose(matrix, np.eye(count))

    def test_identical_nonmatches(self):
        stats = summarise(nc_matrix(np.ones((4, 3))), 0.9)
        self.assertEqual(stats['false_accepts'], 6)
        self.assertEqual(stats['nc_far_percent'], 100)

    def test_boundary_inclusive(self):
        stats = summarise(np.array([[1.0, 0.9], [0.9, 1.0]]), 0.9)
        self.assertEqual(stats['false_accepts'], 1)

    def test_invalid_fingerprint(self):
        for vectors in [np.zeros((4, 3)), np.full((4, 3), np.nan)]:
            with self.assertRaises(ValueError):
                nc_matrix(vectors)
if __name__ == '__main__':
    unittest.main()

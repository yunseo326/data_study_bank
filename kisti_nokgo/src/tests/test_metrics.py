import unittest

import numpy as np

from src.metrics import map_at_k


class MapAtKTest(unittest.TestCase):
    def test_known_answer(self):
        classes = np.array(["a", "b", "c"])
        probabilities = np.array(
            [
                [0.9, 0.08, 0.02],
                [0.6, 0.3, 0.1],
                [0.6, 0.3, 0.1],
            ]
        )
        actual = np.array(["a", "b", "c"])
        self.assertAlmostEqual(map_at_k(actual, probabilities, classes, 3), (1 + 0.5 + 1 / 3) / 3)


if __name__ == "__main__":
    unittest.main()


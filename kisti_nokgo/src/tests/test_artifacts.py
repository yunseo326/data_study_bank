import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.artifacts import write_oof_predictions


class OofArtifactTest(unittest.TestCase):
    def test_parquet_round_trip(self):
        expected = pd.DataFrame(
            {
                "target": ["a", "b"],
                "proba_a": [0.8, 0.2],
                "proba_b": [0.2, 0.8],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oof_predictions.parquet"
            write_oof_predictions(expected, path)
            actual = pd.read_parquet(path)
        pd.testing.assert_frame_equal(actual, expected)

    def test_non_parquet_extension_is_rejected(self):
        with self.assertRaises(ValueError):
            write_oof_predictions(pd.DataFrame({"x": [1]}), "oof.csv")


if __name__ == "__main__":
    unittest.main()


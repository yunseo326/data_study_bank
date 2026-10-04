import time
import unittest

import pandas as pd

from src.features import infer_feature_schema
from src.train import build_training_pipeline
from src.validation import cross_validate


class PipelineSmokeTest(unittest.TestCase):
    def test_classification_end_to_end(self):
        frame = pd.DataFrame(
            {
                "id": range(60),
                "number": [value % 5 for value in range(60)],
                "category": ["x", "y", "z"] * 20,
                "target": ["a", "b"] * 30,
            }
        )
        schema = infer_feature_schema(frame, "target", ["id"])
        pipeline = build_training_pipeline(
            schema,
            "classification",
            "hist_gradient_boosting",
            {"max_iter": 5},
            42,
        )
        result = cross_validate(
            pipeline,
            frame[schema.features],
            frame["target"],
            "classification",
            3,
            True,
            42,
            time.perf_counter() + 10,
        )
        self.assertEqual(result.predictions.shape, (60, 2))
        self.assertIn("accuracy", result.overall_metrics)


if __name__ == "__main__":
    unittest.main()


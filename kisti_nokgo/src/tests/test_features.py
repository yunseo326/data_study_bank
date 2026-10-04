import unittest

import pandas as pd

from src.features import infer_feature_schema


class FeatureSchemaTest(unittest.TestCase):
    def test_target_and_id_are_dropped(self):
        frame = pd.DataFrame(
            {
                "id": [1, 2],
                "number": [1.0, 2.0],
                "category": ["x", "y"],
                "target": [0, 1],
            }
        )
        schema = infer_feature_schema(frame, "target", ["id"])
        self.assertEqual(schema.numeric, ["number"])
        self.assertEqual(schema.categorical, ["category"])
        self.assertEqual(schema.dropped, ["target", "id"])


if __name__ == "__main__":
    unittest.main()


import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.run_experiment import run_experiment


class ExperimentLayoutTest(unittest.TestCase):
    def test_results_and_reproducibility_are_separated_and_verifiable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.csv"
            config_path = root / "config.json"
            frame = pd.DataFrame(
                {
                    "feature": list(range(40)),
                    "category": ["a", "b"] * 20,
                    "target": [0, 1] * 20,
                }
            )
            frame.to_csv(train_path, index=False)
            config_path.write_text(
                json.dumps(
                    {
                        "data": {
                            "train_path": str(train_path),
                            "target": "target",
                            "max_train_rows": 40,
                        },
                        "validation": {"n_splits": 2, "random_state": 42},
                        "model": {
                            "name": "hist_gradient_boosting",
                            "params": {"max_iter": 2},
                        },
                        "runtime": {
                            "time_budget_seconds": 30,
                            "n_threads": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )

            previous = Path.cwd()
            try:
                os.chdir(root)
                run_experiment(str(config_path), "E_TEST")
            finally:
                os.chdir(previous)

            experiment = root / "experiments" / "E_TEST"
            results = experiment / "results"
            reproducibility = experiment / "reproducibility"
            self.assertEqual(
                {path.name for path in results.iterdir()},
                {"experiment.md", "metrics.json", "oof_predictions.parquet"},
            )
            self.assertEqual(
                {path.name for path in reproducibility.iterdir()},
                {"config.json", "run_manifest.json"},
            )

            manifest = json.loads((reproducibility / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["artifact_layout_version"], 2)
            for relative_path, metadata in manifest["verification"]["artifacts"].items():
                content = (experiment / relative_path).read_bytes()
                self.assertEqual(hashlib.sha256(content).hexdigest(), metadata["sha256"])


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from src.EDA.run_eda import run_eda


class EdaTest(unittest.TestCase):
    def _run(self, frame: pd.DataFrame, target: str, render_visuals: bool = False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        data_path = root / "train.csv"
        config_path = root / "config.json"
        output_path = root / "reports"
        frame.to_csv(data_path, index=False)
        config_path.write_text(
            json.dumps(
                {
                    "data": {
                        "train_path": str(data_path),
                        "target": target,
                        "id_columns": ["id"],
                        "max_train_rows": 100,
                        "eda_max_rows": 100,
                    }
                }
            ),
            encoding="utf-8",
        )
        return run_eda(str(config_path), str(output_path), render_visuals), output_path

    def test_regression_target_uses_summary_instead_of_value_counts(self):
        frame = pd.DataFrame({"id": range(60), "x": range(60), "target": [value / 7 for value in range(60)]})
        summary, output = self._run(frame, "target")
        self.assertEqual(summary["task_type"], "regression")
        self.assertEqual(summary["target_distribution"], {})
        self.assertEqual(summary["target_summary"]["count"], 60.0)
        self.assertIn("## Target Summary", (output / "EDA.md").read_text(encoding="utf-8"))

    def test_visuals_create_bounded_svg_artifacts(self):
        frame = pd.DataFrame(
            {
                "id": range(60),
                "x": range(60),
                "y": [value * 2 for value in range(60)],
                "category": ["a", "b"] * 30,
                "target": ["no", "yes"] * 30,
            }
        )
        summary, output = self._run(frame, "target", render_visuals=True)
        self.assertEqual(
            summary["visual_artifacts"],
            ["target_distribution.svg", "numeric_histograms.svg", "correlation_heatmap.svg"],
        )
        for artifact in summary["visual_artifacts"]:
            content = (output / "assets" / artifact).read_text(encoding="utf-8")
            self.assertIn("<svg", content)
            ET.fromstring(content)
        heatmap = (output / "assets" / "correlation_heatmap.svg").read_text(encoding="utf-8")
        self.assertNotIn(">id<", heatmap)


if __name__ == "__main__":
    unittest.main()

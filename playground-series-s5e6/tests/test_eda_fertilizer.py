from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eda_fertilizer import analyze, grouped_top1_accuracy, make_html, total_variation  # noqa: E402


def test_total_variation_is_zero_for_equal_distributions():
    left = pd.Series(["A", "A", "B", "B"])
    right = pd.Series(["B", "A", "B", "A"])
    assert total_variation(left, right) == 0


def test_grouped_top1_accuracy_uses_majority_within_group():
    frame = pd.DataFrame({
        "feature": [1, 1, 1, 2, 2],
        "Fertilizer Name": ["A", "A", "B", "B", "B"],
    })
    assert grouped_top1_accuracy(frame, ["feature"]) == 4 / 5


def test_report_explains_features_objective_and_roadmap():
    report = make_html(analyze())
    assert 'id="features"' in report
    assert "같은 feature도 여러 방식으로 표현할 수 있습니다" in report
    assert 'id="objective"' in report
    assert "MAP@3 ↑ 1.0" in report
    assert "Log loss" in report and "작을수록 좋음" in report
    assert 'id="roadmap"' in report
    assert "현재 위치:" in report

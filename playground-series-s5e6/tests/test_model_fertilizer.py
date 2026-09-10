from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_fertilizer import (  # noqa: E402
    TARGET,
    apply_feature_set,
    make_folds,
    map_at_3,
    per_class_diagnostics,
    target_encode,
    top3_labels,
    validate_probabilities,
    validate_submission,
)


def test_map_at_3_scores_first_second_third_and_missing():
    classes = np.array(["A", "B", "C", "D"], dtype=object)
    probabilities = np.array([
        [0.9, 0.1, 0.0, 0.0],
        [0.5, 0.4, 0.1, 0.0],
        [0.5, 0.3, 0.2, 0.0],
        [0.4, 0.3, 0.2, 0.1],
    ])
    truth = np.array(["A", "B", "C", "D"], dtype=object)
    expected = (1.0 + 0.5 + 1.0 / 3.0 + 0.0) / 4.0
    assert map_at_3(truth, probabilities, classes) == pytest.approx(expected)


def test_top3_is_stable_and_unique_for_distinct_classes():
    classes = np.array(["A", "B", "C", "D"], dtype=object)
    probabilities = np.array([[0.4, 0.3, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25]])
    assert top3_labels(probabilities, classes) == ["A B C", "A B C"]


def test_probability_validation_rejects_bad_rows():
    with pytest.raises(ValueError):
        validate_probabilities(np.array([[0.2, 0.2]]), 1, 2, "bad")
    with pytest.raises(ValueError):
        validate_probabilities(np.array([[1.1, -0.1]]), 1, 2, "bad")


def test_folds_are_deterministic_complete_and_stratified():
    target = pd.Series(list("ABCDEFG") * 20)
    first = make_folds(target, folds=5, seed=326)
    second = make_folds(target, folds=5, seed=326)
    assert np.array_equal(first, second)
    assert set(first) == set(range(5))
    for fold in range(5):
        assert set(target[first == fold]) == set("ABCDEFG")


def test_feature_sets_change_only_the_declared_representation():
    frame = pd.DataFrame({
        "Temparature": [30], "Humidity": [60], "Moisture": [40],
        "Soil Type": ["Sandy"], "Crop Type": ["Wheat"], "Nitrogen": [20],
        "Potassium": [10], "Phosphorous": [15],
    })
    base = apply_feature_set(frame, "base")
    categorical = apply_feature_set(frame, "all_categorical")
    interactions = apply_feature_set(frame, "pair_interactions")
    assert list(base.columns) == list(frame.columns)
    assert all(categorical[column].map(lambda value: isinstance(value, str)).all() for column in categorical)
    assert len(interactions.columns) == len(frame.columns) + 5


def test_target_encoding_uses_only_supplied_training_rows():
    train_x = pd.DataFrame({
        "Temparature": [25, 25, 26, 26], "Humidity": [50, 50, 51, 51],
        "Moisture": [30, 30, 31, 31], "Soil Type": ["A", "A", "B", "B"],
        "Crop Type": ["X", "X", "Y", "Y"], "Nitrogen": [4, 4, 5, 5],
        "Potassium": [0, 0, 1, 1], "Phosphorous": [0, 0, 1, 1],
    })
    train_y = pd.Series(["F1", "F1", "F2", "F2"])
    valid = train_x.iloc[[0]].copy()
    encoded, = target_encode(train_x, train_y, [valid], np.array(["F1", "F2"], dtype=object))
    assert encoded.shape[0] == 1
    assert encoded.filter(like="te_").notna().all().all()
    changed_y = pd.Series(["F2", "F2", "F1", "F1"])
    changed, = target_encode(train_x, changed_y, [valid], np.array(["F1", "F2"], dtype=object))
    assert not np.allclose(encoded.to_numpy(), changed.to_numpy())


def test_training_target_encoding_excludes_each_rows_own_label():
    train_x = pd.DataFrame({
        "Temparature": [25, 26], "Humidity": [50, 51], "Moisture": [30, 31],
        "Soil Type": ["A", "B"], "Crop Type": ["X", "Y"], "Nitrogen": [4, 5],
        "Potassium": [0, 1], "Phosphorous": [0, 1],
    })
    train_y = pd.Series(["F1", "F2"])
    classes = np.array(["F1", "F2"], dtype=object)
    encoded, = target_encode(
        train_x, train_y, [train_x], classes, smoothing=20.0,
        leave_one_out_first=True,
    )
    # Every feature value is unique, so removing the row itself leaves only the
    # shared prior. Its own label must not produce a row-specific boost.
    first_pair = encoded[["te_Temparature_F1", "te_Temparature_F2"]].to_numpy()
    assert np.allclose(first_pair, np.array([[0.5, 0.5], [0.5, 0.5]]))


def test_submission_requires_three_valid_distinct_labels():
    test = pd.DataFrame({"id": [10, 11]})
    classes = np.array(["A", "B", "C", "D"], dtype=object)
    valid = pd.DataFrame({"id": [10, 11], TARGET: ["A B C", "D C B"]})
    validate_submission(valid, test, classes)
    duplicate = pd.DataFrame({"id": [10, 11], TARGET: ["A A C", "D C B"]})
    with pytest.raises(ValueError):
        validate_submission(duplicate, test, classes)


def test_per_class_diagnostics_counts_missing_top3_rows():
    classes = np.array(["A", "B", "C", "D"], dtype=object)
    truth = np.array(["A", "A", "D"], dtype=object)
    probabilities = np.array([
        [0.7, 0.2, 0.1, 0.0],
        [0.1, 0.4, 0.3, 0.2],
        [0.4, 0.3, 0.2, 0.1],
    ])
    rows = {row["label"]: row for row in per_class_diagnostics(truth, probabilities, classes)}
    assert rows["A"]["top1_accuracy"] == 0.5
    assert rows["A"]["top3_recall"] == 0.5
    assert rows["D"]["top1_accuracy"] == 0.0
    assert rows["D"]["top3_recall"] == 0.0

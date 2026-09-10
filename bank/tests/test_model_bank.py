import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_bank import apply_feature_set, lightgbm_parameters, make_folds, ordinal_encode, validate_predictions


class BankFrameworkTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "id": range(20), "duration": range(20), "pdays": [-1, 3] * 10,
            "job": ["a", "b"] * 10, "y": [0] * 16 + [1] * 4,
        })

    def test_feature_sets_change_only_declared_columns(self):
        base = apply_feature_set(self.frame, "base")
        no_duration = apply_feature_set(self.frame, "no_duration")
        previous = apply_feature_set(self.frame, "previous_contacted")
        self.assertEqual(set(base) - {"duration"}, set(no_duration))
        self.assertEqual(set(previous) - {"previous_contacted"}, set(base))
        self.assertListEqual(previous["previous_contacted"].tolist(), [0, 1] * 10)

    def test_stratified_folds_are_complete_and_reproducible(self):
        first = make_folds(self.frame["y"], folds=2, seed=326)
        second = make_folds(self.frame["y"], folds=2, seed=326)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(set(first), {0, 1})

    def test_prediction_validation_rejects_invalid_values(self):
        validate_predictions(np.array([0.1, 0.9]), 2, "valid")
        with self.assertRaises(ValueError):
            validate_predictions(np.array([0.1, np.nan]), 2, "nan")
        with self.assertRaises(ValueError):
            validate_predictions(np.array([1.1]), 1, "range")

    def test_lightgbm_profiles_change_only_the_declared_setting(self):
        baseline, baseline_categorical = lightgbm_parameters("baseline")
        native, native_categorical = lightgbm_parameters("native_categorical")
        native_more_trees, native_more_trees_categorical = lightgbm_parameters("native_cat_more_trees")
        more_leaves, more_leaves_categorical = lightgbm_parameters("native_cat_more_trees_more_leaves")
        larger_child, larger_child_categorical = lightgbm_parameters(
            "native_cat_more_trees_more_leaves_larger_min_child"
        )
        active_bagging, active_bagging_categorical = lightgbm_parameters(
            "native_cat_more_trees_more_leaves_active_bagging"
        )
        trees, trees_categorical = lightgbm_parameters("more_trees")
        self.assertEqual(baseline, native)
        self.assertFalse(baseline_categorical)
        self.assertTrue(native_categorical)
        self.assertTrue(native_more_trees_categorical)
        native_changed = {key for key in native if native[key] != native_more_trees[key]}
        self.assertEqual(native_changed, {"n_estimators"})
        self.assertTrue(more_leaves_categorical)
        leaf_changed = {key for key in native_more_trees if native_more_trees[key] != more_leaves[key]}
        self.assertEqual(leaf_changed, {"num_leaves"})
        self.assertTrue(larger_child_categorical)
        child_changed = {key for key in more_leaves if more_leaves[key] != larger_child[key]}
        self.assertEqual(child_changed, {"min_child_samples"})
        self.assertTrue(active_bagging_categorical)
        bagging_changed = {key for key in more_leaves if more_leaves[key] != active_bagging[key]}
        self.assertEqual(bagging_changed, {"subsample_freq"})
        self.assertFalse(trees_categorical)
        changed = {key for key in baseline if baseline[key] != trees[key]}
        self.assertEqual(changed, {"n_estimators"})

    def test_ordinal_encoder_returns_feature_names_and_categorical_indices(self):
        train = self.frame.drop(columns="y")
        encoded_train, encoded_valid, encoded_test, names, categorical = ordinal_encode(
            train, train.iloc[:4], train.iloc[4:8],
        )
        self.assertEqual(encoded_train.shape, (20, 4))
        self.assertEqual(encoded_valid.shape, (4, 4))
        self.assertEqual(encoded_test.shape, (4, 4))
        self.assertEqual(names, ["id", "duration", "pdays", "job"])
        self.assertEqual(categorical, [3])


if __name__ == "__main__":
    unittest.main()

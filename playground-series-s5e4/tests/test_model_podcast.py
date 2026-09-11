import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_podcast import (
    TARGET,
    anomaly_masks,
    apply_feature_set,
    linear_length_prediction,
    make_folds,
    prepare_fold_features,
    segment_rmse,
    validate_data_contract,
    validate_predictions,
)


class PodcastFrameworkTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({
            "id": [0, 1, 2, 3],
            "Podcast_Name": ["A", "A", "B", "B"],
            "Episode_Title": ["Episode 1", "Episode 2", "Episode 3", "Episode 4"],
            "Episode_Length_minutes": [10.0, np.nan, 121.0, 40.0],
            "Genre": ["News"] * 4,
            "Host_Popularity_percentage": [50.0, 101.0, 30.0, 80.0],
            "Publication_Day": ["Monday"] * 4,
            "Publication_Time": ["Morning"] * 4,
            "Guest_Popularity_percentage": [20.0, np.nan, 105.0, 60.0],
            "Number_of_Ads": [0.0, 1.0, 2.5, 4.0],
            "Episode_Sentiment": ["Neutral"] * 4,
            TARGET: [7.0, 20.0, 80.0, 30.0],
        })
        self.test = self.train.drop(columns=TARGET).copy()
        self.test["id"] += 10
        self.sample = pd.DataFrame({"id": self.test["id"], TARGET: 0.0})

    def test_data_contract_accepts_expected_schema(self):
        validate_data_contract(self.train, self.test, self.sample)
        with self.assertRaises(ValueError):
            validate_data_contract(self.train, self.test, self.sample.iloc[::-1].reset_index(drop=True))

    def test_anomaly_rules_are_explicit(self):
        masks = anomaly_masks(self.train)
        self.assertListEqual(masks["length"].tolist(), [False, False, True, False])
        self.assertListEqual(masks["host"].tolist(), [False, True, False, False])
        self.assertListEqual(masks["guest"].tolist(), [False, False, True, False])
        self.assertListEqual(masks["ads"].tolist(), [False, False, True, True])

    def test_feature_sets_change_only_declared_fields(self):
        base = apply_feature_set(self.train, "base")
        clean = apply_feature_set(self.train, "clean_length")
        changed = base.ne(clean) & ~(base.isna() & clean.isna())
        self.assertEqual(changed.sum().sum(), 1)
        self.assertTrue(pd.isna(clean.loc[2, "Episode_Length_minutes"]))

        numbered = apply_feature_set(self.train, "episode_number")
        self.assertEqual(set(numbered) - {"episode_number"}, set(base))
        self.assertListEqual(numbered["episode_number"].tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_folds_are_complete_balanced_and_reproducible(self):
        first = make_folds(23, folds=5, seed=326)
        second = make_folds(23, folds=5, seed=326)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(set(first), set(range(5)))
        self.assertLessEqual(np.bincount(first).max() - np.bincount(first).min(), 1)

    def test_linear_length_uses_training_median_and_returns_finite_values(self):
        train_x = apply_feature_set(self.train.iloc[:3], "base")
        valid_x = apply_feature_set(self.train.iloc[3:], "base")
        predictions, coefficients = linear_length_prediction(
            train_x, self.train[TARGET].iloc[:3], valid_x,
        )
        self.assertEqual(len(coefficients), 3)
        self.assertEqual(len(predictions[0]), 1)
        self.assertTrue(np.isfinite(predictions[0]).all())

    def test_group_length_imputation_is_fit_on_training_fold(self):
        train_x = apply_feature_set(self.train.iloc[[0, 3]], "length_group_median")
        valid_x = apply_feature_set(self.train.iloc[[1]], "length_group_median")
        test_x = valid_x.copy()
        prepared_train, prepared_valid, prepared_test = prepare_fold_features(
            train_x, valid_x, test_x, "length_group_median",
        )
        self.assertFalse(prepared_train["Episode_Length_minutes"].isna().any())
        self.assertEqual(prepared_valid["Episode_Length_minutes"].iloc[0], 10.0)
        self.assertEqual(prepared_test["Episode_Length_minutes"].iloc[0], 10.0)

    def test_prediction_validation_rejects_nan_and_wrong_length(self):
        validate_predictions(np.array([1.0, 2.0]), 2, "valid")
        with self.assertRaises(ValueError):
            validate_predictions(np.array([1.0, np.nan]), 2, "nan")
        with self.assertRaises(ValueError):
            validate_predictions(np.array([1.0]), 2, "short")

    def test_segment_rmse_handles_missing_length_group(self):
        values = segment_rmse(
            self.train, self.train[TARGET], self.train[TARGET].to_numpy(),
        )
        self.assertIsInstance(values, list)


if __name__ == "__main__":
    unittest.main()

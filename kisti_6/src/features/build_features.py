from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


@dataclass(frozen=True)
class FeatureSchema:
    numeric: list[str]
    categorical: list[str]
    dropped: list[str]

    @property
    def features(self) -> list[str]:
        return self.numeric + self.categorical


def infer_feature_schema(
    frame: pd.DataFrame,
    target: str,
    id_columns: list[str],
) -> FeatureSchema:
    dropped = [column for column in [target, *id_columns] if column in frame.columns]
    candidates = frame.drop(columns=dropped)
    numeric = candidates.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in candidates.columns if column not in numeric]
    if not numeric and not categorical:
        raise ValueError("No usable feature columns remain after dropping target and IDs")
    return FeatureSchema(numeric=numeric, categorical=categorical, dropped=dropped)


def build_preprocessor(schema: FeatureSchema) -> ColumnTransformer:
    transformers = []
    if schema.numeric:
        numeric_pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy="median"))]
        )
        transformers.append(("numeric", numeric_pipeline, schema.numeric))
    if schema.categorical:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        encoded_missing_value=-1,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, schema.categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop")


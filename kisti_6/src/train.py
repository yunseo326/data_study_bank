from __future__ import annotations

from sklearn.pipeline import Pipeline

from src.features import FeatureSchema, build_preprocessor
from src.models import build_baseline_model


def build_training_pipeline(
    schema: FeatureSchema,
    task_type: str,
    model_name: str,
    model_params: dict,
    random_state: int,
) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(schema)),
            (
                "model",
                build_baseline_model(
                    task_type=task_type,
                    model_name=model_name,
                    params=model_params,
                    random_state=random_state,
                ),
            ),
        ]
    )

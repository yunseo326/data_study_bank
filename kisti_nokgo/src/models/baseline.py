from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def build_baseline_model(
    task_type: str,
    model_name: str,
    params: dict[str, Any],
    random_state: int,
):
    if model_name != "hist_gradient_boosting":
        raise ValueError(f"Unsupported baseline model: {model_name}")
    common = {
        "learning_rate": 0.1,
        "max_iter": 50,
        "max_leaf_nodes": 31,
        "l2_regularization": 0.0,
        "early_stopping": True,
        "random_state": random_state,
    }
    common.update(params)
    if task_type == "classification":
        return HistGradientBoostingClassifier(**common)
    if task_type == "regression":
        return HistGradientBoostingRegressor(**common)
    raise ValueError(f"Unknown task type: {task_type}")


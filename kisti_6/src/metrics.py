from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def map_at_k(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    k: int = 3,
) -> float:
    k = min(k, probabilities.shape[1])
    top_indices = np.argsort(-probabilities, axis=1)[:, :k]
    top_labels = classes[top_indices]
    scores = []
    for truth, predictions in zip(y_true, top_labels, strict=True):
        matches = np.flatnonzero(predictions == truth)
        scores.append(0.0 if len(matches) == 0 else 1.0 / (matches[0] + 1))
    return float(np.mean(scores))


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    predictions = classes[np.argmax(probabilities, axis=1)]
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "log_loss": float(log_loss(y_true, probabilities, labels=classes)),
        "map_at_3": map_at_k(y_true, probabilities, classes, k=3),
    }


def regression_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(mean_squared_error(y_true, predictions) ** 0.5),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }


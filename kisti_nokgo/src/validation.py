from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold

from src.metrics import classification_metrics, regression_metrics


@dataclass
class ValidationResult:
    overall_metrics: dict[str, float]
    fold_metrics: list[dict[str, float]]
    predictions: np.ndarray
    classes: np.ndarray | None
    elapsed_seconds: float


def _check_deadline(deadline: float, stage: str) -> None:
    if time.perf_counter() >= deadline:
        raise TimeoutError(f"Time budget exhausted before {stage}")


def cross_validate(
    pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    task_type: str,
    n_splits: int,
    shuffle: bool,
    random_state: int,
    deadline: float,
) -> ValidationResult:
    started = time.perf_counter()
    if task_type == "classification":
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )
        classes = np.sort(target.unique())
        oof = np.zeros((len(target), len(classes)), dtype=float)
        split_iterator = splitter.split(features, target)
    else:
        splitter = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )
        classes = None
        oof = np.zeros(len(target), dtype=float)
        split_iterator = splitter.split(features)

    fold_metrics: list[dict[str, float]] = []
    for fold, (train_indices, valid_indices) in enumerate(split_iterator, start=1):
        _check_deadline(deadline, f"fold {fold}")
        fitted = clone(pipeline)
        fitted.fit(features.iloc[train_indices], target.iloc[train_indices])
        if task_type == "classification":
            fold_probabilities = fitted.predict_proba(features.iloc[valid_indices])
            fold_classes = fitted.named_steps["model"].classes_
            class_positions = {label: index for index, label in enumerate(classes)}
            for source_index, label in enumerate(fold_classes):
                oof[valid_indices, class_positions[label]] = fold_probabilities[:, source_index]
            metrics = classification_metrics(
                target.iloc[valid_indices].to_numpy(),
                oof[valid_indices],
                classes,
            )
        else:
            fold_predictions = fitted.predict(features.iloc[valid_indices])
            oof[valid_indices] = fold_predictions
            metrics = regression_metrics(target.iloc[valid_indices].to_numpy(), fold_predictions)
        metrics["fold"] = float(fold)
        fold_metrics.append(metrics)
        _check_deadline(deadline, f"post-fold {fold}")

    if task_type == "classification":
        overall = classification_metrics(target.to_numpy(), oof, classes)
    else:
        overall = regression_metrics(target.to_numpy(), oof)
    return ValidationResult(
        overall_metrics=overall,
        fold_metrics=fold_metrics,
        predictions=oof,
        classes=classes,
        elapsed_seconds=time.perf_counter() - started,
    )


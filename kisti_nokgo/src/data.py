from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def read_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    return pd.read_csv(file_path)


def infer_target(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    configured_target: str | None = None,
) -> str:
    if configured_target:
        if configured_target not in train.columns:
            raise ValueError(f"Configured target is missing from train: {configured_target}")
        return configured_target
    if test is None:
        raise ValueError("Set data.target when no test data is available")
    candidates = [column for column in train.columns if column not in test.columns]
    if len(candidates) != 1:
        raise ValueError(
            "Target inference requires exactly one train-only column; "
            f"found {candidates}. Set data.target explicitly."
        )
    return candidates[0]


def infer_task_type(target: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(target):
        return "classification"
    unique_count = target.nunique(dropna=True)
    threshold = max(20, int(len(target) * 0.02))
    return "classification" if unique_count <= threshold else "regression"


def infer_id_columns(frame: pd.DataFrame, configured: list[str]) -> list[str]:
    explicit = [column for column in configured if column in frame.columns]
    if explicit:
        return explicit
    inferred = []
    for column in frame.columns:
        normalized = column.strip().lower()
        if normalized in {"id", "index", "row_id"} and frame[column].is_unique:
            inferred.append(column)
    return inferred


def sample_training_data(
    frame: pd.DataFrame,
    target: str,
    max_rows: int,
    task_type: str,
    random_state: int,
) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    stratify = frame[target] if task_type == "classification" else None
    sampled, _ = train_test_split(
        frame,
        train_size=max_rows,
        random_state=random_state,
        stratify=stratify,
    )
    return sampled.reset_index(drop=True)


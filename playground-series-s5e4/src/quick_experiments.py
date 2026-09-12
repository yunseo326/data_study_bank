"""Five-fold screening protocol with a five-minute target per experiment.

This is a direction-finding benchmark, not a replacement for full-data OOF.
It reads train.csv only and keeps its registry and predictions separate from the
formal experiment artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from model_podcast import (
    DATA_DIR,
    EXPERIMENTS,
    ID_COLUMN,
    PROJECT_ROOT,
    SEED,
    TARGET,
    Experiment,
    ModelBudget,
    apply_feature_set,
    fit_fold,
    make_folds,
    prepare_fold_features,
    rmse,
    validate_predictions,
)
from sklearn.model_selection import train_test_split


QUICK_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "quick"
QUICK_REGISTRY_PATH = PROJECT_ROOT / "benchmarks" / "quick_experiments.csv"
QUICK_ROWS = 75_000
QUICK_FOLDS = 5
QUICK_MAX_MINUTES = 5.0
QUICK_MODEL_BUDGET = ModelBudget(
    catboost_iterations=200,
    catboost_early_stopping=30,
    lightgbm_estimators=500,
    lightgbm_early_stopping=40,
    xgboost_estimators=400,
    xgboost_early_stopping=40,
)

QUICK_REGISTRY_FIELDS = [
    "experiment_id", "date", "model", "feature_set", "rows", "folds",
    "cv_rmse", "fold_std", "fold_scores", "runtime_minutes",
    "budget_minutes", "within_budget", "sample_seed", "protocol",
]


def sampling_strata(train: pd.DataFrame) -> pd.Series:
    """Preserve target range and the two large missingness groups in the sample."""
    target_bin = pd.qcut(train[TARGET], q=10, labels=False, duplicates="drop")
    length_missing = train["Episode_Length_minutes"].isna().astype("int8")
    guest_missing = train["Guest_Popularity_percentage"].isna().astype("int8")
    return target_bin.astype(str) + "_L" + length_missing.astype(str) + "_G" + guest_missing.astype(str)


def make_quick_sample_indices(
    train: pd.DataFrame, max_rows: int = QUICK_ROWS, seed: int = SEED,
) -> np.ndarray:
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if len(train) <= max_rows:
        return np.arange(len(train), dtype=np.int64)
    all_indices = np.arange(len(train), dtype=np.int64)
    selected, _ = train_test_split(
        all_indices, train_size=max_rows, random_state=seed,
        shuffle=True, stratify=sampling_strata(train),
    )
    return np.sort(selected)


def upsert_quick_registry(row: dict[str, object]) -> None:
    QUICK_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, str]] = []
    if QUICK_REGISTRY_PATH.exists():
        with QUICK_REGISTRY_PATH.open(encoding="utf-8-sig", newline="") as handle:
            existing = list(csv.DictReader(handle))
    normalized = [
        {field: item.get(field, "") for field in QUICK_REGISTRY_FIELDS}
        for item in existing if item.get("experiment_id") != str(row["experiment_id"])
    ]
    normalized.append({field: row.get(field, "") for field in QUICK_REGISTRY_FIELDS})
    normalized.sort(key=lambda item: item["experiment_id"])
    with QUICK_REGISTRY_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUICK_REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(normalized)


def run_quick_experiment(
    experiment: Experiment,
    max_rows: int = QUICK_ROWS,
    folds: int = QUICK_FOLDS,
    seed: int = SEED,
    max_minutes: float = QUICK_MAX_MINUTES,
    budget: ModelBudget = QUICK_MODEL_BUDGET,
) -> dict:
    started = time.perf_counter()
    stage_seconds: dict[str, float] = {}

    checkpoint = time.perf_counter()
    train = pd.read_csv(DATA_DIR / "train.csv")
    source_rows = len(train)
    stage_seconds["load_train"] = time.perf_counter() - checkpoint

    checkpoint = time.perf_counter()
    sample_indices = make_quick_sample_indices(train, max_rows=max_rows, seed=seed)
    sample = train.iloc[sample_indices].reset_index(drop=True)
    del train
    stage_seconds["stratified_sample"] = time.perf_counter() - checkpoint

    checkpoint = time.perf_counter()
    sample_x = apply_feature_set(sample, experiment.feature_set)
    target = sample[TARGET].astype(float)
    assignments = make_folds(len(sample), folds=folds, seed=seed)
    stage_seconds["features_and_folds"] = time.perf_counter() - checkpoint

    oof = np.zeros(len(sample), dtype=np.float64)
    fold_scores: list[float] = []
    best_iterations: list[int] = []
    fold_timings: list[dict[str, float | int]] = []

    for fold in range(folds):
        fold_started = time.perf_counter()
        valid_mask = assignments == fold
        train_idx = np.flatnonzero(~valid_mask)
        valid_idx = np.flatnonzero(valid_mask)

        prep_started = time.perf_counter()
        fold_train_x, fold_valid_x, _ = prepare_fold_features(
            sample_x.iloc[train_idx], sample_x.iloc[valid_idx], sample_x.iloc[:0],
            experiment.feature_set,
        )
        prepare_seconds = time.perf_counter() - prep_started

        model_started = time.perf_counter()
        valid_prediction, _, best_iteration, _ = fit_fold(
            experiment, fold_train_x, target.iloc[train_idx],
            fold_valid_x, target.iloc[valid_idx], None, seed + fold, budget,
        )
        model_seconds = time.perf_counter() - model_started
        validate_predictions(valid_prediction, len(valid_idx), f"quick fold {fold}")
        oof[valid_idx] = valid_prediction
        score = rmse(target.iloc[valid_idx], valid_prediction)
        fold_scores.append(score)
        best_iterations.append(best_iteration)
        fold_timings.append({
            "fold": fold, "prepare_seconds": prepare_seconds,
            "fit_and_valid_predict_seconds": model_seconds,
            "total_seconds": time.perf_counter() - fold_started,
        })
        print(
            f"quick fold={fold} rmse={score:.6f} best_iteration={best_iteration} "
            f"seconds={fold_timings[-1]['total_seconds']:.1f}",
            flush=True,
        )

    validate_predictions(oof, len(sample), "quick OOF")
    runtime_minutes = (time.perf_counter() - started) / 60
    model_seconds = float(sum(item["fit_and_valid_predict_seconds"] for item in fold_timings))
    total_seconds = runtime_minutes * 60
    stage_seconds["fold_feature_preparation"] = float(sum(item["prepare_seconds"] for item in fold_timings))
    stage_seconds["model_fit_and_valid_prediction"] = model_seconds
    stage_seconds["other"] = max(0.0, total_seconds - sum(stage_seconds.values()))
    bottleneck = max(stage_seconds, key=stage_seconds.get)
    metrics = {
        **asdict(experiment),
        "protocol": "quick-v1",
        "seed": seed,
        "folds": folds,
        "source_rows": source_rows,
        "rows": len(sample),
        "sample_fraction": len(sample) / source_rows,
        "strata": "target_decile × episode_length_missing × guest_popularity_missing",
        "test_prediction": False,
        "model_budget": asdict(budget),
        "cv_rmse": rmse(target, oof),
        "fold_scores": fold_scores,
        "fold_std": float(np.std(fold_scores)),
        "best_iterations": best_iterations,
        "runtime_minutes": runtime_minutes,
        "budget_minutes": max_minutes,
        "within_budget": runtime_minutes <= max_minutes,
        "stage_seconds": stage_seconds,
        "fold_timings": fold_timings,
        "bottleneck": bottleneck,
        "bottleneck_share": stage_seconds[bottleneck] / total_seconds if total_seconds else 0.0,
    }

    output = QUICK_OUTPUT_DIR / experiment.experiment_id
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "oof_predictions.npz", sample_indices=sample_indices,
        folds=assignments, oof=oof,
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    upsert_quick_registry({
        "experiment_id": experiment.experiment_id,
        "date": date.today().isoformat(),
        "model": experiment.model,
        "feature_set": experiment.feature_set,
        "rows": len(sample),
        "folds": folds,
        "cv_rmse": f"{metrics['cv_rmse']:.6f}",
        "fold_std": f"{metrics['fold_std']:.6f}",
        "fold_scores": "|".join(f"{score:.6f}" for score in fold_scores),
        "runtime_minutes": f"{runtime_minutes:.3f}",
        "budget_minutes": f"{max_minutes:.1f}",
        "within_budget": str(metrics["within_budget"]).lower(),
        "sample_seed": seed,
        "protocol": "quick-v1",
    })
    print(json.dumps({
        "experiment_id": experiment.experiment_id,
        "cv_rmse": metrics["cv_rmse"],
        "runtime_minutes": runtime_minutes,
        "within_budget": metrics["within_budget"],
        "bottleneck": bottleneck,
        "bottleneck_share": metrics["bottleneck_share"],
    }, ensure_ascii=False, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--rows", type=int, default=QUICK_ROWS)
    parser.add_argument("--folds", type=int, default=QUICK_FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-minutes", type=float, default=QUICK_MAX_MINUTES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_quick_experiment(
        EXPERIMENTS[args.experiment], max_rows=args.rows, folds=args.folds,
        seed=args.seed, max_minutes=args.max_minutes,
    )


if __name__ == "__main__":
    main()
